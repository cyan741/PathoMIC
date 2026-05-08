"""Build the NCBI taxonomy DAG used by the GNN species channel.

The resulting `.pt` file is a single dict with everything the training-time
``TaxonomyGNN`` needs to operate without any further IO:

  - node2idx          : {taxid_str -> node_idx}                       (dict)     每个节点的整数编号，用于索引节点
  - idx2name          : list[str]   length N                          (list)     每个节点的名称字符串
  - idx2level         : LongTensor[N] in [0..7]                       (tensor)   每个节点的层次级别，0-7分别对应domain, kingdom, phylum, class, order, family, genus, species每节点在 0-7 层级中的位置
  - idx2taxid         : LongTensor[N]                                 (tensor)   每节点的 NCBI taxid
  - edge_index        : LongTensor[2, 2E] (parent->child + reverse)   (tensor)   每个节点的父节点和子节点之间的边，2E表示有2E条边，每条边由两个节点组成，一个是父节点，一个是子节点有向边索引（双向）
  - edge_type         : LongTensor[2E]   (canonical-rank pair index)  (tensor)   每个边的类型，表示父节点和子节点之间的层次关系，0-7分别对应domain, kingdom, phylum, class, order, family, genus, species每节点在 0-7 层级中的位置每条边的类型编码
  - init_features     : FloatTensor[N, 768]   (frozen PubMedBERT)     (tensor)   每节点的冻结 PubMedBERT 向量
  - species2nodeid    : {Target_Species name -> node_idx}             (dict)     名称到节点的快速查找表
  - species_taxid2nodeid : {tax_id (int) -> node_idx}                 (dict)     taxid 到节点的查找表
  - ancestors_per_species : LongTensor[S, 8]                                     每个叶节点在8个层级上的祖先节点编号
        per leaf species, the node index at each canonical rank
        (-1 where NCBI does not assign that rank).                    (tensor)
  - levels            : list[str]   length 8                          (list)

Canonical ranks (index 0 -> 7):
    domain, kingdom, phylum, class, order, family, genus, species

Run once before training (cached on /NAS so it is shared across runs):

    python scripts/build_taxonomy_graph.py \
        --csv_dirs /NAS/luyq/AMP_datasets/splits1 /NAS/luyq/AMP_datasets/splits2 \
        --species_pkl /NAS/luyq/AMP_datasets/species_embeddings.pkl \
        --taxdump_dir /NAS/luyq/AMP_datasets/taxdump \
        --pubmedbert_path /NAS/luyq/huggingface/hub/models--NeuML--pubmedbert-base-embeddings/snapshots/d6eaca8254bc229f3ca42749a5510ae287eb3486 \
        --output /NAS/luyq/AMP_datasets/taxonomy_graph.pt
"""
from __future__ import annotations

import argparse
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import taxopy
from transformers import AutoModel, AutoTokenizer


# Canonical taxonomic ranks we keep (index 0..7).  Other ranks present in NCBI
# (e.g. ``subphylum``, ``species group``) are silently dropped — keeping them
# would make the per-species lineage length variable and wastes capacity on
# ranks that only exist for a few clades.
LEVELS: List[str] = [
    "domain",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]
LEVEL2IDX: Dict[str, int] = {r: i for i, r in enumerate(LEVELS)}
NUM_LEVELS = len(LEVELS)  # 8


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------
def collect_species_from_csvs(csv_dirs: List[str]) -> Dict[str, int]:
    """Return {Target_Species -> Tax_ID} for every species seen in any split."""
    name_to_taxid: Dict[str, int] = {}
    files = ["train.csv", "val.csv", "test.csv", "external_test.csv"]
    for d in csv_dirs:
        for fn in files:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            if "Target_Species" not in df.columns or "Tax_ID" not in df.columns:
                continue
            for sp, tax in zip(df["Target_Species"].astype(str), df["Tax_ID"]):
                if pd.isna(tax):
                    continue
                tax = int(tax)
                if sp not in name_to_taxid:
                    name_to_taxid[sp] = tax
    return name_to_taxid


def load_species_pkl(path: str) -> Dict[str, np.ndarray]:
    """Return {pathogen_name -> 768-d numpy array} from the legacy pkl."""
    with open(path, "rb") as f:
        records = pickle.load(f)
    out = {}
    for r in records:
        emb = np.asarray(r["embedding"]).astype(np.float32)
        out[r["pathogen"]] = emb
    return out


# ---------------------------------------------------------------------------
# Lineage extraction: project NCBI's full-rank lineage onto our 8 canonical
# ranks.  Returns ``[(taxid, name, level_idx)]`` ordered from domain → species,
# skipping ranks that NCBI did not assign for this clade.
# ---------------------------------------------------------------------------
def get_canonical_lineage(taxid: int, taxdb: taxopy.TaxDb):
    try:
        t = taxopy.Taxon(int(taxid), taxdb)
    except (ValueError, KeyError) as exc:
        print(f"  [warn] could not resolve taxid={taxid}: {exc}")
        return None

    rank_to_taxid = t.rank_taxid_dictionary  # OrderedDict, finest -> coarsest
    rank_to_name = t.rank_name_dictionary
    out = []
    for level_idx, rank in enumerate(LEVELS):
        if rank in rank_to_taxid:
            out.append((int(rank_to_taxid[rank]), rank_to_name[rank], level_idx))

    # Sanity: the leaf must be a species rank node and equal the queried id.
    if not out or out[-1][2] != LEVEL2IDX["species"]:
        print(f"  [warn] taxid={taxid} '{t.name}' has rank='{t.rank}' — not 'species'; lineage kept anyway.")
    return out


# ---------------------------------------------------------------------------
# PubMedBERT encoder for taxon names.
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_names_with_pubmedbert(
    names: List[str], model_path: str, batch_size: int = 64, device: str = "cuda:0"
) -> torch.Tensor:
    """Mean-pooled PubMedBERT embedding (768-d) for each input string."""
    if not torch.cuda.is_available():
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path).to(device).eval()
    out = torch.zeros(len(names), mdl.config.hidden_size, dtype=torch.float32)
    for i in range(0, len(names), batch_size):
        chunk = names[i : i + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        h = mdl(**enc).last_hidden_state                       # [b, L, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()     # [b, L, 1]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out[i : i + len(chunk)] = pooled.cpu().float()
    del mdl, tok
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_dirs",
        nargs="+",
        required=True,
        help="One or more directories that each contain train.csv/val.csv/test.csv "
        "(and optionally external_test.csv) with `Target_Species` and `Tax_ID` columns.",
    )
    ap.add_argument(
        "--species_pkl",
        type=str,
        default="/NAS/luyq/AMP_datasets/species_embeddings.pkl",
        help="Existing PubMedBERT-of-wiki-text per species (520 entries). "
        "Used to initialise leaf-species nodes when available.",
    )
    ap.add_argument(
        "--taxdump_dir",
        type=str,
        default="/NAS/luyq/AMP_datasets/taxdump",
        help="Directory holding nodes.dmp / names.dmp / merged.dmp (NCBI taxdump).",
    )
    ap.add_argument(
        "--pubmedbert_path",
        type=str,
        default="/NAS/luyq/huggingface/hub/models--NeuML--pubmedbert-base-embeddings/snapshots/d6eaca8254bc229f3ca42749a5510ae287eb3486",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="/NAS/luyq/AMP_datasets/taxonomy_graph.pt",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Collect species set from CSVs.
    # ------------------------------------------------------------------
    print("[1/6] collecting species from CSVs ...")
    sp_name_to_taxid = collect_species_from_csvs(args.csv_dirs)
    print(f"   -> {len(sp_name_to_taxid)} unique species across all splits")

    # ------------------------------------------------------------------
    # 2. Build TaxDb from the local NCBI dump.
    # ------------------------------------------------------------------
    print("[2/6] loading NCBI taxdump ...")
    taxdb = taxopy.TaxDb(
        nodes_dmp=os.path.join(args.taxdump_dir, "nodes.dmp"),
        names_dmp=os.path.join(args.taxdump_dir, "names.dmp"),
        merged_dmp=os.path.join(args.taxdump_dir, "merged.dmp"),
        keep_files=True,
    )

    # ------------------------------------------------------------------
    # 3. Project each species' full lineage onto the 8 canonical ranks.
    #    Build the (deduplicated) node set, edges, and per-species ancestor
    #    table simultaneously.
    # ------------------------------------------------------------------
    print("[3/6] building taxonomy DAG ...")
    node2idx: Dict[int, int] = OrderedDict()       # taxid -> node_idx
    idx2name: List[str] = []
    idx2level: List[int] = []
    idx2taxid: List[int] = []

    def add_node(taxid: int, name: str, level_idx: int) -> int:
        if taxid in node2idx:
            return node2idx[taxid]
        nid = len(idx2name)
        node2idx[taxid] = nid
        idx2name.append(name)
        idx2level.append(level_idx)
        idx2taxid.append(taxid)
        return nid

    edge_set: set = set()      # (src, dst, edge_type)
    species2nodeid: Dict[str, int] = {}
    species_taxid2nodeid: Dict[int, int] = {}
    ancestors_rows: List[List[int]] = []   # parallel to species_names_sorted
    species_names_sorted: List[str] = []

    skipped = 0
    for sp_name, tax in sorted(sp_name_to_taxid.items()):
        lineage = get_canonical_lineage(tax, taxdb)
        if lineage is None:
            skipped += 1
            continue

        # Add every node in the lineage.
        path_nids: List[Tuple[int, int]] = []   # (level_idx, node_idx)
        for tid, name, level_idx in lineage:
            nid = add_node(tid, name, level_idx)
            path_nids.append((level_idx, nid))

        # Connect consecutive (in canonical order) lineage nodes.
        for (lvl_p, nid_p), (lvl_c, nid_c) in zip(path_nids[:-1], path_nids[1:]):
            # encode the edge type as level pair (parent rank, child rank).
            etype = lvl_p * NUM_LEVELS + lvl_c
            edge_set.add((nid_p, nid_c, etype))

        # Per-species ancestor row at each canonical rank, -1 if absent.
        ancestor_row = [-1] * NUM_LEVELS
        for lvl_idx, nid in path_nids:
            ancestor_row[lvl_idx] = nid
        ancestors_rows.append(ancestor_row)
        species_names_sorted.append(sp_name)

        leaf_nid = path_nids[-1][1]
        species2nodeid[sp_name] = leaf_nid
        species_taxid2nodeid[int(tax)] = leaf_nid

    if skipped:
        print(f"   [warn] skipped {skipped} species whose taxid could not be resolved")
    N = len(idx2name)
    E = len(edge_set)
    print(f"   nodes={N}, unique parent->child edges={E}")
    # Diagnostics: how many nodes per level.
    level_counts = [0] * NUM_LEVELS
    for lvl in idx2level:
        level_counts[lvl] += 1
    for lvl_name, c in zip(LEVELS, level_counts):
        print(f"     level {lvl_name:>9s}: {c:5d} nodes")

    # Materialise edge tensors (directed parent->child, then add reverse).
    src = []
    dst = []
    etypes = []
    for s, d, t in edge_set:
        src.append(s); dst.append(d); etypes.append(t)
        src.append(d); dst.append(s); etypes.append(t)   # reverse, same type
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etypes, dtype=torch.long)

    # ------------------------------------------------------------------
    # 4. Decide each node's PubMedBERT initial vector.
    #    - leaf species nodes whose name is in species_pkl: reuse pkl vector
    #      (encodes wiki / genus-mean / type-mean as in the existing baseline).
    #    - everything else: encode the taxon name with PubMedBERT online.
    # ------------------------------------------------------------------
    print("[4/6] preparing per-node initial features ...")
    pkl_map = load_species_pkl(args.species_pkl)
    # We will batch-encode ONLY the nodes that don't get their vector from pkl.
    needs_encoding_idx: List[int] = []
    needs_encoding_text: List[str] = []
    init_features = torch.zeros(N, 768, dtype=torch.float32)
    n_from_pkl = 0
    for nid, (name, lvl) in enumerate(zip(idx2name, idx2level)):
        if lvl == LEVEL2IDX["species"] and name in pkl_map:
            init_features[nid] = torch.from_numpy(pkl_map[name])
            n_from_pkl += 1
        else:
            needs_encoding_idx.append(nid)
            # Light decoration so PubMedBERT gets a hint about rank context.
            needs_encoding_text.append(f"{LEVELS[lvl]} {name}")
    print(f"   {n_from_pkl} species nodes initialised from species_embeddings.pkl")
    print(f"   {len(needs_encoding_idx)} nodes will be encoded with PubMedBERT")

    # ------------------------------------------------------------------
    # 5. Run PubMedBERT once on all the remaining names.
    # ------------------------------------------------------------------
    print("[5/6] encoding taxon names with PubMedBERT ...")
    if needs_encoding_idx:
        enc = encode_names_with_pubmedbert(
            needs_encoding_text, args.pubmedbert_path, device=args.device
        )
        for k, nid in enumerate(needs_encoding_idx):
            init_features[nid] = enc[k]

    # ------------------------------------------------------------------
    # 6. Save everything.
    # ------------------------------------------------------------------
    print("[6/6] saving graph ...")
    ancestors_per_species = torch.tensor(ancestors_rows, dtype=torch.long)
    out = {
        "node2idx": {str(k): v for k, v in node2idx.items()},   # taxid stringified for JSON-friendliness
        "idx2name": idx2name,
        "idx2level": torch.tensor(idx2level, dtype=torch.long),
        "idx2taxid": torch.tensor(idx2taxid, dtype=torch.long),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "init_features": init_features,
        "species2nodeid": species2nodeid,
        "species_taxid2nodeid": species_taxid2nodeid,
        "ancestors_per_species": ancestors_per_species,
        "species_names": species_names_sorted,
        "levels": LEVELS,
        "num_edge_types": NUM_LEVELS * NUM_LEVELS,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(out, args.output)
    print(f"saved -> {args.output}")
    print(f"   N={N} | E(directed*2)={edge_index.size(1)} | leaves={len(species2nodeid)}")


if __name__ == "__main__":
    main()
