import torch
import torch.nn as nn
from PLM_head import MLP, SpeciesAdapter
from gnn_module import build_species_encoder_from_graph
from fusion_modules import build_fusion
# from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoModel, AutoTokenizer, AlbertTokenizer


# ---------------------------------------------------------------------------
# Deep prefix tuning helper: produces per-layer (K, V) prefix tensors from a
# small re-parameterisation MLP applied to the per-species-level token reps
# produced by the GNN. We rely on HF's `past_key_values` mechanism (already
# present in EsmSelfAttention.forward) so the prefix is *not* prepended to
# the input embeddings at all -- it is concatenated to the K/V of every layer.
# Query length stays at L (peptide only), so hidden_states output is [B,L,H].
# ---------------------------------------------------------------------------
class EsmPrefixKVEncoder(nn.Module):
    """Map species-level tokens [B,N,H] -> per-layer (K,V) prefixes.

    Output shape: list of length ``num_layers`` of tuples ``(K_l, V_l)`` with
    ``K_l`` and ``V_l`` of shape ``[B, num_heads, N, head_dim]`` (already
    transposed for HF EsmSelfAttention which expects post-``transpose_for_scores``
    layout for cached KV).
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int,
                 prefix_kv_hidden: int = 512):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, prefix_kv_hidden),
            nn.Tanh(),
            nn.Linear(prefix_kv_hidden, num_layers * 2 * hidden_size),
        )

    def forward(self, species_tokens: torch.Tensor):
        """species_tokens: [B, N, H] -> list[(K_l, V_l)] of length num_layers."""
        B, N, H = species_tokens.shape
        out = self.mlp(species_tokens)                                       # [B, N, L*2*H]
        out = out.view(B, N, self.num_layers, 2, self.num_heads, self.head_dim)
        # rearrange -> [num_layers, 2, B, num_heads, N, head_dim]
        out = out.permute(2, 3, 0, 4, 1, 5).contiguous()
        return [(out[l, 0], out[l, 1]) for l in range(self.num_layers)]


# Allowed values for ESM2(species_mode=...).
# - 'none'    : no species channel (peptide_emb only).
# - 'adapter' : legacy non-linear adapter on a pre-computed 768-d species
#               vector (original baseline in this repo).
# - 'gnn'     : taxonomy GNN channel, returns a per-batch [B, gnn_out_dim] vec.
# - 'both'    : both 'adapter' and 'gnn' channels concatenated with peptide_emb.
SPECIES_MODES = ("none", "adapter", "gnn", "both")


"""
Check your path of pre-trained models
"""
#### from huggingface directly
cache_dir = "/NAS/luyq/huggingface/hub/"
protbert_checkpoint = "Rostlab/prot_bert"   # "Rostlab/prot_bert_bfd"
amplify_120m_checkpoint = "chandar-lab/AMPLIFY_120M"
amplify_120m_base_checkpoint = "chandar-lab/AMPLIFY_120M_base"
amplify_350m_checkpoint = "chandar-lab/AMPLIFY_350M"
esm2_650m_checkpoint = "facebook/esm2_t33_650M_UR50D"

#### local (downloaded from huggingface website)
protalbert_checkpoint = "/NAS/luyq/huggingface/hub/prot_albert"
esm2_8m_checkpoint = "/NAS/luyq/huggingface/hub/esm2_t6_8M_UR50D"
esm2_35m_checkpoint = "/NAS/luyq/huggingface/hub/esm2_t12_35M_UR50D"
esm2_150m_checkpoint = "/NAS/luyq/huggingface/hub/esm2_t30_150M_UR50D"
esm2_650m_checkpoint = "/NAS/luyq/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
esm2_3b_checkpoint = "/NAS/luyq/esm2_models/esm2_t36_3B_UR50D"


'''
Tokenizer
'''
def load_tokenizer(plm_type):
    if plm_type == "tape":
        tokenizer = TAPETokenizer(vocab='iupac')
    elif plm_type in ['protbert', 'protalbert']:
        if plm_type == 'protbert':
            tokenizer = AutoTokenizer.from_pretrained(protbert_checkpoint, cache_dir=cache_dir,
                                                    do_lower_case=False)
        elif plm_type == 'protalbert':
            tokenizer = AlbertTokenizer.from_pretrained(protalbert_checkpoint,
                                                    do_lower_case=False)
    elif "esm2" in plm_type:
        if plm_type.split('-')[-1] == '8M':
            tokenizer = AutoTokenizer.from_pretrained(esm2_8m_checkpoint)
        elif plm_type.split('-')[-1] == '35M':      # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_35m_checkpoint)
        elif plm_type.split('-')[-1] == '150M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_150m_checkpoint)
        elif plm_type.split('-')[-1] == '650M':     # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_650m_checkpoint, cache_dir=cache_dir)
        elif plm_type.split('-')[-1] == '3B':       # actually same as 8M
            tokenizer = AutoTokenizer.from_pretrained(esm2_3b_checkpoint)
    elif "AMPLIFY" in plm_type:
        if plm_type == 'AMPLIFY-120M':
            tokenizer = AutoTokenizer.from_pretrained(amplify_120m_checkpoint, cache_dir=cache_dir)
        elif plm_type == 'AMPLIFY-120M-base':
            tokenizer = AutoTokenizer.from_pretrained(amplify_120m_base_checkpoint, cache_dir=cache_dir)
        elif plm_type == 'AMPLIFY-350M':
            tokenizer = AutoTokenizer.from_pretrained(amplify_350m_checkpoint, cache_dir=cache_dir)
    return tokenizer


'''
TAPE model
'''
class TAPE(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(TAPE, self).__init__()
        self.tape = ProteinBertModel.from_pretrained('bert-base')
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        
        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.tape.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(768, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(768, [1024, 512, 128, 32, 2])  ## 5layers

    def forward(self, inputs):
        if self.plm_output == 'mean':  # mean of sequence_output
            outputs = self.tape(inputs)[
                0]  # [batch_size, , seq_len,  hidden_size]
            outputs = torch.mean(outputs, dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':  # output of every sample's <cls> token
            outputs = self.tape(
                inputs, input_mask=None)[0][:, 0]  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        return outputs.view(-1, outputs.size(-1))


'''
ProtBert model
'''
class ProtBert(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(ProtBert, self).__init__()
        self.proteinbert = AutoModel.from_pretrained(protbert_checkpoint, cache_dir=cache_dir)

        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.proteinbert.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(1024, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(1024, [1024, 512, 128, 32, 2])  ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.proteinbert(input_ids)

        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))


'''
ProtAlBert model
'''
class ProtAlBert(nn.Module):
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True):
        super(ProtAlBert, self).__init__()
        self.proteinbert = AutoModel.from_pretrained(protalbert_checkpoint)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.proteinbert.parameters():
                param.requires_grad = False

        if head_type == '3MLP':
            self.projection = MLP(4096, [256, 64, 2])               ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(4096, [1024, 512, 128, 32, 2])    ## 5layers
        self.plm_output = plm_output

    def forward(self, input_ids):
        outputs = self.proteinbert(input_ids)

        if self.plm_output == 'mean':
            outputs = outputs[0].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs[0][:, 0]
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))


'''
ESM2 family
'''
class ESM2(nn.Module):
    """ESM2 + optional species side-channel.

    Backwards-compatible wrt. ``use_species`` (legacy bool flag for the
    PubMedBERT-adapter baseline). The new ``species_mode`` argument supersedes
    it and supports four configurations (see ``SPECIES_MODES``).
    """

    def __init__(self,
                 head_type='3MLP',
                 plm_output='mean',
                 finetune_plm=True,
                 esm_size='8M',
                 # ----- legacy adapter knobs (kept for compatibility) -----
                 use_species=False,
                 species_in_dim=768,
                 species_out_dim=128,
                 species_bottleneck=128,
                 species_dropout=0.1,
                 # ----- new species-mode dispatcher -----
                 species_mode=None,
                 # ----- GNN species-channel knobs -----
                 taxo_graph_path=None,
                 gnn_hidden=128,
                 gnn_out_dim=64,
                 gnn_layers=2,
                 gnn_type='gcn',
                 gnn_heads=4,
                 gnn_dropout=0.1,
                gnn_fusion='leaf',
                gnn_hier_levels=('species', 'genus', 'family'),
                # ----- gate_hard fusion (conditional aggregation) -----
                # Species with train count >= gnn_gate_count_threshold take an
                # adapter-identical passthrough branch; the rest use GCN.
                gnn_gate_count_threshold=100,
                species_train_counts=None,
                gnn_freeze_init=True,
                 use_lora_init=False,
                 lora_rank=16,
                 gnn_residual=False,
                 gnn_layernorm=False,
                 # ----- ESM<->GNN fusion strategy (Stage 3) -----
                 fusion_strategy='concat',
                 # ----- Species injection point (Stage 3b) -----
                 # 'post'  : default; GNN species_emb fused with ESM output (existing path)
                 # 'prefix': GNN level_embs prepended as N species tokens to ESM input
                 #           embeddings; no late fusion. Requires gnn_out_dim == self.hidden_size.
                 species_inject='post',
                 # When species_inject='prefix', pool only over peptide tokens
                 # ('peptide') or over all tokens including the species prefix ('all').
                 prefix_pool='peptide',
                 # When species_inject='prefix': 'shallow' = prepend prefix tokens
                 # to the input embeddings (existing _forward_prefix); 'deep' =
                 # inject per-layer (K,V) prefixes (Prefix-Tuning v2). 'deep' uses
                 # HF's past_key_values mechanism so prefix never appears in the
                 # hidden states output (query length stays = peptide length).
                 prefix_depth='shallow',
                 prefix_kv_hidden=512,
                 # ----- Backbone tuning regime ----------
                 freeze_esm=False,
                 use_lora=False,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05,
                 lora_target=('query', 'value')):
        super(ESM2, self).__init__()
        if esm_size == '8M':
            self.checkpoint = esm2_8m_checkpoint
            self.hidden_size = 320
        elif esm_size == '35M':
            self.checkpoint = esm2_35m_checkpoint
            self.hidden_size = 480
        elif esm_size == '150M':
            self.checkpoint = esm2_150m_checkpoint
            self.hidden_size = 640
        elif esm_size == '650M':
            self.checkpoint = esm2_650m_checkpoint
            self.hidden_size = 1280
        elif esm_size == '3B':
            self.checkpoint = esm2_3b_checkpoint
            self.hidden_size = 2560
        else:
            raise ValueError(f"Wrong size of ESM2: {esm_size}")
        self.esm = AutoModel.from_pretrained(self.checkpoint, cache_dir=cache_dir)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm

        # Resolve species_mode. If unset, fall back to the legacy bool so that
        # existing CLI invocations (--use_species) still work unchanged.
        if species_mode is None:
            species_mode = "adapter" if use_species else "none"
        if species_mode not in SPECIES_MODES:
            raise ValueError(f"species_mode must be one of {SPECIES_MODES}, got {species_mode!r}")
        self.species_mode = species_mode
        self.use_species = species_mode != "none"   # kept for backward compat
        print(self.plm_output, self.hidden_size, "species_mode=", species_mode)

        # Freeze the parameters of the PLM if either finetune_plm=False or
        # the explicit --freeze_esm flag is set. LoRA mode also requires a
        # frozen backbone (the LoRA adapter modules are the only trainable
        # path into the ESM layers).
        self.freeze_esm = bool(freeze_esm) or (not finetune_plm) or bool(use_lora)
        if self.freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

        # ----- species channels --------------------------------------------------
        # 'adapter': legacy 768-d -> species_out_dim non-linear adapter.
        if species_mode in ("adapter", "both"):
            self.species_adapter = SpeciesAdapter(
                in_dim=species_in_dim,
                bottleneck=species_bottleneck,
                out_dim=species_out_dim,
                dropout=species_dropout,
            )
        else:
            self.species_adapter = None

        # 'gnn': taxonomy-DAG GNN -> [B, gnn_out_dim].
        if species_mode in ("gnn", "both"):
            if taxo_graph_path is None:
                raise ValueError(
                    "species_mode='%s' requires taxo_graph_path to be provided "
                    "(produced by scripts/build_taxonomy_graph.py)." % species_mode
                )
            if gnn_fusion == "gate_hard" and gnn_out_dim != species_out_dim:
                raise ValueError(
                    f"gnn_fusion='gate_hard' requires gnn_out_dim ({gnn_out_dim}) == "
                    f"species_out_dim ({species_out_dim}) so the passthrough branch "
                    f"matches the species adapter channel width. Pass "
                    f"--gnn_out_dim {species_out_dim}."
                )
            self.species_gnn = build_species_encoder_from_graph(
                taxo_graph_path,
                hidden=gnn_hidden,
                out_dim=gnn_out_dim,
                num_layers=gnn_layers,
                gnn_type=gnn_type,
                heads=gnn_heads,
                dropout=gnn_dropout,
                fusion=gnn_fusion,
                hier_levels=gnn_hier_levels,
                freeze_init=gnn_freeze_init,
                use_lora_init=use_lora_init,
                lora_rank=lora_rank,
                use_residual=gnn_residual,
                use_layernorm=gnn_layernorm,
                species_train_counts=species_train_counts,
                gate_count_threshold=gnn_gate_count_threshold,
                adapter_bottleneck=species_bottleneck,
                adapter_dropout=species_dropout,
            )
        else:
            self.species_gnn = None
        # cached for forward ergonomics
        # When the species encoder reports a different width than the CLI
        # ``gnn_out_dim`` (true for ``hier_raw`` where out_dim is widened to
        # out_dim * len(hier_levels)), trust the encoder.
        if self.species_gnn is not None:
            self.gnn_out_dim = self.species_gnn.out_dim
        else:
            self.gnn_out_dim = gnn_out_dim
        self.species_out_dim = species_out_dim
        self.fusion_strategy = fusion_strategy
        self.species_inject = species_inject
        self.prefix_pool = prefix_pool

        if species_inject not in ("post", "prefix"):
            raise ValueError(f"species_inject must be 'post' or 'prefix', got {species_inject!r}")
        if prefix_pool not in ("peptide", "all"):
            raise ValueError(f"prefix_pool must be 'peptide' or 'all', got {prefix_pool!r}")

        # ----- Stage 3b: species_inject='prefix' setup --------------------------
        # When prefix mode is on, GCN's per-node output width must match ESM's
        # hidden width so we can prepend species tokens directly.
        if prefix_depth not in ("shallow", "deep"):
            raise ValueError(f"prefix_depth must be 'shallow' or 'deep', got {prefix_depth!r}")
        self.prefix_depth = prefix_depth
        self.prefix_kv_hidden = prefix_kv_hidden
        if species_inject == "prefix":
            if species_mode not in ("gnn", "both"):
                raise ValueError(
                    f"species_inject='prefix' requires species_mode in ('gnn','both'); got {species_mode!r}"
                )
            # forward_levels() returns [B, L, per_node_out_dim]; per_node_out_dim must == hidden_size.
            per_node = getattr(self.species_gnn.gnn, "out_dim", gnn_out_dim)
            if per_node != self.hidden_size:
                raise ValueError(
                    f"species_inject='prefix' requires the GCN per-node out_dim ({per_node}) to "
                    f"equal ESM hidden_size ({self.hidden_size}). Pass --gnn_out_dim {self.hidden_size}."
                )
            n_levels = len(self.species_gnn.hier_levels) if self.species_gnn.hier_levels else 0
            if n_levels == 0:
                raise ValueError(
                    "species_inject='prefix' requires gnn_fusion in {hier, hier_attn, hier_raw} "
                    "so the species encoder can expose per-level embeddings."
                )
            self.n_prefix_tokens = n_levels
            self.level_type_emb = nn.Parameter(torch.zeros(n_levels, self.hidden_size))
            nn.init.normal_(self.level_type_emb, std=0.02)

            if prefix_depth == "deep":
                cfg = self.esm.config
                self.prefix_kv_encoder = EsmPrefixKVEncoder(
                    hidden_size=self.hidden_size,
                    num_layers=cfg.num_hidden_layers,
                    num_heads=cfg.num_attention_heads,
                    prefix_kv_hidden=prefix_kv_hidden,
                )
                print(f"[ESM2] deep prefix-tuning enabled: "
                      f"L={cfg.num_hidden_layers} layers x N={n_levels} tokens "
                      f"x ({cfg.num_attention_heads}h x {self.hidden_size // cfg.num_attention_heads}d) "
                      f"via MLP(H={self.hidden_size} -> {prefix_kv_hidden} -> "
                      f"{cfg.num_hidden_layers*2*self.hidden_size})")
            else:
                self.prefix_kv_encoder = None
        else:
            self.n_prefix_tokens = 0
            self.level_type_emb = None
            self.prefix_kv_encoder = None
            if prefix_depth == "deep":
                raise ValueError("prefix_depth='deep' requires species_inject='prefix'.")

        # ----- ESM<->GNN fusion module (Stage 3) -----
        # Only relevant when species_mode includes 'gnn'. Suppressed entirely
        # in prefix-injection mode (no late fusion: species info enters via
        # the prefix tokens only).
        if (species_mode in ("gnn", "both")
                and fusion_strategy != "concat"
                and species_inject != "prefix"):
            self.fusion = build_fusion(
                fusion_strategy,
                esm_dim=self.hidden_size,
                gnn_out_dim=self.gnn_out_dim,
                dropout=gnn_dropout,
            )
        else:
            self.fusion = None

        # ----- compute the input dim of the regression head ---------------------
        if species_inject == "prefix":
            # Pure prefix injection: head sees only the pooled ESM output.
            mlp_in = self.hidden_size
            if species_mode in ("adapter", "both"):
                mlp_in += species_out_dim
        elif species_mode in ("gnn", "both") and self.fusion is not None:
            # Non-concat fusions output a fixed-width vector (= esm_dim).
            mlp_in = self.fusion.out_dim
            if species_mode == "both":
                mlp_in += species_out_dim
        else:
            mlp_in = self.hidden_size
            if species_mode in ("adapter", "both"):
                mlp_in += species_out_dim
            if species_mode in ("gnn", "both"):
                mlp_in += self.gnn_out_dim

        # Last layer has NO activation (the MLP class breaks before adding one)
        # so the scalar output is unrestricted -- essential for log-MIC values
        # that can be negative.
        if head_type == '3MLP':
            self.projection = MLP(mlp_in, [256, 64, 1])
        elif head_type == '5MLP':
            self.projection = MLP(mlp_in, [1024, 512, 128, 32, 1])
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        # ----- Optional LoRA adapter on the ESM backbone ----------------------
        # In-place replacement: peft swaps the matching nn.Linear modules with
        # LoraLinear wrappers; base weights stay frozen (already done above)
        # while the new low-rank adapters are trainable. We KEEP a reference
        # to the underlying EsmModel as ``self._esm_base`` so the shallow-prefix
        # forward path can still call .embeddings / .encoder / get_extended_attention_mask
        # directly. LoraLinear modifies the matching Linear modules in-place,
        # so the same forward path automatically picks them up.
        self.use_lora = bool(use_lora)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target = tuple(lora_target) if not isinstance(lora_target, str) else tuple(
            s.strip() for s in lora_target.split(",") if s.strip()
        )
        self._esm_base = self.esm
        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as exc:
                raise ImportError(
                    "`peft` package is required when use_lora=True. "
                    "Install with `pip install peft`."
                ) from exc
            lora_cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=list(self.lora_target),
                task_type=None,
            )
            self.esm = get_peft_model(self.esm, lora_cfg)
            # peft only marks LoRA params trainable; everything else stays frozen.
            # Refresh _esm_base to point at the unwrapped EsmModel for direct
            # encoder/.embeddings access in the shallow-prefix path.
            self._esm_base = self.esm.get_base_model()
            n_trainable = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
            n_total     = sum(p.numel() for p in self.esm.parameters())
            print(f"[ESM2] LoRA enabled: r={self.lora_r}, alpha={self.lora_alpha}, "
                  f"target_modules={self.lora_target} | trainable {n_trainable/1e6:.2f}M / "
                  f"total {n_total/1e6:.2f}M ESM params")

    def forward(self, input_ids, species_emb=None, species_ids=None, species_names=None,
                attention_mask=None):
        """
        input_ids       : LongTensor [B, L]
        species_emb     : FloatTensor [B, species_in_dim] -- 'adapter' / 'both' modes
        species_ids     : LongTensor [B] of species rows in the GNN graph,
                          OR a Python list of species name strings -- 'gnn' / 'both' modes
        species_names   : alias for ``species_ids`` for readability when passing names.
        attention_mask  : optional [B, L] mask (1=keep, 0=pad). When None we
                          infer it from input_ids != pad_token_id.
        """
        # -------- prefix-injection path (Stage 3b) -----------------------
        if self.species_inject == "prefix":
            if self.prefix_depth == "deep":
                return self._forward_deep_prefix(input_ids, species_emb,
                                                 species_ids, species_names,
                                                 attention_mask)
            return self._forward_prefix(input_ids, species_emb, species_ids,
                                        species_names, attention_mask)

        # cross_attn fusion needs token-level output; for the rest, mean/cls is enough
        need_tokens = (self.fusion is not None) and (self.fusion_strategy == "cross_attn")
        outputs = self.esm(input_ids)
        seq_tokens = outputs[0] if need_tokens else None         # [B, L, hidden_size]
        if self.plm_output == 'mean':
            seq_rep = outputs[0].mean(dim=1)        # [B, hidden_size]
        elif self.plm_output == 'cls':
            seq_rep = outputs[0][:, 0]              # [B, hidden_size]
        else:
            raise ValueError(f"Unknown plm_output: {self.plm_output}")

        # ----- adapter branch (legacy) -----
        adapter_emb = None
        if self.species_mode in ("adapter", "both"):
            if species_emb is None:
                raise ValueError(f"species_mode={self.species_mode!r} but species_emb is None.")
            adapter_emb = self.species_adapter(species_emb)             # [B, species_out_dim]

        # ----- GNN branch -----
        gnn_emb = None
        if self.species_mode in ("gnn", "both"):
            sp_in = species_ids if species_ids is not None else species_names
            if sp_in is None:
                raise ValueError(
                    f"species_mode={self.species_mode!r} but neither species_ids nor "
                    f"species_names was provided."
                )
            gnn_emb = self.species_gnn(sp_in)                           # [B, gnn_out_dim]

        # ----- fuse ESM + GNN -----
        if gnn_emb is not None and self.fusion is not None:
            seq_fused = self.fusion(seq_rep, gnn_emb, seq_tokens=seq_tokens)
            if adapter_emb is not None:
                fused = torch.cat([seq_fused, adapter_emb], dim=-1)
            else:
                fused = seq_fused
        else:
            feats = [seq_rep]
            if adapter_emb is not None:
                feats.append(adapter_emb)
            if gnn_emb is not None:
                feats.append(gnn_emb)
            fused = feats[0] if len(feats) == 1 else torch.cat(feats, dim=-1)

        out = self.projection(fused)
        return out.view(-1, out.size(-1))

    # ------------------------------------------------------------------
    def _forward_prefix(self, input_ids, species_emb, species_ids,
                        species_names, attention_mask):
        """Pure prefix-injection forward.

        HF ``EsmModel.forward`` cannot accept both ``input_ids`` and
        ``inputs_embeds`` (raises), and its ``EsmEmbeddings`` token-dropout
        branch references ``input_ids`` even when ``inputs_embeds`` is given.
        We therefore drive the embeddings + encoder layers manually so that
        we can splice the species prefix into the token stream cleanly while
        still reproducing ESM2's token_dropout / position / layernorm steps
        for the peptide portion (so the peptide stays in-distribution).

        Layout (N = num hier levels, L = peptide length):
            tokens 0 .. N-1            : species prefix tokens (no position info)
            tokens N .. N+L-1          : peptide amino-acid tokens
        Pooling depends on ``self.prefix_pool``.
        """
        sp_in = species_ids if species_ids is not None else species_names
        if sp_in is None:
            raise ValueError("species_inject='prefix' requires species_ids or species_names")

        # 1) GCN per-level embeddings + level_type_emb -> species prefix tokens.
        level_embs = self.species_gnn.forward_levels(sp_in)            # [B, N, H]
        sp_tokens = level_embs + self.level_type_emb.unsqueeze(0)      # [B, N, H]
        B, N, H = sp_tokens.shape

        # 2) Peptide word embeddings.
        esm_base = self._esm_base
        embed_layer = esm_base.embeddings
        word_embeds = embed_layer.word_embeddings(input_ids)           # [B, L, H]
        L = word_embeds.size(1)

        # 3) Build / extend the attention mask.
        if attention_mask is None:
            pad_id = getattr(esm_base.config, "pad_token_id", 1)
            attention_mask = (input_ids != pad_id).long()              # [B, L]
        ones = torch.ones(B, N, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask_ext = torch.cat([ones, attention_mask], dim=1)   # [B, N+L]

        # 4) ESM2 token_dropout on the PEPTIDE only (prefix has no mask tokens).
        if embed_layer.token_dropout:
            mask_id = embed_layer.mask_token_id
            word_embeds = word_embeds.masked_fill(
                (input_ids == mask_id).unsqueeze(-1), 0.0
            )
            src_lengths = attention_mask.sum(-1).float()
            mask_ratio_observed = (input_ids == mask_id).sum(-1).float() / src_lengths.clamp(min=1)
            mask_ratio_train = 0.15 * 0.8
            scale = (1.0 - mask_ratio_train) / (1.0 - mask_ratio_observed).clamp(min=1e-6)
            word_embeds = (word_embeds * scale[:, None, None]).to(word_embeds.dtype)

        # 5) Positional embeddings (absolute). Pin prefix to padding_idx (no
        #    position info, which leaves the peptide's pretrained position
        #    layout undisturbed); peptide gets HF's standard scheme.
        if embed_layer.position_embedding_type == "absolute":
            from transformers.models.esm.modeling_esm import create_position_ids_from_input_ids
            pad_idx = embed_layer.padding_idx
            pep_pos_ids = create_position_ids_from_input_ids(input_ids, pad_idx)  # [B, L]
            pep_pos_emb = embed_layer.position_embeddings(pep_pos_ids)            # [B, L, H]
            word_embeds = word_embeds + pep_pos_emb

            prefix_pos_ids = torch.full((B, N), pad_idx, dtype=torch.long, device=input_ids.device)
            prefix_pos_emb = embed_layer.position_embeddings(prefix_pos_ids)      # [B, N, H]
            sp_tokens = sp_tokens + prefix_pos_emb

        # 6) Concat species prefix + peptide along sequence axis.
        embeddings = torch.cat([sp_tokens, word_embeds], dim=1)        # [B, N+L, H]

        # 7) LayerNorm and attention-mask multiplication (matches EsmEmbeddings tail).
        if embed_layer.layer_norm is not None:
            embeddings = embed_layer.layer_norm(embeddings)
        embeddings = (embeddings * attention_mask_ext.unsqueeze(-1)).to(embeddings.dtype)

        # 8) Encoder forward via the extended-attention-mask machinery.
        extended_attention_mask = esm_base.get_extended_attention_mask(
            attention_mask_ext, embeddings.shape[:-1]
        )
        encoder_outputs = esm_base.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
        )
        hidden = encoder_outputs[0]                                    # [B, N+L, H]
        # ESM2 has a final layer norm in encoder; HF's EsmEncoder already applies
        # it so we don't need to re-apply.

        # 9) Pool.
        if self.prefix_pool == "peptide":
            pep_hidden = hidden[:, N:, :]                              # [B, L, H]
            pool_mask = attention_mask.unsqueeze(-1).float()           # [B, L, 1]
            seq_rep = (pep_hidden * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)
        else:  # all
            pool_mask = attention_mask_ext.unsqueeze(-1).float()       # [B, N+L, 1]
            seq_rep = (hidden * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)

        # Optional legacy adapter channel (only if species_mode='both').
        if self.species_mode == "both" and self.species_adapter is not None:
            if species_emb is None:
                raise ValueError("species_mode='both' but species_emb is None.")
            adapter_emb = self.species_adapter(species_emb)
            seq_rep = torch.cat([seq_rep, adapter_emb], dim=-1)

        out = self.projection(seq_rep)
        return out.view(-1, out.size(-1))

    # ------------------------------------------------------------------
    def _forward_deep_prefix(self, input_ids, species_emb, species_ids,
                             species_names, attention_mask):
        """Deep prefix-tuning (Prefix-Tuning v2 / P-Tuning v2) forward.

        We DON'T touch the input embeddings. Instead, the GCN per-level
        species tokens [B, N, H] are pushed through ``self.prefix_kv_encoder``
        to produce a (K, V) prefix for EVERY transformer layer. These are
        passed via HF's ``past_key_values`` argument so EsmSelfAttention.forward
        concatenates them to its key/value tensors -- the query length stays
        equal to L (peptide), the attention output is therefore [B, L, H] and
        we just mean-pool over peptide tokens for the regression head.

        Note on positional encoding: ESM2 uses rotary positional embeddings
        applied AFTER the past-kv concat, so the prefix gets rotary positions
        [0, N-1] and the peptide gets [N, N+L-1]. Because rotary is relative,
        peptide-to-peptide relative positions are unchanged and the only
        effect is a constant +N offset (small for N=3, ESM-pretrained range).
        """
        sp_in = species_ids if species_ids is not None else species_names
        if sp_in is None:
            raise ValueError("species_inject='prefix' requires species_ids or species_names")
        if self.prefix_kv_encoder is None:
            raise RuntimeError("prefix_depth='deep' but prefix_kv_encoder is None.")

        esm_base = self._esm_base

        # 1) GCN per-level embeddings + level type embedding.
        level_embs = self.species_gnn.forward_levels(sp_in)            # [B, N, H]
        sp_tokens = level_embs + self.level_type_emb.unsqueeze(0)      # [B, N, H]
        B, N, _ = sp_tokens.shape

        # 2) Produce per-layer (K, V) prefixes.
        past_kv = self.prefix_kv_encoder(sp_tokens)                    # list of L_layers (K,V)

        # 3) Build peptide attention mask + extended mask covering [prefix | peptide].
        # We intentionally bypass HF's EsmModel.forward(): EsmEmbeddings expects
        # attention_mask to share length with input_ids (peptide only), while the
        # encoder layers need a mask of length N+L. We therefore call
        # ``embeddings(..., attention_mask=peptide_mask)`` then ``encoder(...,
        # attention_mask=<extended N+L mask>, past_key_values=past_kv)`` manually.
        if attention_mask is None:
            pad_id = getattr(esm_base.config, "pad_token_id", 1)
            attention_mask = (input_ids != pad_id).long()              # [B, L]

        embedding_output = esm_base.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values_length=0,   # keep peptide positions canonical (not offset by N)
        )                                                              # [B, L, H]
        L = embedding_output.size(1)

        ones = torch.ones(B, N, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask_ext = torch.cat([ones, attention_mask], dim=1)  # [B, N+L]
        extended_attention_mask = esm_base.get_extended_attention_mask(
            attention_mask_ext, (B, N + L),
        )                                                              # [B, 1, 1, N+L]

        encoder_outputs = esm_base.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            past_key_values=past_kv,
            use_cache=False,
        )
        hidden = encoder_outputs[0]                                    # [B, L, H]

        # 5) Pool over peptide tokens (the only tokens in the output).
        pool_mask = attention_mask.unsqueeze(-1).float()               # [B, L, 1]
        seq_rep = (hidden * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1)

        # Optional legacy adapter channel (only if species_mode='both').
        if self.species_mode == "both" and self.species_adapter is not None:
            if species_emb is None:
                raise ValueError("species_mode='both' but species_emb is None.")
            adapter_emb = self.species_adapter(species_emb)
            seq_rep = torch.cat([seq_rep, adapter_emb], dim=-1)

        out = self.projection(seq_rep)
        return out.view(-1, out.size(-1))


'''
AMPLIFY family
'''
class AMPLIFY(nn.Module): 
    def __init__(self, head_type='3MLP', plm_output='mean', finetune_plm=True, amplify_type='AMPLIFY-120M'):
        super(AMPLIFY, self).__init__()
        if amplify_type in ['AMPLIFY-120M', 'AMPLIFY-120M-base']:
            self.hidden_size = 640
        elif amplify_type in ['AMPLIFY-350M', 'AMPLIFY-350M-base']:
            self.hidden_size = 960
        else:
            raise ValueError(f"Wrong type of AMPLIFY: {amplify_type}")
        self.encoder = AutoModel.from_pretrained(
            f"chandar-lab/{amplify_type.replace('-', '_')}", trust_remote_code=True, cache_dir=cache_dir)
        self.head_type = head_type
        self.plm_output = plm_output
        self.finetune_plm = finetune_plm
        print(self.plm_output, self.hidden_size)

        # Freeze the parameters of the PLM if finetune_plm is False
        if not finetune_plm:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # print(f"Trainable encoder parameters: {trainable_params:,}")
        
        if head_type == '3MLP':
            self.projection = MLP(self.hidden_size, [256, 64, 2])  ## 3layers
        elif head_type == '5MLP':
            self.projection = MLP(self.hidden_size, [1024, 512, 128, 32, 2])  ## 5layers

    def forward(self, input_ids):
        outputs = self.encoder(input_ids, output_hidden_states=True)
        if self.plm_output == 'mean':
            outputs = outputs.hidden_states[-1].mean(dim=1)  # [batch_size, hidden_size]
            outputs = self.projection(outputs)
        elif self.plm_output == 'cls':
            outputs = outputs.hidden_states[-1][:, 0]
            # print(outputs.shape) #!
            outputs = self.projection(outputs)

        return outputs.view(-1, outputs.size(-1))

