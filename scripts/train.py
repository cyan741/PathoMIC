import os
import math
import warnings
import pandas as pd
import torch
from tqdm import tqdm
import random
import numpy as np
import argparse
from plm_models import ESM2, load_tokenizer
from data_loader import data_loader, seq2token
from losses import build_loss
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import time
# from utils import save_checkpoint
import wandb


def set_seed(seed):
    # Python & Numpy seed
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch seed
    torch.manual_seed(seed)     # default generator
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDNN seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model,
                    optimizer: torch.optim.Optimizer,
                    output_path: str,
                    scheduler=None,
                    epoch: int = None,
                    best_val_loss: float = None,
                    best_ep: int = None,
                    criterion: nn.Module = None) -> None:
    '''
    Save model / optimizer / (optional) scheduler / epoch state to output_path.

    ``best_val_loss`` and ``best_ep`` are also persisted so a resumed run can
    faithfully continue tracking the best checkpoint without having to rerun
    validation from scratch.
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_ep': best_ep,
        'criterion_state_dict': criterion.state_dict() if criterion is not None else None,
    }
    torch.save(checkpoint, output_path)


def build_lr_scheduler(optimizer: torch.optim.Optimizer,
                       num_warmup_steps: int,
                       num_total_steps: int,
                       min_lr_ratio: float = 0.1) -> LambdaLR:
    """
    LR schedule (per training step):
        step < warmup          : linear ramp 0 → 1.0
        warmup ≤ step ≤ total  : cosine decay 1.0 → min_lr_ratio
    multiplier is applied to the optimizer's base LR.

    A small min_lr_ratio ( > 0 ) keeps the model learning slowly at the tail
    instead of freezing at 0.
    """
    num_warmup_steps = max(0, int(num_warmup_steps))
    num_total_steps  = max(num_warmup_steps + 1, int(num_total_steps))

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # linear warmup (start from ~0, avoid divide-by-zero)
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_total_steps - num_warmup_steps)
        )
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return LambdaLR(optimizer, lr_lambda)

def _unpack_batch(batch, device, species_mode, has_meta=False):
    """Decode the variable-arity batch returned by ``MIC_Dataset``.

    Returns:
        (seq_list, species_emb_or_None, species_ids_or_None, mic_values[B,1], meta_dict_or_None)
    """
    species_emb = None
    species_ids = None
    meta = None

    expected = {
        "none": 2, "adapter": 3, "gnn": 3, "both": 4,
    }[species_mode]
    if has_meta:
        expected += 1

    if len(batch) != expected:
        raise ValueError(
            f"Batch tuple length {len(batch)} does not match species_mode={species_mode!r} "
            f"with has_meta={has_meta} (expected {expected})."
        )

    if has_meta:
        meta = batch[-1]
        meta = {k: v.to(device, non_blocking=True) for k, v in meta.items()}
        batch = batch[:-1]

    if species_mode == "none":
        seq_list, mic_values = batch
    elif species_mode == "adapter":
        seq_list, species_emb, mic_values = batch
        species_emb = species_emb.to(device, non_blocking=True)
    elif species_mode == "gnn":
        seq_list, species_ids, mic_values = batch
        species_ids = species_ids.to(device, non_blocking=True)
    else:  # both
        seq_list, species_emb, species_ids, mic_values = batch
        species_emb = species_emb.to(device, non_blocking=True)
        species_ids = species_ids.to(device, non_blocking=True)
    mic_values = mic_values.unsqueeze(1).to(device, non_blocking=True)
    return seq_list, species_emb, species_ids, mic_values, meta


def _model_forward(model, input_ids, species_emb, species_ids, species_mode):
    """Dispatch to ``ESM2.forward`` with the right keyword args per mode."""
    if species_mode == "none":
        return model(input_ids)
    if species_mode == "adapter":
        return model(input_ids, species_emb=species_emb)
    if species_mode == "gnn":
        return model(input_ids, species_ids=species_ids)
    # both
    return model(input_ids, species_emb=species_emb, species_ids=species_ids)


def _build_loss_meta(meta, loss_meta_global, loss_type):
    """Merge per-batch meta with global loss meta (e.g. group_id for GroupDRO)."""
    if meta is None:
        return None
    out = dict(meta)
    if loss_type == "group_dro":
        if loss_meta_global.get("dro_group_by") == "species":
            out["group_id"] = meta["species_group_id"]
        else:
            out["group_id"] = meta["bucket_id"]
    return out


def _parse_dro_adj(adj_spec, num_groups):
    """Parse --loss_dro_adj into a FloatTensor[num_groups] or None.

    Accepts:
      "" / "none"            -> None
      "0.1"                  -> repeat scalar to all groups
      "0.0,0.1,0.2,0.3"      -> explicit per-group vector
    """
    if adj_spec is None:
        return None
    s = str(adj_spec).strip().lower()
    if s in ("", "none"):
        return None
    vals = [float(x.strip()) for x in str(adj_spec).split(",") if x.strip() != ""]
    if len(vals) == 1:
        vals = vals * int(num_groups)
    if len(vals) != int(num_groups):
        raise ValueError(
            f"--loss_dro_adj expects 1 value or {num_groups} comma-separated values, got {len(vals)}"
        )
    return torch.tensor(vals, dtype=torch.float32)


class _FileLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text)

    def flush(self) -> None:
        return


def train_epoch(epoch, model, train_loader, tokenizer, criterion, optimizer,
                device, species_mode="none", scheduler=None,
                loss_type="mse", loss_meta_global=None):
    model.train()
    train_loss = []
    train_epoch_time = 0.0
    has_meta = loss_meta_global is not None and loss_meta_global.get("needs_meta", False)
    if loss_type == "group_dro" and hasattr(criterion, "reset_stats"):
        criterion.reset_stats()
    pbar = tqdm(train_loader)
    pbar.set_description(f"GPU{device} Train epoch-{epoch}")
    print("\n","*" * 30, "Epoch", epoch, "training start...","*" * 30,"\n")
    for batch in pbar:
        seq_list, species_emb, species_ids, mic_values, meta = _unpack_batch(
            batch, device, species_mode, has_meta=has_meta)
        input_ids = seq2token(seq_list, tokenizer, device)
        t1 = time.time()
        outputs = _model_forward(model, input_ids, species_emb, species_ids, species_mode)
        loss_meta = _build_loss_meta(meta, loss_meta_global or {}, loss_type)
        loss = criterion(outputs, mic_values, meta=loss_meta) if loss_meta is not None \
               else criterion(outputs, mic_values, meta=None)
        train_epoch_time += time.time() - t1

        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

    ave_loss = sum(train_loss) / len(train_loss)
    print(f"Epoch {epoch} Train Loss: {ave_loss:.4f}, "
          f"Time: {train_epoch_time:.4f}s, "
          f"LR(end): {optimizer.param_groups[0]['lr']:.2e}")
    return ave_loss, train_loss


def validate_epoch(epoch, model, val_loader, tokenizer, criterion, device,
                   species_mode="none", has_meta=False, loss_type="mse",
                   loss_meta_global=None):
    """Validation always uses MSE (the official metric), regardless of training loss."""
    model.eval()
    val_loss = []
    pbar = tqdm(val_loader)
    pbar.set_description(f"GPU{device} Val epoch-{epoch}")
    print("\n","*" * 30, "Epoch", epoch, "validation start...","*" * 30,"\n")
    mse_eval = nn.MSELoss()
    with torch.no_grad():
        for batch in pbar:
            seq_list, species_emb, species_ids, mic_values, _meta = _unpack_batch(
                batch, device, species_mode, has_meta=has_meta)
            input_ids = seq2token(seq_list, tokenizer, device)
            outputs = _model_forward(model, input_ids, species_emb, species_ids, species_mode)
            loss = mse_eval(outputs, mic_values)
            val_loss.append(loss.item())

    ave_loss = sum(val_loss) / len(val_loss)
    print(f"Epoch {epoch} Val MSE Loss: {ave_loss:.4f}")
    return ave_loss, val_loss


def bucketed_test_eval(model, test_loader, tokenizer, criterion, device,
                       species_mode, train_csv_path,
                       has_meta=False,
                       buckets=((0, 5), (5, 20), (20, 100), (100, float("inf")))):
    """Run inference on the test loader and return per-species-count-bucket MSE."""
    train_df = pd.read_csv(train_csv_path)
    sp_count = train_df["Target_Species"].astype(str).value_counts().to_dict()

    model.eval()
    per_sample = []   # (species_name, abs_err_squared)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="bucketed test"):
            seq_list, species_emb, species_ids, mic_values, _meta = _unpack_batch(
                batch, device, species_mode, has_meta=has_meta)
            input_ids = seq2token(seq_list, tokenizer, device)
            outputs = _model_forward(model, input_ids, species_emb, species_ids, species_mode)
            sq = (outputs - mic_values).pow(2).squeeze(-1).detach().cpu().tolist()
            # Recover per-row species name. We rely on the dataset stashing it
            # in test_loader.dataset.species_names (only set when species_mode != 'none').
            ds = test_loader.dataset
            if ds.species_names is not None:
                start = len(per_sample)
                names = ds.species_names[start : start + len(sq)]
            else:
                names = ["__unknown__"] * len(sq)
            for n, s in zip(names, sq):
                per_sample.append((n, s))

    overall_mse = sum(s for _, s in per_sample) / max(1, len(per_sample))

    bucket_stats = []
    for lo, hi in buckets:
        sums = 0.0; cnt = 0; uniq_sp = set()
        for name, s in per_sample:
            n_train = sp_count.get(name, 0)
            if lo <= n_train < hi:
                sums += s; cnt += 1; uniq_sp.add(name)
        mse = (sums / cnt) if cnt > 0 else float("nan")
        bucket_stats.append({
            "train_count_bucket": f"[{lo},{hi})",
            "n_test_samples": cnt,
            "n_unique_species": len(uniq_sp),
            "mse": mse,
        })

    print("\n" + "=" * 72)
    print(f"BUCKETED TEST EVAL  (overall MSE = {overall_mse:.4f}, N={len(per_sample)})")
    print("=" * 72)
    print(f"{'train-count-bucket':<22}{'#test_samples':>16}{'#unique_sp':>14}{'mse':>14}")
    for b in bucket_stats:
        mse_str = f"{b['mse']:.4f}" if not (b["mse"] != b["mse"]) else "n/a"
        print(f"{b['train_count_bucket']:<22}{b['n_test_samples']:>16}{b['n_unique_species']:>14}{mse_str:>14}")
    print("=" * 72)

    return {"overall_mse": overall_mse, "buckets": bucket_stats}


def _split_test_results_subdir(data_path: str) -> str:
    """e.g. /NAS/.../splits1 -> splits1_test (matches test_scripts layout)."""
    base = os.path.basename(os.path.normpath(data_path))
    return f"{base}_test"


def _checkpoint_basename(plm: str, lr: float, batch_size: int, epoch: int, tag: str = "",
                         seed: int = None) -> str:
    """Stem shared by .pth checkpoints and *_test_results.csv files.

    If ``seed`` is given, ``_sd{seed}`` is appended at the very end so multi-
    seed runs land in distinct files (e.g. ``..._ep16_val_best_sd42.pth``).
    """
    stem = f"{plm}_lr{lr}_bs{batch_size}_ep{epoch}"
    if tag:
        stem = f"{stem}_{tag}"
    if seed is not None:
        stem = f"{stem}_sd{seed}"
    return stem


def load_model_checkpoint(model, ckpt_path: str, device) -> dict:
    """Load ``model_state_dict`` from a train.py checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[Checkpoint] Loading weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys (non-critical): {missing[:5]}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected[:5]}")
    return ckpt


def predict_on_test_loader(model, test_loader, tokenizer, device,
                           species_mode: str, has_meta: bool = False):
    """Run forward on ``test_loader`` (shuffle=False) and return predictions in row order."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test inference"):
            seq_list, species_emb, species_ids, _mic_values, _meta = _unpack_batch(
                batch, device, species_mode, has_meta=has_meta)
            input_ids = seq2token(seq_list, tokenizer, device)
            outputs = _model_forward(model, input_ids, species_emb, species_ids, species_mode)
            preds.extend(outputs.squeeze(-1).detach().cpu().tolist())
    return preds


def save_test_results_csv(test_csv_path: str, predictions, output_path: str) -> str:
    """Write original test.csv columns plus ``predict_MIC`` (log10 MIC)."""
    df = pd.read_csv(test_csv_path)
    if len(predictions) != len(df):
        raise ValueError(
            f"Prediction count ({len(predictions)}) != test rows ({len(df)}). "
            "Check that test_loader order matches test.csv."
        )
    df["predict_MIC"] = predictions
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[test_results] Saved {len(df)} rows -> {output_path}")
    return output_path


def run_post_train_test_eval(
    model,
    test_loader,
    tokenizer,
    criterion,
    device,
    species_mode: str,
    train_csv_path: str,
    has_meta: bool,
    label: str,
):
    """Test-set MSE + optional bucketed breakdown; ``label`` prefixes log lines."""
    print("\n", "*" * 30, f"Testing model ({label})...", "*" * 30, "\n")
    avg_test_loss, _test_loss = validate_epoch(
        0, model, test_loader, tokenizer, criterion, device,
        species_mode=species_mode, has_meta=has_meta,
        loss_type="mse", loss_meta_global=None,
    )
    print(f"[{label}] Test MSE Loss: {avg_test_loss:.4f}")

    bucket_report = None
    if species_mode != "none":
        try:
            print(f"[{label}] Bucketed test eval (by training-set species frequency):")
            bucket_report = bucketed_test_eval(
                model, test_loader, tokenizer, criterion, device,
                species_mode=species_mode,
                train_csv_path=train_csv_path,
                has_meta=has_meta,
            )
        except Exception as exc:
            print(f"[{label}][warn] bucketed eval failed: {exc}")
    else:
        print(f"[{label}] Skipping bucketed eval (species_mode=none).")
    return avg_test_loss, bucket_report


def main():
    # parameters
    parser = argparse.ArgumentParser(description="""Program entry point for amp MIC regression prediction training""")
    parser.add_argument("--data_path",
                        type=str,
                        default="/home/luyq/AMP_datasets/splits1")  
    parser.add_argument("--seed",
                        type=int,
                        default=42)
    parser.add_argument("--plm",
                        type=str,
                        default="esm2-8M")  
    parser.add_argument("--head_type", type=str, default="3MLP")
    parser.add_argument("--plm_output", type=str, default="mean")
    parser.add_argument("--finetune_plm", type=bool, default=True)

    # ----- species channel ----------------------------------------------------
    # The new --species_mode flag dispatches to one of four channels:
    #   none     : peptide-only baseline (no species info)
    #   adapter  : legacy 768-d PubMedBERT vector + non-linear adapter
    #   gnn      : taxonomy-DAG GNN (this PR)
    #   both     : adapter + gnn concatenated (Plan C in the design doc)
    # ``--use_species`` is kept for backwards compatibility: when set, it maps
    # to species_mode='adapter' (overridden if --species_mode is given explicitly).
    parser.add_argument("--use_species", action="store_true",
                        help="[deprecated] same as --species_mode adapter.")
    parser.add_argument("--species_mode", type=str, default=None,
                        choices=["none", "adapter", "gnn", "both"],
                        help="How to inject species information; supersedes --use_species.")
    parser.add_argument("--species_emb_path", type=str,
                        default="/NAS/luyq/AMP_datasets/species_embeddings.pkl")
    parser.add_argument("--species_emb_dim", type=int, default=768)
    parser.add_argument("--species_out_dim", type=int, default=128)
    parser.add_argument("--species_bottleneck", type=int, default=128)
    parser.add_argument("--species_dropout", type=float, default=0.1)

    # ----- taxonomy GNN channel ----------------------------------------------
    parser.add_argument("--taxo_graph_path", type=str,
                        default="/NAS/luyq/AMP_datasets/taxonomy_graph.pt",
                        help="Path to the .pt file produced by build_taxonomy_graph.py.")
    parser.add_argument("--gnn_hidden", type=int, default=128,
                        help="GNN hidden width (recommended 64-128).")
    parser.add_argument("--gnn_out_dim", type=int, default=64,
                        help="Final per-species channel width concatenated with peptide_emb.")
    parser.add_argument("--gnn_layers", type=int, default=2,
                        help="Number of GCN/GAT message-passing layers.")
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat"],
                        help="Graph conv variant.")
    parser.add_argument("--gnn_heads", type=int, default=4,
                        help="GAT only: number of attention heads per layer.")
    parser.add_argument("--gnn_dropout", type=float, default=0.1)
    parser.add_argument("--gnn_fusion", type=str, default="leaf",
                        choices=["leaf", "hier", "hier_attn", "hier_raw"],
                        help="leaf=F1; hier=F2 (configurable levels); hier_attn=attention pool; "
                             "hier_raw=raw concat of per-level embeddings (no hier_proj).")
    parser.add_argument("--gnn_hier_levels", type=str, default="species,genus,family",
                        help="Comma-separated taxonomic levels for hier/hier_attn fusion.")
    parser.add_argument("--gnn_freeze_init", type=int, default=1, choices=[0, 1],
                        help="Freeze the 768-d PubMedBERT init features (1) or fine-tune (0).")
    parser.add_argument("--use_lora_init", action="store_true",
                        help="Add a low-rank residual on top of frozen init features.")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--gnn_residual", action="store_true",
                        help="Skip connection inside each GCN layer.")
    parser.add_argument("--gnn_layernorm", action="store_true",
                        help="Apply LayerNorm between GCN layers.")
    parser.add_argument("--fusion_strategy", type=str, default="concat",
                        choices=["concat", "gated", "film", "cross_attn", "bilinear"],
                        help="How to combine ESM peptide embedding with the GNN species emb.")
    parser.add_argument("--species_inject", type=str, default="post",
                        choices=["post", "prefix"],
                        help="post = default; GNN species_emb fused with ESM output. "
                             "prefix = prepend N species tokens (one per hier_level) to ESM "
                             "input embeddings; pure prefix injection, no late fusion. "
                             "Requires --gnn_out_dim == ESM hidden width (640 for 150M).")
    parser.add_argument("--prefix_pool", type=str, default="peptide",
                        choices=["peptide", "all"],
                        help="When --species_inject=prefix: pool over peptide tokens only "
                             "('peptide') or over all tokens including the species prefix ('all').")
    parser.add_argument("--gnn_lr_mult", type=float, default=1.0,
                        help="Multiplier on the base lr applied ONLY to the GNN+fusion params.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # ----- Loss type ----------------------------------------------------------
    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "huber", "smooth_l1", "bmc", "focal_r", "group_dro"])
    parser.add_argument("--loss_huber_delta", type=float, default=1.0)
    parser.add_argument("--loss_smooth_l1_beta", type=float, default=1.0)
    parser.add_argument("--loss_bmc_noise", type=float, default=1.0)
    parser.add_argument("--loss_bmc_learn_noise", type=int, default=1, choices=[0, 1])
    parser.add_argument("--loss_focal_gamma", type=float, default=2.0)
    parser.add_argument("--loss_focal_base", type=str, default="mse",
                        choices=["mse", "huber", "smooth_l1"])
    parser.add_argument("--loss_dro_eta", type=float, default=0.01)
    parser.add_argument("--loss_dro_base", type=str, default="mse",
                        choices=["mse", "huber", "smooth_l1"])
    parser.add_argument("--loss_dro_group_by", type=str, default="bucket",
                        choices=["bucket", "species"])
    parser.add_argument("--loss_dro_gamma", type=float, default=0.1,
                        help="EMA momentum for GroupDRO's historical group loss.")
    parser.add_argument("--loss_dro_normalize_loss", action="store_true",
                        help="Normalize adjusted group losses before dual update.")
    parser.add_argument("--loss_dro_btl", action="store_true",
                        help="Use BTL variant (alpha-constrained worst-group mixture).")
    parser.add_argument("--loss_dro_alpha", type=float, default=0.2,
                        help="Alpha mass for BTL GroupDRO (effective when --loss_dro_btl).")
    parser.add_argument("--loss_dro_min_var_weight", type=float, default=0.0,
                        help="BTL min-variance mixing weight in [0,1].")
    parser.add_argument("--loss_dro_adj", type=str, default="",
                        help="Group adjustment(s): single float or comma list length=#groups.")
    parser.add_argument("--loss_dro_log_path", type=str, default="",
                        help="If set, append GroupDRO stats to this file each epoch.")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Peak learning rate after warmup.")
    parser.add_argument("--epochs", type=int, default=10)

    # LR scheduler (warmup -> cosine decay)
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Enable warmup + cosine LR schedule.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total training steps used for linear LR warmup.")
    parser.add_argument("--warmup_steps", type=int, default=-1,
                        help="If > 0, overrides --warmup_ratio with an explicit step count.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Final LR as a fraction of --lr (end of cosine decay).")

    # Early-stopping: only kicks in after --es_min_epoch
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--es_min_epoch", type=int, default=20,
                        help="Early stopping is only considered once epoch >= this value.")

    # ----- Resume from a saved checkpoint -------------------------------------
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pth checkpoint to resume training from. "
                             "`--epochs` is interpreted as the TOTAL number of epochs "
                             "(including those already completed in the checkpoint).")
    parser.add_argument("--resume_reset_optimizer", action="store_true",
                        help="If set, do NOT restore optimizer state (use a fresh "
                             "AdamW). Useful when changing learning-rate regime.")
    parser.add_argument("--resume_reset_best", action="store_true",
                        help="If set, ignore the saved best_val_loss / best_ep "
                             "(start tracking from scratch on the resumed run).")
    parser.add_argument("--save_dir", type=str, default="/NAS/luyq/PLM_AMP_Regression/ckp")
    parser.add_argument("--metrics_name", type=str, default="train_metrics.csv")
    parser.add_argument(
        "--test_results_dir",
        type=str,
        default="/home/luyq/PLM_AMP_Regression/test_results",
        help="Root folder for auto-saved test CSVs (default: <repo>/test_results/<split>_test/csv/).",
    )
    parser.add_argument("--device", type=str, default="0")

    # ----- shuffle-species control (null hypothesis test) ------------------
    parser.add_argument(
        "--shuffle_species_control", type=int, default=0, choices=[0, 1],
        help="If 1, randomly permute the (peptide -> species) pairing on the "
             "TRAIN split before training (val/test are untouched). Used to "
             "test whether the species channel actually carries useful signal.",
    )

    # ----- ESM-backbone tuning regime (Prefix-tuning experiments) ----------
    parser.add_argument(
        "--freeze_esm", type=int, default=0, choices=[0, 1],
        help="If 1, freeze ALL ESM2 backbone parameters. Used together with "
             "--species_inject=prefix to study prefix-only tuning.",
    )
    parser.add_argument(
        "--use_lora", type=int, default=0, choices=[0, 1],
        help="If 1, attach a LoRA adapter (via the `peft` library) to the ESM "
             "backbone. Mutually exclusive with --freeze_esm=1 in the same run.",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target", type=str, default="query,value",
        help="Comma-separated list of nn.Linear module names within EsmLayer "
             "to wrap with LoRA (e.g. 'query,value' for Q/V; "
             "'query,key,value' for QKV).",
    )

    # ----- Prefix depth: shallow vs deep (per-layer KV) --------------------
    parser.add_argument(
        "--prefix_depth", type=str, default="shallow",
        choices=["shallow", "deep"],
        help="When --species_inject=prefix: 'shallow' = prepend species tokens "
             "to the input embeddings only (current behaviour); 'deep' = inject "
             "a learned (K,V) pair per layer, prepended to every layer's "
             "self-attention K/V (Prefix-Tuning v2 / P-Tuning v2).",
    )
    parser.add_argument(
        "--prefix_kv_hidden", type=int, default=512,
        help="Hidden width of the per-layer prefix MLP (deep prefix only).",
    )

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AMP-ESM")
    parser.add_argument("--wandb_entity", type=str, default="cyan741741")
    parser.add_argument("--wandb_api_key", default=None, type=str)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")
 
    args = parser.parse_args()

    # Resolve species_mode: explicit CLI arg wins; otherwise fall back to the
    # legacy --use_species bool (True -> 'adapter', False -> 'none').
    if args.species_mode is None:
        args.species_mode = "adapter" if args.use_species else "none"
    print(args)
    set_seed(args.seed)

    # load data
    print("Loading data...")
    data_path = args.data_path
    ckp_path = args.save_dir
    os.makedirs(ckp_path, exist_ok=True)
    train_loader, val_loader, test_loader = data_loader(
        data_path,
        batch_size=args.batch_size,
        num_workers=8,
        seed=args.seed,
        species_mode=args.species_mode,
        species_emb_path=args.species_emb_path if args.species_mode in ("adapter", "both") else None,
        species_emb_dim=args.species_emb_dim,
        taxo_graph_path=args.taxo_graph_path if args.species_mode in ("gnn", "both") else None,
        loss_type=args.loss_type,
        dro_group_by=args.loss_dro_group_by,
        shuffle_species=bool(args.shuffle_species_control),
        shuffle_species_seed=args.seed,
    )
    loss_meta_global = getattr(train_loader, "loss_meta", {"needs_meta": False})
    has_meta = loss_meta_global.get("needs_meta", False)
    # load model and tokenizer
    print("Loading model...")
    hier_levels_tuple = tuple(s.strip() for s in args.gnn_hier_levels.split(",") if s.strip())
    lora_targets_tuple = tuple(
        s.strip() for s in args.lora_target.split(",") if s.strip()
    )
    model = ESM2(
        plm_output=args.plm_output,
        head_type=args.head_type,
        finetune_plm=args.finetune_plm,
        esm_size=args.plm.split('-')[-1],
        species_mode=args.species_mode,
        species_in_dim=args.species_emb_dim,
        species_out_dim=args.species_out_dim,
        species_bottleneck=args.species_bottleneck,
        species_dropout=args.species_dropout,
        taxo_graph_path=args.taxo_graph_path if args.species_mode in ("gnn", "both") else None,
        gnn_hidden=args.gnn_hidden,
        gnn_out_dim=args.gnn_out_dim,
        gnn_layers=args.gnn_layers,
        gnn_type=args.gnn_type,
        gnn_heads=args.gnn_heads,
        gnn_dropout=args.gnn_dropout,
        gnn_fusion=args.gnn_fusion,
        gnn_hier_levels=hier_levels_tuple,
        gnn_freeze_init=bool(args.gnn_freeze_init),
        use_lora_init=args.use_lora_init,
        lora_rank=args.lora_rank,
        gnn_residual=args.gnn_residual,
        gnn_layernorm=args.gnn_layernorm,
        fusion_strategy=args.fusion_strategy,
        species_inject=args.species_inject,
        prefix_pool=args.prefix_pool,
        prefix_depth=args.prefix_depth,
        prefix_kv_hidden=args.prefix_kv_hidden,
        freeze_esm=bool(args.freeze_esm),
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=lora_targets_tuple,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:" + args.device)
    else:        
        device = torch.device("cpu")
    tokenizer = load_tokenizer(args.plm)
    model = model.to(device)

    if args.use_wandb:
        if args.wandb_mode == "online":
            try:
                if args.wandb_api_key:
                    wandb.login(key=args.wandb_api_key, relogin=True)
                else:
                    wandb.login()
            except Exception as exc:
                raise RuntimeError(
                    "W&B login failed. Provide --wandb_api_key or set WANDB_API_KEY in your environment."
                ) from exc

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config=vars(args),
        )

    # train model
    print("Training model...")
    
    # log train loss
    metrics_rows = []


    # ----- Loss --------------------------------------------------------------
    loss_kwargs = dict(
        huber_delta=args.loss_huber_delta,
        smooth_l1_beta=args.loss_smooth_l1_beta,
        bmc_noise=args.loss_bmc_noise,
        bmc_learn_noise=bool(args.loss_bmc_learn_noise),
        focal_gamma=args.loss_focal_gamma,
        focal_base=args.loss_focal_base,
        dro_eta=args.loss_dro_eta,
        dro_base=args.loss_dro_base,
        dro_gamma=args.loss_dro_gamma,
        dro_normalize_loss=args.loss_dro_normalize_loss,
        dro_btl=args.loss_dro_btl,
        dro_alpha=args.loss_dro_alpha,
        dro_min_var_weight=args.loss_dro_min_var_weight,
    )
    if args.loss_type == "group_dro":
        loss_kwargs["dro_num_groups"] = loss_meta_global["dro_num_groups"]
        loss_kwargs["dro_group_counts"] = loss_meta_global.get("dro_group_counts")
        loss_kwargs["dro_adj"] = _parse_dro_adj(
            args.loss_dro_adj,
            loss_meta_global["dro_num_groups"],
        )
    criterion = build_loss(args.loss_type, **loss_kwargs).to(device)
    print(f"[Loss] type={args.loss_type} kwargs={loss_kwargs}")

    # ----- Optimizer with optional split lr (backbone vs new modules) -------
    # "new modules" = anything we add on top of the frozen / LoRA-adapted ESM
    # backbone: the GNN, the fusion module, the legacy species adapter, the
    # regression head, and the prefix-tuning modules (level_type_emb +
    # prefix_kv_encoder). These get ``lr * gnn_lr_mult``; everything else
    # (typically only LoRA adapters when --use_lora is set; the full ESM
    # otherwise) gets the base ``lr``.
    NEW_MODULE_PREFIXES = (
        "species_gnn", "fusion", "species_adapter", "projection",
        "level_type_emb", "prefix_kv_encoder",
    )
    split_lr = (args.gnn_lr_mult != 1.0) or bool(args.freeze_esm) or bool(args.use_lora)
    if split_lr:
        new_params, backbone_params = [], []
        n_new_count = n_bb_count = 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(n.startswith(pref) for pref in NEW_MODULE_PREFIXES):
                new_params.append(p)
                n_new_count += p.numel()
            else:
                backbone_params.append(p)
                n_bb_count += p.numel()
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr,
                 "weight_decay": args.weight_decay},
                {"params": new_params, "lr": args.lr * args.gnn_lr_mult,
                 "weight_decay": args.weight_decay},
            ]
        )
        print(f"[Optimizer] AdamW with split lr: backbone lr={args.lr:.2e} "
              f"({n_bb_count/1e6:.2f}M trainable), "
              f"new-modules lr={args.lr * args.gnn_lr_mult:.2e} "
              f"({n_new_count/1e6:.2f}M trainable), wd={args.weight_decay}")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )
        print(f"[Optimizer] AdamW lr={args.lr:.2e}, wd={args.weight_decay}")

    # ------------------------------------------------------------------
    # Resume: load model / optimizer / best-tracking from a checkpoint.
    # `start_epoch` is the NEXT epoch to train (1-indexed, inclusive).
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float('inf')
    best_ep = -1

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume path does not exist: {args.resume}")
        print(f"[Resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        # -- model --
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"[Resume] Missing keys (left at init): {missing}")
        if unexpected:
            print(f"[Resume] Unexpected keys (ignored): {unexpected}")

        # -- optimizer --
        if (not args.resume_reset_optimizer) and ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                # Override LR with CLI value so the resumed run uses the new peak lr.
                for pg in optimizer.param_groups:
                    pg["lr"]          = args.lr
                    pg["initial_lr"]  = args.lr
                print("[Resume] Optimizer state restored (LR overridden to --lr).")
            except ValueError as e:
                print(f"[Resume] Could not load optimizer state ({e}); using fresh optimizer.")
        else:
            print("[Resume] Using fresh optimizer state.")

        # -- starting epoch --
        saved_ep = int(ckpt.get("epoch") or 0)
        start_epoch = saved_ep + 1
        print(f"[Resume] Saved epoch = {saved_ep} → training will start at epoch {start_epoch}")

        # -- best val loss tracking --
        if (not args.resume_reset_best) and ckpt.get("best_val_loss") is not None:
            best_val_loss = float(ckpt["best_val_loss"])
            best_ep       = int(ckpt.get("best_ep") or saved_ep)
            print(f"[Resume] Restored best_val_loss={best_val_loss:.4f} at epoch {best_ep}")
        else:
            print("[Resume] Not restoring best-val tracker (fresh).")

        if start_epoch > args.epochs:
            raise ValueError(
                f"--epochs={args.epochs} but the checkpoint is already at epoch "
                f"{saved_ep}. Pass a larger --epochs (total, including past) to "
                f"continue training."
            )


    # ------------------------------------------------------------------
    # LR scheduler: linear warmup → cosine decay (per-step update)
    # When resuming we build a FRESH scheduler covering the new [1..epochs]
    # horizon and fast-forward it past the already-completed steps.
    # ------------------------------------------------------------------
    scheduler = None
    if args.use_scheduler:
        total_steps = len(train_loader) * args.epochs
        if args.warmup_steps and args.warmup_steps > 0:
            warmup_steps = args.warmup_steps
        else:
            warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = build_lr_scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_total_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        print(f"[LR scheduler] total_steps={total_steps} | "
              f"warmup_steps={warmup_steps} | "
              f"peak_lr={args.lr:.2e} | "
              f"min_lr={args.lr * args.min_lr_ratio:.2e}")

        # fast-forward to the correct step on the new schedule
        completed_steps = (start_epoch - 1) * len(train_loader)
        if completed_steps > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(completed_steps):
                    scheduler.step()
            print(f"[LR scheduler] fast-forwarded {completed_steps} steps → "
                  f"LR now = {optimizer.param_groups[0]['lr']:.2e}")
    else:
        print("[LR scheduler] disabled (constant LR)")

    if args.resume is not None and ckpt.get("criterion_state_dict") is not None:
        try:
            criterion.load_state_dict(ckpt["criterion_state_dict"], strict=False)
            print("[Resume] Criterion state restored.")
        except Exception as exc:
            print(f"[Resume] Could not load criterion state ({exc}); continuing.")

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs+1):
        last_epoch = epoch
        avg_train_loss, train_loss = train_epoch(
            epoch, model, train_loader, tokenizer, criterion, optimizer, device,
            species_mode=args.species_mode, scheduler=scheduler,
            loss_type=args.loss_type, loss_meta_global=loss_meta_global)

        if args.loss_type == "group_dro" and args.loss_dro_log_path and hasattr(criterion, "log_stats"):
            logger = _FileLogger(args.loss_dro_log_path)
            criterion.log_stats(logger, header=f"[epoch {epoch}]")
        # --- 验证 ---
        avg_val_loss, val_loss = validate_epoch(
            epoch, model, val_loader, tokenizer, criterion, device,
            species_mode=args.species_mode, has_meta=has_meta,
            loss_type=args.loss_type, loss_meta_global=loss_meta_global)
        print(f"Epoch {epoch} Complete. Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        cur_lr = optimizer.param_groups[0]["lr"]
        metrics_rows.append({
            "epoch": epoch,
            "train_mse": avg_train_loss,
            "val_mse": avg_val_loss,
            "lr": cur_lr,
        })

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_mse": avg_train_loss,
                    "val_mse": avg_val_loss,
                    "best_val_mse": min(best_val_loss, avg_val_loss),
                    "lr": cur_lr,
                }
            )
        
        if avg_val_loss < best_val_loss:
            # 保存最佳模型
            # 如果上一个最佳模型存在且不是当前模型，则删除上一个最佳模型文件
            prev_ckp_path = os.path.join(
                ckp_path,
                f"{_checkpoint_basename(args.plm, args.lr, args.batch_size, best_ep, 'val_best', seed=args.seed)}.pth",
            )
            if os.path.exists(prev_ckp_path) and best_ep != epoch:
                os.remove(prev_ckp_path)
            best_val_loss, best_ep = avg_val_loss, epoch
            print(f"New best model found at epoch {best_ep} with Val MSE: {best_val_loss:.4f}. Saving model...")
            save_checkpoint(model, optimizer,
                            os.path.join(
                                ckp_path,
                                f"{_checkpoint_basename(args.plm, args.lr, args.batch_size, epoch, 'val_best', seed=args.seed)}.pth",
                            ),
                            scheduler=scheduler, epoch=epoch,
                            best_val_loss=best_val_loss, best_ep=best_ep,
                            criterion=criterion)

        # ------------------------------------------------------------------
        # Early stopping: only activates AFTER --es_min_epoch (i.e. give the
        # model enough time during warmup + peak-LR stage before killing it).
        # ------------------------------------------------------------------
        es_active = args.early_stopping and epoch >= args.es_min_epoch
        if es_active and (epoch - best_ep) >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no val-loss improvement for {epoch - best_ep} epochs; "
                  f"best was epoch {best_ep} with val MSE {best_val_loss:.4f}).")
            es_stem = f"{args.plm}_lr{args.lr}_bs{args.batch_size}_es_ep{epoch}"
            if args.seed is not None:
                es_stem = f"{es_stem}_sd{args.seed}"
            save_checkpoint(model, optimizer,
                            os.path.join(ckp_path, f"{es_stem}.pth"),
                            scheduler=scheduler, epoch=epoch,
                            best_val_loss=best_val_loss, best_ep=best_ep,
                            criterion=criterion)
            break
        elif args.early_stopping and epoch < args.es_min_epoch and (epoch - best_ep) >= args.early_stop_patience:
            # still in warmup/peak phase – log but do NOT stop
            print(f"[early-stop disabled until epoch {args.es_min_epoch}] "
                  f"val loss hasn't improved for {epoch - best_ep} epochs, continuing.")

        # 最后一个epoch结束保存模型
        if epoch == args.epochs:
            print(f"Training complete. Saving final model at epoch {epoch}. Best Val MSE: {best_val_loss:.4f} at epoch {best_ep}.")
            save_checkpoint(model, optimizer,
                            os.path.join(
                                ckp_path,
                                f"{_checkpoint_basename(args.plm, args.lr, args.batch_size, epoch, seed=args.seed)}.pth",
                            ),
                            scheduler=scheduler, epoch=epoch,
                            best_val_loss=best_val_loss, best_ep=best_ep,
                            criterion=criterion)

    # ------------------------------------------------------------------
    # Post-training evaluation on test.csv
    #   1) val_best checkpoint (early-stop winner): MSE + bucketed + CSV
    #   2) last-epoch checkpoint (or in-memory weights): MSE + bucketed
    # ------------------------------------------------------------------
    train_csv_path = os.path.join(args.data_path, "train.csv")
    test_csv_path = os.path.join(args.data_path, "test.csv")
    os.makedirs(args.test_results_dir, exist_ok=True)
    test_results_csv_dir = os.path.join(
        args.test_results_dir, args.plm, _split_test_results_subdir(args.data_path), "csv",
    )
    os.makedirs(test_results_csv_dir, exist_ok=True)

    val_best_test_mse = float("nan")
    final_test_mse = float("nan")
    bucket_report_val_best = None
    bucket_report_final = None

    val_best_ckp = os.path.join(
        ckp_path,
        f"{_checkpoint_basename(args.plm, args.lr, args.batch_size, best_ep, 'val_best', seed=args.seed)}.pth",
    )
    if best_ep >= 1 and os.path.isfile(val_best_ckp):
        load_model_checkpoint(model, val_best_ckp, device)
        val_best_test_mse, bucket_report_val_best = run_post_train_test_eval(
            model, test_loader, tokenizer, criterion, device,
            species_mode=args.species_mode,
            train_csv_path=train_csv_path,
            has_meta=has_meta,
            label=f"val_best (epoch {best_ep})",
        )
        try:
            preds = predict_on_test_loader(
                model, test_loader, tokenizer, device,
                species_mode=args.species_mode, has_meta=has_meta,
            )
            csv_name = (
                f"{_checkpoint_basename(args.plm, args.lr, args.batch_size, best_ep, 'val_best', seed=args.seed)}"
                f"_test_results.csv"
            )
            save_test_results_csv(
                test_csv_path, preds,
                os.path.join(test_results_csv_dir, csv_name),
            )
        except Exception as exc:
            print(f"[val_best][warn] Could not save test_results CSV: {exc}")
    else:
        print(f"[val_best] No checkpoint at {val_best_ckp}; skipping val_best test eval.")

    # Last-epoch / early-stop snapshot (not necessarily val_best weights).
    _seed_suffix = f"_sd{args.seed}" if args.seed is not None else ""
    final_ckp_candidates = [
        os.path.join(
            ckp_path,
            f"{args.plm}_lr{args.lr}_bs{args.batch_size}_ep{last_epoch}{_seed_suffix}.pth",
        ),
        os.path.join(
            ckp_path,
            f"{args.plm}_lr{args.lr}_bs{args.batch_size}_es_ep{last_epoch}{_seed_suffix}.pth",
        ),
    ]
    final_ckp = next((p for p in final_ckp_candidates if os.path.isfile(p)), None)
    if final_ckp is not None:
        load_model_checkpoint(model, final_ckp, device)
        final_label = f"last_epoch (epoch {last_epoch}, from disk)"
    else:
        print(f"[last_epoch] No saved checkpoint for epoch {last_epoch}; "
              f"using in-memory weights from the training loop.")
        final_label = f"last_epoch (epoch {last_epoch}, in-memory)"

    final_test_mse, bucket_report_final = run_post_train_test_eval(
        model, test_loader, tokenizer, criterion, device,
        species_mode=args.species_mode,
        train_csv_path=train_csv_path,
        has_meta=has_meta,
        label=final_label,
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["best_val_loss"] = best_val_loss
    metrics_df["val_best_test_mse"] = val_best_test_mse
    metrics_df["final_test_mse"] = final_test_mse

    # If resuming, append to an existing metrics file instead of overwriting.
    metrics_path = os.path.join(args.save_dir, args.metrics_name)
    if args.resume is not None and os.path.exists(metrics_path):
        try:
            prev = pd.read_csv(metrics_path)
            metrics_df = pd.concat([prev, metrics_df], ignore_index=True)
            print(f"[Resume] Appended new metrics to existing {metrics_path}")
        except Exception as e:
            print(f"[Resume] Could not read previous metrics ({e}); writing fresh.")
    metrics_df.to_csv(metrics_path, index=False)
    
    if args.use_wandb:
        log_payload = {
            "best_val_mse": best_val_loss,
            "val_best_test_mse": val_best_test_mse,
            "final_test_mse": final_test_mse,
            "test_mse": val_best_test_mse,  # legacy alias
        }
        for prefix, report in (
            ("val_best", bucket_report_val_best),
            ("final", bucket_report_final),
        ):
            if report is None:
                continue
            log_payload[f"{prefix}_test_mse_overall"] = report["overall_mse"]
            for b in report["buckets"]:
                key = (
                    f"{prefix}_test_mse_bucket_"
                    + b["train_count_bucket"].replace("[", "").replace(")", "").replace(",", "_to_")
                )
                log_payload[key] = b["mse"]
                log_payload[key + "_count"] = b["n_test_samples"]
        wandb.log(log_payload)
        wandb.finish()

if __name__ == "__main__":
    main()
