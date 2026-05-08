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
                    best_ep: int = None) -> None:
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

def _unpack_batch(batch, device, species_mode):
    """Decode the variable-arity batch returned by ``MIC_Dataset``.

    Returns: (seq_list, species_emb_or_None, species_ids_or_None, mic_values[B,1]).
    """
    species_emb = None
    species_ids = None
    if species_mode == "none":
        seq_list, mic_values = batch
    elif species_mode == "adapter":
        seq_list, species_emb, mic_values = batch
        species_emb = species_emb.to(device, non_blocking=True)
    elif species_mode == "gnn":
        seq_list, species_ids, mic_values = batch
        species_ids = species_ids.to(device, non_blocking=True)
    elif species_mode == "both":
        seq_list, species_emb, species_ids, mic_values = batch
        species_emb = species_emb.to(device, non_blocking=True)
        species_ids = species_ids.to(device, non_blocking=True)
    else:
        raise ValueError(f"Unknown species_mode: {species_mode!r}")
    mic_values = mic_values.unsqueeze(1).to(device, non_blocking=True)
    return seq_list, species_emb, species_ids, mic_values


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


def train_epoch(epoch, model, train_loader, tokenizer, criterion, optimizer,
                device, species_mode="none", scheduler=None):
    model.train()
    train_loss = []
    train_epoch_time = 0.0
    pbar = tqdm(train_loader)
    pbar.set_description(f"GPU{device} Train epoch-{epoch}")
    print("\n","*" * 30, "Epoch", epoch, "training start...","*" * 30,"\n")
    for batch in pbar:
        seq_list, species_emb, species_ids, mic_values = _unpack_batch(batch, device, species_mode)
        input_ids = seq2token(seq_list, tokenizer, device)
        t1 = time.time()
        outputs = _model_forward(model, input_ids, species_emb, species_ids, species_mode)
        loss = criterion(outputs, mic_values)
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
    print(f"Epoch {epoch} Train MSE Loss: {ave_loss:.4f}, "
          f"Time: {train_epoch_time:.4f}s, "
          f"LR(end): {optimizer.param_groups[0]['lr']:.2e}")
    return ave_loss, train_loss


def validate_epoch(epoch, model, val_loader, tokenizer, criterion, device, species_mode="none"):
    model.eval()
    val_loss = []
    pbar = tqdm(val_loader)
    pbar.set_description(f"GPU{device} Val epoch-{epoch}")
    print("\n","*" * 30, "Epoch", epoch, "validation start...","*" * 30,"\n")

    with torch.no_grad():
        for batch in pbar:
            seq_list, species_emb, species_ids, mic_values = _unpack_batch(batch, device, species_mode)
            input_ids = seq2token(seq_list, tokenizer, device)
            outputs = _model_forward(model, input_ids, species_emb, species_ids, species_mode)
            loss = criterion(outputs, mic_values)
            val_loss.append(loss.item())

    ave_loss = sum(val_loss) / len(val_loss)
    print(f"Epoch {epoch} Val MSE Loss: {ave_loss:.4f}")
    return ave_loss, val_loss


def bucketed_test_eval(model, test_loader, tokenizer, criterion, device,
                       species_mode, train_csv_path,
                       buckets=((0, 5), (5, 20), (20, 100), (100, float("inf")))):
    """Run inference on the test loader and return per-species-count-bucket MSE.

    The bucketing is computed from the training set's ``Target_Species``
    frequency (the same training set the model was trained on). This is the
    metric the plan calls for to validate the long-tail story.
    """
    train_df = pd.read_csv(train_csv_path)
    sp_count = train_df["Target_Species"].astype(str).value_counts().to_dict()

    model.eval()
    per_sample = []   # (species_name, abs_err_squared)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="bucketed test"):
            seq_list, species_emb, species_ids, mic_values = _unpack_batch(batch, device, species_mode)
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
    parser.add_argument("--gnn_fusion", type=str, default="leaf", choices=["leaf", "hier"],
                        help="leaf=F1 (species-only); hier=F2 (species+genus+family).")
    parser.add_argument("--gnn_freeze_init", type=int, default=1, choices=[0, 1],
                        help="Freeze the 768-d PubMedBERT init features (1) or fine-tune (0).")

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
    parser.add_argument("--device", type=str, default="0")

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
        num_workers=4,
        seed=args.seed,
        species_mode=args.species_mode,
        species_emb_path=args.species_emb_path if args.species_mode in ("adapter", "both") else None,
        species_emb_dim=args.species_emb_dim,
        taxo_graph_path=args.taxo_graph_path if args.species_mode in ("gnn", "both") else None,
    )
    # load model and tokenizer
    print("Loading model...")
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
        gnn_freeze_init=bool(args.gnn_freeze_init),
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


    criterion = nn.MSELoss()    # MSE loss for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

    for epoch in range(start_epoch, args.epochs+1):
        avg_train_loss, train_loss = train_epoch(
            epoch, model, train_loader, tokenizer, criterion, optimizer, device,
            species_mode=args.species_mode, scheduler=scheduler)
        # --- 验证 ---
        avg_val_loss, val_loss = validate_epoch(
            epoch, model, val_loader, tokenizer, criterion, device,
            species_mode=args.species_mode)
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
            prev_ckp_path = os.path.join(ckp_path, f"{args.plm}_lr{args.lr}_bs{args.batch_size}_ep{best_ep}_val_best.pth")
            if os.path.exists(prev_ckp_path) and best_ep != epoch:
                os.remove(prev_ckp_path)    
            best_val_loss, best_ep = avg_val_loss, epoch
            print(f"New best model found at epoch {best_ep} with Val MSE: {best_val_loss:.4f}. Saving model...")
            save_checkpoint(model, optimizer,
                            os.path.join(ckp_path, f"{args.plm}_lr{args.lr}_bs{args.batch_size}_ep{epoch}_val_best.pth"),
                            scheduler=scheduler, epoch=epoch,
                            best_val_loss=best_val_loss, best_ep=best_ep)

        # ------------------------------------------------------------------
        # Early stopping: only activates AFTER --es_min_epoch (i.e. give the
        # model enough time during warmup + peak-LR stage before killing it).
        # ------------------------------------------------------------------
        es_active = args.early_stopping and epoch >= args.es_min_epoch
        if es_active and (epoch - best_ep) >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no val-loss improvement for {epoch - best_ep} epochs; "
                  f"best was epoch {best_ep} with val MSE {best_val_loss:.4f}).")
            save_checkpoint(model, optimizer,
                            os.path.join(ckp_path, f"{args.plm}_lr{args.lr}_bs{args.batch_size}_es_ep{epoch}.pth"),
                            scheduler=scheduler, epoch=epoch,
                            best_val_loss=best_val_loss, best_ep=best_ep)
            break
        elif args.early_stopping and epoch < args.es_min_epoch and (epoch - best_ep) >= args.early_stop_patience:
            # still in warmup/peak phase – log but do NOT stop
            print(f"[early-stop disabled until epoch {args.es_min_epoch}] "
                  f"val loss hasn't improved for {epoch - best_ep} epochs, continuing.")

        # 最后一个epoch结束保存模型
        if epoch == args.epochs:
            print(f"Training complete. Saving final model at epoch {epoch}. Best Val MSE: {best_val_loss:.4f} at epoch {best_ep}.")
            save_checkpoint(model, optimizer,
                            os.path.join(ckp_path, f"{args.plm}_lr{args.lr}_bs{args.batch_size}_ep{epoch}.pth"),
                            scheduler=scheduler, epoch=epoch,
                            best_val_loss=best_val_loss, best_ep=best_ep)
    # --- 测试 ---
    print("\n","*"*30, "Testing model...", "*"*30,"\n")
    avg_test_loss, test_loss = validate_epoch(
        epoch, model, test_loader, tokenizer, criterion, device,
        species_mode=args.species_mode)
    print(f"Test MSE Loss: {avg_test_loss:.4f}")

    # ------------------------------------------------------------------
    # Long-tail diagnostic: per-species-count bucketed test MSE.
    # Bucketing is by the species's frequency in the *training* CSV,
    # which is the right denominator for evaluating long-tail behaviour.
    # ------------------------------------------------------------------
    bucket_report = None
    if args.species_mode != "none":
        try:
            bucket_report = bucketed_test_eval(
                model, test_loader, tokenizer, criterion, device,
                species_mode=args.species_mode,
                train_csv_path=os.path.join(args.data_path, "train.csv"),
            )
        except Exception as exc:
            print(f"[warn] bucketed eval failed: {exc}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["best_val_loss"] = best_val_loss
    metrics_df["final_test_loss"] = avg_test_loss

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
        log_payload = {"test_mse": avg_test_loss, "best_val_mse": best_val_loss}
        if bucket_report is not None:
            log_payload["test_mse_overall"] = bucket_report["overall_mse"]
            for b in bucket_report["buckets"]:
                # use a wandb-friendly key; '<' gets stripped to keep panels clean
                key = "test_mse_bucket_" + b["train_count_bucket"].replace("[","").replace(")","").replace(",","_to_")
                log_payload[key] = b["mse"]
                log_payload[key + "_count"] = b["n_test_samples"]
        wandb.log(log_payload)
        wandb.finish()

if __name__ == "__main__":
    main()
