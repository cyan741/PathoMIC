import os
import math
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
                    epoch: int = None) -> None:
    '''
    save model / optimizer / (optional) scheduler / epoch state to output_path.
    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
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

def _unpack_batch(batch, device, use_species):
    """Return (seq_list, species_emb_or_None, mic_values[B,1])."""
    if use_species:
        seq_list, species_emb, mic_values = batch
        species_emb = species_emb.to(device, non_blocking=True)
    else:
        seq_list, mic_values = batch
        species_emb = None
    mic_values = mic_values.unsqueeze(1).to(device, non_blocking=True)
    return seq_list, species_emb, mic_values


def train_epoch(epoch, model, train_loader, tokenizer, criterion, optimizer,
                device, use_species=False, scheduler=None):
    # training for one epoch
    model.train()
    # parameters to monitor
    train_loss = []
    train_epoch_time = 0.0
    pbar = tqdm(train_loader)
    pbar.set_description(f"GPU{device} Train epoch-{epoch}")
    print("\n","*" * 30, "Epoch", epoch, "training start...","*" * 30,"\n")
    for batch in pbar:
        seq_list, species_emb, mic_values = _unpack_batch(batch, device, use_species)
        input_ids = seq2token(seq_list, tokenizer, device)  # tensor shape: [batch_size, seq_len]
        t1 = time.time()
        outputs = model(input_ids, species_emb=species_emb) if use_species else model(input_ids)
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

def validate_epoch(epoch, model, val_loader, tokenizer, criterion, device, use_species=False):
    # validation for one epoch
    model.eval()
    val_loss = []
    pbar = tqdm(val_loader)
    pbar.set_description(f"GPU{device} Val epoch-{epoch}")
    print("\n","*" * 30, "Epoch", epoch, "validation start...","*" * 30,"\n")

    with torch.no_grad():
        for batch in pbar:
            seq_list, species_emb, mic_values = _unpack_batch(batch, device, use_species)
            input_ids = seq2token(seq_list, tokenizer, device)  # tensor shape: [batch_size, seq_len]
            outputs = model(input_ids, species_emb=species_emb) if use_species else model(input_ids)
            loss = criterion(outputs, mic_values)
            val_loss.append(loss.item())

    ave_loss = sum(val_loss) / len(val_loss)
    print(f"Epoch {epoch} Val MSE Loss: {ave_loss:.4f}")

    return ave_loss, val_loss

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

    # species embedding options
    parser.add_argument("--use_species", action="store_true",
                        help="Inject species embedding via a non-linear adapter.")
    parser.add_argument("--species_emb_path", type=str,
                        default="/home/luyq/AMP_datasets/species_embeddings.pkl")
    parser.add_argument("--species_emb_dim", type=int, default=768)
    parser.add_argument("--species_out_dim", type=int, default=128)
    parser.add_argument("--species_bottleneck", type=int, default=128)
    parser.add_argument("--species_dropout", type=float, default=0.1)

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
    parser.add_argument("--save_dir", type=str, default="/NAS/luyq/PLM_AMP_Regression/ckp")
    parser.add_argument("--metrics_name", type=str, default="train_metrics.csv")
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="AMP-ESM")
    parser.add_argument("--wandb_entity", type=str, default="cyan741741")
    parser.add_argument("--wandb_api_key", default=None, type=str)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")
 
    args = parser.parse_args()
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
        species_emb_path=args.species_emb_path if args.use_species else None,
        species_emb_dim=args.species_emb_dim,
    )
    # load model and tokenizer
    print("Loading model...")
    model = ESM2(
        plm_output=args.plm_output,
        head_type=args.head_type,
        finetune_plm=args.finetune_plm,
        esm_size=args.plm.split('-')[-1],
        use_species=args.use_species,
        species_in_dim=args.species_emb_dim,
        species_out_dim=args.species_out_dim,
        species_bottleneck=args.species_bottleneck,
        species_dropout=args.species_dropout,
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
    # LR scheduler: linear warmup → cosine decay (per-step update)
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
    else:
        print("[LR scheduler] disabled (constant LR)")

    best_val_loss = float('inf')
    best_ep = -1
    for epoch in range(1, args.epochs+1):
        avg_train_loss, train_loss = train_epoch(
            epoch, model, train_loader, tokenizer, criterion, optimizer, device,
            use_species=args.use_species, scheduler=scheduler)
        # --- 验证 ---
        avg_val_loss, val_loss = validate_epoch(
            epoch, model, val_loader, tokenizer, criterion, device,
            use_species=args.use_species)
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
                            scheduler=scheduler, epoch=epoch)

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
                            scheduler=scheduler, epoch=epoch)
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
                            scheduler=scheduler, epoch=epoch)
    # --- 测试 ---
    print("\n","*"*30, "Testing model...", "*"*30,"\n")
    avg_test_loss, test_loss = validate_epoch(
        epoch, model, test_loader, tokenizer, criterion, device,
        use_species=args.use_species)
    print(f"Test MSE Loss: {avg_test_loss:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["best_val_loss"] = best_val_loss
    metrics_df["final_test_loss"] = avg_test_loss
    metrics_df.to_csv(os.path.join(args.save_dir, args.metrics_name), index=False)
    
    if args.use_wandb:
        wandb.log({"test_mse": avg_test_loss, "best_val_mse": best_val_loss})
        wandb.finish()

if __name__ == "__main__":
    main()
