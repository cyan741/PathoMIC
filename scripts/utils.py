from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import torch
def transform(x:np.ndarray, mean:float, std:float) -> np.ndarray:
    return (x - mean) / std if std != 0 else x

def inverse_transform(x:np.ndarray, original_mean:float, original_std:float) -> np.ndarray:
    return x * original_std + original_mean

def evaluate_metrics(y_true, y_pred):
    """
    y_true: 真实标签 (numpy array or list)
    y_pred: 模型预测值 (numpy array or list)
    """
    # 确保输入是 numpy 数组且为一维且为 float
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    # 过滤 nan 和 inf
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    # 长度不足时返回 nan
    if len(y_true) < 2 or len(y_pred) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
    
    # 1. 误差指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 2. 拟合优度
    r2 = r2_score(y_true, y_pred)
    
    # 3. 相关性指标 (注意：这两个函数返回 (correlation, p-value))
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    
    print(f"--- Evaluation Metrics ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Pearson R:  {pearson_corr:.4f}")
    print(f"Spearman R: {spearman_corr:.4f}")
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }

# --- 使用示例 ---
# 在验证集跑完后：
# all_preds = [2.1, 1.9, ...]
# all_labels = [2.0, 1.8, ...]
# metrics = evaluate_metrics(all_labels, all_preds)
# wandb.log(metrics)  <-- 可以直接 log 到 wandb


def plot_scatter(args, epoch, y_true, y_pred, title="Predicted vs True MIC"):
    # 将输入标准化为一维 float 数组并过滤异常值
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # 如果没有数据则直接返回
    if len(y_true) == 0 or len(y_pred) == 0:
        print("plot_scatter: no valid data to plot")
        return

    # 自动拼接参数信息
    extra = f"{getattr(args, 'plm', 'NA')}_lr={getattr(args, 'lr', 'NA')}_bs={getattr(args, 'batch_size', 'NA')}"
    full_title = f"{title} ({extra})"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    # 画对角线
    m = min(float(np.min(y_true)), float(np.min(y_pred)))
    M = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([m, M], [m, M], '--r', lw=2)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(full_title)
    plt.tight_layout()
    # 保存图片名也带参数
    save_dir = "/data/luyq/PLM_AMP_Regression/figs"
    os.makedirs(save_dir, exist_ok=True)
    # sanitize extra for filename
    safe_extra = extra.replace('=', '').replace('.', 'p')
    save_path = f"{save_dir}/scatter_{safe_extra}.png"
    plt.savefig(save_path)
    # 上传到 wandb（如果初始化了）
    
    wandb.define_metric("epoch")
    wandb.define_metric("scatter_plot", step_metric="epoch")
    try:
        wandb.log({"scatter_plot": wandb.Image(save_path),
                   "epoch":epoch})
    except Exception:
        print("wandb not initialized, skipping log.")
    plt.close()


def save_checkpoint(model, optimizer: torch.optim.Optimizer, output_path: str) -> None:
    '''
    save 
    model state dict 
    optimizer state dict
    epoch number
    to output_path

    '''
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # save scheduler state dict if scheduler is not None
    # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None 
}
    torch.save(checkpoint, output_path)
