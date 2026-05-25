"""Regression losses for the AMP-MIC long-tail / OOD setting.

All losses share a uniform interface so the training loop can stay agnostic:

    loss_fn(y_pred, y_true, meta=None) -> scalar tensor

where ``meta`` is an optional dict used by losses that need extra info:
    meta['group_id']   : LongTensor [B]   group id for GroupDRO (e.g. species or bucket)

Implemented losses:
    - MSE                            (baseline)
    - HuberLoss(delta)               (robust)
    - SmoothL1Loss(beta)             (robust)
    - BalancedMSE / BMC              (long-tail, batch-level normalization)
    - FocalR(gamma=2.0)              (long-tail; reweight by error magnitude)
    - GroupDRO(num_groups, eta=0.01) (OOD / worst-group; needs meta['group_id'])

Each is an nn.Module so optimizer state for any internal trainable parameters
(e.g. BMC's noise variance) is checkpointed correctly.
"""
from __future__ import annotations

from typing import Dict, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, meta=None):
        return F.mse_loss(y_pred, y_true)


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true, meta=None):
        # PyTorch's smooth_l1 with beta=delta is equivalent to Huber.
        return F.huber_loss(y_pred, y_true, delta=self.delta)


class SmoothL1(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, y_pred, y_true, meta=None):
        return F.smooth_l1_loss(y_pred, y_true, beta=self.beta)




# ---------------------------------------------------------------------------
class BalancedMSE(nn.Module):
    """Balanced MSE / BMC (batch-level, Ren et al. CVPR 2022).

    Treat the regression targets as labels; compute the implicit posterior
        p_hat(y_b | x) ∝ exp(- (y_b - y_pred)^2 / (2 * sigma^2))
    where {y_b} are the labels in the current batch (sometimes called the
    BNI variant of BMC). The balanced loss is the cross-entropy of the
    one-hot target index against this posterior.

    Critical: works best with bs >= 16 and reasonably diverse y in batch.
    """

    def __init__(self, init_noise_sigma: float = 1.0, learn_noise: bool = True):
        super().__init__()
        if learn_noise:
            self.log_noise = nn.Parameter(torch.tensor(math.log(init_noise_sigma)))
        else:
            self.register_buffer("log_noise", torch.tensor(math.log(init_noise_sigma)))
        self.learn_noise = learn_noise

    def forward(self, y_pred, y_true, meta=None):
        # Both [B, 1]
        sigma2 = torch.exp(2.0 * self.log_noise)
        # Distance matrix: pred_i vs y_b (B x B)
        diff = y_pred - y_true.transpose(0, 1)                     # [B, B]
        logits = -(diff.pow(2)) / (2.0 * sigma2)
        target = torch.arange(y_pred.size(0), device=y_pred.device)
        return F.cross_entropy(logits, target)


# ---------------------------------------------------------------------------
class FocalR(nn.Module):
    """Focal regression: w_i = (sigmoid(|err_i|))^gamma.

    Up-weights samples the model is currently bad at, similar in spirit to
    classification focal loss. Works on top of MSE.
    """

    def __init__(self, gamma: float = 2.0, base: str = "mse", delta: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.base = base
        self.delta = delta

    def forward(self, y_pred, y_true, meta=None):
        err = (y_pred - y_true).abs()
        weight = torch.sigmoid(err).pow(self.gamma)
        if self.base == "mse":
            per = (y_pred - y_true).pow(2)
        elif self.base == "huber":
            per = F.huber_loss(y_pred, y_true, delta=self.delta, reduction="none")
        else:
            per = F.smooth_l1_loss(y_pred, y_true, beta=self.delta, reduction="none")
        return (per * weight).mean()


# ---------------------------------------------------------------------------
class GroupDRO(nn.Module):
    """Distributionally Robust Optimization (Sagawa et al. 2020).

    Maintains a softmax-weighted distribution over groups; updates weights
    multiplicatively so worst-performing groups get more attention.

    Args:
        num_groups: total number of groups (e.g. 4 species-count buckets).
        eta:        step size for the dual variable update.
        base:       per-sample loss to use ('mse' or 'huber').

    Expected meta:
        meta['group_id'] : LongTensor [B] in [0, num_groups)
    """

    def __init__(self, num_groups: int, eta: float = 0.01,
                 base: str = "mse", delta: float = 1.0,
                 group_counts: Optional[torch.Tensor] = None,
                 gamma: float = 0.1,
                 normalize_loss: bool = False,
                 btl: bool = False,
                 alpha: Optional[float] = None,
                 min_var_weight: float = 0.0,
                 adj: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_groups = num_groups
        self.eta = eta
        self.base = base
        self.delta = delta
        self.gamma = gamma
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.alpha = alpha
        self.min_var_weight = min_var_weight

        # Adversary distribution over groups (uniform init).
        self.register_buffer("group_weights", torch.ones(num_groups) / num_groups)        # 不需要梯度更新的张量

        if group_counts is None:
            group_counts = torch.ones(num_groups, dtype=torch.float32) # 维度[num_groups]
        self.register_buffer("group_counts", group_counts.float().clamp(min=1.0))
        self.register_buffer("group_frac", self.group_counts / self.group_counts.sum())# 维度[num_groups]

        if adj is None:
            # Optional adjustment term added to group losses before exponentiating.
            adj = torch.zeros(num_groups, dtype=torch.float32) # 维度[num_groups]
        if adj.numel() != num_groups:
            raise ValueError(f"adj length ({adj.numel()}) must equal num_groups ({num_groups})")
        self.register_buffer("adj", adj.float())

        self.register_buffer("exp_avg_loss", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("exp_avg_initialized", torch.zeros(num_groups, dtype=torch.bool))
        self.register_buffer("last_group_loss", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("last_group_count", torch.zeros(num_groups, dtype=torch.float32))

        # Stats for monitoring (saved in state_dict).
        self.register_buffer("processed_data_counts", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("update_data_counts", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("update_batch_counts", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("avg_group_loss", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("avg_per_sample_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("avg_actual_loss", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("batch_count", torch.tensor(0.0, dtype=torch.float32))

    def _per_sample(self, y_pred, y_true):
        if self.base == "mse":
            return (y_pred - y_true).pow(2).squeeze(-1) # [B]
        elif self.base == "huber":
            return F.huber_loss(y_pred, y_true, delta=self.delta, reduction="none").squeeze(-1)
        return F.smooth_l1_loss(y_pred, y_true, beta=self.delta, reduction="none").squeeze(-1)

    def _compute_group_avg(self, per_sample_loss, group_id):
        group_losses = torch.zeros(self.num_groups, device=per_sample_loss.device)
        group_counts = torch.zeros(self.num_groups, device=per_sample_loss.device)
        group_losses.index_add_(0, group_id, per_sample_loss) # index_add_()函数是PyTorch中用于在指定维度上根据索引将值添加到张量中的函数。
        # 在这里，group_losses是一个大小为num_groups的张量，group_id是一个大小为B的长整型张量，表示每个样本所属的组ID，而per_sample_loss是一个大小为B的张量，表示每个样本的损失值。
        # 通过调用group_losses.index_add_(0, group_id, per_sample_loss)，我们将per_sample_loss中的每个元素根据group_id中的索引添加到group_losses的相应位置上，从而计算出每个组的总损失。
        group_counts.index_add_(0, group_id, torch.ones_like(per_sample_loss))
        group_loss_avg = group_losses / group_counts.clamp(min=1.0) # torch.clamp()函数是PyTorch中用于限制张量元素的函数。以避免除以零的情况发生。
        return group_loss_avg, group_counts

    def _update_exp_avg_loss(self, group_loss, group_count):

        is_present = (group_count > 0)
        prev_weight = (1.0 - self.gamma * is_present.float()) * self.exp_avg_initialized.float()
        curr_weight = 1.0 - prev_weight
        self.exp_avg_loss = self.exp_avg_loss * prev_weight + group_loss * curr_weight
        self.exp_avg_initialized = self.exp_avg_initialized | is_present # 逐位或运算符，更新哪些组已经被初始化过了

    def _compute_robust_loss(self, group_loss):
        adjusted = group_loss
        if torch.all(self.adj > 0):
            adjusted = adjusted + self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted = adjusted / adjusted.sum().clamp(min=1e-8)
        with torch.no_grad():
            self.group_weights = self.group_weights * torch.exp(self.eta * adjusted.detach()) # self.eta是更新步长，adjusted.detach()表示在计算梯度时不考虑adjusted的变化，只使用其当前值。
            self.group_weights = self.group_weights / self.group_weights.sum().clamp(min=1e-8)
        robust = torch.dot(group_loss, self.group_weights)
        return robust

    def _compute_robust_loss_greedy(self, group_loss, ref_loss):
        if self.alpha is None or self.alpha <= 0:
            raise ValueError("GroupDRO with btl=True requires alpha > 0")
        sorted_idx = torch.argsort(ref_loss, descending=True)
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = int(mask.sum().item())
        if last_idx < weights.numel():
            weights[last_idx] = 1.0 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1.0 - self.min_var_weight)

        robust = torch.dot(sorted_loss, weights)
        _, unsort_idx = torch.sort(sorted_idx)
        unsorted_weights = weights[unsort_idx]
        return robust, unsorted_weights

    def _compute_robust_loss_btl(self, group_loss):
        # BTL variant: use the previous step's group loss as the reference for sorting, instead of the current loss. This is a more stable "greedy" update that doesn't require tuning eta.
        adjusted = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts)
        return self._compute_robust_loss_greedy(group_loss, adjusted)

    def reset_stats(self):
        self.processed_data_counts.zero_()
        self.update_data_counts.zero_()
        self.update_batch_counts.zero_()
        self.avg_group_loss.zero_()
        self.avg_per_sample_loss.zero_()
        self.avg_actual_loss.zero_()
        self.batch_count.zero_()

    def _update_stats(self, actual_loss, group_loss, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom = denom + (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # batch-wise average actual loss
        denom = self.batch_count + 1.0
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1.0 / denom) * actual_loss

        # counts
        self.processed_data_counts = self.processed_data_counts + group_count
        if weights is not None:
            self.update_data_counts = self.update_data_counts + group_count * (weights > 0).float()
            self.update_batch_counts = self.update_batch_counts + ((group_count * weights) > 0).float()
        else:
            self.update_data_counts = self.update_data_counts + group_count
            self.update_batch_counts = self.update_batch_counts + (group_count > 0).float()
        self.batch_count = self.batch_count + 1.0

        group_frac = self.processed_data_counts / self.processed_data_counts.sum().clamp(min=1e-8)
        self.avg_per_sample_loss = torch.dot(group_frac, self.avg_group_loss)

    def get_stats(self):
        stats = {
            "avg_actual_loss": float(self.avg_actual_loss.item()),
            "avg_per_sample_loss": float(self.avg_per_sample_loss.item()),
        }
        for idx in range(self.num_groups):
            stats[f"avg_loss_group:{idx}"] = float(self.avg_group_loss[idx].item())
            stats[f"exp_avg_loss_group:{idx}"] = float(self.exp_avg_loss[idx].item())
            stats[f"processed_data_count_group:{idx}"] = float(self.processed_data_counts[idx].item())
            stats[f"update_data_count_group:{idx}"] = float(self.update_data_counts[idx].item())
            stats[f"update_batch_count_group:{idx}"] = float(self.update_batch_counts[idx].item())
        return stats

    def log_stats(self, logger, header: Optional[str] = None):
        if logger is None:
            return
        if header:
            logger.write(header + "\n")
        logger.write(f"Average incurred loss: {self.avg_per_sample_loss.item():.4f}\n")
        logger.write(f"Average sample loss: {self.avg_actual_loss.item():.4f}\n")
        for group_idx in range(self.num_groups):
            adj = self.adj[group_idx] / torch.sqrt(self.group_counts)[group_idx]
            logger.write(
                f"  group {group_idx} "
                f"[n = {int(self.processed_data_counts[group_idx])}]:\t"
                f"loss = {self.avg_group_loss[group_idx]:.4f}  "
                f"exp loss = {self.exp_avg_loss[group_idx]:.4f}  "
                f"adjusted loss = {(self.exp_avg_loss[group_idx] + adj):.4f}  "
                f"adv prob = {self.group_weights[group_idx]:.4f}\n"
            )
        logger.flush()

    def forward(self, y_pred, y_true, meta=None):
        if meta is None or "group_id" not in meta:
            # Fallback: behaves like vanilla MSE.
            return self._per_sample(y_pred, y_true).mean()
        per = self._per_sample(y_pred, y_true)
        group_id = meta["group_id"].to(per.device)
        group_loss_avg, group_counts = self._compute_group_avg(per, group_id)
        self._update_exp_avg_loss(group_loss_avg.detach(), group_counts.detach())

        if self.btl:
            robust, weights = self._compute_robust_loss_btl(group_loss_avg)
            with torch.no_grad():
                self.group_weights = weights
        else:
            robust = self._compute_robust_loss(group_loss_avg)
            weights = self.group_weights

        with torch.no_grad():
            self.last_group_loss = group_loss_avg.detach()
            self.last_group_count = group_counts.detach()
        self._update_stats(robust.detach(), group_loss_avg.detach(), group_counts.detach(), weights.detach())
        return robust


# ---------------------------------------------------------------------------
def build_loss(loss_type: str, **kwargs) -> nn.Module:
    """Factory for losses chosen via CLI."""
    s = loss_type.lower()
    if s == "mse":
        return MSELoss()
    if s == "huber":
        return HuberLoss(delta=kwargs.get("huber_delta", 1.0))
    if s == "smooth_l1":
        return SmoothL1(beta=kwargs.get("smooth_l1_beta", 1.0))
    if s == "bmc":
        return BalancedMSE(init_noise_sigma=kwargs.get("bmc_noise", 1.0),
                           learn_noise=kwargs.get("bmc_learn_noise", True))
    if s == "focal_r":
        return FocalR(gamma=kwargs.get("focal_gamma", 2.0),
                      base=kwargs.get("focal_base", "mse"),
                      delta=kwargs.get("huber_delta", 1.0))
    if s == "group_dro":
        return GroupDRO(num_groups=kwargs["dro_num_groups"],
                        eta=kwargs.get("dro_eta", 0.01),
                        base=kwargs.get("dro_base", "mse"),
                        delta=kwargs.get("huber_delta", 1.0),
                        group_counts=kwargs.get("dro_group_counts"),
                        gamma=kwargs.get("dro_gamma", 0.1),
                        normalize_loss=kwargs.get("dro_normalize_loss", False),
                        btl=kwargs.get("dro_btl", False),
                        alpha=kwargs.get("dro_alpha", None),
                        min_var_weight=kwargs.get("dro_min_var_weight", 0.0),
                        adj=kwargs.get("dro_adj", None))
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
