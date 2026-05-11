"""Regression losses for the AMP-MIC long-tail / OOD setting.

All losses share a uniform interface so the training loop can stay agnostic:

    loss_fn(y_pred, y_true, meta=None) -> scalar tensor

where ``meta`` is an optional dict used by losses that need extra info:
    meta['bin_idx']    : LongTensor [B]   bin id for LDS (precomputed by data_loader)
    meta['bin_weight'] : FloatTensor [num_bins]   1/p_LDS(y) per bin
    meta['group_id']   : LongTensor [B]   group id for GroupDRO (e.g. species or bucket)

Implemented losses:
    - MSE                            (baseline)
    - HuberLoss(delta)               (robust)
    - SmoothL1Loss(beta)             (robust)
    - LDSWeighted(base, sigma=2.0)   (long-tail; reweights MSE/Huber by inverse smoothed label freq)
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
# Helper: build LDS weights from the training labels and a number of bins.
# ---------------------------------------------------------------------------
def compute_lds_weights(y: np.ndarray, num_bins: int = 50,
                        sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_edges, weight_per_bin) for Label Distribution Smoothing.

    weight_per_bin[i] = 1 / smoothed_density[i], normalised to mean 1.
    """
    from scipy.ndimage import gaussian_filter1d
    counts, bin_edges = np.histogram(y, bins=num_bins)
    smoothed = gaussian_filter1d(counts.astype(np.float32), sigma=sigma)
    smoothed = np.clip(smoothed, 1e-3, None)
    inv = 1.0 / smoothed
    inv = inv / inv.mean()                                # normalise to mean 1
    return bin_edges, inv.astype(np.float32)


def assign_bin(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map each y to its 0-indexed bin idx (clamped to valid range)."""
    idx = np.digitize(y, bin_edges, right=False) - 1
    idx = np.clip(idx, 0, len(bin_edges) - 2)
    return idx.astype(np.int64)


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
class LDSWeighted(nn.Module):
    """Wrap any per-sample base loss with LDS reweighting.

    base_loss : str in {'mse', 'huber', 'smooth_l1'}
    Per-sample weight comes from meta['bin_weight'][meta['bin_idx']] (computed
    by the data loader once at startup).
    """

    def __init__(self, base: str = "huber", delta: float = 1.0):
        super().__init__()
        self.base = base
        self.delta = delta

    def _per_sample(self, y_pred, y_true):
        diff = y_pred - y_true
        if self.base == "mse":
            return diff.pow(2)
        if self.base == "huber":
            abs_d = diff.abs()
            quad = 0.5 * abs_d.pow(2)
            lin  = self.delta * (abs_d - 0.5 * self.delta)
            return torch.where(abs_d <= self.delta, quad, lin)
        if self.base == "smooth_l1":
            return F.smooth_l1_loss(y_pred, y_true, beta=self.delta, reduction="none")
        raise ValueError(self.base)

    def forward(self, y_pred, y_true, meta=None):
        per = self._per_sample(y_pred, y_true).squeeze(-1)         # [B]
        if meta is None or "bin_idx" not in meta or "bin_weight" not in meta:
            return per.mean()
        weights = meta["bin_weight"][meta["bin_idx"]].to(per.device)
        return (per * weights).mean()


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
                 base: str = "mse", delta: float = 1.0):
        super().__init__()
        self.num_groups = num_groups
        self.eta = eta
        self.base = base
        self.delta = delta
        # Adversary's distribution over groups (uniform init).
        self.register_buffer("group_weights", torch.ones(num_groups) / num_groups)

    def _per_sample(self, y_pred, y_true):
        if self.base == "mse":
            return (y_pred - y_true).pow(2).squeeze(-1)
        elif self.base == "huber":
            return F.huber_loss(y_pred, y_true, delta=self.delta, reduction="none").squeeze(-1)
        return F.smooth_l1_loss(y_pred, y_true, beta=self.delta, reduction="none").squeeze(-1)

    def forward(self, y_pred, y_true, meta=None):
        if meta is None or "group_id" not in meta:
            # Fallback: behaves like vanilla MSE.
            return self._per_sample(y_pred, y_true).mean()
        per = self._per_sample(y_pred, y_true)
        group_id = meta["group_id"].to(per.device)
        # group_loss[g] = mean of per[i] for i in group g.
        group_losses = torch.zeros(self.num_groups, device=per.device)
        group_counts = torch.zeros(self.num_groups, device=per.device)
        group_losses.index_add_(0, group_id, per)
        group_counts.index_add_(0, group_id, torch.ones_like(per))
        group_loss_avg = group_losses / group_counts.clamp(min=1.0)
        # Update the dual variable q (weights over groups).
        with torch.no_grad():
            self.group_weights = self.group_weights * torch.exp(self.eta * group_loss_avg)
            self.group_weights = self.group_weights / self.group_weights.sum()
        # Mask out empty groups so they don't contribute.
        active = (group_counts > 0).float()
        active_w = self.group_weights * active
        active_w = active_w / active_w.sum().clamp(min=1e-8)
        return (active_w * group_loss_avg).sum()


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
    if s == "lds":
        return LDSWeighted(base=kwargs.get("lds_base", "huber"),
                           delta=kwargs.get("huber_delta", 1.0))
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
                        delta=kwargs.get("huber_delta", 1.0))
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
