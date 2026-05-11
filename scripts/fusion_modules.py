"""Five fusion strategies for combining ESM peptide embedding with the
TaxonomyGNN species embedding.

All fusion modules expose the SAME interface:
    forward(seq_rep, species_emb, seq_tokens=None) -> [B, out_dim]
where:
    seq_rep      : [B, esm_dim]       pooled peptide embedding (mean / cls)
    species_emb  : [B, gnn_out_dim]   per-batch species feature
    seq_tokens   : [B, L, esm_dim]    full token-level ESM output (only used by cross_attn)

The downstream MLP head sees ``out_dim`` channels.

Choice of fusion sets a compatibility constraint on `gnn_out_dim`:
    concat       :  gnn_out_dim free; out_dim = esm_dim + gnn_out_dim
    gated/film   :  gnn_out_dim must == esm_dim (so they can be combined per-channel)
    cross_attn   :  gnn_out_dim must == esm_dim (used as a single Query)
    bilinear     :  gnn_out_dim free; both projected to a common dim internally
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_fusion(
    strategy: str,
    esm_dim: int,
    gnn_out_dim: int,
    *,
    bilinear_inner: int = 256,
    xattn_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory: return the selected fusion module."""
    s = strategy.lower()
    if s == "concat":
        return ConcatFusion(esm_dim, gnn_out_dim)
    if s == "gated":
        return GatedFusion(esm_dim, gnn_out_dim)
    if s == "film":
        return FiLMFusion(esm_dim, gnn_out_dim)
    if s == "cross_attn":
        return CrossAttnFusion(esm_dim, gnn_out_dim, num_heads=xattn_heads, dropout=dropout)
    if s == "bilinear":
        return BilinearFusion(esm_dim, gnn_out_dim, inner=bilinear_inner)
    raise ValueError(f"Unknown fusion strategy: {strategy!r}")


# ---------------------------------------------------------------------------
class ConcatFusion(nn.Module):
    """``[seq_rep ; species_emb]`` -> output of width esm_dim + gnn_out_dim."""

    def __init__(self, esm_dim: int, gnn_out_dim: int):
        super().__init__()
        self.out_dim = esm_dim + gnn_out_dim

    def forward(self, seq_rep: torch.Tensor, species_emb: torch.Tensor,
                seq_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.cat([seq_rep, species_emb], dim=-1)


# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """Per-channel sigmoid gate that interpolates between ESM and GNN.

    Requires gnn_out_dim == esm_dim.
        gate = sigmoid(Linear([seq_rep ; species_emb]))   in [0,1]^esm_dim
        fused = gate * seq_rep + (1 - gate) * species_emb
    """

    def __init__(self, esm_dim: int, gnn_out_dim: int):
        super().__init__()
        if gnn_out_dim != esm_dim:
            raise ValueError(
                f"GatedFusion requires gnn_out_dim == esm_dim "
                f"({gnn_out_dim} vs {esm_dim}). Set --gnn_out_dim {esm_dim}."
            )
        self.out_dim = esm_dim
        self.gate_proj = nn.Linear(2 * esm_dim, esm_dim)

    def forward(self, seq_rep, species_emb, seq_tokens=None):
        gate = torch.sigmoid(self.gate_proj(torch.cat([seq_rep, species_emb], dim=-1)))
        return gate * seq_rep + (1.0 - gate) * species_emb


# ---------------------------------------------------------------------------
class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation:
        gamma, beta = Linear(species_emb)
        fused       = (1 + gamma) * seq_rep + beta

    Requires gnn_out_dim == esm_dim.
    """

    def __init__(self, esm_dim: int, gnn_out_dim: int):
        super().__init__()
        if gnn_out_dim != esm_dim:
            raise ValueError(
                f"FiLMFusion requires gnn_out_dim == esm_dim "
                f"({gnn_out_dim} vs {esm_dim}). Set --gnn_out_dim {esm_dim}."
            )
        self.out_dim = esm_dim
        self.gamma_proj = nn.Linear(esm_dim, esm_dim)
        self.beta_proj  = nn.Linear(esm_dim, esm_dim)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, seq_rep, species_emb, seq_tokens=None):
        gamma = self.gamma_proj(species_emb)
        beta  = self.beta_proj(species_emb)
        return (1.0 + gamma) * seq_rep + beta


# ---------------------------------------------------------------------------
class CrossAttnFusion(nn.Module):
    """Multi-head cross attention with the GNN embedding as Query.

    species_emb -> Q [B, 1, esm_dim]
    seq_tokens  -> K, V [B, L, esm_dim]
    out         -> [B, esm_dim] (attn output squeezed)

    Requires gnn_out_dim == esm_dim AND seq_tokens to be supplied.
    """

    def __init__(self, esm_dim: int, gnn_out_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if gnn_out_dim != esm_dim:
            raise ValueError(
                f"CrossAttnFusion requires gnn_out_dim == esm_dim "
                f"({gnn_out_dim} vs {esm_dim}). Set --gnn_out_dim {esm_dim}."
            )
        self.out_dim = esm_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=esm_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # Mix the attention output with the original pooled rep so we don't
        # discard the strong sequence signal entirely.
        self.mix = nn.Linear(2 * esm_dim, esm_dim)

    def forward(self, seq_rep, species_emb, seq_tokens=None):
        if seq_tokens is None:
            raise ValueError("CrossAttnFusion needs ``seq_tokens`` ([B,L,D]).")
        q = species_emb.unsqueeze(1)                                # [B, 1, D]
        attn_out, _ = self.attn(q, seq_tokens, seq_tokens, need_weights=False)
        attn_out = attn_out.squeeze(1)                              # [B, D]
        return self.mix(torch.cat([seq_rep, attn_out], dim=-1))


# ---------------------------------------------------------------------------
class BilinearFusion(nn.Module):
    """Low-rank Hadamard fusion:
        f = Linear_e(seq_rep) * Linear_g(species_emb)        # element-wise
    Then a final Linear projects to esm_dim for the head's input.
    """

    def __init__(self, esm_dim: int, gnn_out_dim: int, inner: int = 256):
        super().__init__()
        self.out_dim = esm_dim
        self.proj_e = nn.Linear(esm_dim, inner)
        self.proj_g = nn.Linear(gnn_out_dim, inner)
        self.proj_out = nn.Linear(inner, esm_dim)

    def forward(self, seq_rep, species_emb, seq_tokens=None):
        e = self.proj_e(seq_rep)
        g = self.proj_g(species_emb)
        return self.proj_out(F.relu(e * g))
