"""
PairUpdate — pure PyTorch implementation
Based on Algorithm 7 from the AlphaFold 3 paper.

Replaces the simplified PairUpdate stub in main_trunk.py.

Steps
-----
1. d_ij = ||r_i^center − r_j^center||           scalar pairwise distances
2. b_ij = LinearNoBias(Transform_RBF(d_ij))      RBF-discretized distance bias  ∈ R^c
3. z_ij += DropoutRowwise_0.25(TriangleAttentionStartingNodeWithBias(z_ij, b_ij))
4. z_ij += DropoutColumnwise_0.25(TriangleAttentionEndingNodeWithBias(z_ij, b_ij))
5. z_ij += Transition(z_ij)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, jaxtyped

from architecture.atom_transformers import LinearNoBias


# ---------------------------------------------------------------------------
# RBF transform  (Transform_RBF in step 2)
# ---------------------------------------------------------------------------

class TransformRBF(nn.Module):
    """
    Converts a scalar distance d into a fixed-size RBF feature vector,
    then projects to R^c via LinearNoBias.

    Centers are evenly spaced in [d_min, d_max]; width = spacing.
    """
    centers: torch.Tensor  # registered buffer; narrowed from Tensor | Module

    def __init__(self, c: int, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 22.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(d_min, d_max, n_rbf))
        self.sigma = (d_max - d_min) / n_rbf
        self.proj  = LinearNoBias(n_rbf, c)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        d: Float[torch.Tensor, "B N_res N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        rbf: Float[torch.Tensor, "B N_res N_res n_rbf"] = torch.exp(
            -((rearrange(d, "b n_i n_j -> b n_i n_j 1") - self.centers) ** 2) / self.sigma ** 2
        )
        return self.proj(rbf)


# ---------------------------------------------------------------------------
# TriangleAttentionStartingNode  (step 3)
# "Starting node" = row-wise gated self-attention on z_ij,
#  biased by b_ij (coordinate pair bias).
# ---------------------------------------------------------------------------

class TriangleAttentionStartingNodeWithBias(nn.Module):
    """
    For each row i, attend over all j using queries/keys/values from z_ij,
    with an additive pair bias b_ij projected to per-head scalars.
    Gate with a sigmoid on z_ij.
    """
    def __init__(self, c: int, n_heads: int = 4):
        super().__init__()
        assert c % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = c // n_heads

        self.norm     = nn.LayerNorm(c)
        self.to_q     = LinearNoBias(c, c)
        self.to_k     = LinearNoBias(c, c)
        self.to_v     = LinearNoBias(c, c)
        self.to_g     = nn.Linear(c, c)           # gating (bias allowed)
        self.to_b     = LinearNoBias(c, n_heads)  # bias projection b_ij → n_heads
        self.norm_b   = nn.LayerNorm(c)
        self.to_out   = LinearNoBias(c, c)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        b: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        H, D = self.n_heads, self.head_dim

        zn: Float[torch.Tensor, "B N_res N_res c_pair"] = self.norm(z)
        Q: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(self.to_q(zn), "b i j (h d) -> b i j h d", h=H)
        K: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(self.to_k(zn), "b i j (h d) -> b i j h d", h=H)
        V: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(self.to_v(zn), "b i j (h d) -> b i j h d", h=H)
        G: Float[torch.Tensor, "B N_res N_res c_pair"] = torch.sigmoid(self.to_g(zn))

        # Starting node: fix i, attend j (queries) over k (keys)
        attn: Float[torch.Tensor, "B N_res n_heads N_res N_res"] = einsum(Q, K, "b i j h d, b i k h d -> b i h j k") / math.sqrt(D)

        # Bias b[j, k, h] is independent of starting node i → broadcast over i
        bias: Float[torch.Tensor, "B 1 n_heads N_res N_res"] = rearrange(self.to_b(self.norm_b(b)), "b n_j n_k h -> b 1 h n_j n_k")
        attn = F.softmax(attn + bias, dim=-1)

        out: Float[torch.Tensor, "B N_res N_res c_pair"] = rearrange(
            einsum(attn, V, "b i h j k, b i k h d -> b i j h d"),
            "b i j h d -> b i j (h d)",
        )
        out = G * out
        return self.to_out(out)


# ---------------------------------------------------------------------------
# TriangleAttentionEndingNode  (step 4)
# "Ending node" = column-wise gated self-attention on z_ij,
#  biased by b_ij.
# ---------------------------------------------------------------------------

class TriangleAttentionEndingNodeWithBias(nn.Module):
    """
    For each column j, attend over all i using queries/keys/values from z_ij,
    biased by b_ij.
    """
    def __init__(self, c: int, n_heads: int = 4):
        super().__init__()
        assert c % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = c // n_heads

        self.norm   = nn.LayerNorm(c)
        self.to_q   = LinearNoBias(c, c)
        self.to_k   = LinearNoBias(c, c)
        self.to_v   = LinearNoBias(c, c)
        self.to_g   = nn.Linear(c, c)
        self.to_b   = LinearNoBias(c, n_heads)
        self.norm_b = nn.LayerNorm(c)
        self.to_out = LinearNoBias(c, c)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        b: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        H, D = self.n_heads, self.head_dim

        zn: Float[torch.Tensor, "B N_res N_res c_pair"] = self.norm(z)
        # Transpose to column-first so ending node j leads
        Q: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(self.to_q(zn), "b n_i n_j (h d) -> b n_j n_i h d", h=H)
        K: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(self.to_k(zn), "b n_i n_j (h d) -> b n_j n_i h d", h=H)
        V: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(self.to_v(zn), "b n_i n_j (h d) -> b n_j n_i h d", h=H)
        G: Float[torch.Tensor, "B N_res N_res c_pair"] = torch.sigmoid(self.to_g(zn))

        # Ending node: fix j, attend i (queries) over k (keys)
        attn: Float[torch.Tensor, "B N_res n_heads N_res N_res"] = einsum(Q, K, "b n_j n_i h d, b n_j n_k h d -> b n_j h n_i n_k") / math.sqrt(D)

        # Bias b[i, j, h] for query i ending at fixed j, broadcast over k
        bias: Float[torch.Tensor, "B N_res n_heads N_res 1"] = rearrange(self.to_b(self.norm_b(b)), "b n_i n_j h -> b n_j h n_i 1")
        attn = F.softmax(attn + bias, dim=-1)

        # Weighted sum, then transpose back to (B, N_i, N_j, C)
        out: Float[torch.Tensor, "B N_res N_res c_pair"] = rearrange(
            einsum(attn, V, "b n_j h n_i n_k, b n_j n_k h d -> b n_j n_i h d"),
            "b n_j n_i h d -> b n_i n_j (h d)",
        )
        out = G * out
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transition  (step 5)
# ---------------------------------------------------------------------------

class Transition(nn.Module):
    def __init__(self, c: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.ff1  = LinearNoBias(c, c * expansion)
        self.ff2  = LinearNoBias(c * expansion, c)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Works for any leading dims (B N_res N_res c) or (B N_res c)
        zn = self.norm(z)
        hidden = F.relu(self.ff1(zn))
        return self.ff2(hidden)


# ---------------------------------------------------------------------------
# DropoutRowwise / DropoutColumnwise
# ---------------------------------------------------------------------------

class DropoutRowwise(nn.Module):
    """Drops entire rows (dim 1) with probability p during training."""
    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask_shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
        mask = torch.ones(mask_shape, device=x.device)
        mask = F.dropout(mask, p=self.p, training=True) * (1 - self.p)
        return x * mask


class DropoutColumnwise(nn.Module):
    """Drops entire columns (dim 2) with probability p during training."""
    def __init__(self, p: float = 0.25):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask_shape = (x.size(0), 1, x.size(2)) + (1,) * (x.dim() - 3)
        mask = torch.ones(mask_shape, device=x.device)
        mask = F.dropout(mask, p=self.p, training=True) * (1 - self.p)
        return x * mask


# ---------------------------------------------------------------------------
# PairUpdate — Algorithm 7
# ---------------------------------------------------------------------------

class PairUpdate(nn.Module):
    """
    Parameters
    ----------
    c       : pair embedding dim  (default 128)
    n_rbf   : number of RBF centres
    n_heads : attention heads
    dropout : rowwise/columnwise dropout probability (0.25 per paper)
    """

    def __init__(
        self,
        c:       int   = 128,
        n_rbf:   int   = 16,
        n_heads: int   = 4,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Step 2
        self.rbf      = TransformRBF(c, n_rbf=n_rbf)

        # Step 3
        self.tri_start   = TriangleAttentionStartingNodeWithBias(c, n_heads)
        self.drop_row    = DropoutRowwise(dropout)

        # Step 4
        self.tri_end     = TriangleAttentionEndingNodeWithBias(c, n_heads)
        self.drop_col    = DropoutColumnwise(dropout)

        # Step 5
        self.transition  = Transition(c)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z:        Float[torch.Tensor, "B N_res N_res c_pair"],
        r_center: Float[torch.Tensor, "B N_res 3"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:

        # ------------------------------------------------------------------
        # Step 1: d_ij = ||r_i^center − r_j^center||
        # ------------------------------------------------------------------
        diff: Float[torch.Tensor, "B N_res N_res 3"] = rearrange(r_center, "b n d -> b n 1 d") - rearrange(r_center, "b n d -> b 1 n d")
        d_ij: Float[torch.Tensor, "B N_res N_res"]   = diff.norm(dim=-1)

        # ------------------------------------------------------------------
        # Step 2: b_ij = LinearNoBias(Transform_RBF(d_ij))
        # ------------------------------------------------------------------
        b_ij: Float[torch.Tensor, "B N_res N_res c_pair"] = self.rbf(d_ij)

        # ------------------------------------------------------------------
        # Step 3: z_ij += DropoutRowwise(TriangleAttentionStartingNode(z, b))
        # ------------------------------------------------------------------
        z = z + self.drop_row(self.tri_start(z, b_ij))

        # ------------------------------------------------------------------
        # Step 4: z_ij += DropoutColumnwise(TriangleAttentionEndingNode(z, b))
        # ------------------------------------------------------------------
        z = z + self.drop_col(self.tri_end(z, b_ij))

        # ------------------------------------------------------------------
        # Step 5: z_ij += Transition(z_ij)
        # ------------------------------------------------------------------
        z = z + self.transition(z)

        return z
