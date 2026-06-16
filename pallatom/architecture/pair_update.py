"""PairUpdate — based on Algorithm 7 from the AlphaFold 3 paper.

Replaces the simplified PairUpdate stub in main_trunk.py.

Steps
-----
1. d_ij = ||r_i^center - r_j^center||           scalar pairwise distances
2. b_ij = LinearNoBias(Transform_RBF(d_ij)) RBF-discretized distance bias ∈ R^c
3. z_ij += DropoutRowwise_0.25(TriangleAttentionStartingNodeWithBias(z_ij,
b_ij))
4. z_ij += DropoutColumnwise_0.25(TriangleAttentionEndingNodeWithBias(z_ij,
b_ij))
5. z_ij += Transition(z_ij)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.errors import InvalidPairHeadDimensionError
from architecture.layers import LayerNorm, LinearNoBias, TypedLinear
from beartype import beartype
from einops import einsum, rearrange, reduce
from jaxtyping import Float, jaxtyped
from typing_extensions import override

# ---------------------------------------------------------------------------
# RBF transform  (Transform_RBF in step 2)
# ---------------------------------------------------------------------------


class TransformRBF(nn.Module):
    """Converts scalar distance into RBF feature vector then to pair embed dim.

    Centers are evenly spaced in [d_min, d_max]; width = spacing.
    """

    def __init__(
        self,
        c: int,
        n_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 22.0,
    ) -> None:
        super().__init__()
        centers: Float[torch.Tensor, "n_rbf"] = torch.linspace(
            d_min,
            d_max,
            n_rbf,
        )
        self.centers: Float[torch.Tensor, "n_rbf"]
        self.register_buffer("centers", centers)
        self.sigma: float = (d_max - d_min) / n_rbf
        self.proj: LinearNoBias = LinearNoBias(n_rbf, c)

    @override
    def __call__(
        self,
        d: Float[torch.Tensor, "B N_res N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(d)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        d: Float[torch.Tensor, "B N_res N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Project pairwise distances with RBF basis into pair embedding space.

        Args:
            d: Pairwise distance matrix of shape ``(B, N_res, N_res)``.

        Returns:
            Pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
        """
        rbf: Float[torch.Tensor, "B N_res N_res n_rbf"] = torch.exp(
            -((rearrange(d, "b n_i n_j -> b n_i n_j 1") - self.centers) ** 2)
            / self.sigma**2,
        )
        result: Float[torch.Tensor, "B N_res N_res c_pair"] = self.proj(rbf)
        return result


# ---------------------------------------------------------------------------
# TriangleAttentionStartingNode  (step 3)
# "Starting node" = row-wise gated self-attention on z_ij,
#  biased by b_ij (coordinate pair bias).
# ---------------------------------------------------------------------------


class TriangleAttentionStartingNodeWithBias(nn.Module):
    """Triangle attention over rows with pair bias and gating.

    For each row i, attend over all j using queries/keys/values from z_ij,
    with an additive pair bias b_ij projected to per-head scalars.
    Gate with a sigmoid on z_ij.
    """

    def __init__(self, c_pair: int, n_heads: int = 4) -> None:
        super().__init__()
        if c_pair % n_heads != 0:
            raise InvalidPairHeadDimensionError(c_pair, n_heads)
        self.n_heads: int = n_heads
        self.head_dim: int = c_pair // n_heads

        self.layer_norm: LayerNorm = LayerNorm(c_pair)
        self.to_q: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.to_k: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.to_v: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.to_g: TypedLinear = TypedLinear(
            c_pair,
            c_pair,
        )  # gating (bias allowed)
        self.to_out: LinearNoBias = LinearNoBias(c_pair, c_pair)

    @override
    def __call__(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        b: Float[torch.Tensor, "B N_res N_res n_heads"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(z, b)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        b: Float[torch.Tensor, "B N_res N_res n_heads"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Apply row gated biased self-attention to update pair embeddings.

        Args:
            z: Pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
            b: Per-head additive attention bias of shape
                ``(B, N_res, N_res, n_heads)``.

        Returns:
            Updated pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
        """
        zn: Float[torch.Tensor, "B N_res N_res c_pair"] = self.layer_norm(z)
        q: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            self.to_q(zn),
            "B n_i n_j (n_heads head_dim) -> B n_i n_j n_heads head_dim",
            n_heads=self.n_heads,
        )
        k: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            self.to_k(zn),
            "B n_i n_j (n_heads head_dim) -> B n_i n_j n_heads head_dim",
            n_heads=self.n_heads,
        )
        v: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            self.to_v(zn),
            "B n_i n_j (n_heads head_dim) -> B n_i n_j n_heads head_dim",
            n_heads=self.n_heads,
        )
        g: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            torch.sigmoid(self.to_g(zn)),
            "B n_i n_j (n_heads head_dim) -> B n_i n_j n_heads head_dim",
            n_heads=self.n_heads,
        )

        # Starting node: fix i, attend j (queries) over k (keys)
        attn: Float[torch.Tensor, "B N_res n_heads N_res N_res"] = einsum(
            q,
            k,
            "b i j h d, b i k h d -> b i h j k",
        ) / math.sqrt(self.head_dim)

        attn = F.softmax(
            attn + rearrange(b, "b n_j n_k h -> b 1 h n_j n_k"),
            dim=-1,
        )
        intermediate: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = (
            einsum(attn, v, "b i h j k, b i k h d -> b i j h d")
        )
        intermediate = g * intermediate
        out: Float[torch.Tensor, "B N_res N_res c_pair"] = rearrange(
            intermediate,
            "b i j h d -> b i j (h d)",
        )
        return self.to_out(out)


# ---------------------------------------------------------------------------
# TriangleAttentionEndingNode  (step 4)
# "Ending node" = column-wise gated self-attention on z_ij,
#  biased by b_ij.
# ---------------------------------------------------------------------------


class TriangleAttentionEndingNodeWithBias(nn.Module):
    """Column-wise triangle attention over pair embeddings, biased by b_ij.

    For each column j, attends over all i using queries/keys/values from
    z_ij with an additive per-head bias derived from b_ij.
    """

    def __init__(self, c_pair: int, n_heads: int = 4) -> None:
        super().__init__()
        if c_pair % n_heads != 0:
            raise InvalidPairHeadDimensionError(c_pair, n_heads)
        self.n_heads: int = n_heads
        self.head_dim: int = c_pair // n_heads

        self.layer_norm: LayerNorm = LayerNorm(c_pair)
        self.to_q: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.to_k: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.to_v: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.to_g: TypedLinear = TypedLinear(c_pair, c_pair)
        self.to_out: LinearNoBias = LinearNoBias(c_pair, c_pair)

    @override
    def __call__(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        b: Float[torch.Tensor, "B N_res N_res n_heads"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(z, b)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        b: Float[torch.Tensor, "B N_res N_res n_heads"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Apply column gated biased self-attention to update pair embeddings.

        Args:
            z: Pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
            b: Per-head additive attention bias of shape
                ``(B, N_res, N_res, n_heads)``.

        Returns:
            Updated pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
        """
        zn: Float[torch.Tensor, "B N_res N_res c_pair"] = self.layer_norm(z)
        # Transpose to column-first so ending node j leads
        q: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            self.to_q(zn),
            "b n_i n_j (h d) -> b n_j n_i h d",
            h=self.n_heads,
        )
        k: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            self.to_k(zn),
            "b n_i n_j (h d) -> b n_j n_i h d",
            h=self.n_heads,
        )
        v: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            self.to_v(zn),
            "b n_i n_j (h d) -> b n_j n_i h d",
            h=self.n_heads,
        )
        g: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = rearrange(
            torch.sigmoid(self.to_g(zn)),
            "B n_i n_j (n_heads head_dim) -> B n_j n_i n_heads head_dim",
            n_heads=self.n_heads,
        )

        # Ending node: fix j, attend i (queries) over k (keys)
        attn: Float[torch.Tensor, "B N_res n_heads N_res N_res"] = einsum(
            q,
            k,
            "b n_j n_i h d, b n_j n_k h d -> b n_j h n_i n_k",
        ) / math.sqrt(self.head_dim)

        attn = F.softmax(
            attn + rearrange(b, "b n_i n_k h -> b 1 h n_i n_k"),
            dim=-1,
        )
        intermediate: Float[torch.Tensor, "B N_res N_res n_heads head_dim"] = (
            einsum(attn, v, "b n_j h n_i n_k, b n_j n_k h d -> b n_j n_i h d")
        )
        intermediate = g * intermediate
        # Weighted sum, then transpose back to (B, N_i, N_j, C)
        out: Float[torch.Tensor, "B N_res N_res c_pair"] = rearrange(
            intermediate,
            "b n_j n_i h d -> b n_i n_j (h d)",
        )
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transition  (step 5)
# ---------------------------------------------------------------------------


class Transition(nn.Module):
    """Two-layer feed-forward transition block applied to pair embeddings.

    Uses a SwiGLU-style gate: ``silu(W1·x) ⊙ W2·x`` projected back to ``c``.
    """

    def __init__(self, c: int, expansion: int = 4) -> None:
        super().__init__()
        self.layer_norm: LayerNorm = LayerNorm(c)
        self.x_to_a: LinearNoBias = LinearNoBias(c, c * expansion)
        self.x_to_b: LinearNoBias = LinearNoBias(c, c * expansion)
        self.hidden_to_out: LinearNoBias = LinearNoBias(c * expansion, c)

    @override
    def __call__(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(x)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Apply SwiGLU two-layer FFN to any leading-dimension tensor.

        Normalises ``x``, computes ``silu(W1·x) ⊙ W2·x``, and projects the
        result back to the original embedding dimension.

        Args:
            x: Input tensor of any shape ``(..., c)`` where the last dimension
                matches the embedding size ``c`` supplied at construction.

        Returns:
            Tensor of the same shape as ``x`` after the feed-forward transform.
        """
        # Works for any leading dims (B N_res N_res c) or (B N_res c)
        x = self.layer_norm(x)
        a = self.x_to_a(x)
        b = self.x_to_b(x)
        hidden = F.silu(a) * b
        return self.hidden_to_out(hidden)


# ---------------------------------------------------------------------------
# DropoutRowwise / DropoutColumnwise
# ---------------------------------------------------------------------------


class DropoutRowwise(nn.Module):
    """Drops entire rows (dim 1) with probability p during training."""

    def __init__(self, p: float = 0.25) -> None:
        super().__init__()
        self.p: float = p

    @override
    def __call__(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(x)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Drop entire rows of ``x`` with probability ``p`` during training.

        At inference time (``self.training`` is ``False``) or when ``p == 0``
        the input is returned unchanged.

        Args:
            x: Input tensor of shape ``(B, N_rows, ...)`` where dim 1 holds
                the rows to be dropped.

        Returns:
            Tensor of the same shape as ``x`` with randomly zeroed rows (and
            remaining rows rescaled to preserve expected values).
        """
        if not self.training or self.p == 0:
            return x
        mask_shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
        mask = torch.ones(mask_shape, device=x.device)
        mask = F.dropout(mask, p=self.p, training=True) * (1 - self.p)
        return x * mask


class DropoutColumnwise(nn.Module):
    """Drops entire columns (dim 2) with probability p during training."""

    def __init__(self, p: float = 0.25) -> None:
        super().__init__()
        self.p: float = p

    @override
    def __call__(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(x)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Drop entire columns of ``x`` with probability ``p`` during training.

        At inference time (``self.training`` is ``False``) or when ``p == 0``
        the input is returned unchanged.

        Args:
            x: Input tensor of shape ``(B, N_rows, N_cols, ...)`` where dim 2
                holds the columns to be dropped.

        Returns:
            Tensor of the same shape as ``x`` with randomly zeroed columns
            (and remaining columns rescaled to preserve expected values).
        """
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
    """Parameters

    ----------
    c       : pair embedding dim  (default 128)
    n_rbf   : number of RBF centres
    n_heads : attention heads
    dropout : rowwise/columnwise dropout probability (0.25 per paper)
    """

    def __init__(
        self,
        c: int = 128,
        n_rbf: int = 16,
        n_heads: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Step 2
        self.rbf: TransformRBF = TransformRBF(c, n_rbf=n_rbf)
        self.b_proj: LinearNoBias = LinearNoBias(c, n_heads)

        # Step 3
        self.tri_start: TriangleAttentionStartingNodeWithBias = (
            TriangleAttentionStartingNodeWithBias(c, n_heads)
        )
        self.drop_row: DropoutRowwise = DropoutRowwise(dropout)

        # Step 4
        self.tri_end: TriangleAttentionEndingNodeWithBias = (
            TriangleAttentionEndingNodeWithBias(c, n_heads)
        )
        self.drop_col: DropoutColumnwise = DropoutColumnwise(dropout)

        # Step 5
        self.transition: Transition = Transition(c)

    @override
    def __call__(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        r_center: Float[torch.Tensor, "B N_res 3"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(z, r_center)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        r_center: Float[torch.Tensor, "B N_res 3"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Update pair embeddings with triangle attention and RBF bias.

        Implements Algorithm 7 from the AlphaFold 3 paper in five steps:
        compute pairwise distances, project them through an RBF bias, apply
        rowwise triangle attention, apply columnwise triangle attention, and
        apply a feed-forward transition.

        Args:
            z: Current pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
            r_center: Center atom coordinates per residue of shape
                ``(B, N_res, 3)`` used to derive the coordinate pair bias.

        Returns:
            Updated pair embeddings of shape ``(B, N_res, N_res, c_pair)``.
        """
        # ------------------------------------------------------------------
        # Step 1: d_ij = ||r_i^center - r_j^center||
        # ------------------------------------------------------------------
        diff: Float[torch.Tensor, "B N_res N_res 3"] = rearrange(
            r_center,
            "b n d -> b n 1 d",
        ) - rearrange(r_center, "b n d -> b 1 n d")
        d_ij: Float[torch.Tensor, "B N_res N_res"] = torch.sqrt(
            reduce(diff**2, "b n m d -> b n m", "sum").clamp(min=1e-8),
        )

        # ------------------------------------------------------------------
        # Step 2: b_ij = LinearNoBias(Transform_RBF(d_ij))
        # ------------------------------------------------------------------
        b_ij: Float[torch.Tensor, "B N_res N_res n_heads"] = self.b_proj(
            self.rbf(d_ij),
        )

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
        return z + self.transition(z)
