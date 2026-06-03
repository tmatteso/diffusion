"""Node update modules for single-representation refinement."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.layers import LayerNorm, LinearNoBias, TypedLinear
from architecture.pair_update import DropoutRowwise, Transition
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, Int, jaxtyped
from typing_extensions import override


class AdaLN(nn.Module):
    """Adaptive LayerNorm — Algorithm 26.

    Normalises a conditioned on s:
        norm_a(a) scaled and shifted by functions of norm_s(s).
    """

    def __init__(self, c_a: int, c_s: int) -> None:
        super().__init__()
        self.norm_a: LayerNorm = LayerNorm(c_a, elementwise_affine=False)
        self.norm_s: LayerNorm = LayerNorm(c_s, bias=False)
        self.to_scale: TypedLinear = TypedLinear(c_s, c_a)
        self.to_shift: LinearNoBias = LinearNoBias(c_s, c_a)

    @override
    def __call__(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(a, s)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_a"],
        s: Float[torch.Tensor, "B N_res c_s"],
    ) -> Float[torch.Tensor, "B N_res c_a"]:
        """Apply adaptive layer norm: scale and shift `a` using gated projections of `s`."""
        a = self.norm_a(a)
        s = self.norm_s(s)
        return torch.sigmoid(self.to_scale(s)) * a + self.to_shift(s)


# ---------------------------------------------------------------------------
# AttentionPairBias
# ---------------------------------------------------------------------------
# Standard single-sequence self-attention where z_ij provides a per-head
# additive bias on the attention logits, and t_i is injected as a bias on
# the queries (time / noise-level conditioning).
#
# With β_ij = 0 the pair bias is purely additive (no learned gating).
# ---------------------------------------------------------------------------


class AttentionPairBias(nn.Module):
    """Self-attention on node embeddings s_i biased by pair embeddings z_ij.

    Parameters
    ----------
    c_res   : single embedding dim  (default 256)
    c_pair  : pair   embedding dim
    n_heads : number of attention heads (default 8 per Algorithm 6)
    """

    def __init__(self, c_res: int, c_pair: int, n_heads: int = 8) -> None:
        super().__init__()
        assert c_res % n_heads == 0, "c_res must be divisible by n_heads"
        self.n_heads: int = n_heads
        self.head_dim: int = c_res // n_heads

        self.adaLN: AdaLN = AdaLN(c_a=c_res, c_s=c_res)
        # all actual model callers (NodeUpdate, DiffusionTransformer) always pass a real s,
        # so norm_a is never called
        self.norm_a: LayerNorm = LayerNorm(c_res, elementwise_affine=False)
        self.a_to_q: TypedLinear = TypedLinear(c_res, c_res)
        self.a_to_k: TypedLinear = TypedLinear(c_res, c_res)
        self.a_to_v: TypedLinear = TypedLinear(c_res, c_res)
        self.z_to_b: LinearNoBias = LinearNoBias(c_pair, self.n_heads)
        self.a_to_g: LinearNoBias = LinearNoBias(c_res, c_res)
        self.s_to_a: TypedLinear = TypedLinear(c_res, c_res)  # biasinit=-2.0
        _: Float[torch.Tensor, "..."] = nn.init.constant_(self.s_to_a.bias, -2.0)
        self.out_to_a: LinearNoBias = LinearNoBias(c_res, c_res)

        self.norm_z: LayerNorm = LayerNorm(c_pair)

    @override
    def __call__(
        self,
        a: Float[torch.Tensor, "B N_res c_res"],
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_j c_pair"],
        beta: Float[torch.Tensor, "B N_res N_j"] | None = None,
        neighbor_idx: Int[torch.Tensor, "B N_res N_j"] | None = None,
    ) -> Float[torch.Tensor, "B N_res c_res"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(a, s, z, beta, neighbor_idx)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_res"],
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_j c_pair"],
        beta: Float[torch.Tensor, "B N_res N_j"] | None = None,
        neighbor_idx: Int[torch.Tensor, "B N_res N_j"] | None = None,
    ) -> Float[torch.Tensor, "B N_res c_res"]:
        """Compute time-conditioned pair-biased attention over residue single embeddings.

        Supports both dense ``[B, N, N, c_pair]`` and sparse ``[B, N, K, c_pair]`` pair
        tensors.  When ``neighbor_idx`` is provided the attention is sparse: for each query
        position i, only its K neighbours listed in ``neighbor_idx[i]`` are used as
        keys/values, keeping the logit shape ``(B, n_heads, N, K)`` consistent with the
        sparse pair bias derived from ``z``.  When ``neighbor_idx`` is ``None``, standard
        dense self-attention over all N positions is used.

        Args:
            a: Node embeddings of shape ``(B, N, c_res)``.
            s: Optional conditioning embeddings of shape ``(B, N, c_res)``; if ``None``
                a plain LayerNorm is applied instead of AdaLN.
            z: Pair embeddings — dense ``(B, N, N, c_pair)`` or sparse ``(B, N, K, c_pair)``.
            beta: Optional additive attention bias of shape ``(B, N, N_j)`` matching z.
            neighbor_idx: Sparse neighbour indices ``(B, N, K)``; required when z is sparse.

        Returns:
            Updated node embeddings of shape ``(B, N, c_res)``.
        """
        a = self.adaLN(a, s) if s is not None else self.norm_a(a)

        q: Float[torch.Tensor, "B N_res n_heads head_dim"] = rearrange(
            self.a_to_q(a),
            "B N_res (n_heads head_dim) -> B N_res n_heads head_dim",
            n_heads=self.n_heads,
        )
        k: Float[torch.Tensor, "B N_res n_heads head_dim"] = rearrange(
            self.a_to_k(a),
            "B N_res (n_heads head_dim) -> B N_res n_heads head_dim",
            n_heads=self.n_heads,
        )
        v: Float[torch.Tensor, "B N_res n_heads head_dim"] = rearrange(
            self.a_to_v(a),
            "B N_res (n_heads head_dim) -> B N_res n_heads head_dim",
            n_heads=self.n_heads,
        )
        pair_bias: Float[torch.Tensor, "B n_heads N_res N_j"] = rearrange(
            self.z_to_b(self.norm_z(z)),
            "B N_i N_j n_heads -> B n_heads N_i N_j",
        )
        if beta is not None:
            pair_bias = pair_bias + rearrange(beta, "B N_i N_j -> B 1 N_i N_j")

        g: Float[torch.Tensor, "B N_res n_heads head_dim"] = torch.sigmoid(
            rearrange(
                self.a_to_g(a),
                "B N_res (n_heads head_dim) -> B N_res n_heads head_dim",
                n_heads=self.n_heads,
            )
        )

        if neighbor_idx is not None:
            B = k.shape[0]
            batch_idx = torch.arange(B, device=k.device)[:, None, None]  # (B, 1, 1) broadcasts
            k_gathered: Float[torch.Tensor, "B N_res N_j n_heads head_dim"] = k[
                batch_idx, neighbor_idx
            ]
            attn: Float[torch.Tensor, "B n_heads N_res N_j"] = einsum(
                q, k_gathered, "B N h d, B N K h d -> B h N K"
            ) / math.sqrt(self.head_dim)
            attn = F.softmax(attn + pair_bias, dim=-1)
            v_gathered: Float[torch.Tensor, "B N_res N_j n_heads head_dim"] = v[
                batch_idx, neighbor_idx
            ]
            intermediate: Float[torch.Tensor, "B N_res n_heads head_dim"] = einsum(
                attn, v_gathered, "B h N K, B N K h d -> B N h d"
            )
        else:
            # Dense path: full N-by-N self-attention.
            attn = einsum(
                q, k, "B N_q n_heads head_dim, B N_k n_heads head_dim -> B n_heads N_q N_k"
            ) / math.sqrt(self.head_dim)
            attn = F.softmax(attn + pair_bias, dim=-1)
            intermediate = einsum(
                attn, v, "B n_heads N_q N_k, B N_k n_heads head_dim -> B N_q n_heads head_dim"
            )

        intermediate = g * intermediate
        out: Float[torch.Tensor, "B N_res c_res"] = rearrange(
            intermediate,
            "B N_q n_heads head_dim -> B N_q (n_heads head_dim)",
        )
        a = self.out_to_a(out)

        if s is not None:
            return torch.sigmoid(self.s_to_a(s)) * a
        return a


# ---------------------------------------------------------------------------
# NodeUpdate — Algorithm 6
# ---------------------------------------------------------------------------


class NodeUpdate(nn.Module):
    """Parameters

    ----------
    c       : single embedding dim
    c_pair  : pair   embedding dim
    n_heads : attention heads       (default 8, per Algorithm 6)
    dropout : rowwise dropout prob  (default 0.25, per Algorithm 6)
    """

    def __init__(
        self,
        c: int,
        c_pair: int,
        n_heads: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Step 1
        self.attn_pair_bias: AttentionPairBias = AttentionPairBias(c, c_pair, n_heads)
        self.dropout_row: DropoutRowwise = DropoutRowwise(dropout)

        # Step 2
        self.transition: Transition = Transition(c)

    @override
    def __call__(
        self,
        s: Float[torch.Tensor, "B N_res c_res"],
        t: Float[torch.Tensor, "B N_res c_res"],
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res c_res"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(s, t, z)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        s: Float[torch.Tensor, "B N_res c_res"],
        t: Float[torch.Tensor, "B N_res c_res"],
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res c_res"]:
        """Apply pair-biased attention and transition to update the residue single embedding."""
        # ------------------------------------------------------------------
        # Step 1: s_i += DropoutRowwise_0.25(AttentionPairBias(s, t, z, β=0, N_head=8))
        # ------------------------------------------------------------------
        # DropoutRowwise expects leading (B, rows, ...) — expand s to (B, N_res, 1, c)
        # so it drops rows independently per batch item, then squeeze back.
        attn_out: Float[torch.Tensor, "B N_res c_res"] = self.attn_pair_bias(s, t, z)
        s = s + rearrange(
            self.dropout_row(rearrange(attn_out, "b n c -> b n 1 c")),
            "b n 1 c -> b n c",
        )

        # ------------------------------------------------------------------
        # Step 2: s_i += Transition(s_i)
        # ------------------------------------------------------------------
        return s + self.transition(s)
