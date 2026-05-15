"""Node update modules for single-representation refinement."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.atom_transformers import LinearNoBias
from architecture.pair_update import DropoutRowwise, Transition
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, jaxtyped


class AdaLN(nn.Module):
    """Adaptive LayerNorm — Algorithm 26.

    Normalises a conditioned on s:
        norm_a(a) scaled and shifted by functions of norm_s(s).
    """

    def __init__(self, c_a: int, c_s: int) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(c_a, elementwise_affine=False)
        self.norm_s = nn.LayerNorm(c_s, bias=False)
        self.to_scale = nn.Linear(c_s, c_a)
        self.to_shift = LinearNoBias(c_s, c_a)

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
        self.n_heads = n_heads
        self.head_dim = c_res // n_heads

        self.adaLN = AdaLN(c_a=c_res, c_s=c_res)
        self.norm_a = nn.LayerNorm(c_res)
        self.a_to_q = nn.Linear(c_res, c_res)
        self.a_to_k = nn.Linear(c_res, c_res)
        self.a_to_v = nn.Linear(c_res, c_res)
        self.z_to_b = LinearNoBias(c_pair, self.n_heads)
        self.a_to_g = LinearNoBias(c_res, c_res)
        self.s_to_a = nn.Linear(c_res, c_res)  # biasinit=-2.0
        self.out_to_a = LinearNoBias(c_res, c_res)

        self.norm_z = nn.LayerNorm(c_pair)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        a: Float[torch.Tensor, "B N_res c_res"],
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
        # beta_ij? (β_ij = 0 means purely additive, no extra gate)
    ) -> Float[torch.Tensor, "B N_res c_res"]:
        """Compute time-conditioned pair-biased attention over residue single embeddings."""
        # Inject time conditioning into queries
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
        bias: Float[torch.Tensor, "B n_heads N_res N_res"] = rearrange(
            self.z_to_b(self.norm_z(z)),
            "B N_i N_j n_heads -> B n_heads N_i N_j",
        )  # + beta_ij would go here if added
        g: Float[torch.Tensor, "B N_res n_heads head_dim"] = torch.sigmoid(
            rearrange(
                self.a_to_g(a),
                "B N_res (n_heads head_dim) -> B N_res n_heads head_dim",
                n_heads=self.n_heads,
            )
        )
        # Attention logits: (B, h, N_q, N_k)
        attn: Float[torch.Tensor, "B n_heads N_res N_res"] = einsum(
            q, k, "B N_q n_heads head_dim, B N_k n_heads head_dim -> B n_heads N_q N_k"
        ) / math.sqrt(self.head_dim)

        attn: Float[torch.Tensor, "B n_heads N_res N_res"] = F.softmax(attn + bias, dim=-1)
        # attn @ v
        intermediate: Float[torch.Tensor, "B N_res n_heads head_dim"] = einsum(
            attn, v, "B n_heads N_q N_k, B N_k n_heads head_dim -> B N_q n_heads head_dim"
        )
        # then multiply by g
        intermediate = g * intermediate
        # concat over n_heads
        out: Float[torch.Tensor, "B N_res c_res"] = rearrange(
            intermediate,
            "B N_q n_heads head_dim -> B N_q (n_heads head_dim)",  # c_res = n_heads * head_dim
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
    c       : single embedding dim  (default 256)
    c_pair  : pair   embedding dim
    n_heads : attention heads       (default 8, per Algorithm 6)
    dropout : rowwise dropout prob  (default 0.25, per Algorithm 6)
    """

    def __init__(
        self,
        c: int = 256,
        c_pair: int = 128,
        n_heads: int = 8,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Step 1
        self.attn_pair_bias = AttentionPairBias(c, c_pair, n_heads)
        self.dropout_row = DropoutRowwise(dropout)

        # Step 2
        self.transition = Transition(c)

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
