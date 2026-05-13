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
    c       : single embedding dim  (default 256)
    c_pair  : pair   embedding dim
    n_heads : number of attention heads (default 8 per Algorithm 6)
    """

    def __init__(self, c: int, c_pair: int, n_heads: int = 8) -> None:
        super().__init__()
        assert c % n_heads == 0, "c must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = c // n_heads

        self.norm_s = nn.LayerNorm(c)
        self.norm_t = nn.LayerNorm(c)
        self.norm_z = nn.LayerNorm(c_pair)

        # Time conditioning: project t_i → query bias
        self.proj_t = LinearNoBias(c, c)

        # QKV projections (no bias, as throughout AF3)
        self.to_q = LinearNoBias(c, c)
        self.to_k = LinearNoBias(c, c)
        self.to_v = LinearNoBias(c, c)

        # Pair → per-head scalar bias  (β_ij = 0 means purely additive, no extra gate)
        self.to_bias = LinearNoBias(c_pair, n_heads)

        # Gating on output (standard in AF3 attention)
        self.to_g = nn.Linear(c, c)  # gating (bias allowed)
        self.to_out = LinearNoBias(c, c)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        s: Float[torch.Tensor, "B N_res c_res"],
        t: Float[torch.Tensor, "B N_res c_res"],
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res c_res"]:
        """Compute time-conditioned pair-biased attention over residue single embeddings."""
        H, D = self.n_heads, self.head_dim

        # Inject time conditioning into queries
        sn: Float[torch.Tensor, "B N_res c_res"] = self.norm_s(s) + self.proj_t(self.norm_t(t))

        Q: Float[torch.Tensor, "B N_res n_heads head_dim"] = rearrange(
            self.to_q(sn), "b n (h d) -> b n h d", h=H
        )
        K: Float[torch.Tensor, "B N_res n_heads head_dim"] = rearrange(
            self.to_k(sn), "b n (h d) -> b n h d", h=H
        )
        V: Float[torch.Tensor, "B N_res n_heads head_dim"] = rearrange(
            self.to_v(sn), "b n (h d) -> b n h d", h=H
        )
        G: Float[torch.Tensor, "B N_res c_res"] = torch.sigmoid(self.to_g(sn))

        # Attention logits: (B, h, N_q, N_k)
        attn: Float[torch.Tensor, "B n_heads N_res N_res"] = einsum(
            Q, K, "b n_q h d, b n_k h d -> b h n_q n_k"
        ) / math.sqrt(D)

        # Additive pair bias: (B, N_res, N_res, n_heads) → (B, h, N_q, N_k)
        bias: Float[torch.Tensor, "B n_heads N_res N_res"] = rearrange(
            self.to_bias(self.norm_z(z)), "b n_q n_k h -> b h n_q n_k"
        )
        attn: Float[torch.Tensor, "B n_heads N_res N_res"] = F.softmax(attn + bias, dim=-1)

        # Weighted sum → (B, N_res, c_res)
        out: Float[torch.Tensor, "B N_res c_res"] = rearrange(
            einsum(attn, V, "b h n_q n_k, b n_k h d -> b n_q h d"),
            "b n h d -> b n (h d)",
        )
        out: Float[torch.Tensor, "B N_res c_res"] = G * out
        return self.to_out(out)


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
