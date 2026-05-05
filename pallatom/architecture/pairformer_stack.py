import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, jaxtyped

from architecture.atom_transformers import LinearNoBias

# ---------------------------------------------------------------------------
# PairformerStack — simplified pair-only stack used in step 5
# Each block: LayerNorm → triangle updates (outer-product mean approximation
# via row/col attention) → transition FFN, all on pair representation only.
# ---------------------------------------------------------------------------


class PairformerBlock(nn.Module):
    """
    One block of the Pairformer operating purely on pair embeddings v_ij.

    Uses:

    - Row-wise gated self-attention  (triangle attention, rows)
    - Column-wise gated self-attention (triangle attention, cols)
    - Pair transition FFN
    """
    def __init__(self, c: int, n_heads: int = 4):
        super().__init__()
        assert c % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = c // n_heads

        # Row-wise attention
        self.norm_row  = nn.LayerNorm(c)
        self.q_row     = LinearNoBias(c, c)
        self.k_row     = LinearNoBias(c, c)
        self.v_row     = LinearNoBias(c, c)
        self.g_row     = nn.Linear(c, c)         # gating (with bias)
        self.out_row   = LinearNoBias(c, c)

        # Column-wise attention
        self.norm_col  = nn.LayerNorm(c)
        self.q_col     = LinearNoBias(c, c)
        self.k_col     = LinearNoBias(c, c)
        self.v_col     = LinearNoBias(c, c)
        self.g_col     = nn.Linear(c, c)
        self.out_col   = LinearNoBias(c, c)

        # Transition FFN
        self.norm_ff   = nn.LayerNorm(c)
        self.ff1       = LinearNoBias(c, c * 4)
        self.ff2       = LinearNoBias(c * 4, c)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        v: Float[torch.Tensor, "B N_res N_res c"],
    ) -> Float[torch.Tensor, "B N_res N_res c"]:
        H, D = self.n_heads, self.head_dim

        # ------------------------------------------------------------------
        # Row-wise: for each (b, i), attend over j
        # ------------------------------------------------------------------
        xn: Float[torch.Tensor, "B N_res N_res c"] = self.norm_row(v)
        Q: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(self.q_row(xn), "b n m (h d) -> b n h m d", h=H)
        K: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(self.k_row(xn), "b n m (h d) -> b n h m d", h=H)
        V: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(self.v_row(xn), "b n m (h d) -> b n h m d", h=H)
        G: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(torch.sigmoid(self.g_row(xn)), "b n m (h d) -> b n h m d", h=H)
        attn: Float[torch.Tensor, "B N_res H N_res N_res"] = torch.softmax(
            einsum(Q, K, "b n h m_q d, b n h m_k d -> b n h m_q m_k") * (D ** -0.5),
            dim=-1,
        )
        out: Float[torch.Tensor, "B N_res N_res c"] = rearrange(
            G * einsum(attn, V, "b n h m_q m_k, b n h m_k d -> b n h m_q d"),
            "b n h m d -> b n m (h d)",
        )
        v = v + self.out_row(out)

        # ------------------------------------------------------------------
        # Column-wise: for each (b, j), attend over i
        # ------------------------------------------------------------------
        xn: Float[torch.Tensor, "B N_res N_res c"] = self.norm_col(v)
        Q: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(self.q_col(xn), "b n m (h d) -> b m h n d", h=H)
        K: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(self.k_col(xn), "b n m (h d) -> b m h n d", h=H)
        V: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(self.v_col(xn), "b n m (h d) -> b m h n d", h=H)
        G: Float[torch.Tensor, "B N_res H N_res D"] = rearrange(torch.sigmoid(self.g_col(xn)), "b n m (h d) -> b m h n d", h=H)
        attn: Float[torch.Tensor, "B N_res H N_res N_res"] = torch.softmax(
            einsum(Q, K, "b m h n_q d, b m h n_k d -> b m h n_q n_k") * (D ** -0.5),
            dim=-1,
        )
        out: Float[torch.Tensor, "B N_res N_res c"] = rearrange(
            G * einsum(attn, V, "b m h n_q n_k, b m h n_k d -> b m h n_q d"),
            "b m h n d -> b n m (h d)",
        )
        v = v + self.out_col(out)

        # ------------------------------------------------------------------
        # Transition FFN
        # ------------------------------------------------------------------
        v = v + self.ff2(F.relu(self.ff1(self.norm_ff(v))))
        return v


class PairformerStack(nn.Module):
    def __init__(self, c: int, n_blocks: int = 2, n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([PairformerBlock(c, n_heads) for _ in range(n_blocks)])

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        v: Float[torch.Tensor, "B N_res N_res c"],
    ) -> Float[torch.Tensor, "B N_res N_res c"]:
        for block in self.blocks:
            v: Float[torch.Tensor, "B N_res N_res c"] = block(v)
        return v
