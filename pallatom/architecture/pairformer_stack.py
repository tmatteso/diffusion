"""Pairformer stack layers for pair and single representation refinement."""

from typing import overload

import torch
import torch.nn as nn
from architecture.layers import LayerNorm, LinearNoBias, TypedLinear, TypedModuleList
from architecture.node_update import AttentionPairBias
from architecture.pair_update import (
    DropoutColumnwise,
    DropoutRowwise,
    Transition,
    TriangleAttentionEndingNodeWithBias,
    TriangleAttentionStartingNodeWithBias,
)
from beartype import beartype
from einops import einsum
from jaxtyping import Float, jaxtyped
from typing_extensions import override


# ---------------------------------------------------------------------------
# TriangleMultiplicationOutgoing  (Alg 17 step 2)
# m_ij = gate(z_ij) · out(norm(Σ_k a_ik ⊙ b_kj))
# ---------------------------------------------------------------------------
class TriangleMultiplicationOutgoing(nn.Module):
    """Outgoing triangle multiplicative update on pair embeddings."""

    def __init__(self, c: int, c_hidden: int | None = None) -> None:
        super().__init__()
        c_hidden = c_hidden or c
        self.norm_z: LayerNorm = LayerNorm(c)
        self.proj_a: LinearNoBias = LinearNoBias(c, c_hidden)
        self.gate_a: TypedLinear = TypedLinear(c, c_hidden)
        self.proj_b: LinearNoBias = LinearNoBias(c, c_hidden)
        self.gate_b: TypedLinear = TypedLinear(c, c_hidden)
        self.gate: TypedLinear = TypedLinear(c, c)
        self.norm_out: LayerNorm = LayerNorm(c_hidden)
        self.proj_out: LinearNoBias = LinearNoBias(c_hidden, c)

    @override
    def __call__(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(z)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Compute outgoing triangle product: m[i,j] = Σ_k a[i,k] ⊙ b[k,j]."""
        zn: Float[torch.Tensor, "B N_res N_res c_pair"] = self.norm_z(z)
        a: Float[torch.Tensor, "B N_res N_res c_hidden"] = torch.sigmoid(
            self.gate_a(zn)
        ) * self.proj_a(zn)
        b: Float[torch.Tensor, "B N_res N_res c_hidden"] = torch.sigmoid(
            self.gate_b(zn)
        ) * self.proj_b(zn)
        g: Float[torch.Tensor, "B N_res N_res c_pair"] = torch.sigmoid(self.gate(z))
        m: Float[torch.Tensor, "B N_res N_res c_hidden"] = einsum(
            a, b, "b i k c, b k j c -> b i j c"
        )
        return g * self.proj_out(self.norm_out(m))


# ---------------------------------------------------------------------------
# TriangleMultiplicationIncoming  (Alg 17 step 3)
# m_ij = gate(z_ij) · out(norm(Σ_k a_ki ⊙ b_jk))
# ---------------------------------------------------------------------------


class TriangleMultiplicationIncoming(nn.Module):
    """Incoming triangle multiplicative update on pair embeddings."""

    def __init__(self, c: int, c_hidden: int | None = None) -> None:
        super().__init__()
        c_hidden = c_hidden or c
        self.norm_z: LayerNorm = LayerNorm(c)
        self.proj_a: LinearNoBias = LinearNoBias(c, c_hidden)
        self.gate_a: TypedLinear = TypedLinear(c, c_hidden)
        self.proj_b: LinearNoBias = LinearNoBias(c, c_hidden)
        self.gate_b: TypedLinear = TypedLinear(c, c_hidden)
        self.gate: TypedLinear = TypedLinear(c, c)
        self.norm_out: LayerNorm = LayerNorm(c_hidden)
        self.proj_out: LinearNoBias = LinearNoBias(c_hidden, c)

    @override
    def __call__(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(z)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Compute incoming triangle product: m[i,j] = Σ_k a[k,i] ⊙ b[j,k]."""
        zn: Float[torch.Tensor, "B N_res N_res c_pair"] = self.norm_z(z)
        a: Float[torch.Tensor, "B N_res N_res c_hidden"] = torch.sigmoid(
            self.gate_a(zn)
        ) * self.proj_a(zn)
        b: Float[torch.Tensor, "B N_res N_res c_hidden"] = torch.sigmoid(
            self.gate_b(zn)
        ) * self.proj_b(zn)
        g: Float[torch.Tensor, "B N_res N_res c_pair"] = torch.sigmoid(self.gate(z))
        m: Float[torch.Tensor, "B N_res N_res c_hidden"] = einsum(
            a, b, "b k i c, b j k c -> b i j c"
        )
        return g * self.proj_out(self.norm_out(m))


class PairformerBlock(nn.Module):
    """One block of the Pairformer, refining pair and optional single embeddings.

    Applies the full AF3 Pairformer update sequence to the pair representation
    ``z`` and, when provided, the residue single representation ``s``:

    1. Outgoing triangle multiplication  (row dropout)
    2. Incoming triangle multiplication  (row dropout)
    3. Triangle attention starting node  (row dropout)
    4. Triangle attention ending node    (column dropout)
    5. Transition FFN on ``z``
    6. Attention with pair bias on ``s`` + transition FFN on ``s``  (if ``s`` given)

    Args:
        c_pair: Channel dimension of the pair embedding ``z``.
        c_res: Channel dimension of the residue single embedding ``s``. Optional
        n_heads: Number of attention heads for triangle attention and pair-bias
            attention.  Must evenly divide ``c_pair``.
    """

    def __init__(self, c_pair: int, c_res: int | None, n_heads: int = 4) -> None:
        super().__init__()
        assert c_pair % n_heads == 0
        self.n_heads: int = n_heads
        self.head_dim: int = c_pair // n_heads
        self.c_hidden: int = c_pair // 2

        self.row_dropout: DropoutRowwise = DropoutRowwise(p=0.25)
        self.column_dropout: DropoutColumnwise = DropoutColumnwise(p=0.25)
        self.b_proj_start: LinearNoBias = LinearNoBias(c_pair, n_heads // 4)
        self.b_proj_end: LinearNoBias = LinearNoBias(c_pair, n_heads // 4)

        self.triangle_mult_outgoing: TriangleMultiplicationOutgoing = (
            TriangleMultiplicationOutgoing(c=c_pair, c_hidden=self.c_hidden)
        )
        self.traingle_mult_incoming: TriangleMultiplicationIncoming = (
            TriangleMultiplicationIncoming(c=c_pair, c_hidden=self.c_hidden)
        )
        self.triangle_attn_starting_node: TriangleAttentionStartingNodeWithBias = (
            TriangleAttentionStartingNodeWithBias(c_pair=c_pair, n_heads=n_heads // 4)
        )
        self.triangle_attn_ending_node: TriangleAttentionEndingNodeWithBias = (
            TriangleAttentionEndingNodeWithBias(c_pair=c_pair, n_heads=n_heads // 4)
        )
        self.transition1: Transition = Transition(c_pair)

        if c_res is not None:
            self.attn_pair_bias: AttentionPairBias = AttentionPairBias(
                c_res=c_res, c_pair=c_pair, n_heads=self.n_heads
            )  # n head = 16 as per AF3.

            self.transition2: Transition = Transition(c_res)

    @override
    def __call__(
        self,
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B N_res c_res"] | None, Float[torch.Tensor, "B N_res N_res c_pair"]
    ]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(s, z)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B N_res c_res"] | None, Float[torch.Tensor, "B N_res N_res c_pair"]
    ]:
        """Apply row-wise and column-wise triangle attention with transition FFN."""
        z = z + self.row_dropout(self.triangle_mult_outgoing(z))
        z = z + self.row_dropout(self.traingle_mult_incoming(z))
        b: Float[torch.Tensor, "B N_res N_res n_heads"] = self.b_proj_start(z)
        z = z + self.row_dropout(self.triangle_attn_starting_node(z, b))
        b = self.b_proj_end(z)
        z = z + self.column_dropout(self.triangle_attn_ending_node(z, b))
        z = z + self.transition1(z)
        if s is not None:
            s = s + self.attn_pair_bias(a=s, s=None, z=z)
            s = s + self.transition2(s)
            return s, z
        return None, z


class PairformerStack(nn.Module):
    """Stack of PairformerBlock modules applied sequentially to pair embeddings."""

    def __init__(
        self, c: int, n_blocks: int = 2, n_heads: int = 4, c_res: int | None = None
    ) -> None:
        super().__init__()
        self.blocks: TypedModuleList[PairformerBlock] = TypedModuleList(
            [PairformerBlock(c, c_res, n_heads) for _ in range(n_blocks)]
        )

    @overload
    def __call__(
        self,
        s: None,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]: ...

    @overload
    def __call__(
        self,
        s: Float[torch.Tensor, "B N_res c_res"],
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> tuple[
        Float[torch.Tensor, "B N_res c_res"], Float[torch.Tensor, "B N_res N_res c_pair"]
    ]: ...

    @override
    def __call__(
        self,
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> (
        tuple[Float[torch.Tensor, "B N_res c_res"], Float[torch.Tensor, "B N_res N_res c_pair"]]
        | Float[torch.Tensor, "B N_res N_res c_pair"]
    ):
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(s, z)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        s: Float[torch.Tensor, "B N_res c_res"] | None,
        z: Float[torch.Tensor, "B N_res N_res c_pair"],
    ) -> (
        tuple[Float[torch.Tensor, "B N_res c_res"], Float[torch.Tensor, "B N_res N_res c_pair"]]
        | Float[torch.Tensor, "B N_res N_res c_pair"]
    ):
        """Pass pair embeddings through all Pairformer blocks sequentially."""
        for block in self.blocks:
            s, z = block(s=s, z=z)
        if s is not None:
            return s, z
        return z  # the PairformerStack in AF3 can operate on z alone or z and s.
