"""Template embedder for encoding structural template information.

Contains ``TemplateEmbedder``, which implements Algorithm 3 of the Pallatom
architecture: it projects distogram-based template representation together with
time-conditioning and trunk pair embedding into fixed-size pair output tensor.
"""

from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.layers import LayerNorm, LinearNoBias
from architecture.pairformer_stack import PairformerStack
from beartype import beartype
from einops import einsum, rearrange
from jaxtyping import Float, jaxtyped

# ---------------------------------------------------------------------------
# TemplateEmbedder — Algorithm 3
# ---------------------------------------------------------------------------


class TemplateEmbedder(nn.Module):
    """Parameters

    ----------
    n_bins   : number of distogram bins  (f_distogram last dim)
    c_z      : trunk pair embedding dim  (z_ij last dim)
    c        : internal pair dim         (default 64)
    d        : output pair dim           (default 128)
    n_blocks : PairformerStack depth     (default 2)
    n_heads  : attention heads           (default 4)
    """

    def __init__(
        self,
        n_bins: int,
        c_z: int,
        c: int,
        d: int,
        n_blocks: int,
        n_heads: int,
    ) -> None:
        super().__init__()

        # Step 4 projections
        # a_ij dim = n_bins + 1 (b_mask) + 1 (b_time)
        a_dim = n_bins + 1 + 1
        self.norm_z: LayerNorm = LayerNorm(c_z)
        self.proj_z: LinearNoBias = LinearNoBias(
            c_z,
            c,
        )  # LinearNoBias(LayerNorm(z_ij))
        self.proj_a: LinearNoBias = LinearNoBias(a_dim, c)  # LinearNoBias(a_ij)

        # Step 5
        self.pairformer: PairformerStack = PairformerStack(c, n_blocks, n_heads)

        # Step 6
        self.norm_v: LayerNorm = LayerNorm(c)
        self.proj_out: LinearNoBias = LinearNoBias(c, d)

    @override
    def __call__(
        self,
        f_distogram: Float[torch.Tensor, "B N_res N_res n_bins"],
        f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
        z_ij: Float[torch.Tensor, "B N_res N_res c_z"],
        t: Float[torch.Tensor, "B N_res N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res d"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(f_distogram, f_pseudo_beta_mask, z_ij, t)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        f_distogram: Float[torch.Tensor, "B N_res N_res n_bins"],
        f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
        z_ij: Float[torch.Tensor, "B N_res N_res c_z"],
        t: Float[torch.Tensor, "B N_res N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res d"]:
        """Embed distogram and pair features into pair representation.

        Implements Algorithm 3: constructs a per-pair feature vector ``a_ij``
        from the distogram bins, pseudo-beta mask outer product, and
        time-conditioned mask, adds a projected trunk pair embedding, runs a
        ``PairformerStack``, and projects the result to the output
        dimension ``d``.

        Args:
            f_distogram: Pairwise C-beta distance distribution over ``n_bins``,
                shape ``(B, N_res, N_res, n_bins)``.
            f_pseudo_beta_mask: Binary mask indicating residues with a valid
                pseudo-beta carbon in the template, shape ``(B, N_res)``.
            z_ij: Trunk pair embedding to be combined with template features,
                shape ``(B, N_res, N_res, c_z)``.
            t: Time-conditioning signal broadcast over the pair dimension,
                shape ``(B, N_res, N_res)``.

        Returns:
            Time-conditioned template pair embedding of shape
                ``(B, N_res, N_res, d)``.
        """
        # ------------------------------------------------------------------
        # Step 1: b_ij^mask = f_i^mask · f_j^mask
        # ------------------------------------------------------------------
        b_mask: Float[torch.Tensor, "B N_res N_res 1"] = rearrange(
            einsum(f_pseudo_beta_mask, f_pseudo_beta_mask, "b i, b j -> b i j"),
            "b i j -> b i j 1",
        )

        # ------------------------------------------------------------------
        # Step 2: b_ij^time = t ⊙ f_ij^mask    (broadcast t over i and j)
        # ------------------------------------------------------------------
        b_time = rearrange(
            t * b_mask[..., 0],  # b_mask is (B, N, N, 1); squeeze and multiply
            "b i j -> b i j 1",
        )

        # ------------------------------------------------------------------
        # Step 3: a_ij = concat(f_distogram, b_mask, b_time)
        # ------------------------------------------------------------------
        a_ij: Float[torch.Tensor, "B N_res N_res a_dim"] = torch.cat(
            [f_distogram, b_mask, b_time],
            dim=-1,
        )

        # This part has to be done once for each template.

        # ------------------------------------------------------------------
        # Step 4: v_ij = LinearNoBias(LayerNorm(z_ij)) + LinearNoBias(a_ij)
        # ------------------------------------------------------------------
        v_ij: Float[torch.Tensor, "B N_res N_res c"] = self.proj_z(
            self.norm_z(z_ij),
        ) + self.proj_a(a_ij)

        # ------------------------------------------------------------------
        # Step 5: v_ij = PairformerStack(v_ij, N_block)
        # ------------------------------------------------------------------
        v_ij = self.pairformer(s=None, z=v_ij)

        # ------------------------------------------------------------------
        # Step 6: u_ij = LinearNoBias(ReLU(LayerNorm(v_ij)))
        # ------------------------------------------------------------------
        u_ij: Float[torch.Tensor, "B N_res N_res d"] = self.proj_out(
            F.relu(self.norm_v(v_ij)),
        )

        # if you had multiple templates,
        # you would average the u_ij across templates after the layer norm
        # then fire a LinearNoBias(ReLu(u_ij))

        return u_ij
