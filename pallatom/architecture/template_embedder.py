"""Template embedder for encoding structural template information."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.atom_transformers import LinearNoBias
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
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_z = LinearNoBias(c_z, c)  # LinearNoBias(LayerNorm(z_ij))
        self.proj_a = LinearNoBias(a_dim, c)  # LinearNoBias(a_ij)

        # Step 5
        self.pairformer = PairformerStack(c, n_blocks, n_heads)

        # Step 6
        self.norm_v = nn.LayerNorm(c)
        self.proj_out = LinearNoBias(c, d)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        f_distogram: Float[torch.Tensor, "B N_res N_res n_bins"],
        f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
        z_ij: Float[torch.Tensor, "B N_res N_res c_z"],
        t: Float[torch.Tensor, "B N_res N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res d"]:
        """Embed template distogram and pair features into time-conditioned pair representation."""
        # ------------------------------------------------------------------
        # Step 1: b_ij^mask = f_i^mask · f_j^mask
        # ------------------------------------------------------------------
        b_mask: Float[torch.Tensor, "B N_res N_res 1"] = rearrange(
            einsum(f_pseudo_beta_mask, f_pseudo_beta_mask, "b i, b j -> b i j"),
            "b i j -> b i j 1",
        )

        # ------------------------------------------------------------------
        # Step 2: b_ij^time = t ⊙ f_i^mask    (broadcast t over j)
        # ------------------------------------------------------------------
        b_time: Float[torch.Tensor, "B N_res N_res 1"] = rearrange(
            t * rearrange(f_pseudo_beta_mask, "b i -> b i 1"), "b i j -> b i j 1"
        )

        # ------------------------------------------------------------------
        # Step 3: a_ij = concat(f_distogram, b_mask, b_time)
        # ------------------------------------------------------------------
        a_ij: Float[torch.Tensor, "B N_res N_res a_dim"] = torch.cat(
            [f_distogram, b_mask, b_time], dim=-1
        )

        # This part has to be done once for each template.

        # ------------------------------------------------------------------
        # Step 4: v_ij = LinearNoBias(LayerNorm(z_ij)) + LinearNoBias(a_ij)
        # ------------------------------------------------------------------
        v_ij: Float[torch.Tensor, "B N_res N_res c"] = self.proj_z(self.norm_z(z_ij)) + self.proj_a(
            a_ij
        )

        # ------------------------------------------------------------------
        # Step 5: v_ij = PairformerStack(v_ij, N_block)
        # ------------------------------------------------------------------
        v_ij = self.pairformer(s=None, z=v_ij)

        # ------------------------------------------------------------------
        # Step 6: u_ij = LinearNoBias(ReLU(LayerNorm(v_ij)))
        # ------------------------------------------------------------------
        u_ij: Float[torch.Tensor, "B N_res N_res d"] = self.proj_out(F.relu(self.norm_v(v_ij)))

        # if you had multiple templates,
        # you would average the u_ij across templates after the layer norm
        # then fire a LinearNoBias(ReLu(u_ij))

        return u_ij
