"""EDM sampling from a trained MainTrunk denoising network."""

import math
import dataclasses
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Int, jaxtyped

from architecture.main_trunk import MainTrunk
from helpers.featurize import FeaturizedBatch
from helpers.atom_utils import (
    ATOM5_ELEMENTS,
    ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB,
    Protein,
    to_pdb,
    rigid_group_atom_positions,
    ATOM5_NAMES,
)
from architecture.atom_transformers import build_sparse_pairs, WINDOW_SIZE

# atom5 slot → atom37 index (used when writing PDB via atom37 representation)
# atom5: N=0, CA=1, C=2, O=3, CB=4  →  atom37: N=0, CA=1, C=2, O=3, CB=4
ATOM5_TO_ATOM37 = [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB]
NATOM = 5  # atoms per residue


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EDMPrecond  —  adapts MainTrunk to the D_θ(r, σ) interface
#
#  EDMSampler calls  D_cur = denoiser(r_noisy, sigma)
#  MainTrunk.forward takes a FeaturizedBatch.
#  EDMPrecond holds the static per-protein context and rebuilds the batch
#  at each denoising step by swapping in the current (r_input, t_hat).
#
#  For unconditional generation, gt_res_distogram and f_pseudo_beta_mask
#  are zeros (no template conditioning).
# ─────────────────────────────────────────────────────────────────────────────

class EDMPrecond(nn.Module):
    """
    Wraps MainTrunk as an EDM-compatible denoiser D_θ(r_noisy, σ) → r_denoised.

    Parameters
    ----------
    model     : trained MainTrunk
    context   : FeaturizedBatch with static fields filled in; r_input and
                t_hat are replaced at every forward call
    sigma_min : lower σ bound, used only to compute t_normalized
    sigma_max : upper σ bound, used only to compute t_normalized
    """
    def __init__(
        self,
        model:     MainTrunk,
        context:   FeaturizedBatch,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        super().__init__()
        self.model     = model
        self.context   = context
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        r_input: Float[torch.Tensor, "B N_atom 3"],
        t_hat:   float,
    ) -> Float[torch.Tensor, "B N_atom 3"]:
        t_normalized: float = (
            (math.log(t_hat) - math.log(self.sigma_min))
            / (math.log(self.sigma_max) - math.log(self.sigma_min))
        )
        batch = dataclasses.replace(
            self.context,
            r_input=r_input,
            t_hat=t_hat,
            t_normalized=t_normalized,
        )
        r_denoised: Float[torch.Tensor, "B N_atom 3"]
        r_denoised, *_ = self.model(batch)
        return r_denoised


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EDM Sampler  —  deterministic Heun ODE  (Algorithm 2 in the paper)
#
#  ODE:  dr/dσ = (r − D_θ(r,σ)) / σ   (the "probability flow ODE")
#  Heun = one Euler predictor + one corrector for 2nd-order accuracy.
#
#  Optional stochasticity: inject a small amount of noise at each step
#  (S_churn > 0) to get the SDE variant, analogous to DDIM η > 0.
# ─────────────────────────────────────────────────────────────────────────────

class EDMSampler:
    """
    Karras et al. 2022 deterministic (Heun) sampler.

    Parameters
    ----------
    denoiser  : EDMPrecond wrapping a trained MainTrunk
    sigma_min : float  smallest noise level  (paper: 0.002)
    sigma_max : float  largest  noise level  (paper: 80.0)
    rho       : float  schedule exponent     (paper: 7.0)
    S_churn   : float  stochastic noise injected per step (0 = deterministic)
    S_tmin    : float  only inject noise in [S_tmin, S_tmax]
    S_tmax    : float
    S_noise   : float  scaling of injected noise
    """
    def __init__(
        self,
        denoiser:  EDMPrecond,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho:       float = 7.0,
        S_churn:   float = 0.0,
        S_tmin:    float = 0.0,
        S_tmax:    float = float("inf"),
        S_noise:   float = 1.003,
    ):
        self.denoiser  = denoiser
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho       = rho
        self.S_churn   = S_churn
        self.S_tmin    = S_tmin
        self.S_tmax    = S_tmax
        self.S_noise   = S_noise

    @jaxtyped(typechecker=beartype)
    def _sigma_schedule(
        self,
        steps:  int,
        device: torch.device | str,
    ) -> Float[torch.Tensor, "S"]:  # S = steps + 1
        """
        Karras σ schedule: σ_i = (σ_max^(1/ρ) + i/(N-1)·(σ_min^(1/ρ)−σ_max^(1/ρ)))^ρ
        Returns a tensor of length (steps+1,) with σ_N=0 appended.
        """
        rho: float = self.rho
        i: Float[torch.Tensor, "steps"] = torch.arange(steps, device=device).float()
        t: Float[torch.Tensor, "steps"] = (
            self.sigma_max ** (1 / rho)
            + i / (steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho
        return torch.cat([t, t.new_zeros(1)])   # (S,)  S = steps + 1

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        shape: tuple[int, int, int],     # (B, N_atom, 3)
        steps: int = 40,
        device: torch.device | str = "cpu",
    ) -> Float[torch.Tensor, "B N_atom 3"]:
        """
        shape : (B, N_atom, 3)  — batch of atom coordinate tensor shapes
        steps : number of ODE steps  (40 is usually plenty)
        """
        sigmas: Float[torch.Tensor, "S"] = self._sigma_schedule(steps, device)

        # pure noise initialised at σ_max — independent per batch item
        z: Float[torch.Tensor, "B N_atom 3"] = torch.randn(shape, device=device) * sigmas[0]

        for i in range(steps):
            sigma_cur:  Float[torch.Tensor, ""] = sigmas[i]
            sigma_next: Float[torch.Tensor, ""] = sigmas[i + 1]

            # ── optional stochastic noise injection (S_churn) ──────────────
            sigma_hat: Float[torch.Tensor, ""]
            if self.S_churn > 0 and self.S_tmin <= sigma_cur <= self.S_tmax:
                gamma:    float = min(self.S_churn / steps, math.sqrt(2) - 1)
                sigma_hat = sigma_cur * (1 + gamma)
                z = z + (sigma_hat ** 2 - sigma_cur ** 2).sqrt() \
                      * self.S_noise * torch.randn_like(z)
            else:
                sigma_hat = sigma_cur

            # ── first derivative (Euler predictor) ─────────────────────────
            D_cur:  Float[torch.Tensor, "B N_atom 3"] = self.denoiser(z, sigma_hat.item())
            d_cur:  Float[torch.Tensor, "B N_atom 3"] = (z - D_cur) / sigma_hat
            z_next: Float[torch.Tensor, "B N_atom 3"] = z + (sigma_next - sigma_hat) * d_cur

            # ── second derivative (Heun corrector), skip at last step ──────
            if sigma_next > 0:
                D_next: Float[torch.Tensor, "B N_atom 3"] = self.denoiser(z_next, sigma_next.item())
                d_next: Float[torch.Tensor, "B N_atom 3"] = (z_next - D_next) / sigma_next
                d_avg:  Float[torch.Tensor, "B N_atom 3"] = (d_cur + d_next) / 2.0
                z_next = z + (sigma_next - sigma_hat) * d_avg

            z = z_next

        return z   # (B, N_atom, 3)  denoised atom5 coordinates


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Context builder  —  constructs the static FeaturizedBatch fields
#
#  MainTrunk expects several per-protein tensors that don't change across
#  denoising steps.  This helper builds them for unconditional generation
#  (no template: zero distogram, zero pseudo-beta mask).
# ─────────────────────────────────────────────────────────────────────────────

def build_sampling_context(
    N_res:         int,
    batch_size:    int = 1,
    n_templ_bins:  int = 38,
    n_atom_bins:   int = 22,
    c_res_embed:   int = 32,        # must match the dim used in featurize_batch
    device:        str = "cpu",
) -> FeaturizedBatch:
    """
    Build the static context FeaturizedBatch for unconditional backbone generation.

    All batch items share identical static context (same reference conformer, same
    indices). r_input and t_hat are placeholder zeros; EDMPrecond.forward replaces
    them at each denoising step via dataclasses.replace.

    Parameters
    ----------
    N_res      : number of residues
    batch_size : B — number of protein samples to generate in parallel
    """
    B = batch_size
    N_atom: int = N_res * NATOM

    # ── Ala reference conformer tiled over all residues ──────────────────────
    def _ala_ref_pos() -> Float[torch.Tensor, "5 3"]:
        pos_by_name = {
            name: pos for name, _, pos in rigid_group_atom_positions["ALA"]
        }
        return torch.tensor(
            [pos_by_name.get(name, (0.0, 0.0, 0.0)) for name in ATOM5_NAMES],
            dtype=torch.float32,
        )

    ref_pos_single: Float[torch.Tensor, "N_atom 3"] = rearrange(
        repeat(_ala_ref_pos().to(device), "a d -> n a d", n=N_res),
        "n a d -> (n a) d",
    )

    ref_element_single: Float[torch.Tensor, "N_atom E"] = rearrange(
        repeat(ATOM5_ELEMENTS.float().to(device), "a e -> n a e", n=N_res),
        "n a e -> (n a) e",
    )

    # ── Index tensors (shared across batch — identical for all items) ────────
    tok_idx_single:    Int[torch.Tensor, "N_atom"] = (
        torch.arange(N_res, dtype=torch.long, device=device).repeat_interleave(NATOM)
    )
    center_uid_single: Int[torch.Tensor, "N_res"] = (
        torch.arange(N_res, dtype=torch.long, device=device) * NATOM + 1  # CA slot
    )

    # ── Residue position embedding ───────────────────────────────────────────
    residue_index:   Int[torch.Tensor,   "N_res"]           = torch.arange(N_res, dtype=torch.long, device=device)
    index_embedding: nn.Embedding                            = nn.Embedding(256, c_res_embed).to(device)
    with torch.no_grad():
        f_residue_idx_single: Float[torch.Tensor, "N_res c_res_embed"] = index_embedding(residue_index)

    # ── Dummy atom distogram (not used by forward, only by loss) ─────────────
    _, valid_nbr = build_sparse_pairs(tok_idx_single, WINDOW_SIZE)
    K: int = valid_nbr.shape[1]

    # ── Tile single-item tensors along new B dim ─────────────────────────────
    def tile(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(B, *t.shape).contiguous()

    return FeaturizedBatch(
        ref_pos=tile(ref_pos_single),
        ref_element=tile(ref_element_single),
        ref_space_uid=torch.zeros(B, N_atom, dtype=torch.long, device=device),
        gt_res_distogram=torch.zeros(B, N_res, N_res, n_templ_bins, dtype=torch.long, device=device),
        f_pseudo_beta_mask=torch.zeros(B, N_res, dtype=torch.long, device=device),
        f_residue_idx=tile(f_residue_idx_single),
        r_input=torch.zeros(B, N_atom, 3, dtype=torch.float32, device=device),
        r_gt=torch.zeros(B, N_atom, 3, dtype=torch.float32, device=device),
        atom5_mask=torch.ones(B, N_atom, dtype=torch.bool, device=device),
        aa_indices=torch.zeros(B, N_res, dtype=torch.long, device=device),
        residue_mask=torch.ones(B, N_res, dtype=torch.bool, device=device),
        t_hat=1.0,
        t_normalized=0.5,
        tok_idx=tile(tok_idx_single),
        center_uid=tile(center_uid_single),
        gt_atom_distogram_sparse=torch.zeros(B, N_atom, K, n_atom_bins, dtype=torch.float32, device=device),
        gt_atom_distogram_mask_sparse=torch.zeros(B, N_atom, K, dtype=torch.bool, device=device),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PDB helper  —  atom5 (N_res, 5, 3) → atom37 (N_res, 37, 3)
# ─────────────────────────────────────────────────────────────────────────────

@jaxtyped(typechecker=beartype)
def atom5_to_atom37(
    coords_5: Float[np.ndarray, "N_res 5 3"],
    mask_5:   Optional[Float[np.ndarray, "N_res 5"]] = None,
) -> tuple[Float[np.ndarray, "N_res 37 3"], Float[np.ndarray, "N_res 37"]]:
    """
    Map atom5 coordinates back into the full atom37 layout.

    Returns
    -------
    x_37   : (N_res, 37, 3)
    mask_37: (N_res, 37)
    """
    N_res:   int                              = coords_5.shape[0]
    x_37:    Float[np.ndarray, "N_res 37 3"] = np.zeros((N_res, 37, 3), dtype=np.float32)
    mask_37: Float[np.ndarray, "N_res 37"]   = np.zeros((N_res, 37),    dtype=np.float32)

    for atom5_slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        x_37[:, atom37_idx, :]  = coords_5[:, atom5_slot, :]
        mask_37[:, atom37_idx]  = mask_5[:, atom5_slot] if mask_5 is not None else 1.0

    return x_37, mask_37


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main sampling script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json as _json
    from pathlib import Path as _Path

    from sample.sample_config import SampleConfig

    parser = argparse.ArgumentParser(description="Sample protein structures from a trained PallAtom model")
    parser.add_argument("--config", required=True, help="Path to SampleConfig JSON")
    args = parser.parse_args()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    scfg:      SampleConfig = SampleConfig.model_validate(_json.loads(_Path(args.config).read_text()))
    mp         = scfg.model
    noise      = scfg.noise
    sampler_p  = scfg.sampler
    gen        = scfg.generation

    model = MainTrunk(
        f_ref_dim=mp.f_ref_dim,
        n_bins=mp.n_bins,
        c_atom=mp.c_atom,
        c_pair=mp.c_pair,
        c_res=mp.c_res,
        c_atompair=mp.c_atompair,
        K_unit=mp.K_unit,
        sigma_data=noise.sigma_data,
    ).to(device)
    model.load_state_dict(torch.load(scfg.checkpoint.checkpoint_path, map_location=device))
    model.eval()

    N_RES:     int = gen.n_res
    N_atom:    int = N_RES * NATOM
    B_SAMPLE:  int = gen.n_samples  # generate all samples in one batched call

    context:     FeaturizedBatch = build_sampling_context(
        N_RES, batch_size=B_SAMPLE, n_templ_bins=mp.n_bins, device=device
    )
    edm_precond: EDMPrecond      = EDMPrecond(
        model, context,
        sigma_min=noise.sigma_min,
        sigma_max=noise.sigma_max,
    ).to(device)
    edm_precond.eval()

    edm_sampler: EDMSampler = EDMSampler(
        edm_precond,
        sigma_min=noise.sigma_min,
        sigma_max=noise.sigma_max,
        rho=sampler_p.rho,
        S_churn=sampler_p.S_churn,
        S_tmin=sampler_p.S_tmin,
        S_tmax=sampler_p.S_tmax,
        S_noise=sampler_p.S_noise,
    )

    coords_batch: Float[torch.Tensor, "B N_atom 3"] = edm_sampler.sample(
        shape=(B_SAMPLE, N_atom, 3),
        steps=sampler_p.ddim_steps,
        device=device,
    )

    pdb_strings: list[str] = []
    for b in range(B_SAMPLE):
        coords_np: Float[np.ndarray, "N_res 5 3"] = rearrange(
            coords_batch[b].cpu().numpy(), "(n a) d -> n a d", n=N_RES, a=NATOM
        )
        x_37, mask_37 = atom5_to_atom37(coords_np)
        prot = Protein(
            atom_positions=x_37,
            atom_mask=mask_37,
            residue_index=np.arange(N_RES, dtype=np.int32),
            aatype=np.zeros(N_RES, dtype=np.int32),
            chain_index=np.zeros(N_RES, dtype=np.int32),
            b_factors=np.ones((N_RES, 37), dtype=np.float32),
        )
        pdb_strings.append(to_pdb(prot))

    _Path(scfg.output.output_path).write_text(_json.dumps(pdb_strings))
