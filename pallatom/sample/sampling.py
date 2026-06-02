"""EDM sampling from a trained MainTrunk denoising network."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import structlog
import torch
from architecture.atom_transformers import WINDOW_SIZE, build_sparse_pairs
from architecture.main_trunk import MainTrunk
from beartype import beartype
from einops import rearrange, reduce, repeat
from helpers.alignment import centre_random_augment
from helpers.atom_utils import (
    ATOM5_ELEMENTS,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
    Protein,
    atom37_to_atom5,
    atom37_to_cb,
    protein_from_pdb,
    restype_order,
    to_pdb,
)
from helpers.context_managers import FatalOnError, StructlogConfig
from helpers.featurize import Distogram, FeaturizedBatch, ref_pos_for_residue
from jaxtyping import Bool, Float, Int, jaxtyped
from sample.sample_config import SampleConfig, SamplerParams
from train.train_config import NoiseScheduleParams

# atom5 slot → atom37 index (used when writing PDB via atom37 representation)
# atom5: N=0, CA=1, C=2, O=3, CB=4  →  atom37: N=0, CA=1, C=2, O=3, CB=4
ATOM5_TO_ATOM37 = [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB]
NATOM = 5  # atoms per residue


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class AllAtomContext:  # N_atom = N_res * 5 for atom5 representation
    """Batched all-atom conditioning context combining structure, sequence, and distogram fields."""

    # structure input
    r_gt: Float[torch.Tensor, "B N_atom 3"]  # gt atom_positions
    atom5_mask: Bool[torch.Tensor, "B N_atom"]
    residue_mask: Bool[torch.Tensor, "B N_res"]
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"]

    # amino acid input -- aa indices is the seq itself, f_residue_idx is the residue index
    aa_indices: Int[torch.Tensor, "B N_res"]
    f_residue_idx: Int[torch.Tensor, "B N_res"]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class TemplateContext:
    """Batched template conditioning tensors for the sampling loop.

    Attributes:
        f_template_distogram: Pairwise Cβ distance bin distribution over the
            template structure, shape (B, N_RES, N_RES, N_TEMPL_BINS).
        f_pseudo_beta_mask: Binary mask indicating residues with a valid
            pseudo-β carbon in the template, shape (B, N_RES).
    """

    f_template_distogram: Int[torch.Tensor, "B N_RES N_RES N_TEMPL_BINS"]
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_RES"]


@jaxtyped(typechecker=beartype)
def build_AA_context(
    atom_37_coordinate_tensor: Float[torch.Tensor, "N_res 37 3"],
    atom_37_mask: Float[torch.Tensor, "N_res 37"],
    residue_index: Float[torch.Tensor, "N_res"],
    aa_sequence: str,
    atom_distogram_fn: Distogram,
    batch_size: int,
    device: str,
) -> AllAtomContext:
    """Build the all-atom conditioning context from a single ground-truth structure.

    Converts atom37 coordinates to the compact atom5 representation, maps the
    amino-acid sequence string to integer indices (using mask token 20 for unknown
    residues), stores the raw integer residue index, and precomputes the sparse
    atom-pair distogram over a local K-neighbour window.  All single-protein
    tensors are replicated along the batch dimension to produce a batch of size
    ``batch_size``, ready to be consumed by the sampling loop.

    Args:
        atom_37_coordinate_tensor: Ground-truth all-atom coordinates in the atom37
            layout, shape (N_res, 37, 3).
        atom_37_mask: Binary mask indicating which atom37 slots are present,
            shape (N_res, 37).
        residue_index: Per-residue index, shape (N_res,).
        aa_sequence: One-letter amino-acid sequence string; residues not found in
            the standard vocabulary are mapped to mask token 20.
        atom_distogram_fn: Callable that accepts batched atom positions and masks
            and returns a dense (B, N_atom, N_atom, n_bins) distogram.
        batch_size: Number of parallel sampling trajectories; all context tensors
            are tiled to this batch dimension.
        device: PyTorch device string for tensor allocation.

    Returns:
        An AllAtomContext with batched ground-truth atom positions, atom masks,
        residue masks, sequence indices, integer residue indices, and sparse
        atom-pair distogram labels.
    """
    N_res: int = atom_37_coordinate_tensor.shape[0]
    _aa_vals = [restype_order.get(r, 20) for r in aa_sequence]
    aa_indices_i: Int[torch.Tensor, N_res] = torch.tensor(_aa_vals, dtype=torch.long, device=device)
    f_residue_idx_i: Int[torch.Tensor, "N_res"] = residue_index.long().to(device)

    atom5_pos, atom5_mask = atom37_to_atom5(
        rearrange(atom_37_coordinate_tensor, "n a d -> 1 n a d"),
        rearrange(atom_37_mask, "n a -> 1 n a"),
    )
    atom5_pos = rearrange(atom5_pos, "1 n a d -> n a d")  # (N_res_i, 5, 3)
    atom5_mask = rearrange(atom5_mask, "1 n a -> n a")  # (N_res_i, 5)
    residue_mask_i: Bool[torch.Tensor, N_res] = atom5_mask.any(dim=-1)
    packed_flat_pos_i: Float[torch.Tensor, "N_atom 3"] = rearrange(atom5_pos, "n a d -> (n a) d")
    packed_atom_mask_i: Bool[torch.Tensor, "N_atom"] = repeat(residue_mask_i, "n -> (n a)", a=NATOM)

    # stack all to be of shape batch_size
    r_gt: Float[torch.Tensor, "B N_atom 3"] = repeat(
        packed_flat_pos_i, "n d -> b n d", b=batch_size
    )
    atom5_mask: Bool[torch.Tensor, "B N_atom"] = repeat(
        packed_atom_mask_i, "n -> b n", b=batch_size
    )
    residue_mask: Bool[torch.Tensor, "B N_res"] = repeat(residue_mask_i, "n -> b n", b=batch_size)
    aa_indices: Int[torch.Tensor, "B N_res"] = repeat(aa_indices_i, "n -> b n", b=batch_size)
    f_residue_idx: Int[torch.Tensor, "B N_res"] = repeat(f_residue_idx_i, "n -> b n", b=batch_size)
    # atom distogram
    # ── Sparse atom distogram (batched) ──────────────────────────────────────
    _tok_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        N_res, dtype=torch.long, device=device
    ).repeat_interleave(NATOM)
    neighbor_idx, _ = build_sparse_pairs(_tok_single, WINDOW_SIZE)  # (N_atom, K)

    # atom_distogram_fn supports batched input: (B, N_atom, 3) → (B, N_atom, N_atom, n_bins)
    gt_atom_disto_dense, gt_atom_mask_dense = atom_distogram_fn(r_gt, atom5_mask)
    n_atom_bins: int = gt_atom_disto_dense.shape[-1]

    # Vectorised sparse gather: result[b, l, k] = dense[b, l, neighbor_idx[l, k]]
    nbr_b: Int[torch.Tensor, "B N_atom K"] = repeat(neighbor_idx, "n k -> b n k", b=batch_size)
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"] = (
        gt_atom_disto_dense.gather(2, repeat(nbr_b, "b n k -> b n k d", d=n_atom_bins))
    )
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"] = (
        gt_atom_mask_dense.long().gather(2, nbr_b).bool()
    )
    del gt_atom_disto_dense, gt_atom_mask_dense

    return AllAtomContext(
        # structure input
        r_gt=r_gt,
        atom5_mask=atom5_mask,
        residue_mask=residue_mask,
        # amino acid input -- aa indices is the seq itself, f_residue_idx is the residue index
        aa_indices=aa_indices,
        f_residue_idx=f_residue_idx,
        # atom distogram
        gt_atom_distogram_sparse=gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=gt_atom_distogram_mask_sparse,
    )


@jaxtyped(typechecker=beartype)
def build_template_context(
    prot: Protein,
    batch_size: int,
    distogram_fn: Distogram,
    device: torch.device | str,
) -> TemplateContext:
    """Build a batch of template distogram contexts from a list of Protein objects.

    Pads all proteins in the list to the same maximum residue count, extracts the
    pseudo-β carbon positions via ``atom37_to_cb``, and passes those positions through
    ``distogram_fn`` to produce the inter-residue distance distribution used as
    structural conditioning.  The batch size equals the number of proteins supplied,
    so callers must ensure the list length matches the model's expected batch size.

    Args:
        prot: Protein dataclass instance; contains
            ``atom_positions`` (N_res, 37, 3) and ``atom_mask`` (N_res, 37)
            numpy arrays.  Proteins with fewer residues than the batch maximum
            are zero-padded on the C-terminal end.
        batch_size: Number of copies of the template to stack into the batch dimension.
        distogram_fn: Callable that maps batched Cβ positions of shape
            (B, N_res, 3) and a residue mask of shape (B, N_res) to a
            (B, N_res, N_res, n_bins) distogram tensor.
        device: PyTorch device string for tensor allocation; defaults to ``"cpu"``.

    Returns:
        A TemplateContext containing the integer-quantised distogram
        ``f_template_distogram`` of shape (B, N_res, N_res, n_templ_bins) and the
        binary pseudo-β mask ``f_pseudo_beta_mask`` of shape (B, N_res).
    """
    pos: Float[torch.Tensor, "N_res 37 3"] = torch.tensor(
        prot.atom_positions, dtype=torch.float64, device=device
    )
    mask: Float[torch.Tensor, "N_res 37"] = torch.tensor(
        prot.atom_mask, dtype=torch.float64, device=device
    )

    # repeat to get batch dim
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"] = repeat(
        pos, "n a d -> b n a d", b=batch_size
    )
    atom37_mask: Float[torch.Tensor, "B N_res 37"] = repeat(mask, "n a -> b n a", b=batch_size)
    residue_mask: Bool[torch.Tensor, "B N_res"] = atom37_mask[:, :, ATOM37_CA].bool()

    pseudo_beta_carbon_positions: Float[torch.Tensor, "B N_res 3"]
    pseudo_beta_carbon_positions, _ = atom37_to_cb(
        atom37_positions=atom37_positions,
        atom37_mask=atom37_mask,
    )

    f_disto, _ = distogram_fn(pseudo_beta_carbon_positions, residue_mask)

    gt_res_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"] = f_disto.long()
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"] = residue_mask.long()

    return TemplateContext(
        f_template_distogram=gt_res_distogram,
        f_pseudo_beta_mask=f_pseudo_beta_mask,
    )


@jaxtyped(typechecker=beartype)
def build_sampling_context(
    atom_positions: Float[torch.Tensor, "N_res 37 3"],
    atom_mask: Float[torch.Tensor, "N_res 37"],
    residue_index: Float[torch.Tensor, "N_res"],
    seq: str,
    pdb_file_path: str | None,
    atom_distogram_fn: "Distogram",
    templ_distogram_fn: "Distogram",
    batch_size: int,
    device: str,
) -> FeaturizedBatch:
    """Build the static context FeaturizedBatch for conditional or unconditional sampling.

    Parameters
    ----------
    atom_positions    : (N_res, 37, 3) reference atom coordinates in atom37 layout
    atom_mask         : (N_res, 37) float mask; 1 where atom is present
    residue_index     : (N_res,) per-residue position index
    seq               : amino-acid sequence string of length N_res
    pdb_file_str      : PDB str used as template; None is no template
    atom_distogram_fn : Distogram for atom-level pairwise distances
    templ_distogram_fn: Distogram for template Cβ pairwise distances
    batch_size        : B — number of parallel samples; all share the same context
    device            : torch device string
    """
    N_res: int = atom_positions.shape[0]
    B: int = batch_size

    # ── AllAtomContext (structure + sequence + atom distogram) ───────────────
    with torch.no_grad():
        aa_ctx: AllAtomContext = build_AA_context(
            atom_37_coordinate_tensor=atom_positions.to(device),
            atom_37_mask=atom_mask.to(device),
            residue_index=residue_index.to(device),
            aa_sequence=seq,
            atom_distogram_fn=atom_distogram_fn,
            batch_size=B,
            device=device,
        )  # aa_ctx currently does nothing during sampling.

    # ── TemplateContext (residue-level distogram from PDB templates) ─────────
    n_templ_bins: int = templ_distogram_fn.n_bins + int(templ_distogram_fn.overflow_bin)
    if not pdb_file_path:
        gt_res_distogram = torch.zeros(
            B, N_res, N_res, n_templ_bins, dtype=torch.long, device=device
        )
        f_pseudo_beta_mask = torch.zeros(B, N_res, dtype=torch.long, device=device)
        template_context = TemplateContext(
            f_template_distogram=gt_res_distogram,
            f_pseudo_beta_mask=f_pseudo_beta_mask,
        )
    else:
        protein = protein_from_pdb(pdb_file_path)
        with torch.no_grad():
            template_context = build_template_context(
                prot=protein, batch_size=B, distogram_fn=templ_distogram_fn, device=device
            )

    # ── Ala reference conformer tiled over all residues ──────────────────────
    ala_ref_pos: Float[torch.Tensor, "5 3"] = ref_pos_for_residue("ALA").to(device)

    ref_pos_single: Float[torch.Tensor, "N_atom 3"] = rearrange(
        repeat(ala_ref_pos, "a d -> n a d", n=N_res),
        "n a d -> (n a) d",
    )
    ref_element_single: Float[torch.Tensor, "N_atom E"] = rearrange(
        repeat(ATOM5_ELEMENTS.float().to(device), "a e -> n a e", n=N_res),
        "n a e -> (n a) e",
    )

    # ── Index tensors ────────────────────────────────────────────────────────
    tok_idx_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        N_res, dtype=torch.long, device=device
    ).repeat_interleave(NATOM)
    center_uid_single: Int[torch.Tensor, "N_atom"] = (
        torch.arange(N_res, dtype=torch.long, device=device) * NATOM + 1  # CA slot
    ).repeat_interleave(NATOM)
    ref_space_uid_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        N_res, dtype=torch.long, device=device
    ).repeat_interleave(NATOM)

    def tile(t: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "B ..."]:
        return repeat(t, "... -> b ...", b=B)

    packed_t_hat = 1.0 * torch.ones(B)
    # Apply zero-centered noise to turn r_gt into r_input.
    noise = torch.randn_like(aa_ctx.r_gt)
    noise = noise - reduce(noise, "b n d -> b 1 d", "mean")  # match sampling convention
    r_input: Float[torch.Tensor, "B N_atom 3"] = (
        aa_ctx.r_gt + rearrange(packed_t_hat, "b -> b 1 1") * noise
    )

    return FeaturizedBatch(
        ref_pos=tile(ref_pos_single),
        ref_element=tile(ref_element_single),
        ref_space_uid=tile(ref_space_uid_single),
        t_hat=packed_t_hat,  # these are spoofed inputs, they modified during sampling
        t_normalized=0.5 * torch.ones(B, N_res, N_res),  # spoofed inputs, modified during sampling
        tok_idx=tile(tok_idx_single),
        center_uid=tile(center_uid_single),
        gt_res_distogram=template_context.f_template_distogram,
        f_pseudo_beta_mask=template_context.f_pseudo_beta_mask,
        r_gt=aa_ctx.r_gt,
        r_gt_noised=r_input,  # spoofed inputs, modified during sampling
        atom5_mask=aa_ctx.atom5_mask,
        gt_atom_distogram_sparse=aa_ctx.gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=aa_ctx.gt_atom_distogram_mask_sparse,
        aa_indices=aa_ctx.aa_indices,
        f_residue_idx=aa_ctx.f_residue_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EDM Sampler  —  deterministic Heun ODE  (Algorithm 2 in the paper)
#
#  ODE:  dr/dsigma = (r - D_θ(r,sigma)) / sigma   (the "probability flow ODE")
#  Heun = one Euler predictor + one corrector for 2nd-order accuracy.
#
#  Optional stochasticity: inject a small amount of noise at each step
#  (S_churn > 0) to get the SDE variant, analogous to DDIM η > 0.
# ─────────────────────────────────────────────────────────────────────────────


class EDMSampler:
    """Karras et al. 2022 deterministic (Heun) sampler.

    Parameters
    ----------
    model     : trained MainTrunk
    context   : FeaturizedBatch with static per-protein fields; r_gt and t_hat
                are replaced at every denoising step
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
        model: MainTrunk,
        context: FeaturizedBatch,
        template_distogram_fn: Distogram,
        sampler_params: SamplerParams,
        noise_params: NoiseScheduleParams,
    ) -> None:
        self.model = model
        self.context = context
        self.template_distogram_fn = template_distogram_fn
        self.total_timesteps = sampler_params.ddim_steps
        self.sigma_data = noise_params.sigma_data
        self.sigma_min = noise_params.sigma_min
        self.sigma_max = noise_params.sigma_max
        self.rho = sampler_params.rho
        self.S_churn = sampler_params.S_churn
        self.S_tmin = sampler_params.S_tmin
        self.S_tmax = sampler_params.S_tmax
        self.S_noise = sampler_params.S_noise
        self.eta_step_scale = sampler_params.eta_step_scale
        self.seq_temperature = sampler_params.seq_temperature
        self.device = context.r_gt.device

    @jaxtyped(typechecker=beartype)
    def noise_schedule(self, timestep: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
        """AF3 noise schedule following the closed-form noise level sequence.

        The formula is:

        t_hat = sigma_data * (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        """
        t_hat: Float[torch.Tensor, ""] = (
            self.sigma_data
            * (
                self.sigma_max ** (1 / self.rho)
                + timestep * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            )
            ** self.rho
        )

        return t_hat

    @jaxtyped(typechecker=beartype)
    def denoise(
        self,
        *,
        r_noisy: Float[torch.Tensor, "B N_atom 3"],
        f_template_distogram: Float[torch.Tensor, "B N_res N_res n_bins"] | None,
        normalized_timestep: Float[torch.Tensor, ""],
        t_hat: Float[torch.Tensor, ""],
    ) -> tuple[Float[torch.Tensor, "B N_atom 3"], Float[torch.Tensor, "B N_res n_amino"]]:
        """Denoise noisy coordinates at noise level t_hat using the wrapped MainTrunk."""
        B: int = r_noisy.shape[0]
        N_res: int = self.context.aa_indices.shape[1]
        t_hat_batch: Float[torch.Tensor, "B"] = repeat(t_hat, " -> b", b=B)
        t_normalized: Float[torch.Tensor, "B N_res N_res"] = repeat(
            normalized_timestep,
            " -> b n1 n2",
            b=B,
            n1=N_res,
            n2=N_res,
        )

        batch: FeaturizedBatch = dataclasses.replace(
            self.context,
            r_gt_noised=r_noisy,
            t_hat=t_hat_batch,
            t_normalized=t_normalized,
        )
        if f_template_distogram is not None:
            batch = dataclasses.replace(
                batch,
                gt_res_distogram=f_template_distogram.long(),
            )
        predicted_outputs = self.model(batch)
        return predicted_outputs.r_denoised, predicted_outputs.seq_logits

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        shape: tuple[int, int, int],  # (B, N_atom, 3)
    ) -> tuple[Float[torch.Tensor, "B N_atom 3"], Float[torch.Tensor, "B N_res"]]:
        """Pallatom sampler, Algorithm 1."""
        # B, N_res = shape[0], shape[1]

        delta_t: float = 1 / self.total_timesteps
        # c_T = NoiseSchedule(1 - uniform(0, 1) * delta_t)
        c_T: Float[torch.Tensor, ""] = self.noise_schedule(
            1 - (torch.rand((), device=self.device)) * delta_t
        )
        # r_l ~ c_T * N(0, I), centred so the prior matches zero-COM training data
        r_l: Float[torch.Tensor, "B N_atom 3"] = c_T * torch.randn((shape), device=self.device)
        r_l = r_l - reduce(r_l, "b n d -> b 1 d", "mean")

        # init to sentinel values to please the type checker.
        r_denoised: Float[torch.Tensor, "B N_atom 3"] = r_l
        seq_logits: Float[torch.Tensor, "B N_res n_amino"] = torch.zeros(
            (shape[0], self.context.aa_indices.shape[1], 20), device=self.device
        )
        for timestep in range(1, self.total_timesteps - 1):
            t_p: Float[torch.Tensor, ""] = (
                timestep / self.total_timesteps - (torch.rand((), device=self.device)) * delta_t
            )
            c_T = self.noise_schedule(t_p)
            c_T_minus_one: Float[torch.Tensor, ""] = self.noise_schedule(t_p - delta_t)
            r_l = centre_random_augment(coords=r_l)
            # ── optional stochastic noise injection (S_churn) ──────────────
            gamma: float
            if self.S_tmin <= timestep / self.total_timesteps <= self.S_tmax:
                gamma = self.S_churn
            else:
                gamma = 0.0
            # Select temporarily increased noise level t_hat
            # clamp to never introduce more new noise than what is present in train data.
            t_hat: Float[torch.Tensor, ""] = torch.clamp(
                c_T * (gamma + 1),
                max=self.sigma_data * self.sigma_max,
            )

            # Add new noise to move from t_p to t_hat
            eps = torch.randn((shape), device=self.device)
            # zero the center of mass of the noise.
            eps = eps - reduce(eps, "b n d -> b 1 d", "mean")
            noisy_r_l = r_l + self.S_noise * torch.sqrt(t_hat**2 - c_T**2) * eps

            # update the self condition feature
            r_denoised, seq_logits = self.denoise(
                r_noisy=noisy_r_l,
                f_template_distogram=None,  # the self-condition feature is init earlier.
                t_hat=t_hat,
                normalized_timestep=t_p,
            )
            ca_idx: Int[torch.Tensor, "B N_res"] = rearrange(
                self.context.center_uid, "b (n a) -> b n a", a=NATOM
            )[
                :, :, 0
            ]  # I don't like this
            ca_positions: Float[torch.Tensor, "B N_res 3"] = torch.gather(
                r_denoised, 1, repeat(ca_idx, "b n -> b n d", d=3)
            )
            f_template_distogram: Float[torch.Tensor, "B N_res N_res n_bins"]
            f_template_distogram, _ = self.template_distogram_fn(ca_positions)
            # calculate the score function.
            r_denoised, seq_logits = self.denoise(
                r_noisy=noisy_r_l,
                f_template_distogram=f_template_distogram,
                t_hat=t_hat,
                normalized_timestep=t_p,
            )
            # Evaluate dx/dt at t_hat
            delta_l = (noisy_r_l - r_denoised) / t_hat
            dt = c_T_minus_one - t_hat
            # take Euler step from t_hat to next t_p
            r_l = noisy_r_l + self.eta_step_scale * dt * delta_l

            # if we are using churn, why aren't we applying a second order correction?
            # speed. 2x the number of NFE. could add later if you want.
            # ── second derivative (Heun corrector), skip at last step ──────
            # pallatom doesn't use a Heun corrector.
            # if timestep != self.total_timesteps - 1:

        # focus only on the sequence distribution decoded by the network in final sampling step
        # use low-temperature softmax to decode amino acid sequence
        decode_seqs: Float[torch.Tensor, "B N_res"] = torch.argmax(
            torch.softmax((seq_logits) / self.seq_temperature, dim=-1), dim=-1
        ).float()
        r_denoised = r_denoised - reduce(r_denoised, "b n d -> b 1 d", "mean")
        return r_denoised, decode_seqs


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PDB helper  —  atom5 (N_res, 5, 3) → atom37 (N_res, 37, 3)
# ─────────────────────────────────────────────────────────────────────────────


@jaxtyped(typechecker=beartype)
def atom5_to_atom37(
    coords_5: Float[torch.Tensor, "N_res 5 3"],
    mask_5: Float[torch.Tensor, "N_res 5"] | None = None,
) -> tuple[Float[torch.Tensor, "N_res 37 3"], Float[torch.Tensor, "N_res 37"]]:
    """Map atom5 coordinates back into the full atom37 layout.

    Returns:
    -------
    x_37   : (N_res, 37, 3)
    mask_37: (N_res, 37).
    """
    N_res: int = coords_5.shape[0]
    x_37: Float[torch.Tensor, "N_res 37 3"] = torch.zeros((N_res, 37, 3), dtype=torch.float64)
    mask_37: Float[torch.Tensor, "N_res 37"] = torch.zeros((N_res, 37), dtype=torch.float64)

    for atom5_slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        x_37[:, atom37_idx, :] = coords_5[:, atom5_slot, :]
        mask_37[:, atom37_idx] = mask_5[:, atom5_slot] if mask_5 is not None else 1.0

    return x_37, mask_37


def main(args: argparse.Namespace, scfg: SampleConfig, device: str) -> None:
    """Runs EDM Sampling for the Pallatom model."""
    with StructlogConfig(is_rank_zero=True, log_file=args.log_file), FatalOnError():
        log = structlog.get_logger()
        mp = scfg.model
        noise = scfg.noise
        gen = scfg.generation
        log.info("config loaded", config=args.config, n_res=gen.n_res, n_samples=gen.n_samples)

        model = MainTrunk(
            f_ref_dim=mp.f_ref_dim,
            n_bins=scfg.distogram_res.n_bins,
            n_atom_bins=scfg.distogram_atom.n_bins,
            c_atom=mp.c_atom,
            c_pair=mp.c_pair,
            c_res=mp.c_res,
            c_atompair=mp.c_atompair,
            K_unit=mp.K_unit,
            sigma_data=noise.sigma_data,
            residue_number=mp.max_residues,
            n_amino=mp.n_amino,
            n_blocks_atom_transformer_encoder=mp.n_blocks_atom_transformer_encoder,
            n_heads_atom_transformer_encoder=mp.n_heads_atom_transformer_encoder,
            n_blocks_atom_transformer_decoder=mp.n_blocks_atom_transformer_decoder,
            n_heads_atom_transformer_decoder=mp.n_heads_atom_transformer_decoder,
            n_pairformer_blocks_template_embedder=mp.n_pairformer_blocks_template_embedder,
            n_paiformer_heads_template_embedder=mp.n_paiformer_heads_template_embedder,
        ).to(device)
        ckpt = torch.load(scfg.checkpoint.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        log.info("model loaded", checkpoint=scfg.checkpoint.checkpoint_path, device=device)

        N_RES: int = gen.n_res
        N_atom: int = N_RES * NATOM
        B_SAMPLE: int = gen.n_samples

        _atom_disto = Distogram(
            n_bins=scfg.distogram_atom.n_bins,
            min_dist=scfg.distogram_atom.min_dist,
            max_dist=scfg.distogram_atom.max_dist,
            overflow_bin=False,
        ).to(device)
        _templ_disto = Distogram(
            n_bins=scfg.distogram_res.n_bins - 1,
            min_dist=scfg.distogram_res.min_dist,
            max_dist=scfg.distogram_res.max_dist,
            overflow_bin=True,
        ).to(device)
        context: FeaturizedBatch = build_sampling_context(
            atom_positions=torch.zeros(N_RES, 37, 3, device=device),
            atom_mask=torch.ones(N_RES, 37, device=device),
            residue_index=torch.arange(N_RES, dtype=torch.float, device=device),
            seq="A" * N_RES,
            pdb_file_path=None,  # single template only. for now it is None.
            atom_distogram_fn=_atom_disto,
            templ_distogram_fn=_templ_disto,
            batch_size=B_SAMPLE,
            device=device,
        )
        edm_sampler: EDMSampler = EDMSampler(
            model=model,
            context=context,
            template_distogram_fn=_templ_disto,
            sampler_params=scfg.sampler,
            noise_params=noise,
        )

        log.info("sampling", n_res=N_RES, n_samples=B_SAMPLE, ddim_steps=scfg.sampler.ddim_steps)
        coords_batch: Float[torch.Tensor, "B N_atom 3"]
        seqs_batch: Float[torch.Tensor, "B N_res"]
        coords_batch, seqs_batch = edm_sampler.sample(
            shape=(B_SAMPLE, N_atom, 3),
        )
        log.info("sampling complete", n_res=N_RES, n_samples=B_SAMPLE)

        pdb_strings: list[str] = []
        for b in range(B_SAMPLE):
            coords_t: Float[torch.Tensor, "N_res 5 3"] = rearrange(
                coords_batch[b].cpu(), "(n a) d -> n a d", n=N_RES, a=NATOM
            )
            x_37, mask_37 = atom5_to_atom37(coords_t)
            atom_positions: npt.NDArray[np.float64] = np.asarray(x_37, dtype=np.float64)
            atom_mask: npt.NDArray[np.float64] = np.asarray(mask_37, dtype=np.float64)
            aatype: npt.NDArray[np.intp] = np.asarray(seqs_batch[b].cpu(), dtype=np.intp)
            prot = Protein(
                atom_positions=atom_positions,
                atom_mask=atom_mask,
                residue_index=np.arange(N_RES, dtype=np.intp),
                aatype=aatype,
                chain_index=np.zeros(N_RES, dtype=np.intp),  # obvious problem
                b_factors=np.ones((N_RES, 37), dtype=np.float64),
            )
            pdb_strings.append(to_pdb(prot))

        Path(scfg.output.output_path).write_text(json.dumps(pdb_strings))
        log.info("output written", path=scfg.output.output_path, n_structures=B_SAMPLE)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main sampling script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample protein structures from a trained PallAtom model"
    )
    parser.add_argument("--config", required=True, help="path to SampleConfig JSON")
    parser.add_argument("--log_file", required=True, help="path to write structured JSON log lines")
    args = parser.parse_args()

    scfg: SampleConfig = SampleConfig.model_validate(json.loads(Path(args.config).read_text()))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    main(args=args, scfg=scfg, device=device)
