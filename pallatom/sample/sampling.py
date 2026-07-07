"""EDM sampling from a trained MainTrunk denoising network.

Contains the EDMSampler class implementing the Karras et al. 2022 Heun ODE
sampler, context-building helpers (AllAtomContext, TemplateContext), and the
main sampling entry point.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import structlog
import torch
from architecture.atom_transformers import WINDOW_SIZE, build_sparse_pairs
from architecture.main_trunk import MainTrunk, PredictedOutputs
from beartype import beartype
from einops import rearrange, repeat
from helpers.alignment import centre_random_augment, masked_com
from helpers.atom_utils import (
    ATOM5_ELEMENTS,
    ATOM37_CA,
    Protein,
    atom5_to_atom37,
    atom37_to_atom5,
    atom37_to_cb,
    protein_from_pdb,
    to_pdb,
)
from helpers.context_managers import FatalOnError, StructlogConfig
from helpers.data import Distogram, FeaturizedBatch, ref_pos_for_residue
from jaxtyping import Bool, Float, Int, jaxtyped
from sample.sample_config import SampleConfig, SamplerParams
from structlog.typing import FilteringBoundLogger
from train.train_config import NoiseScheduleParams

# atom5 slot → atom37 index (used when writing PDB via atom37 representation)
# atom5: N=0, CA=1, C=2, O=3, CB=4  →  atom37: N=0, CA=1, C=2, O=3, CB=4
NATOM = 5  # atoms per residue


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class AllAtomContext:  # N_atom = N_res * 5 for atom5 representation
    """Batched all-atom conditioning context.

    Stores the ground-truth atom5 positions, per-atom and per-residue masks,
    sparse atom-pair distogram labels, amino-acid sequence indices, and integer
    residue indices for a batch of B identical protein copies used during
    sampling.
    """

    # structure input
    r_gt: Float[torch.Tensor, "B N_atom 3"]  # gt atom_positions
    atom5_mask: Bool[torch.Tensor, "B N_atom"]
    residue_mask: Bool[torch.Tensor, "B N_res"]
    gt_atom_distogram_sparse: Int[torch.Tensor, "B N_atom K"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"]

    # amino acid input -- aa indices is the seq itself,
    # f_residue_idx is the residue index.
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

    f_template_distogram: Float[torch.Tensor, "B N_RES N_RES N_TEMPL_BINS"]
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_RES"]


@jaxtyped(typechecker=beartype)
def build_aa_context(
    prot: Protein,
    atom_distogram_fn: Distogram,
    batch_size: int,
    device: str,
) -> AllAtomContext:
    """Build the all-atom conditioning context from a ground-truth structure.

    Converts atom37 coordinates from ``prot`` to the compact atom5
    representation, derives residue and atom masks, and precomputes a sparse
    atom-pair distogram over a local K-neighbour window. All per-protein
    tensors are tiled to ``batch_size`` along the leading batch dimension,
    ready to be consumed by the sampling loop.

    Args:
        prot: Protein dataclass containing ``atom_positions`` (N_res, 37, 3),
            ``atom_mask`` (N_res, 37), ``aatype`` (N_res,) integer indices,
            and ``residue_index`` (N_res,).
        atom_distogram_fn: Callable that accepts batched atom positions and
            masks and returns a dense
            (B, N_atom, N_atom, n_bins) distogram tensor.
        batch_size: Number of parallel sampling trajectories; all context
            tensors are tiled to this batch dimension.
        device: PyTorch device string for tensor allocation.

    Returns:
        An AllAtomContext with batched ground-truth atom positions, atom
        masks, residue masks, amino-acid indices, integer residue indices,
        and sparse atom-pair distogram labels and masks.
    """
    atom_37_coordinate_tensor: Float[torch.Tensor, "N_res 37 3"] = torch.tensor(
        prot.atom_positions,
        dtype=torch.float,
        device=device,
    )
    atom_37_mask: Float[torch.Tensor, "N_res 37 3"] = torch.tensor(
        prot.atom_mask,
        dtype=torch.float,
        device=device,
    )

    aa_indices_i: Int[torch.Tensor, "N_res n_amino"] = torch.tensor(
        prot.aatype,
        dtype=torch.long,
        device=device,
    )
    f_residue_idx_i: Int[torch.Tensor, "N_res"] = torch.tensor(
        prot.residue_index,
        dtype=torch.long,
        device=device,
    )

    atom5_pos, atom5_mask = atom37_to_atom5(
        rearrange(atom_37_coordinate_tensor, "n a d -> 1 n a d"),
        rearrange(atom_37_mask, "n a -> 1 n a"),
    )
    atom5_pos = rearrange(atom5_pos, "1 n a d -> n a d")  # (N_res_i, 5, 3)
    atom5_mask = rearrange(atom5_mask, "1 n a -> n a")  # (N_res_i, 5)
    residue_mask_i: Bool[torch.Tensor, "N_res"] = atom5_mask.any(dim=-1)
    packed_flat_pos_i: Float[torch.Tensor, "N_atom 3"] = rearrange(
        atom5_pos,
        "n a d -> (n a) d",
    )
    packed_atom_mask_i: Bool[torch.Tensor, "N_atom"] = repeat(
        residue_mask_i,
        "n -> (n a)",
        a=NATOM,
    )

    # stack all to be of shape batch_size
    r_gt: Float[torch.Tensor, "B N_atom 3"] = repeat(
        packed_flat_pos_i,
        "n d -> b n d",
        b=batch_size,
    )
    atom5_mask: Bool[torch.Tensor, "B N_atom"] = repeat(
        packed_atom_mask_i,
        "n -> b n",
        b=batch_size,
    )
    residue_mask: Bool[torch.Tensor, "B N_res"] = repeat(
        residue_mask_i,
        "n -> b n",
        b=batch_size,
    )
    aa_indices: Int[torch.Tensor, "B N_res"] = repeat(
        aa_indices_i,
        "n -> b n",
        b=batch_size,
    )
    f_residue_idx: Int[torch.Tensor, "B N_res"] = repeat(
        f_residue_idx_i,
        "n -> b n",
        b=batch_size,
    )
    # atom distogram
    # ── Sparse atom distogram (batched) ──────────────────────────────────────
    _tok_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        f_residue_idx_i.shape[0],
        dtype=torch.long,
        device=device,
    ).repeat_interleave(NATOM)

    # its okay that we repeat the unbatched neighbor_idx as all members of
    # the sampling batch do not have padding
    neighbor_idx, _ = build_sparse_pairs(
        _tok_single,
        WINDOW_SIZE,
    )  # (N_atom, K)

    gt_atom_disto_dense: Float[torch.Tensor, "B N_atom N_atom n_atom_bins"]
    gt_atom_mask_dense: Bool[torch.Tensor, "B N_atom N_atom"]
    gt_atom_disto_dense, gt_atom_mask_dense = atom_distogram_fn(
        r_gt,
        atom5_mask,
    )
    n_atom_bins: int = gt_atom_disto_dense.shape[-1]

    # sparse gather: result[b, l, k] = dense[b, l, neighbor_idx[l, k]]
    nbr_b: Int[torch.Tensor, "B N_atom K"] = repeat(
        neighbor_idx,
        "n k -> b n k",
        b=batch_size,
    )
    gt_atom_distogram_sparse: Int[torch.Tensor, "B N_atom K"] = (
        gt_atom_disto_dense.gather(
            2,
            repeat(nbr_b, "b n k -> b n k d", d=n_atom_bins),
        ).argmax(dim=-1)
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
        # amino acid input -- aa indices is the seq itself,
        # f_residue_idx is the residue index.
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
    """Build a batch of template distogram contexts from a Protein object.

    Pads all proteins in the list to the same maximum residue count, extracts
    the pseudo-β carbon positions via ``atom37_to_cb``, and passes those
    positions through ``distogram_fn`` to produce the inter-residue distance
    distribution used as structural conditioning. The batch size equals the
    number of proteins supplied, so callers must ensure the list length
    matches the model's expected batch size.

    Args:
        prot: Protein dataclass instance; contains
            ``atom_positions`` (N_res, 37, 3) and ``atom_mask`` (N_res, 37)
            numpy arrays.  Proteins with fewer residues than the batch maximum
            are zero-padded on the C-terminal end.
        batch_size: Number of copies of the template to stack into the batch
            dimension.
        distogram_fn: Callable that maps batched Cβ positions of shape
            (B, N_res, 3) and a residue mask of shape (B, N_res) to a
            (B, N_res, N_res, n_bins) distogram tensor.
        device: PyTorch device string for tensor allocation; defaults to
            ``"cpu"``.

    Returns:
        A TemplateContext containing the integer-quantised distogram
        ``f_template_distogram`` of shape (B, N_res, N_res, n_templ_bins) and
        the
        binary pseudo-β mask ``f_pseudo_beta_mask`` of shape (B, N_res).
    """
    pos: Float[torch.Tensor, "N_res 37 3"] = torch.tensor(
        prot.atom_positions,
        dtype=torch.float64,
        device=device,
    )
    mask: Float[torch.Tensor, "N_res 37"] = torch.tensor(
        prot.atom_mask,
        dtype=torch.float64,
        device=device,
    )

    # repeat to get batch dim
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"] = repeat(
        pos,
        "n a d -> b n a d",
        b=batch_size,
    )
    atom37_mask: Float[torch.Tensor, "B N_res 37"] = repeat(
        mask,
        "n a -> b n a",
        b=batch_size,
    )
    residue_mask: Bool[torch.Tensor, "B N_res"] = atom37_mask[
        :,
        :,
        ATOM37_CA,
    ].bool()

    pseudo_beta_carbon_positions: Float[torch.Tensor, "B N_res 3"]
    pseudo_beta_carbon_positions, _ = atom37_to_cb(
        atom37_positions=atom37_positions,
        atom37_mask=atom37_mask,
    )

    f_disto: Float[torch.Tensor, "B N_res N_res n_templ_bins"]
    f_disto, _ = distogram_fn(pseudo_beta_carbon_positions, residue_mask)

    noised_res_distogram: Float[torch.Tensor, "B N_res N_res n_templ_bins"] = (
        f_disto
    )
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"] = residue_mask.long()

    return TemplateContext(
        f_template_distogram=noised_res_distogram,
        f_pseudo_beta_mask=f_pseudo_beta_mask,
    )


@jaxtyped(typechecker=beartype)
def build_sampling_context(
    pdb_file_path: Path | None,
    atom_distogram_fn: "Distogram",
    templ_distogram_fn: "Distogram",
    residue_number: int,
    batch_size: int,
    device: str,
) -> FeaturizedBatch:
    """Build static context FeaturizedBatch for sampling.

    Args:
        pdb_file_path: Path to a PDB file used as template; None for no
            template (zero-initialised coordinates are used instead).
        atom_distogram_fn: Distogram for atom-level pairwise distances.
        templ_distogram_fn: Distogram for template Cβ pairwise distances.
        residue_number: Number of residues to generate.
        batch_size: Number of parallel samples; all share the same context.
        device: PyTorch device string (e.g. "cpu" or "cuda:0").

    Returns:
        A FeaturizedBatch with static context fields populated and spoofed
        r_gt_noised / t_hat / t_normalized placeholders ready to be
        overwritten at each sampling step.
    """
    B: int = batch_size

    # ── TemplateContext (residue-level distogram from PDB templates) ─────────
    protein = (
        Protein(
            atom_positions=np.zeros((residue_number, 37, 3)),
            atom_mask=np.ones((residue_number, 37)),
            residue_index=np.arange(residue_number, dtype=np.intp),
            aatype=np.zeros(residue_number, dtype=np.intp),
            chain_index=np.zeros(residue_number, dtype=np.intp),
            b_factors=np.zeros((residue_number, 37)),
        )
        if not pdb_file_path
        else protein_from_pdb(pdb_file_path)
    )
    with torch.no_grad():
        template_context = build_template_context(
            prot=protein,
            batch_size=B,
            distogram_fn=templ_distogram_fn,
            device=device,
        )
        aa_ctx: AllAtomContext = build_aa_context(
            prot=protein,
            atom_distogram_fn=atom_distogram_fn,
            batch_size=B,
            device=device,
        )

    # ── Ala reference conformer tiled over all residues ──────────────────────
    ala_ref_pos: Float[torch.Tensor, "5 3"] = ref_pos_for_residue("ALA").to(
        device,
    )

    ref_pos_single: Float[torch.Tensor, "N_atom 3"] = rearrange(
        repeat(ala_ref_pos, "a d -> n a d", n=residue_number),
        "n a d -> (n a) d",
    )
    ref_element_single: Float[torch.Tensor, "N_atom E"] = rearrange(
        repeat(
            ATOM5_ELEMENTS.float().to(device),
            "a e -> n a e",
            n=residue_number,
        ),
        "n a e -> (n a) e",
    )

    # ── Index tensors ───────────────────────────────────────────────────────
    tok_idx_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        residue_number,
        dtype=torch.long,
        device=device,
    ).repeat_interleave(NATOM)
    center_uid_single: Int[torch.Tensor, "N_atom"] = (
        torch.arange(residue_number, dtype=torch.long, device=device) * NATOM
        + 1  # CA slot
    ).repeat_interleave(NATOM)
    ref_space_uid_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        residue_number,
        dtype=torch.long,
        device=device,
    ).repeat_interleave(NATOM)

    def tile(t: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "B ..."]:
        return repeat(t, "... -> b ...", b=B)

    packed_t_hat = 1.0 * torch.ones(B, device=device)
    # Apply zero-centered noise to turn r_gt into r_input. The mean is taken
    # over valid atoms only, so padded atom slots don't bias it toward the
    # origin (match sampling convention).
    noise = torch.randn_like(aa_ctx.r_gt, device=device)
    noise = noise - masked_com(noise, mask=aa_ctx.atom5_mask)
    r_input: Float[torch.Tensor, "B N_atom 3"] = (
        aa_ctx.r_gt + rearrange(packed_t_hat, "b -> b 1 1") * noise
    )

    return FeaturizedBatch(
        ref_pos=tile(ref_pos_single),
        ref_element=tile(ref_element_single),
        ref_space_uid=tile(ref_space_uid_single),
        t_hat=packed_t_hat,  # spoofed inputs, modified during sampling
        t_normalized=0.5
        * torch.ones(
            B,
            residue_number,
            residue_number,
        ),  # spoofed inputs, modified during sampling
        tok_idx=tile(tok_idx_single),
        center_uid=tile(center_uid_single),
        gt_res_distogram_indices=torch.zeros(
            B,
            residue_number,
            residue_number,
            dtype=torch.long,
            device=device,
        ),
        noised_res_distogram=template_context.f_template_distogram,
        f_pseudo_beta_mask=template_context.f_pseudo_beta_mask,
        r_gt=aa_ctx.r_gt,
        r_gt_noised=r_input,  # spoofed inputs, modified during sampling
        atom5_mask=aa_ctx.atom5_mask,
        gt_atom_distogram_sparse=aa_ctx.gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=aa_ctx.gt_atom_distogram_mask_sparse,
        aa_indices=aa_ctx.aa_indices,
        f_residue_idx=aa_ctx.f_residue_idx,
    )


# ────────────────────────────────────────────────────────────────────────────
# EDM Sampler  —  deterministic Heun ODE  (Algorithm 2 in the paper)
#
#  ODE:  dr/dsigma = (r - D_θ(r,sigma)) / sigma   (the "probability flow ODE")
#  Heun = one Euler predictor + one corrector for 2nd-order accuracy.
#
#  Optional stochasticity: inject a small amount of noise at each step
#  (S_churn > 0) to get the SDE variant, analogous to DDIM η > 0.
# ────────────────────────────────────────────────────────────────────────────


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
        self.model: MainTrunk = model
        self.context: FeaturizedBatch = context
        self.template_distogram_fn: Distogram = template_distogram_fn
        self.total_timesteps: int = sampler_params.ddim_steps
        self.sigma_data: float = noise_params.sigma_data
        self.sigma_min: float = noise_params.sigma_min
        self.sigma_max: float = noise_params.sigma_max
        self.rho: float = sampler_params.rho
        self.S_churn: float = sampler_params.S_churn
        self.S_tmin: float = sampler_params.S_tmin
        self.S_tmax: float = sampler_params.S_tmax
        self.S_noise: float = sampler_params.S_noise
        self.eta_step_scale: float = sampler_params.eta_step_scale
        self.seq_temperature: float = sampler_params.seq_temperature
        self.device: torch.device = context.r_gt.device

    @jaxtyped(typechecker=beartype)
    def noise_schedule(
        self,
        timestep: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, ""]:
        """AF3 noise schedule following closed-form noise level sequence.

        The formula is:

        t_hat = sigma_data * (sigma_max^(1/rho) + t * (sigma_min^(1/rho) -
        sigma_max^(1/rho)))^rho
        """
        t_hat: Float[torch.Tensor, ""] = cast(
            'Float[torch.Tensor, ""]',
            self.sigma_data
            * (
                self.sigma_max ** (1 / self.rho)
                + timestep
                * (
                    self.sigma_min ** (1 / self.rho)
                    - self.sigma_max ** (1 / self.rho)
                )
            )
            ** self.rho,
        )

        return t_hat

    @jaxtyped(typechecker=beartype)
    def denoise(
        self,
        *,
        r_noisy: Float[torch.Tensor, "B N_atom 3"],
        f_template_distogram: (
            Float[torch.Tensor, "B N_res N_res n_bins"] | None
        ),
        normalized_timestep: Float[torch.Tensor, ""],
        t_hat: Float[torch.Tensor, ""],
    ) -> tuple[
        Float[torch.Tensor, "B N_atom 3"],
        Float[torch.Tensor, "B N_res n_amino"],
    ]:
        """Denoise noisy coordinates at noise level t_hat using MainTrunk.

        Constructs a FeaturizedBatch by injecting the current noisy positions
        and noise level into the static sampling context, optionally
        overriding the template distogram, and returns the denoised
        coordinates together with per-residue sequence logits.

        Args:
            r_noisy: Noisy atom positions at the current denoising step,
                shape (B, N_atom, 3).
            f_template_distogram: Optional template pairwise distance
                distribution to inject into the batch; pass None to keep the
                distogram already stored in the context.
            normalized_timestep: Diffusion time in [0, 1) as a scalar tensor,
                broadcast to a (B, N_res, N_res) pair field.
            t_hat: Current noise level sigma as a scalar tensor.

        Returns:
            A tuple of (r_denoised, seq_logits) where r_denoised has shape
            (B, N_atom, 3) and seq_logits has shape (B, N_res, n_amino).
        """
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
                noised_res_distogram=f_template_distogram,
            )
        predicted_outputs: PredictedOutputs = self.model(batch)
        return predicted_outputs.r_denoised, predicted_outputs.seq_logits

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        shape: tuple[int, int, int],  # (B, N_atom, 3)
    ) -> tuple[
        Float[torch.Tensor, "B N_atom 3"],
        Float[torch.Tensor, "B N_res"],
    ]:
        """Pallatom sampler, Algorithm 1.

        Runs the full denoising trajectory from sigma_max down to sigma_min
        using the Euler-step ODE solver with optional stochastic noise
        injection (S_churn). Each iteration applies a self-conditioning step
        followed by a full denoising call to obtain the score estimate.

        Args:
            shape: Tuple (B, N_atom, 3) specifying the batch size, total atom
                count, and spatial dimension of the output coordinate tensor.

        Returns:
            A tuple of (r_denoised, decode_seqs) where r_denoised has shape
            (B, N_atom, 3) containing centre-of-mass-zeroed coordinates and
            decode_seqs has shape (B, N_res) containing integer amino-acid
            indices decoded via low-temperature softmax.
        """
        delta_t: float = 1 / self.total_timesteps
        # from pallatom, c_T = NoiseSchedule(1 - uniform(0, 1) * delta_t)
        c_T: Float[torch.Tensor, ""] = self.noise_schedule(
            1 - (torch.rand((), device=self.device)) * delta_t,
        )
        # r_l ~ c_T * N(0, I), centred so prior matches zero center of mass
        # training data. Centring uses valid atoms only so padded atom slots
        # don't bias it toward the origin.
        r_l: Float[torch.Tensor, "B N_atom 3"] = c_T * torch.randn(
            (shape),
            device=self.device,
        )
        r_l = r_l - masked_com(r_l, mask=self.context.atom5_mask)

        # init to sentinel values to please the type checker.
        r_denoised: Float[torch.Tensor, "B N_atom 3"] = r_l
        seq_logits: Float[torch.Tensor, "B N_res n_amino"] = torch.zeros(
            (shape[0], self.context.aa_indices.shape[1], 20),
            device=self.device,
        )
        for timestep in range(1, self.total_timesteps - 1):
            t_p: Float[torch.Tensor, ""] = (
                timestep / self.total_timesteps
                - (torch.rand((), device=self.device)) * delta_t
            )
            c_T = self.noise_schedule(t_p)
            c_T_minus_one: Float[torch.Tensor, ""] = self.noise_schedule(
                t_p - delta_t,
            )
            r_l = centre_random_augment(
                coords=r_l,
                mask=self.context.atom5_mask,
            )
            # ── optional stochastic noise injection (S_churn) ──────────────
            gamma: float = (
                self.S_churn
                if self.S_tmin <= timestep / self.total_timesteps <= self.S_tmax
                else 0.0
            )
            # Select temporarily increased noise level t_hat
            t_hat: Float[torch.Tensor, ""] = c_T * (gamma + 1)

            # Add new noise to move from t_p to t_hat
            eps = torch.randn((shape), device=self.device)
            # zero the center of mass of the noise, over valid atoms only.
            eps = eps - masked_com(eps, mask=self.context.atom5_mask)
            noisy_r_l = r_l + (
                self.S_noise * torch.sqrt(t_hat**2 - c_T**2) * eps
            )

            # update the self condition feature
            # the self-condition template distogram is init earlier.
            r_denoised, seq_logits = self.denoise(
                r_noisy=noisy_r_l,
                f_template_distogram=None,
                t_hat=t_hat,
                normalized_timestep=t_p,
            )

            atom37_positions: Float[torch.Tensor, "B N_res 37 3"]
            atom37_mask: Float[torch.Tensor, "B N_res 37"]
            atom37_positions, atom37_mask = atom5_to_atom37(
                rearrange(r_denoised, "b (n a) d -> b n a d", a=NATOM),
                rearrange(
                    self.context.atom5_mask,
                    "b (n a) -> b n a",
                    a=NATOM,
                ).float(),
            )
            cb_positions: Float[torch.Tensor, "B N_res 3"]
            cb_positions, _ = atom37_to_cb(atom37_positions, atom37_mask)
            self_condition_residue_mask: Bool[torch.Tensor, "B N_res"] = (
                self.context.f_pseudo_beta_mask.bool()
            )
            f_template_distogram: Float[torch.Tensor, "B N_res N_res n_bins"]
            f_template_distogram, _ = self.template_distogram_fn(
                cb_positions,
                self_condition_residue_mask,
            )
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

            # if using churn, why not apply a second order correction?
            # speed. 2x the number of NFE. could add later if you want.
            # ── second derivative (Heun corrector), skip at last step ──────
            # pallatom doesn't use a Heun corrector.
            # if timestep != self.total_timesteps - 1:

        # focus only on the sequence distribution decoded by the network
        # in final sampling step use low-temperature softmax to decode
        # amino acid sequence
        decode_seqs: Float[torch.Tensor, "B N_res"] = torch.argmax(
            torch.softmax((seq_logits) / self.seq_temperature, dim=-1),
            dim=-1,
        ).float()
        r_denoised = r_denoised - masked_com(
            r_denoised,
            mask=self.context.atom5_mask,
        )
        return r_denoised, decode_seqs


@dataclasses.dataclass
class SamplingArgs:
    """Parsed command-line arguments for the sampling entry point.

    Wraps the two required CLI flags so downstream code can reference them with
    static types rather than relying on the untyped argparse Namespace object.
    """

    config: Path
    log_file: Path


def _parse_args() -> SamplingArgs:
    """Build the argument parser and return a fully typed SamplingArgs.

    Returns:
        A SamplingArgs dataclass populated from sys.argv containing the path
        to the JSON SampleConfig file and the path for the structured log
        output.
    """
    _ = parser = argparse.ArgumentParser(
        description="Sample protein structures from a trained PallAtom model",
    )
    _ = parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="path to SampleConfig JSON",
    )
    _ = parser.add_argument(
        "--log_file",
        required=True,
        type=Path,
        help="path to write structured JSON log lines",
    )
    ns = parser.parse_args()
    return SamplingArgs(
        config=cast("Path", ns.config),
        log_file=cast("Path", ns.log_file),
    )


def main(args: SamplingArgs, scfg: SampleConfig, device: str) -> None:
    """Run EDM sampling for the Pallatom model end-to-end.

    Loads the model checkpoint, builds the sampling context, runs the EDM
    sampler, converts the resulting atom5 coordinates to atom37 PDB format,
    and writes all sampled structures as a JSON list of PDB strings to the
    configured output path.

    Args:
        args: Parsed command-line arguments containing the config path and log
            file path.
        scfg: Fully validated SampleConfig instance with model, noise, sampler,
            and output sub-configurations.
        device: PyTorch device string (e.g. "cuda" or "cpu") on which the model
            and tensors are allocated.
    """
    with (
        StructlogConfig(is_rank_zero=True, log_file=args.log_file),
        FatalOnError(),
    ):
        log: FilteringBoundLogger = cast(
            "FilteringBoundLogger",
            structlog.get_logger(),
        )
        mp = scfg.model
        noise = scfg.noise
        gen = scfg.generation
        log.info(
            "config loaded",
            config=args.config,
            n_res=gen.n_res,
            n_samples=gen.n_samples,
        )

        model = MainTrunk(
            model_params=mp,
            res_distogram_params=scfg.distogram_res,
            atom_distogram_params=scfg.distogram_atom,
            noise_params=noise,
        ).to(device)
        ckpt = cast(
            "dict[str, dict[str, Float[torch.Tensor, ...]]]",
            torch.load(scfg.checkpoint.checkpoint_path, map_location=device),
        )
        _ = model.load_state_dict(ckpt["model"])
        _ = model.eval()
        log.info(
            "model loaded",
            checkpoint=scfg.checkpoint.checkpoint_path,
            device=device,
        )

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
            pdb_file_path=None,  # single template only. for now it is None.
            atom_distogram_fn=_atom_disto,
            templ_distogram_fn=_templ_disto,
            residue_number=N_RES,
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

        log.info(
            "sampling",
            n_res=N_RES,
            n_samples=B_SAMPLE,
            ddim_steps=scfg.sampler.ddim_steps,
        )
        coords_batch: Float[torch.Tensor, "B N_atom 3"]
        seqs_batch: Float[torch.Tensor, "B N_res"]
        coords_batch, seqs_batch = edm_sampler.sample(
            shape=(B_SAMPLE, N_atom, 3),
        )
        log.info("sampling complete", n_res=N_RES, n_samples=B_SAMPLE)

        pdb_strings: list[str] = []
        for b in range(B_SAMPLE):
            coords_t: Float[torch.Tensor, "1 N_res 5 3"] = rearrange(
                coords_batch[b].cpu(),
                "(n a) d -> 1 n a d",
                n=N_RES,
                a=NATOM,
            )
            x_37_b, mask_37_b = atom5_to_atom37(coords_t)
            atom_positions: npt.NDArray[np.float64] = np.asarray(
                rearrange(x_37_b, "1 n a d -> n a d"),
                dtype=np.float64,
            )
            atom_mask: npt.NDArray[np.float64] = np.asarray(
                rearrange(mask_37_b, "1 n a -> n a"),
                dtype=np.float64,
            )
            aatype: npt.NDArray[np.intp] = np.asarray(
                seqs_batch[b].cpu(),
                dtype=np.intp,
            )
            prot = Protein(
                atom_positions=atom_positions,
                atom_mask=atom_mask,
                residue_index=np.arange(N_RES, dtype=np.intp),
                aatype=aatype,
                chain_index=np.zeros(N_RES, dtype=np.intp),  # obvious problem
                b_factors=np.ones((N_RES, 37), dtype=np.float64),
            )
            pdb_strings.append(to_pdb(prot))

        _ = Path(scfg.output.output_path).write_text(
            json.dumps(pdb_strings),
            encoding="utf-8",
        )
        log.info(
            "output written",
            path=scfg.output.output_path,
            n_structures=B_SAMPLE,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main sampling script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _args = _parse_args()
    _scfg: SampleConfig = SampleConfig.model_validate(
        json.loads(Path(_args.config).read_text(encoding="utf-8")),
    )
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    main(args=_args, scfg=_scfg, device=_device)
