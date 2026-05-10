"""EDM sampling from a trained MainTrunk denoising network."""

import dataclasses
import math

import numpy as np
import structlog
import torch
import torch.nn as nn
from architecture.atom_transformers import WINDOW_SIZE, build_sparse_pairs
from architecture.main_trunk import MainTrunk
from beartype import beartype
from einops import rearrange, repeat
from helpers.atom_utils import (
    ATOM5_ELEMENTS,
    ATOM5_NAMES,
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
    rigid_group_atom_positions,
    to_pdb,
)
from helpers.featurize import Distogram, FeaturizedBatch
from jaxtyping import Bool, Float, Int, jaxtyped

log = structlog.get_logger()

# atom5 slot → atom37 index (used when writing PDB via atom37 representation)
# atom5: N=0, CA=1, C=2, O=3, CB=4  →  atom37: N=0, CA=1, C=2, O=3, CB=4
ATOM5_TO_ATOM37 = [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB]
NATOM = 5  # atoms per residue

# write sampling contexts and APIs for:
# unconditional sampling -- done I believe
# conditional sampling from amino acid sequence alone (no templates)
# conditional sampling from amino acid sequence + partial template
# conditional sampling from no amino acid sequence + partial template -- done I believe
# conditional sampling from no amino acid sequence + full template


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
        model: MainTrunk,
        context: FeaturizedBatch,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        super().__init__()
        self.model = model
        self.context = context
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        r_input: Float[torch.Tensor, "B N_atom 3"],
        t_hat: float,
    ) -> Float[torch.Tensor, "B N_atom 3"]:
        t_normalized: float = (math.log(t_hat) - math.log(self.sigma_min)) / (
            math.log(self.sigma_max) - math.log(self.sigma_min)
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


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class AllAtomContext:  # N_atom = N_res * 5 for atom5 representation
    # structure input
    r_gt: Float[torch.Tensor, "B N_atom 3"]  # gt atom_positions
    atom5_mask: Bool[torch.Tensor, "B N_atom"]
    residue_mask: Bool[torch.Tensor, "B N_res"]
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"]

    # amino acid input -- aa indices is the seq itself, f_residue_idx is the residue index
    aa_indices: Int[torch.Tensor, "B N_res"]
    f_residue_idx: Float[torch.Tensor, "B N_res c_res"]


@jaxtyped(typechecker=beartype)
def build_AA_context(
    atom_37_coordinate_tensor: Float[torch.Tensor, "N_res 37 3"],
    atom_37_mask: Float[torch.Tensor, "N_res 37"],
    residue_index: Float[torch.Tensor, "N_res"],
    index_embedding: nn.Embedding,
    aa_sequence: str,
    atom_distogram_fn: Distogram,
    batch_size: int,
    device: str,
) -> AllAtomContext:
    _aa_vals = [restype_order.get(r, 20) for r in aa_sequence]
    aa_indices_i: Int[torch.Tensor, N_res] = torch.tensor(_aa_vals, dtype=torch.long, device=device)
    f_residue_idx_i: Float[torch.Tensor, "N_res c_res"] = index_embedding(residue_index.long())

    atom5_pos, atom5_mask = atom37_to_atom5(
        rearrange(atom_37_coordinate_tensor, "n a d -> 1 n a d"),
        rearrange(atom_37_mask, "n a -> 1 n a"),
    )
    atom5_pos = rearrange(atom5_pos, "1 n a d -> n a d")  # (N_res_i, 5, 3)
    atom5_mask = rearrange(atom5_mask, "1 n a -> n a")  # (N_res_i, 5)

    N_res: int = atom_37_coordinate_tensor.shape[0]
    residue_mask_i: Bool[torch.Tensor, N_res] = atom5_mask.any(dim=-1)
    packed_flat_pos_i: Float[torch.Tensor, "N_atom 3"] = rearrange(atom5_pos, "n a d -> (n a) d")
    packed_atom_mask_i: Bool[torch.Tensor, N_atom] = repeat(residue_mask_i, "n -> (n a)", a=NATOM)

    # stack all to be of shape batch_size
    r_gt: Float[torch.Tensor, "B N_atom 3"] = repeat(
        packed_flat_pos_i, "n d -> b n d", b=batch_size
    )
    atom5_mask: Bool[torch.Tensor, "B N_atom"] = repeat(
        packed_atom_mask_i, "n -> b n", b=batch_size
    )
    residue_mask: Bool[torch.Tensor, "B N_res"] = repeat(residue_mask_i, "n -> b n", b=batch_size)
    aa_indices: Int[torch.Tensor, "B N_res"] = repeat(aa_indices_i, "n -> b n", b=batch_size)
    f_residue_idx: Float[torch.Tensor, "B N_res c_res"] = repeat(
        f_residue_idx_i, "n c -> b n c", b=batch_size
    )
    # atom distogram
    # ── Sparse atom distogram (batched) ──────────────────────────────────────
    _tok_single: Int[torch.Tensor, N_atom] = torch.arange(
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
@dataclasses.dataclass(frozen=True)
class TemplateContext:
    f_template_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"]
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"]


@jaxtyped(typechecker=beartype)
def build_template_context(
    ls_of_proteins: list[Protein],
    distogram_fn: Distogram,
    device: str = "cpu",
) -> TemplateContext:
    max_n_res: int = max(p.atom_positions.shape[0] for p in ls_of_proteins)

    pos_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    for prot in ls_of_proteins:
        n_res: int = prot.atom_positions.shape[0]
        pad: int = max_n_res - n_res
        pos_i = torch.tensor(prot.atom_positions, dtype=torch.float32, device=device)
        mask_i = torch.tensor(prot.atom_mask, dtype=torch.float32, device=device)
        if pad > 0:
            pos_i = torch.cat(
                [pos_i, torch.zeros(pad, 37, 3, dtype=torch.float32, device=device)], dim=0
            )
            mask_i = torch.cat(
                [mask_i, torch.zeros(pad, 37, dtype=torch.float32, device=device)], dim=0
            )
        pos_list.append(pos_i)
        mask_list.append(mask_i)

    atom37_positions: Float[torch.Tensor, "B N_res 37 3"] = torch.stack(pos_list)
    atom37_mask: Float[torch.Tensor, "B N_res 37"] = torch.stack(mask_list)

    residue_mask: Bool[torch.Tensor, "B N_res"] = atom37_mask[:, :, ATOM37_CA].bool()

    pseudo_beta_carbon_positions: Float[torch.Tensor, "B N_res 3"]
    beta_carbon_mask: Bool[torch.Tensor, "B N_res"]
    pseudo_beta_carbon_positions, beta_carbon_mask = atom37_to_cb(
        atom37_positions=atom37_positions,
        atom37_mask=atom37_mask,
    )

    f_disto, _ = distogram_fn(pseudo_beta_carbon_positions, residue_mask)

    gt_res_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"] = f_disto.long()
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"] = beta_carbon_mask.long()

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
    pdb_files: list[str],
    index_embedding: nn.Embedding,
    atom_distogram_fn: "Distogram",
    templ_distogram_fn: "Distogram",
    batch_size: int = 1,
    device: str = "cpu",
) -> FeaturizedBatch:
    """
    Build the static context FeaturizedBatch for conditional or unconditional sampling.

    Parameters
    ----------
    atom_positions    : (N_res, 37, 3) reference atom coordinates in atom37 layout
    atom_mask         : (N_res, 37) float mask; 1 where atom is present
    residue_index     : (N_res,) per-residue position index fed to index_embedding
    seq               : amino-acid sequence string of length N_res
    pdb_files         : PDB paths used as templates; empty list → unconditioned templates
    index_embedding   : nn.Embedding mapping residue indices to c_res-dim vectors
    atom_distogram_fn : Distogram for atom-level pairwise distances
    templ_distogram_fn: Distogram for template Cβ pairwise distances
    batch_size        : B — number of parallel samples; all share the same context
    device            : torch device string
    """
    N_res: int = atom_positions.shape[0]
    N_atom: int = N_res * NATOM
    B: int = batch_size

    # ── AllAtomContext (structure + sequence + atom distogram) ───────────────
    with torch.no_grad():
        aa_ctx: AllAtomContext = build_AA_context(
            atom_37_coordinate_tensor=atom_positions.to(device),
            atom_37_mask=atom_mask.to(device),
            residue_index=residue_index.to(device),
            index_embedding=index_embedding,
            aa_sequence=seq,
            atom_distogram_fn=atom_distogram_fn,
            batch_size=B,
            device=device,
        )

    # ── TemplateContext (residue-level distogram from PDB templates) ─────────
    n_templ_bins: int = templ_distogram_fn.n_bins + int(templ_distogram_fn.overflow_bin)
    if pdb_files:
        proteins: list[Protein] = [protein_from_pdb(p) for p in pdb_files]
        templ_ctx: TemplateContext = build_template_context(
            proteins, templ_distogram_fn, device=device
        )
        n_templ_res: int = templ_ctx.f_template_distogram.shape[1]
        if n_templ_res < N_res:
            # Template covers fewer residues than the target; pad remainder with zeros.
            disto_padded = torch.zeros(
                1, N_res, N_res, n_templ_bins, dtype=torch.long, device=device
            )
            disto_padded[:, :n_templ_res, :n_templ_res, :] = templ_ctx.f_template_distogram[:1]
            mask_padded = torch.zeros(1, N_res, dtype=torch.long, device=device)
            mask_padded[:, :n_templ_res] = templ_ctx.f_pseudo_beta_mask[:1]
            gt_res_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"] = repeat(
                disto_padded, "1 n m k -> b n m k", b=B
            )
            f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"] = repeat(
                mask_padded, "1 n -> b n", b=B
            )
        else:
            gt_res_distogram = repeat(
                templ_ctx.f_template_distogram[:1, :N_res, :N_res, :], "1 n m k -> b n m k", b=B
            )
            f_pseudo_beta_mask = repeat(templ_ctx.f_pseudo_beta_mask[:1, :N_res], "1 n -> b n", b=B)
    else:
        gt_res_distogram = torch.zeros(
            B, N_res, N_res, n_templ_bins, dtype=torch.long, device=device
        )
        f_pseudo_beta_mask = torch.zeros(B, N_res, dtype=torch.long, device=device)

    # ── Ala reference conformer tiled over all residues ──────────────────────
    def _ala_ref_pos() -> Float[torch.Tensor, "5 3"]:
        pos_by_name = {name: pos for name, _, pos in rigid_group_atom_positions["ALA"]}
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

    # ── Index tensors ────────────────────────────────────────────────────────
    tok_idx_single: Int[torch.Tensor, N_atom] = torch.arange(
        N_res, dtype=torch.long, device=device
    ).repeat_interleave(NATOM)
    center_uid_single: Int[torch.Tensor, N_res] = (
        torch.arange(N_res, dtype=torch.long, device=device) * NATOM + 1  # CA slot
    )

    def tile(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(B, *t.shape).contiguous()

    return FeaturizedBatch(
        ref_pos=tile(ref_pos_single),
        ref_element=tile(ref_element_single),
        ref_space_uid=torch.zeros(B, N_atom, dtype=torch.long, device=device),
        t_hat=1.0,
        t_normalized=0.5,
        tok_idx=tile(tok_idx_single),
        center_uid=tile(center_uid_single),
        gt_res_distogram=gt_res_distogram,
        f_pseudo_beta_mask=f_pseudo_beta_mask,
        r_input=torch.zeros(B, N_atom, 3, dtype=torch.float32, device=device),
        r_gt=aa_ctx.r_gt,
        atom5_mask=aa_ctx.atom5_mask,
        residue_mask=aa_ctx.residue_mask,
        gt_atom_distogram_sparse=aa_ctx.gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=aa_ctx.gt_atom_distogram_mask_sparse,
        aa_indices=aa_ctx.aa_indices,
        f_residue_idx=aa_ctx.f_residue_idx,
    )


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
        denoiser: EDMPrecond,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_tmin: float = 0.0,
        S_tmax: float = float("inf"),
        S_noise: float = 1.003,
    ):
        self.denoiser = denoiser
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @jaxtyped(typechecker=beartype)
    def _sigma_schedule(
        self,
        steps: int,
        device: torch.device | str,
    ) -> Float[torch.Tensor, "S"]:  # S = steps + 1
        """
        Karras σ schedule: σ_i = (σ_max^(1/ρ) + i/(N-1)·(σ_min^(1/ρ)−σ_max^(1/ρ)))^ρ
        Returns a tensor of length (steps+1,) with σ_N=0 appended.
        """
        rho: float = self.rho
        i: Float[torch.Tensor, steps] = torch.arange(steps, device=device).float()
        t: Float[torch.Tensor, steps] = (
            self.sigma_max ** (1 / rho)
            + i / (steps - 1) * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho
        return torch.cat([t, t.new_zeros(1)])  # (S,)  S = steps + 1

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def sample(
        self,
        shape: tuple[int, int, int],  # (B, N_atom, 3)
        steps: int = 40,
        device: torch.device | str = "cpu",
    ) -> Float[torch.Tensor, "B N_atom 3"]:
        """
        shape : (B, N_atom, 3)  — batch of atom coordinate tensor shapes
        steps : number of ODE steps  (40 is usually plenty)
        """
        sigmas: Float[torch.Tensor, S] = self._sigma_schedule(steps, device)

        # pure noise initialised at σ_max — independent per batch item
        z: Float[torch.Tensor, "B N_atom 3"] = torch.randn(shape, device=device) * sigmas[0]

        for i in range(steps):
            sigma_cur: Float[torch.Tensor, ""] = sigmas[i]
            sigma_next: Float[torch.Tensor, ""] = sigmas[i + 1]

            # ── optional stochastic noise injection (S_churn) ──────────────
            sigma_hat: Float[torch.Tensor, ""]
            if self.S_churn > 0 and self.S_tmin <= sigma_cur <= self.S_tmax:
                gamma: float = min(self.S_churn / steps, math.sqrt(2) - 1)
                sigma_hat = sigma_cur * (1 + gamma)
                z = z + (sigma_hat**2 - sigma_cur**2).sqrt() * self.S_noise * torch.randn_like(z)
            else:
                sigma_hat = sigma_cur

            # ── first derivative (Euler predictor) ─────────────────────────
            D_cur: Float[torch.Tensor, "B N_atom 3"] = self.denoiser(z, sigma_hat.item())
            d_cur: Float[torch.Tensor, "B N_atom 3"] = (z - D_cur) / sigma_hat
            z_next: Float[torch.Tensor, "B N_atom 3"] = z + (sigma_next - sigma_hat) * d_cur

            # ── second derivative (Heun corrector), skip at last step ──────
            if sigma_next > 0:
                D_next: Float[torch.Tensor, "B N_atom 3"] = self.denoiser(z_next, sigma_next.item())
                d_next: Float[torch.Tensor, "B N_atom 3"] = (z_next - D_next) / sigma_next
                d_avg: Float[torch.Tensor, "B N_atom 3"] = (d_cur + d_next) / 2.0
                z_next = z + (sigma_next - sigma_hat) * d_avg

            z = z_next

        return z  # (B, N_atom, 3)  denoised atom5 coordinates


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PDB helper  —  atom5 (N_res, 5, 3) → atom37 (N_res, 37, 3)
# ─────────────────────────────────────────────────────────────────────────────


@jaxtyped(typechecker=beartype)
def atom5_to_atom37(
    coords_5: Float[np.ndarray, "N_res 5 3"],
    mask_5: Float[np.ndarray, "N_res 5"] | None = None,
) -> tuple[Float[np.ndarray, "N_res 37 3"], Float[np.ndarray, "N_res 37"]]:
    """
    Map atom5 coordinates back into the full atom37 layout.

    Returns
    -------
    x_37   : (N_res, 37, 3)
    mask_37: (N_res, 37)
    """
    N_res: int = coords_5.shape[0]
    x_37: Float[np.ndarray, "N_res 37 3"] = np.zeros((N_res, 37, 3), dtype=np.float32)
    mask_37: Float[np.ndarray, "N_res 37"] = np.zeros((N_res, 37), dtype=np.float32)

    for atom5_slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        x_37[:, atom37_idx, :] = coords_5[:, atom5_slot, :]
        mask_37[:, atom37_idx] = mask_5[:, atom5_slot] if mask_5 is not None else 1.0

    return x_37, mask_37


# ─────────────────────────────────────────────────────────────────────────────
# 5.  File log processor
# ─────────────────────────────────────────────────────────────────────────────


class _FileLogProcessor:
    """Structlog processor that appends JSON lines to a file, then passes the event dict through."""

    def __init__(self, path: str) -> None:
        import json as _json

        self._f = open(path, "w", buffering=1)  # line-buffered
        self._json = _json

    def __call__(self, logger, method, event_dict):
        self._f.write(self._json.dumps(event_dict) + "\n")
        return event_dict


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main sampling script
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json as _json
    import traceback as _tb
    from pathlib import Path as _Path

    from sample.sample_config import SampleConfig

    parser = argparse.ArgumentParser(
        description="Sample protein structures from a trained PallAtom model"
    )
    parser.add_argument("--config", required=True, help="path to SampleConfig JSON")
    parser.add_argument("--log_file", required=True, help="path to write structured JSON log lines")
    args = parser.parse_args()

    _processors = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        _FileLogProcessor(args.log_file),
        structlog.dev.ConsoleRenderer(),
    ]
    structlog.configure(
        processors=_processors,
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    try:
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

        scfg: SampleConfig = SampleConfig.model_validate(
            _json.loads(_Path(args.config).read_text())
        )
        mp = scfg.model
        noise = scfg.noise
        sampler_p = scfg.sampler
        gen = scfg.generation
        log.info("config loaded", config=args.config, n_res=gen.n_res, n_samples=gen.n_samples)

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
        ckpt = torch.load(scfg.checkpoint.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        index_embedding = nn.Embedding(mp.max_residues, mp.c_res).to(device)
        index_embedding.load_state_dict(ckpt["index_embedding"])
        index_embedding.eval()
        model.eval()
        log.info("model loaded", checkpoint=scfg.checkpoint.checkpoint_path, device=device)

        N_RES: int = gen.n_res
        N_atom: int = N_RES * NATOM
        B_SAMPLE: int = gen.n_samples

        from helpers.featurize import Distogram as _Distogram

        _atom_disto = _Distogram(n_bins=22, min_dist=2.0, max_dist=22.0).to(device)
        _templ_disto = _Distogram(
            n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
        ).to(device)
        context: FeaturizedBatch = build_sampling_context(
            atom_positions=torch.zeros(N_RES, 37, 3, device=device),
            atom_mask=torch.ones(N_RES, 37, device=device),
            residue_index=torch.arange(N_RES, dtype=torch.float, device=device),
            seq="A" * N_RES,
            pdb_files=[],
            index_embedding=index_embedding,
            atom_distogram_fn=_atom_disto,
            templ_distogram_fn=_templ_disto,
            batch_size=B_SAMPLE,
            device=device,
        )
        edm_precond: EDMPrecond = EDMPrecond(
            model,
            context,
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

        log.info("sampling", n_res=N_RES, n_samples=B_SAMPLE, ddim_steps=sampler_p.ddim_steps)
        coords_batch: Float[torch.Tensor, "B N_atom 3"] = edm_sampler.sample(
            shape=(B_SAMPLE, N_atom, 3),
            steps=sampler_p.ddim_steps,
            device=device,
        )
        log.info("sampling complete", n_res=N_RES, n_samples=B_SAMPLE)

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
        log.info("output written", path=scfg.output.output_path, n_structures=B_SAMPLE)
    except Exception as _exc:
        log.error("fatal", error=str(_exc), traceback=_tb.format_exc())
        raise SystemExit(1) from _exc
