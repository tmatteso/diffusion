"""Featurization utilities for converting raw atom data into model inputs."""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.atom_transformers import WINDOW_SIZE, build_sparse_pairs
from beartype import beartype
from einops import rearrange, reduce, repeat
from helpers.atom_utils import (
    ATOM5_CA,
    ATOM5_ELEMENTS,
    ATOM5_NAMES,
    atom37_to_atom5,
    atom37_to_cb,
    restype_order,
    rigid_group_atom_positions,
)
from helpers.batch_types import FeaturizedBatch, FeaturizedItem, ProteinBatch
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import TrainConfig
from typing_extensions import override

_DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# template distogram: Pairwise distogram of pseudo Cβ are discretized into 38 bins of
# equal width between of bin min=3.25A, bin ˚ max=50.75A, one more ˚
# bin contains any larger distances

# template cb mask: Mask indicating if the Cβ atom has coordinates for the template at
# this residue, where 1 indicates existing residues and 0 is used for
# padding residues.


# the distogram is a N x N x bin_num matrix where each entry is a 1 if a
# residue distance is in the right num_bin bucket. the point of the distogram
# is to keep learned features geometrically grounded. otherwise we can get
# weird output coords like your VAE
class Distogram(nn.Module):
    """Pairwise distogram module.

    Precomputes bin edges once at init; forward() maps per-residue coordinates
    → one-hot distogram + validity mask.  Accepts either:

    - ``(..., N, 3)``    — single atom per residue (e.g. pseudo-Cβ),
      auto-expanded to ``(..., N, 1, 3)``
    - ``(..., N, A, 3)`` — A atoms per residue (e.g. atom5, atom37)

    Args:
        n_bins:       Number of distance bins.
        min_dist:     Lower edge of first bin in Ångströms (default 2.0).
        max_dist:     Upper edge of last bin in Ångströms (default 22.0).
        overflow_bin: If True, adds one extra bin capturing distances > max_dist,
                      making the output shape (..., n_bins + 1) instead of (..., n_bins).
    """

    def __init__(
        self,
        *,
        n_bins: int,
        overflow_bin: bool,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
    ) -> None:
        super().__init__()
        self.n_bins: int = n_bins
        self.min_dist: float = min_dist
        self.max_dist: float = max_dist
        self.overflow_bin: bool = overflow_bin

        edges: Float[torch.Tensor, "n_bins_plus_1"] = torch.linspace(min_dist, max_dist, n_bins + 1)
        self.edges: Float[torch.Tensor, "n_bins_plus_1"]
        self.register_buffer("edges", edges)

    # ------------------------------------------------------------------
    @override
    def extra_repr(self) -> str:
        """Return a human-readable summary of the binning configuration."""
        return (
            f"n_bins={self.n_bins}, "
            f"min_dist={self.min_dist}, "
            f"max_dist={self.max_dist}, "
            f"overflow_bin={self.overflow_bin}"
        )

    # ------------------------------------------------------------------
    @override
    def __call__(
        self,
        coords: Float[torch.Tensor, "... total_atom_count 3"],
        coords_mask: Bool[torch.Tensor, "... total_atom_count"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "... total_atom_count total_atom_count n_bins"],
        Bool[torch.Tensor, "... total_atom_count total_atom_count"],
    ]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(coords, coords_mask)

    # ------------------------------------------------------------------
    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        coords: Float[torch.Tensor, "... total_atom_count 3"],
        coords_mask: Bool[torch.Tensor, "... total_atom_count"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "... total_atom_count total_atom_count n_bins"],
        Bool[torch.Tensor, "... total_atom_count total_atom_count"],
    ]:
        """Forward pass maps per-residue coordinates → one-hot distogram + validity mask.

        Args:
            coords:      (..., total_atom_count, 3)
            coords_mask: (..., total_atom_count) — 1 where valid; all-ones if None.

        Returns:
            f_distogram: (..., total_atom_count, total_atom_count, n_bins [+1])
                         — one-hot bin assignment; last bin is the overflow bin
                         when overflow_bin=True.
            f_pair_mask: (..., total_atom_count, total_atom_count) bool — True where pair is valid.
                         overflow_bin=True:  valid atom pairs only.
                         overflow_bin=False: valid atom pairs AND dist <= max_dist.
        """
        # ---- 1. Pairwise distances (..., total_atom_count, total_atom_count) ------------------
        diff: Float[torch.Tensor, "... total_atom_count total_atom_count 3"] = rearrange(
            coords, "... n d -> ... n 1 d"
        ) - rearrange(coords, "... n d -> ... 1 n d")
        dist: Float[torch.Tensor, "... total_atom_count total_atom_count"] = torch.sqrt(
            reduce(diff**2, "... n m d -> ... n m", "sum").clamp(min=1e-8)
        )

        # ---- 2. Bin assignment & one-hot (n_bins[+1]) ----
        bin_idx: Int[torch.Tensor, "... total_atom_count total_atom_count"] = torch.bucketize(
            dist, self.edges[1:]
        )
        if self.overflow_bin:
            bin_idx = bin_idx.clamp(min=0)
            n_classes = self.n_bins + 1
        else:
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)
            n_classes = self.n_bins
        f_distogram = F.one_hot(bin_idx, num_classes=n_classes).float()

        # ---- 3. Pair mask (..., total_atom_count, total_atom_count) ---------------------------
        if coords_mask is not None:
            atom_valid = coords_mask.bool()
        else:
            atom_valid = torch.ones(coords.shape[:-1], dtype=torch.bool, device=coords.device)

        f_pair_mask = rearrange(atom_valid, "... n -> ... n 1") & rearrange(
            atom_valid, "... n -> ... 1 n"
        )

        if not self.overflow_bin:
            f_pair_mask = f_pair_mask & (dist <= self.max_dist)

        return f_distogram, f_pair_mask


@jaxtyped(typechecker=beartype)
def sinusoidal_encoding(
    positions: Float[torch.Tensor, "batch N_res"], dim: int = 32
) -> Float[torch.Tensor, "batch N_res dim"]:
    """Sinusoidal positional encoding. positions: (batch, N_res) → (batch, N_res, dim)."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / (half - 1))
    ).to(positions.device)
    pos = rearrange(positions.float(), "batch n_res -> batch n_res 1")
    args = pos * freqs  # (batch, N_res, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, N_res, dim)


# batch = torch.load('test_dict.pt') # this is a good test batch


@jaxtyped(typechecker=beartype)
def ref_pos_for_residue(resname: str) -> Float[torch.Tensor, "5 3"]:
    """Return reference atom positions for the 5 ATOM5 atoms of a residue.

    Args:
        resname: Three-letter residue name (e.g. "ALA", "GLY").

    Returns:
        Tensor of shape (5, 3) with XYZ coordinates for each ATOM5 atom;
        atoms absent from rigid_group_atom_positions default to (0, 0, 0).
    """
    pos_by_name = {name: pos for name, _, pos in rigid_group_atom_positions[resname]}
    return torch.tensor(
        [pos_by_name.get(name, (0.0, 0.0, 0.0)) for name in ATOM5_NAMES],
        dtype=torch.float32,
    )


@jaxtyped(typechecker=beartype)
def featurize_single_item(
    atom37_positions: Float[torch.Tensor, "N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "N_res 37"],
    index: Float[torch.Tensor, "N_res"],
    aa_sequence: str,
    ala_ref_pos: Float[torch.Tensor, "5 3"],
    ala_ref_elem: Float[torch.Tensor, "5 4"],
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
    sigma_data: float,
    P_std: float,
    P_mean: float,
    max_seq_len_in_batch: int,
    device: str = _DEFAULT_DEVICE,
) -> FeaturizedItem:
    """Featurize a single protein from atom37 representation to model-ready tensors.

    Converts atom37 coordinates to the compact atom5 representation, computes the
    Cβ pseudo-beta distogram, builds sinusoidal residue-index encodings, tiles
    the ALA reference positions and elements across all residues, and samples a
    noise level from the lognormal diffusion schedule.

    Args:
        atom37_positions: Atom positions in atom37 layout, shape ``(N_res, 37, 3)``,
            already on ``device``.
        atom37_mask: Atom validity mask in atom37 layout, shape ``(N_res, 37)``,
            already on ``device``.
        index: Per-residue index tensor used for sinusoidal encoding, shape ``(N_res,)``,
            already on ``device``.
        aa_sequence: One-letter amino-acid sequence string.
        ala_ref_pos: ALA reference atom positions, shape ``(5, 3)``, already on ``device``.
        ala_ref_elem: ALA atom element one-hots, shape ``(5, 4)``, already on ``device``.
        c_beta_distogram_fn: Callable mapping Cβ positions and residue mask to a
            ``(N_res, N_res, n_bins)`` distogram tensor.
        atom_distogram_fn: Callable mapping atom positions and atom mask to a
            ``(N_res, N_res, n_atom_bins)`` distogram tensor.
        sigma_data: Data standard deviation constant used to scale the sampled noise level.
        P_std: Standard deviation of the lognormal noise schedule (ln sigma ~ N(P_mean, P_std²)).
        P_mean: Mean of the lognormal noise schedule (ln sigma ~ N(P_mean, P_std²)).
        max_seq_len_in_batch: Padded sequence len of the batch (``batch.atom_positions.shape[1]``).
            Used to pad ``aa_indices`` to ``N_res`` with ``-100`` (CE ignore index) for residues
            that are padding rather than real amino acids.
        device: PyTorch device string for newly created tensors.

    Returns:
        A :class:`FeaturizedItem` containing flat atom positions, masks, distogram
        labels, reference geometry, sequence indices, sinusoidal residue encodings,
        and the sampled diffusion noise level.
    """
    Natom: int = 5
    N_res: int = atom37_positions.shape[0]

    # Noise schedule lognormal. ln(sigma) ~ N(Pmean, Pstd**2),
    # Pmean = -1.2, Pstd = 1.5, sigma_data = 16,
    # ~ N(0, I) * Pstd + Pmean == ~ N(Pmean, Pstd**2). the reparameterization trick.
    ln_sigma: Float[torch.Tensor, ""] = torch.randn((), device=device) * P_std + P_mean
    sigma: Float[torch.Tensor, ""] = torch.exp(ln_sigma)
    # sigma_data is a constant determined by the variance of the data (default 16)
    # t_hat is the sampled noise level. This according to AF3.
    t_hat: Float[torch.Tensor, ""] = sigma_data * sigma

    # t_normalized is drawn from uniform(0, 1) and broadcast to every residue pair.
    t_scalar: Float[torch.Tensor, ""] = torch.rand((), device=device)
    t_template: Float[torch.Tensor, "N_res N_res"] = repeat(t_scalar, "-> n m", n=N_res, m=N_res)

    aa_vals = [restype_order[r] for r in aa_sequence]
    aa_indices: Int[torch.Tensor, "N_res"] = torch.full(
        (max_seq_len_in_batch,), -100, dtype=torch.long, device=device
    )
    aa_indices[: len(aa_vals)] = torch.tensor(aa_vals, dtype=torch.long, device=device)
    atom5_pos: Float[torch.Tensor, "N_res 5 3"]
    atom5_mask: Float[torch.Tensor, "N_res 5"]
    atom5_pos, atom5_mask = atom37_to_atom5(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    atom5_pos = rearrange(atom5_pos, "1 n a d -> n a d")
    atom5_mask = rearrange(atom5_mask, "1 n a -> n a")
    residue_mask: Bool[torch.Tensor, "N_res"] = atom5_mask.any(dim=-1)
    # f_pseudo_beta: Mask indicating if the Cβ atom has coordinates for the template at
    # this residue, where 1 indicates existing residues and 0 is used for
    # padding residues. f_pseudo_beta == residue_mask

    c_beta_pos: Float[torch.Tensor, "B N_res 5 3"]
    c_beta_pos, _ = atom37_to_cb(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    c_beta_pos = rearrange(c_beta_pos, "1 n d -> n d")

    gt_res_distogram: Float[torch.Tensor, "N_atom N_atom n_template_bins"]
    gt_res_distogram, _ = c_beta_distogram_fn(c_beta_pos, residue_mask)

    flat_pos: Float[torch.Tensor, "N_atom 3"] = rearrange(atom5_pos, "n a d -> (n a) d")
    atom_mask_flat: Bool[torch.Tensor, "N_atom"] = rearrange(atom5_mask.bool(), "n a -> (n a)")

    ref_pos: Float[torch.Tensor, "N_atom 3"] = rearrange(
        repeat(ala_ref_pos, "a d -> n a d", n=N_res), "n a d -> (n a) d"
    )
    ref_elem: Float[torch.Tensor, "N_atom 4"] = rearrange(
        repeat(ala_ref_elem, "a e -> n a e", n=N_res), "n a e -> (n a) e"
    )
    # The pdb residue number for calculating relative positional embedding
    f_residue_idx: Int[torch.Tensor, "N_res"] = index.long()
    # atom to residue map
    token_idx: Int[torch.Tensor, "N_atom"] = torch.arange(N_res, device=device).repeat_interleave(
        Natom
    )
    # center is alpha carbon for standard protein residues, C1prime for nucleic acid residues
    center_single: Int[torch.Tensor, "N_atom"] = (
        torch.arange(N_res, device=device) * 5 + ATOM5_CA
    ).repeat_interleave(Natom)
    # ref_space_uid is the numerical encoding of the chain id and residue index associated with
    # this reference conformer. Each (chain id, residue index) tuple is assigned an integer
    # on first appearance.
    ref_space_uid: Int[torch.Tensor, "B N_atom"] = f_residue_idx.repeat_interleave(Natom)
    # because we are only using one chain right now, f_residue_idx == ref_space_uid

    neighbor_idx: Float[torch.Tensor, "N, K"]
    valid_mask: Float[torch.Tensor, "N, K"]  # True where the slot is a real neighbour (not padding)
    neighbor_idx, valid_mask = build_sparse_pairs(token_idx, WINDOW_SIZE)  # (N_atom, K)

    # atom_distogram_fn
    gt_atom_disto_dense: Float[torch.Tensor, "N_atom N_atom n_atom_bins"]
    gt_atom_mask_dense: Bool[torch.Tensor, "N_atom N_atom"]
    gt_atom_disto_dense, gt_atom_mask_dense = atom_distogram_fn(flat_pos, atom_mask_flat)
    n_atom_bins: int = gt_atom_disto_dense.shape[-1]

    # Vectorised sparse gather: result[l, k] = dense[l, neighbor_idx[l, k]]
    # index must be the same shape as the output
    gt_atom_distogram_sparse: Float[torch.Tensor, "N_atom K n_atom_bins"] = (
        gt_atom_disto_dense.gather(1, repeat(neighbor_idx, "n k -> n k d", d=n_atom_bins))
    )
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "N_atom K"] = (
        gt_atom_mask_dense.long().gather(1, neighbor_idx).bool() & valid_mask
    )
    del gt_atom_disto_dense, gt_atom_mask_dense

    return FeaturizedItem(
        flat_pos=flat_pos,
        atom_mask_flat=atom_mask_flat,
        f_pseudo_beta=residue_mask.long(),
        gt_res_distogram=gt_res_distogram.long(),
        aa_indices=aa_indices,
        ref_pos=ref_pos,
        ref_element=ref_elem,
        f_residue_idx=f_residue_idx,
        t_hat=t_hat,
        t_template=t_template,
        ref_space_uid=ref_space_uid,
        tok_idx=token_idx,
        center_uid=center_single,
        gt_atom_distogram_sparse=gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=gt_atom_distogram_mask_sparse,
    )


@jaxtyped(typechecker=beartype)
def featurize_batch(
    batch: ProteinBatch,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> FeaturizedBatch:
    """Convert a raw ProteinBatch into a FeaturizedBatch ready for model input.

    Draws a shared log-normal noise level sigma for the entire batch, converts atom37
    coordinates to the compact atom5 representation, computes Cβ and atom-level
    distograms, builds sinusoidal residue-index encodings, adds isotropic Gaussian
    noise to ground-truth positions, and gathers sparse atom-pair distogram labels
    over a local K-neighbour window.  All per-item tensors are padded to the same
    N_res (guaranteed by the DataLoader's collation) and stacked into a single
    batched FeaturizedBatch dataclass.

    Args:
        batch: Raw protein batch with atom37 coordinates, masks, sequences, and
            residue indices for B proteins (all pre-padded to the same length).
        tcfg: Training config supplying noise schedule parameters and model width.
        distogram_res: Residue-level Cβ distogram head.
        distogram_atom: Atom-level sparse distogram head.

    Returns:
        A FeaturizedBatch containing noisy input coordinates, ground-truth
        positions, distogram labels, atom masks, sequence indices, sinusoidal
        residue encodings, and the sampled noise level sigma.
    """
    B: int = len(batch.seq)
    device: str = str(distogram_res.edges.device)

    # ── Shared noise for the whole batch ──────────────────────────────────────
    P_std, P_mean = tcfg.noise.P_std, tcfg.noise.P_mean
    sigma_data = tcfg.noise.sigma_data

    # ── Shared helpers reused across all items ────────────────────────────────
    ala_ref_pos: Float[torch.Tensor, "5 3"] = ref_pos_for_residue("ALA").to(device)  # (5, 3)
    ala_ref_elem: Float[torch.Tensor, "5 4"] = ATOM5_ELEMENTS.float().to(device)  # (5, 4)

    # ── Per-item featurization ────────────────────────────────────────────────
    items: list[FeaturizedItem] = [
        featurize_single_item(
            atom37_positions=batch.atom_positions[ix].to(device),
            atom37_mask=batch.atom_mask[ix].to(device),
            index=batch.residue_index[ix].to(device),
            aa_sequence=batch.seq[ix],
            ala_ref_pos=ala_ref_pos,
            ala_ref_elem=ala_ref_elem,
            c_beta_distogram_fn=distogram_res,
            atom_distogram_fn=distogram_atom,
            sigma_data=sigma_data,
            P_std=P_std,
            P_mean=P_mean,
            max_seq_len_in_batch=batch.atom_positions.shape[1],
            device=device,
        )
        for ix in range(B)
    ]

    # ── Stack per-item tensors into a single batched tensor ──────────────────
    packed_flat_pos = torch.stack([it.flat_pos for it in items])  # (B, N_atom, 3)
    packed_atom_mask = torch.stack([it.atom_mask_flat for it in items])  # (B, N_atom)
    packed_pseudo_beta = torch.stack([it.f_pseudo_beta for it in items])  # (B, N_res)
    packed_aa = torch.stack([it.aa_indices for it in items])  # (B, N_res)
    packed_ref_pos = torch.stack([it.ref_pos for it in items])  # (B, N_atom, 3)
    packed_ref_elem = torch.stack([it.ref_element for it in items])  # (B, N_atom, 4)
    packed_res_idx = torch.stack([it.f_residue_idx for it in items])  # (B, N_res)
    packed_t_hat = torch.stack([it.t_hat for it in items])  # (B, )
    packed_t_temp = torch.stack([it.t_template for it in items])  # (B, N_res, N_res)
    packed_gt_res_distogram = torch.stack(
        [it.gt_res_distogram for it in items]
    )  # (B, N_res, N_res, n_templ_bins)
    packed_ref_space_uid = torch.stack([it.ref_space_uid for it in items])
    packed_center_uid = torch.stack([it.center_uid for it in items])
    packed_tok_idx = torch.stack([it.tok_idx for it in items])
    packed_gt_atom_distogram_sparse = torch.stack([it.gt_atom_distogram_sparse for it in items])
    packed_gt_atom_distogram_mask_sparse = torch.stack(
        [it.gt_atom_distogram_mask_sparse for it in items]
    )
    # step 0:
    # Apply zero-centered noise to turn r_gt into r_input
    noise = torch.randn_like(packed_flat_pos)
    noise = noise - reduce(noise, "b n d -> b 1 d", "mean")  # match sampling convention
    r_input: Float[torch.Tensor, "B N_atom 3"] = (
        packed_flat_pos + rearrange(packed_t_hat, "b -> b 1 1") * noise
    )

    return FeaturizedBatch(
        ref_pos=packed_ref_pos,
        ref_element=packed_ref_elem,
        ref_space_uid=packed_ref_space_uid,
        gt_res_distogram=packed_gt_res_distogram,
        f_pseudo_beta_mask=packed_pseudo_beta,
        f_residue_idx=packed_res_idx,
        r_gt=packed_flat_pos,
        r_gt_noised=r_input,
        atom5_mask=packed_atom_mask,
        aa_indices=packed_aa,
        t_hat=packed_t_hat,
        t_normalized=packed_t_temp,
        tok_idx=packed_tok_idx,
        center_uid=packed_center_uid,
        gt_atom_distogram_sparse=packed_gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=packed_gt_atom_distogram_mask_sparse,
    )


@jaxtyped(typechecker=beartype)
def apply_conditioning_dropout(
    batch: FeaturizedBatch,
    p_distogram: float,
    p_atom: float,
    p_seq: float,
    device: str,
) -> FeaturizedBatch:
    """Randomly zero out conditioning signals to enable classifier-free guidance.

    For each residue independently, three Bernoulli masks are sampled — one per
    conditioning modality — and the corresponding features are ablated:

    - **Distogram**: both the row and column of dropped residues are zeroed in
      the symmetric Cβ distogram matrix and the pseudo-β mask is cleared.
    - **Atom mask**: the atom5_mask for dropped residues is set to False,
      preventing the model from attending to those atom positions.
    - **Sequence**: amino-acid indices for dropped residues are replaced with
      the mask token (index 20, "X").

    Dropout is applied only to residues present in the batch (gated by
    ``f_pseudo_beta_mask``), so padding positions are never affected.

    Args:
        batch: Featurized batch whose conditioning tensors will be partially zeroed.
        p_distogram: Per-residue Bernoulli drop probability for the distogram
            and pseudo-β conditioning signals.
        p_atom: Per-residue Bernoulli drop probability for atom-level position
            conditioning (atom5_mask).
        p_seq: Per-residue Bernoulli drop probability for the sequence conditioning
            (aa_indices replaced with mask token 20).
        device: PyTorch device on which to allocate the Bernoulli sample tensors.

    Returns:
        A new FeaturizedBatch with the same fields as the input except
        ``gt_res_distogram``, ``f_pseudo_beta_mask``, ``atom5_mask``, and
        ``aa_indices`` are partially zeroed according to the sampled masks.
    """
    f_pseudo_beta: Bool[torch.Tensor, "B N_res"] = batch.f_pseudo_beta_mask.bool()
    B, N_res = f_pseudo_beta.shape

    drop_d: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_distogram, device=device)).bool() & f_pseudo_beta
    )
    drop_a: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_atom, device=device)).bool() & f_pseudo_beta
    )
    drop_s: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_seq, device=device)).bool() & f_pseudo_beta
    )

    # Distogram: zero rows AND columns for dropped residues (matrix is symmetric)
    keep_d: Bool[torch.Tensor, "B N_res"] = ~drop_d
    disto_mask: Bool[torch.Tensor, "B N_res N_res"] = rearrange(keep_d, "b i -> b i 1") & rearrange(
        keep_d, "b j -> b 1 j"
    )
    new_distogram = batch.gt_res_distogram * rearrange(disto_mask.long(), "b i j -> b i j 1")
    new_pseudo_beta_mask = batch.f_pseudo_beta_mask * keep_d.long()

    # Atom mask: expand residue-level drop to atom level (5 atoms per residue)
    drop_a_expanded: Bool[torch.Tensor, "B N_atom"] = repeat(drop_a, "b n -> b (n a)", a=5)
    new_atom5_mask: Bool[torch.Tensor, "B N_atom"] = batch.atom5_mask & ~drop_a_expanded

    # Sequence: replace dropped tokens with mask-token index 20 ("X")
    new_aa_indices: Int[torch.Tensor, "B N_res"] = batch.aa_indices.masked_fill(drop_s, 20)

    return dataclasses.replace(
        batch,
        gt_res_distogram=new_distogram,
        f_pseudo_beta_mask=new_pseudo_beta_mask,
        atom5_mask=new_atom5_mask,
        aa_indices=new_aa_indices,
    )
