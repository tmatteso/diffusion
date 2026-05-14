"""Featurization utilities for converting raw atom data into model inputs."""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.atom_transformers import WINDOW_SIZE, build_sparse_pairs
from beartype import beartype
from einops import rearrange, repeat
from helpers.atom_utils import (
    ATOM5_ELEMENTS,
    ATOM5_NAMES,
    atom37_to_atom5,
    atom37_to_cb,
    restype_order,
    rigid_group_atom_positions,
)
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import TrainConfig

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
        n_bins: int,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        overflow_bin: bool = False,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.overflow_bin = overflow_bin

        edges = torch.linspace(min_dist, max_dist, n_bins + 1)
        self.register_buffer("edges", edges)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        """Return a human-readable summary of the binning configuration."""
        return (
            f"n_bins={self.n_bins}, "
            f"min_dist={self.min_dist}, "
            f"max_dist={self.max_dist}, "
            f"overflow_bin={self.overflow_bin}"
        )

    # ------------------------------------------------------------------
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
        diff = rearrange(coords, "... n d -> ... n 1 d") - rearrange(coords, "... n d -> ... 1 n d")
        dist = diff.norm(dim=-1)

        # ---- 2. Bin assignment & one-hot (n_bins[+1]) ----
        bin_idx = torch.bucketize(dist, self.edges[1:])
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
    pos = positions.float().unsqueeze(-1)  # (batch, N_res, 1)
    args = pos * freqs  # (batch, N_res, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (batch, N_res, dim)


# batch = torch.load('test_dict.pt') # this is a good test batch


@jaxtyped(typechecker=beartype)
def _ref_pos_for_residue(resname: str) -> Float[torch.Tensor, "5 3"]:
    pos_by_name = {name: pos for name, _, pos in rigid_group_atom_positions[resname]}
    return torch.tensor(
        [pos_by_name.get(name, (0.0, 0.0, 0.0)) for name in ATOM5_NAMES],
        dtype=torch.float32,
    )


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class ProteinBatch:
    """Raw per-protein data collated into a single batch by the DataLoader."""

    atom_positions: Float[torch.Tensor, "B N_res 37 3"]
    atom_mask: Float[torch.Tensor, "B N_res 37"]
    residue_index: Float[torch.Tensor, "B N_res"]
    seq: list[str]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class FeaturizedBatch:
    """Model-ready batch produced by featurize_batch, noisy inputs and ground-truth labels."""

    ref_pos: Float[torch.Tensor, "B N_atom 3"]
    ref_element: Float[torch.Tensor, "B N_atom 4"]
    ref_space_uid: Int[torch.Tensor, "B N_atom"]
    gt_res_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"]
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"]
    f_residue_idx: Float[torch.Tensor, "B N_res c_res"]
    r_input: Float[torch.Tensor, "B N_atom 3"]
    r_gt: Float[torch.Tensor, "B N_atom 3"]
    atom5_mask: Bool[torch.Tensor, "B N_atom"]
    aa_indices: Int[torch.Tensor, "B N_res"]
    residue_mask: Bool[torch.Tensor, "B N_res"]
    t_hat: float
    t_normalized: float
    tok_idx: Int[torch.Tensor, "B N_atom"]
    center_uid: Int[torch.Tensor, "B N_res"]
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class FeaturizedItem:
    """Per-protein features produced by featurize_single_item before batching."""

    N_res: int
    flat_pos: Float[torch.Tensor, "N_atom 3"]
    atom_mask_flat: Bool[torch.Tensor, "N_atom"]
    residue_mask: Bool[torch.Tensor, "N_res"]
    f_pseudo_beta: Int[torch.Tensor, "N_res"]
    gt_res_distogram: Int[torch.Tensor, "N_res N_res n_templ_bins"]
    aa_indices: Int[torch.Tensor, "N_res"]
    ref_pos: Float[torch.Tensor, "N_atom 3"]
    ref_element: Float[torch.Tensor, "N_atom 4"]
    f_residue_idx: Float[torch.Tensor, "N_res c_res"]


@jaxtyped(typechecker=beartype)
def featurize_single_item(
    atom37_positions: Float[torch.Tensor, "N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "N_res 37"],
    index: Float[torch.Tensor, "N_res"],
    aa_sequence: str,
    ala_ref_pos: Float[torch.Tensor, "5 3"],
    ala_ref_elem: Float[torch.Tensor, "5 4"],
    c_res: int,
    c_beta_distogram_fn: Distogram,
    device: str = _DEFAULT_DEVICE,
) -> FeaturizedItem:
    """Featurize a single protein from atom37 representation to model-ready tensors.

    Converts atom37 coordinates to the compact atom5 representation, computes the
    Cβ pseudo-beta distogram, builds sinusoidal residue-index encodings, and tiles
    the ALA reference positions and elements across all residues.

    Args:
        atom37_positions: Atom positions in atom37 layout, already on ``device``.
        atom37_mask: Atom validity mask in atom37 layout, already on ``device``.
        index: Per-residue index tensor used for sinusoidal encoding, already on ``device``.
        aa_sequence: One-letter amino-acid sequence string.
        ala_ref_pos: ALA reference atom positions ``(5, 3)``, already on ``device``.
        ala_ref_elem: ALA atom element one-hots ``(5, 4)``, already on ``device``.
        c_res: Residue embedding dimension (determines sinusoidal encoding width).
        c_beta_distogram_fn: Callable mapping Cβ positions and residue mask to a
            ``(N_res, N_res, n_bins)`` distogram tensor.
        device: PyTorch device string for newly created tensors.

    Returns:
        A :class:`FeaturizedItem` containing flat atom positions, masks, distogram
        labels, reference geometry, sequence indices, and sinusoidal residue encodings.
    """
    Natom: int = 5
    N_res_i: int = atom37_positions.shape[0]

    _seq_len = min(len(aa_sequence), N_res_i)
    _aa_vals = [restype_order[r] for r in aa_sequence[:_seq_len]]
    _aa_vals += [-100] * (N_res_i - _seq_len)
    aa_indices_i: Int[torch.Tensor, "N_res"] = torch.tensor(
        _aa_vals, dtype=torch.long, device=device
    )

    atom5_pos, atom5_msk = atom37_to_atom5(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    atom5_pos = rearrange(atom5_pos, "1 n a d -> n a d")
    atom5_msk = rearrange(atom5_msk, "1 n a -> n a")
    residue_mask_i: Bool[torch.Tensor, "N_res"] = atom5_msk.any(dim=-1)

    c_beta_pos, f_pseudo_beta_i = atom37_to_cb(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    c_beta_pos = rearrange(c_beta_pos, "1 n d -> n d")
    f_pseudo_beta_i = rearrange(f_pseudo_beta_i, "1 n -> n")

    gt_res_disto_i, _ = c_beta_distogram_fn(c_beta_pos, residue_mask_i)

    flat_pos_i: Float[torch.Tensor, "N_atom 3"] = rearrange(atom5_pos, "n a d -> (n a) d")
    atom_mask_flat_i: Bool[torch.Tensor, "N_atom"] = repeat(residue_mask_i, "n -> (n a)", a=Natom)

    ref_pos_i: Float[torch.Tensor, "N_atom 3"] = rearrange(
        repeat(ala_ref_pos, "a d -> n a d", n=N_res_i), "n a d -> (n a) d"
    )
    ref_elem_i: Float[torch.Tensor, "N_atom 4"] = rearrange(
        repeat(ala_ref_elem, "a e -> n a e", n=N_res_i), "n a e -> (n a) e"
    )

    f_residue_idx_i: Float[torch.Tensor, "N_res c_res"] = sinusoidal_encoding(
        index.unsqueeze(0), dim=c_res
    ).squeeze(0)

    return FeaturizedItem(
        N_res=N_res_i,
        flat_pos=flat_pos_i,
        atom_mask_flat=atom_mask_flat_i,
        residue_mask=residue_mask_i,
        f_pseudo_beta=f_pseudo_beta_i.long(),
        gt_res_distogram=gt_res_disto_i.long(),
        aa_indices=aa_indices_i,
        ref_pos=ref_pos_i,
        ref_element=ref_elem_i,
        f_residue_idx=f_residue_idx_i,
    )


@jaxtyped(typechecker=beartype)
def featurize_batch(
    batch: ProteinBatch,
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
    device: str = _DEFAULT_DEVICE,
) -> FeaturizedBatch:
    """Convert a raw ProteinBatch into a FeaturizedBatch ready for model input.

    Draws a shared log-normal noise level σ for the entire batch, converts atom37
    coordinates to the compact atom5 representation, computes Cβ and atom-level
    distograms, builds sinusoidal residue-index encodings, adds isotropic Gaussian
    noise to ground-truth positions, and gathers sparse atom-pair distogram labels
    over a local K-neighbour window.  All per-item tensors are padded to the same
    N_res (guaranteed by the DataLoader's collation) and stacked into a single
    batched FeaturizedBatch dataclass.

    Args:
        batch: Raw protein batch with atom37 coordinates, masks, sequences, and
            residue indices for B proteins (all pre-padded to the same length).
        tcfg: Training configuration supplying noise schedule parameters
            (sigma_min, sigma_max, P_std, P_mean) and model width (c_res).
        c_beta_distogram_fn: Callable that maps Cβ positions and a residue mask
            to a (N_res, N_res, n_bins) distogram tensor.
        atom_distogram_fn: Callable that maps batched atom positions and masks to a
            (B, N_atom, N_atom, n_atom_bins) atom-level distogram tensor.
        device: PyTorch device string; defaults to CUDA when available.

    Returns:
        A FeaturizedBatch containing noisy input coordinates, ground-truth
        positions, distogram labels, atom masks, sequence indices, sinusoidal
        residue encodings, and the sampled noise level σ.
    """
    B: int = len(batch.seq)
    Natom: int = 5
    c_res: int = tcfg.model.c_res

    # ── Shared noise for the whole batch ──────────────────────────────────────
    sigma_min, sigma_max = tcfg.noise.sigma_min, tcfg.noise.sigma_max
    P_std, P_mean = tcfg.noise.P_std, tcfg.noise.P_mean
    ln_sigma: Float[torch.Tensor, 1] = torch.randn(1, device=device) * P_std + P_mean
    sigma: Float[torch.Tensor, 1] = torch.exp(ln_sigma)
    t_hat: float = sigma.item()
    t_normalized: float = (math.log(t_hat) - math.log(sigma_min)) / (
        math.log(sigma_max) - math.log(sigma_min)
    )

    # ── Shared helpers reused across all items ────────────────────────────────
    ala_ref_pos = _ref_pos_for_residue("ALA").to(device)  # (5, 3)
    ala_ref_elem = ATOM5_ELEMENTS.float().to(device)  # (5, 4)

    # ── Per-item featurization ────────────────────────────────────────────────
    items: list[FeaturizedItem] = [
        featurize_single_item(
            atom37_positions=batch.atom_positions[ix].to(device),
            atom37_mask=batch.atom_mask[ix].to(device),
            index=batch.residue_index[ix].to(device),
            aa_sequence=batch.seq[ix],
            ala_ref_pos=ala_ref_pos,
            ala_ref_elem=ala_ref_elem,
            c_res=c_res,
            c_beta_distogram_fn=c_beta_distogram_fn,
            device=device,
        )
        for ix in range(B)
    ]

    # All items must have the same N_res (ProteinBatch is pre-padded by the DataLoader).
    N_res_total: int = items[0].N_res
    N_atom_total: int = N_res_total * Natom

    # ── Stack per-item tensors into a single batched tensor ──────────────────
    packed_flat_pos = torch.stack([it.flat_pos for it in items])  # (B, N_atom, 3)
    packed_atom_mask = torch.stack([it.atom_mask_flat for it in items])  # (B, N_atom)
    packed_res_mask = torch.stack([it.residue_mask for it in items])  # (B, N_res)
    packed_pseudo_beta = torch.stack([it.f_pseudo_beta for it in items])  # (B, N_res)
    packed_aa = torch.stack([it.aa_indices for it in items])  # (B, N_res)
    packed_ref_pos = torch.stack([it.ref_pos for it in items])  # (B, N_atom, 3)
    packed_ref_elem = torch.stack([it.ref_element for it in items])  # (B, N_atom, 4)
    packed_res_idx = torch.stack([it.f_residue_idx for it in items])  # (B, N_res, c_res)
    gt_res_distogram = torch.stack(
        [it.gt_res_distogram for it in items]
    )  # (B, N_res, N_res, n_templ_bins)

    # Each item is a single chain; ref_space_uid = 0 everywhere (no separators needed).
    ref_space_uid: Int[torch.Tensor, "B N_atom"] = torch.zeros(
        B, N_atom_total, dtype=torch.long, device=device
    )

    # tok_idx and center_uid are identical for all items (same N_res after padding).
    _tok_single: Int[torch.Tensor, "N_atom"] = torch.arange(
        N_res_total, dtype=torch.long, device=device
    ).repeat_interleave(Natom)
    _center_single: Int[torch.Tensor, "N_res"] = (
        torch.arange(N_res_total, dtype=torch.long, device=device) * Natom + 1
    )  # should always be the alpha carbon
    tok_idx: Int[torch.Tensor, "B N_atom"] = _tok_single.unsqueeze(0).expand(B, -1).contiguous()
    center_uid: Int[torch.Tensor, "B N_res"] = (
        _center_single.unsqueeze(0).expand(B, -1).contiguous()
    )

    # ── Noisy atom positions ──────────────────────────────────────────────────
    epsilon: Float[torch.Tensor, "B N_atom 3"] = torch.randn_like(packed_flat_pos)
    r_input: Float[torch.Tensor, "B N_atom 3"] = packed_flat_pos + sigma * epsilon

    # ── Sparse atom distogram (batched) ──────────────────────────────────────
    neighbor_idx, _ = build_sparse_pairs(_tok_single, WINDOW_SIZE)  # (N_atom, K)

    # atom_distogram_fn supports batched input: (B, N_atom, 3) → (B, N_atom, N_atom, n_bins)
    gt_atom_disto_dense, gt_atom_mask_dense = atom_distogram_fn(packed_flat_pos, packed_atom_mask)
    n_atom_bins: int = gt_atom_disto_dense.shape[-1]

    # Vectorised sparse gather: result[b, l, k] = dense[b, l, neighbor_idx[l, k]]
    nbr_b: Int[torch.Tensor, "B N_atom K"] = repeat(neighbor_idx, "n k -> b n k", b=B)
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"] = (
        gt_atom_disto_dense.gather(2, repeat(nbr_b, "b n k -> b n k d", d=n_atom_bins))
    )
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"] = (
        gt_atom_mask_dense.long().gather(2, nbr_b).bool()
    )
    del gt_atom_disto_dense, gt_atom_mask_dense

    return FeaturizedBatch(
        ref_pos=packed_ref_pos,
        ref_element=packed_ref_elem,
        ref_space_uid=ref_space_uid,
        gt_res_distogram=gt_res_distogram,
        f_pseudo_beta_mask=packed_pseudo_beta,
        f_residue_idx=packed_res_idx,
        r_input=r_input,
        r_gt=packed_flat_pos,
        atom5_mask=packed_atom_mask,
        aa_indices=packed_aa,
        residue_mask=packed_res_mask,
        t_hat=t_hat,
        t_normalized=t_normalized,
        tok_idx=tok_idx,
        center_uid=center_uid,
        gt_atom_distogram_sparse=gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=gt_atom_distogram_mask_sparse,
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
    ``residue_mask``), so padding positions are never affected.

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
    residue_mask: Bool[torch.Tensor, "B N_res"] = batch.residue_mask
    B, N_res = residue_mask.shape

    drop_d: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_distogram, device=device)).bool() & residue_mask
    )
    drop_a: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_atom, device=device)).bool() & residue_mask
    )
    drop_s: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_seq, device=device)).bool() & residue_mask
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
