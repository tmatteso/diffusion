"""Dataset and data loading utilities for protein structure files.

Contains ProteinDataset and ClusteredProteinDataset for lazy-loading protein
structures from JSONL files, collate helpers for variable-length batching, and
a factory function that assembles bucketed train/val/test DataLoaders with
optional DDP support.
"""

# ruff: noqa: ERA001

import dataclasses
import hashlib
import io
import math
import multiprocessing as mp
import queue
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, cast, override

import numpy as np
import numpy.typing as npt
import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import webdataset as wds
from architecture.atom_transformers import build_sparse_pairs
from beartype import beartype
from einops import rearrange, reduce, repeat
from helpers.alignment import centre_random_augment, masked_com
from helpers.atom_utils import (
    ATOM5_CA,
    ATOM5_ELEMENTS,
    ATOM5_NAMES,
    Protein,
    atom5_to_atom37,
    atom37_to_atom5,
    atom37_to_cb,
    center_positions,
    make_fixed_size,
    make_np_example,
    restype_order,
    rigid_group_atom_positions,
    truncate_to_length,
)
from helpers.context_managers import (
    FatalOnError,
    ShardWorkerNotInitializedError,
    ShardWorkerState,
)
from jaxtyping import Bool, Float, Int, jaxtyped
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
)
from structlog.typing import FilteringBoundLogger
from torch.utils.data.distributed import DistributedSampler
from train.train_config import TrainArgs, TrainConfig
from webdataset.writer import TarWriter


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class FeaturizedItem:
    """Per-protein features produced by featurize_single_item before batching.

    Stores all tensors for a single protein after featurization but before the
    DataLoader collates them into a batch.  These tensors are later stacked
    into a ``FeaturizedBatch`` by ``featurize_batch``.

    Attributes:
        r_gt: Ground-truth atom positions in flat (atom-indexed) layout.
        r_gt_noised: Noised atom positions in flat (atom-indexed) layout.
        atom5_mask: Boolean mask indicating which of the 5 backbone+Cβ atoms
            are present for each residue.
        f_pseudo_beta_mask: Boolean mask indicating residues that have a valid
            pseudo-β carbon, used to build the template Cβ distogram.
        gt_res_distogram_indices: Integer bin index per residue pair from
            ground-truth (unnoised) coordinates, used only as the distogram
            loss target.
        noised_res_distogram: Float one-hot distance distribution from noised
            coordinates, used as the self-conditioning template fed to the
            model, binned into ``n_templ_bins`` distance categories.
        aa_indices: Integer amino-acid class index for each residue (vocabulary
            size 20).
        ref_pos: Reference atom positions drawn from the ground-truth
            structure, used to initialise atom-pair features in the model.
        ref_element: One-hot element identity per atom (C / N / O / UNK),
            encoded as a float vector of length 4.
        f_residue_idx: Integer residue index for each residue, projected to a
            sinusoidal embedding inside the model.
        t_hat: Scalar noise level sigma sampled for this item during training.
        t_normalized: Pairwise normalised time/template weights over residue
            pairs, used for time-conditional template weighting.
        ref_space_uid: Chain / space identifier per atom, used to determine
            covalent bonding in relative position encoding.
        tok_idx: Maps each atom to its parent residue index in ``[0, N_res)``.
        center_uid: For each atom, the index of its residue's designated center
            atom; used to extract per-residue center positions.
        gt_atom_distogram_sparse: Ground-truth pairwise atom distance
            distribution over the sparse local ``K``-neighbour window, binned
            into ``n_atom_bins`` distance categories.
        gt_atom_distogram_mask_sparse: Boolean mask indicating valid neighbour
            entries in ``gt_atom_distogram_sparse``.
    """

    r_gt: Float[torch.Tensor, "N_atom 3"]
    r_gt_noised: Float[torch.Tensor, "N_atom 3"]
    atom5_mask: Bool[torch.Tensor, "N_atom"]
    f_pseudo_beta_mask: Int[torch.Tensor, "N_res"]
    gt_res_distogram_indices: Int[torch.Tensor, "N_res N_res"]
    noised_res_distogram: Float[torch.Tensor, "N_res N_res n_templ_bins"]
    aa_indices: Int[torch.Tensor, "N_res"]
    ref_pos: Float[torch.Tensor, "N_atom 3"]
    ref_element: Float[torch.Tensor, "N_atom 4"]
    f_residue_idx: Int[torch.Tensor, "N_res"]
    t_hat: Float[torch.Tensor, ""]
    t_normalized: Float[torch.Tensor, "N_res N_res"]
    ref_space_uid: Int[torch.Tensor, "N_atom"]
    tok_idx: Int[torch.Tensor, "N_atom"]
    center_uid: Int[torch.Tensor, "N_atom"]
    gt_atom_distogram_sparse: Int[torch.Tensor, "N_atom K"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "N_atom K"]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class FeaturizedBatch:
    """Model-ready batch produced by featurize_batch or build_sampling_context.

    Collates the outputs of ``featurize_single_item`` for a full mini-batch and
    adds a leading batch dimension ``B`` to every tensor.  Fields include both
    the noisy diffusion input (``r_gt_noised``) and the clean ground-truth
    target (``r_gt``) so that a single forward pass can compute the denoising
    loss.

    Attributes:
        ref_pos: Reference atom positions from the ground-truth structure, with
            batch dimension prepended.
        ref_element: One-hot element identity per atom (C / N / O / UNK) with
            batch dimension prepended.
        ref_space_uid: Chain / space identifier per atom with batch dimension
            prepended.
        gt_res_distogram_indices: Integer bin index per residue pair from
            ground-truth (unnoised) coordinates, used only as the distogram
            loss target, with batch dimension prepended.
        noised_res_distogram: Float one-hot distance distribution from noised
            coordinates, used as the self-conditioning template fed to the
            model, with batch dimension prepended.
        f_pseudo_beta_mask: Binary mask indicating residues that have a valid
            pseudo-β carbon in the template.
        f_residue_idx: Per-residue index for sinusoidal positional encoding,
            with batch dimension prepended.
        r_gt: Clean ground-truth atom positions (Å), with batch dimension
            prepended.
        r_gt_noised: Noisy version of ``r_gt`` at noise level ``t_hat``, used
            as the diffusion model input.
        atom5_mask: Boolean mask for the five canonical backbone atoms per
            residue after flattening.
        aa_indices: Integer amino-acid class indices per residue (vocabulary
            size 20) with batch dimension prepended.
        t_hat: Per-item noise level sigma, one scalar per batch element.
        t_normalized: Pairwise normalised time/template weights over residue
            pairs with batch dimension prepended.
        tok_idx: Maps each atom to its parent residue index ``[0, N_res)``
            with batch dimension prepended.
        center_uid: Per-atom index of the residue's designated center atom
            with batch dimension prepended.
        gt_atom_distogram_sparse: Ground-truth pairwise atom distance
            distribution over the sparse local ``K``-neighbour window with
            batch dimension prepended.
        gt_atom_distogram_mask_sparse: Boolean mask for valid entries in
            ``gt_atom_distogram_sparse`` with batch dimension prepended.
    """

    ref_pos: Float[torch.Tensor, "B N_atom 3"]
    ref_element: Float[torch.Tensor, "B N_atom 4"]
    ref_space_uid: Int[torch.Tensor, "B N_atom"]
    gt_res_distogram_indices: Int[torch.Tensor, "B N_res N_res"]
    noised_res_distogram: Float[torch.Tensor, "B N_res N_res n_templ_bins"]
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"]
    f_residue_idx: Int[torch.Tensor, "B N_res"]
    r_gt: Float[torch.Tensor, "B N_atom 3"]
    r_gt_noised: Float[torch.Tensor, "B N_atom 3"]
    atom5_mask: Bool[torch.Tensor, "B N_atom"]
    aa_indices: Int[torch.Tensor, "B N_res"]
    t_hat: Float[torch.Tensor, "B"]
    t_normalized: Float[torch.Tensor, "B N_res N_res"]
    tok_idx: Int[torch.Tensor, "B N_atom"]
    center_uid: Int[torch.Tensor, "B N_atom"]
    gt_atom_distogram_sparse: Int[torch.Tensor, "B N_atom K"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"]

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool,
    ) -> "FeaturizedBatch":
        """Return a new FeaturizedBatch with all tensors transferred to device.

        Args:
            device: Target device string or torch.device.
            non_blocking: If True, the transfer is asynchronous with respect to
                the host. Requires pinned source memory for the overlap to be
                effective.

        Returns:
            A new FeaturizedBatch whose tensors reside on device.
        """
        return dataclasses.replace(
            self,
            ref_pos=self.ref_pos.to(device, non_blocking=non_blocking),
            ref_element=self.ref_element.to(device, non_blocking=non_blocking),
            ref_space_uid=self.ref_space_uid.to(
                device,
                non_blocking=non_blocking,
            ),
            gt_res_distogram_indices=self.gt_res_distogram_indices.to(
                device,
                non_blocking=non_blocking,
            ),
            noised_res_distogram=self.noised_res_distogram.to(
                device,
                non_blocking=non_blocking,
            ),
            f_pseudo_beta_mask=self.f_pseudo_beta_mask.to(
                device,
                non_blocking=non_blocking,
            ),
            f_residue_idx=self.f_residue_idx.to(
                device,
                non_blocking=non_blocking,
            ),
            r_gt=self.r_gt.to(device, non_blocking=non_blocking),
            r_gt_noised=self.r_gt_noised.to(device, non_blocking=non_blocking),
            atom5_mask=self.atom5_mask.to(device, non_blocking=non_blocking),
            aa_indices=self.aa_indices.to(device, non_blocking=non_blocking),
            t_hat=self.t_hat.to(device, non_blocking=non_blocking),
            t_normalized=self.t_normalized.to(
                device,
                non_blocking=non_blocking,
            ),
            tok_idx=self.tok_idx.to(device, non_blocking=non_blocking),
            center_uid=self.center_uid.to(device, non_blocking=non_blocking),
            gt_atom_distogram_sparse=self.gt_atom_distogram_sparse.to(
                device,
                non_blocking=non_blocking,
            ),
            gt_atom_distogram_mask_sparse=(
                self.gt_atom_distogram_mask_sparse.to(
                    device,
                    non_blocking=non_blocking,
                )
            ),
        )

    def pin_memory(self) -> "FeaturizedBatch":
        """Return a new FeaturizedBatch with all tensors in pinned memory.

        Called automatically by the DataLoader when ``pin_memory=True``.
        Pinned (page-locked) memory enables truly asynchronous CPU→GPU
        transfers when ``non_blocking=True`` is passed to ``.to()``.

        Returns:
            A new FeaturizedBatch whose tensors are in page-locked memory.
        """
        return dataclasses.replace(
            self,
            ref_pos=self.ref_pos.pin_memory(),
            ref_element=self.ref_element.pin_memory(),
            ref_space_uid=self.ref_space_uid.pin_memory(),
            gt_res_distogram_indices=self.gt_res_distogram_indices.pin_memory(),
            noised_res_distogram=self.noised_res_distogram.pin_memory(),
            f_pseudo_beta_mask=self.f_pseudo_beta_mask.pin_memory(),
            f_residue_idx=self.f_residue_idx.pin_memory(),
            r_gt=self.r_gt.pin_memory(),
            r_gt_noised=self.r_gt_noised.pin_memory(),
            atom5_mask=self.atom5_mask.pin_memory(),
            aa_indices=self.aa_indices.pin_memory(),
            t_hat=self.t_hat.pin_memory(),
            t_normalized=self.t_normalized.pin_memory(),
            tok_idx=self.tok_idx.pin_memory(),
            center_uid=self.center_uid.pin_memory(),
            gt_atom_distogram_sparse=self.gt_atom_distogram_sparse.pin_memory(),
            gt_atom_distogram_mask_sparse=(
                self.gt_atom_distogram_mask_sparse.pin_memory()
            ),
        )


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
        n_bins: Number of distance bins.
        overflow_bin: If True, adds one extra bin capturing distances >
            max_dist, making the output shape (..., n_bins + 1) instead of
            (..., n_bins).
        min_dist: Lower edge of first bin in Ångströms (default 2.0).
        max_dist: Upper edge of last bin in Ångströms (default 22.0).
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

        edges: Float[torch.Tensor, "n_bins_plus_1"] = torch.linspace(
            min_dist,
            max_dist,
            n_bins + 1,
        )
        self.edges: Float[torch.Tensor, "n_bins_plus_1"]
        self.register_buffer("edges", edges)

    # ------------------------------------------------------------------
    @override
    def extra_repr(self) -> str:
        """Return a human-readable summary of the binning configuration.

        Returns:
            Comma-separated string listing n_bins, min_dist, max_dist, and
            overflow_bin for display in ``repr(module)``.
        """
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
        """Call forward; typed override so call-site return types are not Any.

        Args:
            coords: Per-atom coordinates of shape
                ``(..., total_atom_count, 3)``.
            coords_mask: Boolean validity mask of shape
                ``(..., total_atom_count)``; all atoms are treated as valid
                when ``None``.

        Returns:
            A tuple of (distogram one-hot tensor, pair validity mask boolean
            tensor) with the same leading batch dimensions as ``coords``.
        """
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
        """Maps per-residue coordinates to distogram + validity mask.

        Args:
            coords:      (..., total_atom_count, 3)
            coords_mask: (..., total_atom_count) — 1 where valid; all-ones if
                None.

        Returns:
            f_distogram: (..., total_atom_count, total_atom_count, n_bins [+1])
                         — one-hot bin assignment; last bin is the overflow bin
                         when overflow_bin=True.
            f_pair_mask: (..., total_atom_count, total_atom_count) bool — True
                where pair is valid.
                         overflow_bin=True:  valid atom pairs only.
                         overflow_bin=False: valid atom pairs AND dist <=
                         max_dist.
        """
        # ---- 1. Pairwise distances (..., total_atom_count, total_atom_count)
        diff: Float[torch.Tensor, "... total_atom_count total_atom_count 3"] = (
            rearrange(coords, "... n d -> ... n 1 d")
            - rearrange(coords, "... n d -> ... 1 n d")
        )
        pairwise_dist_mat: Float[
            torch.Tensor,
            "... total_atom_count total_atom_count",
        ] = torch.sqrt(
            reduce(diff**2, "... n m d -> ... n m", "sum").clamp(min=1e-8),
        )

        # ---- 2. Bin assignment & one-hot (n_bins[+1]) ----
        bin_idx: Int[torch.Tensor, "... total_atom_count total_atom_count"] = (
            torch.bucketize(pairwise_dist_mat, self.edges[1:])
        )
        if self.overflow_bin:
            bin_idx = bin_idx.clamp(min=0)
            n_classes = self.n_bins + 1
        else:
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)
            n_classes = self.n_bins
        f_distogram = F.one_hot(bin_idx, num_classes=n_classes).float()

        # ---- 3. Pair mask (..., total_atom_count, total_atom_count) --------
        atom_valid = (
            coords_mask.bool()
            if coords_mask is not None
            else torch.ones(
                coords.shape[:-1],
                dtype=torch.bool,
                device=coords.device,
            )
        )

        f_pair_mask = rearrange(atom_valid, "... n -> ... n 1") & rearrange(
            atom_valid,
            "... n -> ... 1 n",
        )

        if not self.overflow_bin:
            f_pair_mask = f_pair_mask & (pairwise_dist_mat <= self.max_dist)

        return f_distogram, f_pair_mask


@jaxtyped(typechecker=beartype)
def sinusoidal_encoding(
    positions: Float[torch.Tensor, "batch N_res"],
    dim: int = 32,
) -> Float[torch.Tensor, "batch N_res dim"]:
    """Sinusoidal positional encoding mapping residue indices to embeddings.

    Args:
        positions: Per-residue scalar indices of shape ``(batch, N_res)``.
        dim: Output embedding dimension; must be even (split equally between
            sin and cos components, default 32).

    Returns:
        Embedding tensor of shape ``(batch, N_res, dim)`` where the first
        ``dim // 2`` channels are sine projections and the last ``dim // 2``
        are cosine projections across log-spaced frequencies.
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1)),
    ).to(positions.device)
    pos = rearrange(positions.float(), "batch n_res -> batch n_res 1")
    args = pos * freqs  # (batch, N_res, half)
    return torch.cat(
        [torch.sin(args), torch.cos(args)],
        dim=-1,
    )  # (batch, N_res, dim)


@jaxtyped(typechecker=beartype)
def ref_pos_for_residue(resname: str) -> Float[torch.Tensor, "5 3"]:
    """Return reference atom positions for the 5 ATOM5 atoms of a residue.

    Args:
        resname: Three-letter residue name (e.g. "ALA", "GLY").

    Returns:
        Tensor of shape (5, 3) with XYZ coordinates for each ATOM5 atom;
        atoms absent from rigid_group_atom_positions default to (0, 0, 0).
    """
    pos_by_name = {
        name: pos for name, _, pos in rigid_group_atom_positions[resname]
    }
    return torch.tensor(
        [pos_by_name.get(name, (0.0, 0.0, 0.0)) for name in ATOM5_NAMES],
        dtype=torch.float32,
    )


@jaxtyped(typechecker=beartype)
def featurize_single_item(
    prot: Protein,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
    tcfg: TrainConfig,
    max_seq_len_in_batch: int,
) -> FeaturizedItem:
    """Featurize one protein into model-ready tensors.

    Converts atom37 coordinates to atom5, pads to max_seq_len_in_batch,
    computes ground-truth and self-conditioning Cβ distograms, builds the
    sparse atom-pair distogram over the local residue window, tiles ALA
    reference geometry across residues, and samples a diffusion noise level
    from the lognormal schedule.

    Args:
        prot: Protein with atom37 coordinates, mask, sequence,
            and residue indices.
        c_beta_distogram_fn: Residue-level Cβ distogram head used for both
            the GT distogram loss target and the noised self-conditioning
            template.
        atom_distogram_fn: Atom-level sparse distogram head used to bucket
            ground-truth pairwise atom distances within the neighbour window.
        tcfg: Training configuration; provides the lognormal noise schedule
            parameters (``tcfg.noise``) and the residue-level neighbour
            window size (``tcfg.model.window_size``).
        max_seq_len_in_batch: Padded sequence length; all tensors are
            zero-padded to this length so items in a batch are uniform.

    Returns:
        FeaturizedItem holding flat atom positions (ground-truth and noised),
        atom and residue masks, GT and noised Cβ distogram targets, sparse
        atom-pair distogram targets, ALA reference geometry, sequence indices,
        the sampled noise level ``t_hat``, and the uniform template weight
        ``t_normalized``.
    """
    noise_params = tcfg.noise
    window_size = tcfg.model.window_size
    # you need to pad the pos, res_idx, and mask now
    # do this first, then fix featurize batch
    atom37_positions: Float[torch.Tensor, "N_res 37 3"] = torch.tensor(
        prot.atom_positions,
        dtype=torch.float32,
    )
    atom37_mask: Int[torch.Tensor, "N_res 37"] = torch.tensor(
        prot.atom_mask,
    )
    # The pdb residue number for calculating relative positional embedding
    f_residue_idx: Int[torch.Tensor, "N_res"] = torch.tensor(prot.residue_index)
    Natom: int = 5
    unpadded_N_res: int = atom37_positions.shape[0]

    pad = max_seq_len_in_batch - unpadded_N_res
    if pad > 0:
        atom37_positions = F.pad(atom37_positions, (0, 0, 0, 0, 0, pad))
        atom37_mask = F.pad(atom37_mask, (0, 0, 0, pad))
        f_residue_idx = F.pad(f_residue_idx, (0, pad))
    N_res: int = max_seq_len_in_batch

    aa_indices: Int[torch.Tensor, "N_res"] = torch.full(
        (max_seq_len_in_batch,),
        -100,
        dtype=torch.long,
    )
    aa_indices[:unpadded_N_res] = torch.tensor(prot.aatype, dtype=torch.long)

    ala_ref_pos: Float[torch.Tensor, "5 3"] = ref_pos_for_residue("ALA")
    ala_ref_elem: Float[torch.Tensor, "5 4"] = ATOM5_ELEMENTS.float()
    # Noise schedule lognormal. ln(sigma) ~ N(Pmean, Pstd**2),
    # Pmean = -1.2, Pstd = 1.5, sigma_data = 16,
    # ~ N(0, I) * Pstd + Pmean == ~ N(Pmean, Pstd**2).
    # The reparameterization trick. t_hat is the sampled noise level per AF3.
    t_hat: Float[torch.Tensor, ""] = noise_params.sigma_data * torch.exp(
        torch.randn(()) * noise_params.P_std + noise_params.P_mean,
    )

    # t_normalized is drawn from uniform(0, 1) and broadcast to every
    # residue pair.
    t_template: Float[torch.Tensor, "N_res N_res"] = repeat(
        torch.rand(()),
        "-> n m",
        n=N_res,
        m=N_res,
    )

    atom5_pos, atom5_mask = atom37_to_atom5(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    atom5_pos = rearrange(atom5_pos, "1 n a d -> n a d")
    atom5_mask = rearrange(atom5_mask, "1 n a -> n a")
    residue_mask: Bool[torch.Tensor, "N_res"] = atom5_mask.any(dim=-1)
    # f_pseudo_beta: Mask indicating if the Cβ atom has coordinates for the
    # template at this residue, where 1 indicates existing residues and 0 is
    # used for padding residues. f_pseudo_beta == residue_mask

    atom_mask_flat: Bool[torch.Tensor, "N_atom"] = rearrange(
        atom5_mask.bool(),
        "n a -> (n a)",
    )

    # Random rigid augmentation (AF3 Algorithm 20): centre + Haar-uniform
    # SO(3) rotation and translation, applied before noising so r_gt and
    # r_gt_noised share the same augmented frame. The mask keeps padded
    # atom slots from biasing the centroid toward the origin.
    flat_pos: Float[torch.Tensor, "N_atom 3"] = rearrange(
        centre_random_augment(
            rearrange(atom5_pos, "n a d -> 1 (n a) d"),
            mask=rearrange(atom_mask_flat, "n -> 1 n"),
        ),
        "1 n d -> n d",
    )

    # GT Cβ from unnoised atom37_positions → loss target. Deliberately NOT
    # taken from the post-augmentation flat_pos: centre_random_augment is a
    # rigid transform (rotation + translation, det forced to +1), so pairwise
    # distances — and therefore this distogram — are identical either way.
    # Don't "fix" this into recomputing from the augmented frame; it's
    # unnecessary work for a numerically identical result.
    c_beta_pos_clean, _ = atom37_to_cb(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    c_beta_pos_clean = rearrange(c_beta_pos_clean, "1 n d -> n d")
    gt_res_disto_onehot, _ = c_beta_distogram_fn(c_beta_pos_clean, residue_mask)
    gt_res_distogram_indices: Int[torch.Tensor, "N_res N_res"] = (
        gt_res_disto_onehot.argmax(dim=-1)
    )
    noise = torch.randn_like(flat_pos)  # (N_atom, 3)
    # zero CoM over valid atoms only, so padded atom slots don't bias it
    r_gt_noised = flat_pos + t_hat * (
        noise
        - rearrange(
            masked_com(
                rearrange(noise, "n d -> 1 n d"),
                mask=rearrange(atom_mask_flat, "n -> 1 n"),
            ),
            "1 1 d -> 1 d",
        )
    )

    # Noised Cβ → self-conditioning template
    atom37_noised, _ = atom5_to_atom37(
        rearrange(r_gt_noised, "(n a) d -> 1 n a d", a=Natom),  # add B=1
        rearrange(atom5_mask, "n a -> 1 n a"),  # add B=1
    )
    atom37_noised = rearrange(atom37_noised, "1 n a d -> n a d")  # strip B

    c_beta_pos_noised, _ = atom37_to_cb(
        rearrange(atom37_noised, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    c_beta_pos_noised = rearrange(c_beta_pos_noised, "1 n d -> n d")
    noised_res_distogram, _ = c_beta_distogram_fn(
        c_beta_pos_noised,
        residue_mask,
    )

    ref_pos: Float[torch.Tensor, "N_atom 3"] = rearrange(
        repeat(ala_ref_pos, "a d -> n a d", n=N_res),
        "n a d -> (n a) d",
    )
    ref_elem: Float[torch.Tensor, "N_atom 4"] = rearrange(
        repeat(ala_ref_elem, "a e -> n a e", n=N_res),
        "n a e -> (n a) e",
    )
    # atom to residue map
    token_idx: Int[torch.Tensor, "N_atom"] = torch.arange(
        N_res,
    ).repeat_interleave(Natom)
    # center is alpha carbon for standard protein residues,
    # C1prime for nucleic acid residues.
    center_single: Int[torch.Tensor, "N_atom"] = (
        torch.arange(N_res) * 5 + ATOM5_CA
    ).repeat_interleave(Natom)
    # ref_space_uid is the numerical encoding of the chain id and residue
    # index associated with this reference conformer. Each (chain id, residue
    # index) tuple is assigned an integer on first appearance.
    ref_space_uid: Int[torch.Tensor, "N_atom"] = (
        f_residue_idx.repeat_interleave(Natom)
    )
    # because we are only using one chain right now,
    # f_residue_idx == ref_space_uid.

    neighbor_idx, valid_mask = build_sparse_pairs(
        token_idx,
        window_size,
    )  # (N_atom, K)

    # Sparse atom distogram: compute distances only for the K known neighbours,
    # avoiding the O(N²) dense intermediate the full Distogram.forward would
    # allocate. build_sparse_pairs already identified the neighbour indices, so
    # we gather their positions and bucketize directly into bin indices.
    neighbor_pos: Float[torch.Tensor, "N_atom K 3"] = flat_pos[neighbor_idx]
    diff_sparse: Float[torch.Tensor, "N_atom K 3"] = (
        rearrange(flat_pos, "n d -> n 1 d") - neighbor_pos
    )
    sparse_dist: Float[torch.Tensor, "N_atom K"] = torch.sqrt(
        reduce(diff_sparse**2, "n k d -> n k", "sum").clamp(min=1e-8),
    )
    gt_atom_distogram_sparse: Int[torch.Tensor, "N_atom K"] = torch.bucketize(
        sparse_dist,
        atom_distogram_fn.edges[1:],
    ).clamp(
        0,
        (
            atom_distogram_fn.n_bins + 1
            if atom_distogram_fn.overflow_bin
            else atom_distogram_fn.n_bins
        )
        - 1,
    )
    neighbor_valid: Bool[torch.Tensor, "N_atom K"] = atom_mask_flat[
        neighbor_idx
    ]
    within_range: Bool[torch.Tensor, "N_atom K"] = (
        (sparse_dist <= atom_distogram_fn.max_dist)
        if not atom_distogram_fn.overflow_bin
        else torch.ones_like(valid_mask)
    )
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "N_atom K"] = (
        rearrange(atom_mask_flat, "n -> n 1")
        & neighbor_valid
        & within_range
        & valid_mask
    )

    return FeaturizedItem(
        r_gt=flat_pos,
        r_gt_noised=r_gt_noised,
        atom5_mask=atom_mask_flat,
        f_pseudo_beta_mask=residue_mask.long(),
        gt_res_distogram_indices=gt_res_distogram_indices,
        noised_res_distogram=noised_res_distogram,
        aa_indices=aa_indices,
        ref_pos=ref_pos,
        ref_element=ref_elem,
        f_residue_idx=f_residue_idx,
        t_hat=t_hat,
        t_normalized=t_template,
        ref_space_uid=ref_space_uid,
        tok_idx=token_idx,
        center_uid=center_single,
        gt_atom_distogram_sparse=gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=gt_atom_distogram_mask_sparse,
    )


@jaxtyped(typechecker=beartype)
def featurize_batch(
    batch: list[Protein],
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> FeaturizedBatch:
    """Featurize a list of proteins into a batched FeaturizedBatch.

    Calls featurize_single_item per protein, stacks the results, then
    adds zero-centered isotropic Gaussian noise to the ground-truth
    coordinates to produce noisy input positions.

    Args:
        batch: List of B proteins, all pre-padded to the same
            sequence length.
        tcfg: Training config supplying noise schedule parameters.
        distogram_res: Residue-level Cβ distogram module.
        distogram_atom: Atom-level sparse distogram module.

    Returns:
        FeaturizedBatch with noisy inputs, ground-truth positions,
        distogram labels, masks, and the sampled noise level sigma.
    """
    B: int = len(batch)
    max_seq_len_in_batch: int = max(len(prot.atom_positions) for prot in batch)

    # ── Per-item featurization ────────────────────────────────────────────────
    items: list[FeaturizedItem] = [
        featurize_single_item(
            prot=batch[ix],
            c_beta_distogram_fn=distogram_res,
            atom_distogram_fn=distogram_atom,
            tcfg=tcfg,
            max_seq_len_in_batch=max_seq_len_in_batch,
        )
        for ix in range(B)
    ]

    # ── Stack per-item tensors into batched tensors ──────────────────
    packed = {
        f.name: torch.stack([getattr(it, f.name) for it in items])
        for f in dataclasses.fields(items[0])
    }
    return FeaturizedBatch(
        ref_pos=packed["ref_pos"],
        ref_element=packed["ref_element"],
        ref_space_uid=packed["ref_space_uid"],
        gt_res_distogram_indices=packed["gt_res_distogram_indices"],
        noised_res_distogram=packed["noised_res_distogram"],
        f_pseudo_beta_mask=packed["f_pseudo_beta_mask"],
        f_residue_idx=packed["f_residue_idx"],
        r_gt=packed["r_gt"],
        r_gt_noised=packed["r_gt_noised"],
        atom5_mask=packed["atom5_mask"],
        aa_indices=packed["aa_indices"],
        t_hat=packed["t_hat"],
        t_normalized=packed["t_normalized"],
        tok_idx=packed["tok_idx"],
        center_uid=packed["center_uid"],
        gt_atom_distogram_sparse=packed["gt_atom_distogram_sparse"],
        gt_atom_distogram_mask_sparse=packed["gt_atom_distogram_mask_sparse"],
    )


@dataclasses.dataclass
class FeaturizeCollate:
    """Picklable collation callable wrapping featurize_batch for workers.

    Captures the three featurization dependencies so the collate_fn
    contract ``(batch: list[T]) -> CollatedT`` is satisfied. Implemented
    as a dataclass so it survives pickle round-trips required by
    multi-worker DataLoaders.

    Attributes:
        tcfg: Training configuration supplying noise schedule parameters.
        distogram_res: Residue-level Cβ distogram module.
        distogram_atom: Atom-level sparse distogram module.
    """

    tcfg: TrainConfig
    distogram_res: Distogram
    distogram_atom: Distogram

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: list[Protein]) -> FeaturizedBatch:
        """Featurize a pre-assembled protein batch.

        Args:
            batch: List of proteins assembled by the dataset iterator.

        Returns:
            FeaturizedBatch with noisy inputs, ground-truth positions,
            and labels.
        """
        return featurize_batch(
            batch,
            self.tcfg,
            self.distogram_res,
            self.distogram_atom,
        )


class ProteinEntry(BaseModel):
    """Minimal schema for protein JSONL entries.

    Validates the three fields required by ProteinDataset and
    ClusteredProteinDataset — name, seq, and coords.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    name: str
    seq: str
    coords: dict[str, list[list[float]]]


class ProteinNamesManifest(RootModel[list[str]]):
    """Pydantic root model holding a flat list of protein entry names."""

    root: list[str]


class DatasetSplitsManifest(BaseModel):
    """Train/validation/test split lists, plus optional CATH topology mapping.

    Attributes:
        model_config: Pydantic config; ``extra="ignore"`` silently drops
            unknown fields encountered when loading the manifest.
        train: Protein entry names in the training split.
        validation: Protein entry names in the validation split.
        test: Protein entry names in the test split.
        cath_nodes: Optional mapping from protein chain name to its CATH
            topology codes (e.g. ``"2fyz.A": ["1.20.5"]``). Empty when the
            manifest does not include CATH metadata.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    train: list[str]
    validation: list[str]
    test: list[str]
    cath_nodes: dict[str, list[str]] = Field(default_factory=dict)


class ProteinDataset(
    torch.utils.data.Dataset[Protein],
):
    """Lazy-loading Dataset backed by a JSONL file.

    Scans the file once at construction to build a name→byte-offset index
    (only offsets are kept in RAM, not the protein data).  Each __getitem__
    seeks to the relevant line and parses only that entry.

    Compatible with num_workers > 0: the open file handle is excluded from
    pickling and re-opened lazily inside each worker process.

    JSONL format expected per line:
        {"name": "1abc.A", "seq": "ACDEF...", "coords": {"N": [[x,y,z],...],
         "CA": [...], "C": [...], "O": [...]}, ...}

    Args:
        jsonl_path:     Path to the JSONL file.
        names:          List of entry names (e.g. "1abc.A") to include.
        max_seq_length: Sequences longer than this are truncated; shorter ones
                        are zero-padded to this length.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        names: list[str],
        max_seq_length: int = 256,
    ) -> None:
        self.jsonl_path: Path = Path(jsonl_path)
        self.max_seq_length: int = max_seq_length
        self._file: io.BufferedReader | None = None

        name_set = set(names)
        offsets: list[int] = []
        byte_pos = 0
        with self.jsonl_path.open("rb") as f:
            for raw_line in f:
                if ProteinEntry.model_validate_json(raw_line).name in name_set:
                    offsets.append(byte_pos)
                byte_pos += len(raw_line)

        self._offsets: list[int] = offsets

    # ------------------------------------------------------------------
    # File-handle lifecycle — excluded from pickle so multiprocessing works

    def _open(self) -> io.BufferedReader:
        if self._file is None:
            self._file = self.jsonl_path.open("rb")
        return self._file

    @override
    def __getstate__(self) -> dict[str, object]:
        """Return picklable state with the open file handle set to None.

        Replaces the open ``_file`` handle with ``None`` before pickling so
        that the object can be serialised and sent to DataLoader worker
        processes, which will re-open the file lazily via ``_open``.

        Returns:
            A copy of ``__dict__`` with ``_file`` set to ``None``.
        """
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self) -> None:
        """Close the underlying file handle on deletion.

        Ensures the JSONL file descriptor is released when the dataset object
        is garbage-collected, even if ``__getstate__`` was never called.
        """
        if self._file is not None:
            self._file.close()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries in the dataset.

        Returns:
            Count of protein entries whose names were present in the supplied
            ``names`` list and found in the JSONL file.
        """
        return len(self._offsets)

    @override
    def __getitem__(
        self,
        idx: int,
    ) -> Protein:
        """Return the parsed JSON entry at the given index.

        Seeks directly to the pre-computed byte offset for ``idx``, parses the
        JSONL line, centres and pads the coordinates to ``max_seq_length``.

        Args:
            idx: Integer index in ``[0, len(self))``.

        Returns:
            Protein with atom37 coordinates truncated to at most
            max_seq_length residues.
        """
        f = self._open()
        _ = f.seek(self._offsets[idx])
        entry = ProteinEntry.model_validate_json(f.readline())

        np_example = make_np_example(entry.coords)
        center_positions(np_example)
        make_fixed_size(np_example, self.max_seq_length)
        truncate_to_length(np_example, self.max_seq_length)

        n_res: int = len(np_example["atom_positions"])

        raw = [
            restype_order.get(aa, restype_order["X"])
            for aa in entry.seq[:n_res]
        ]
        aatype = np.array(
            raw + [restype_order["X"]] * (n_res - len(raw)),
            dtype=np.intp,
        )

        return Protein(
            atom_positions=np_example["atom_positions"].astype(np.float64),
            aatype=aatype,
            atom_mask=np_example["atom_mask"].astype(np.float64),
            residue_index=np_example["residue_index"].astype(np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )


class ShardMetadata(BaseModel):
    """Persisted record of the parameters used to build a shard directory.

    Serialised as ``shard_metadata.json`` alongside the shard tars and read
    back at train time to verify that an existing shard directory matches
    the current configuration before re-use.

    Attributes:
        model_config: Pydantic config — frozen to prevent mutation after init.
        names_hash: SHA-256 hex digest of the sorted protein names list.
        token_budget: Maximum padded token cost per batch used during sharding.
        shard_size: Maximum number of proteins per shard.
        n_shards: Total number of shard tars written.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    names_hash: str
    token_budget: int
    shard_size: int
    n_shards: int


class FFDWorkerPlan(BaseModel):
    """One DataLoader worker's per-epoch plan for FFD batch streaming.

    Attributes:
        shard_ids: Global shard IDs this worker streams this epoch, in
            iteration order.
        permutations: For each shard, a list mapping streaming position
            (i.e. WebDataset arrival order) to the protein's sorted rank
            in the prepended-and-permuted local sequence.
        batch_ends: For each shard, a list of cumulative batch-end
            positions in the prepended-and-permuted sequence
            ``carry_in_sizes[k] + permuted shard k proteins``.
        carry_in_sizes: For each shard, the number of proteins prepended
            from the previous shard's carry-over. Always zero for length-
            disjoint shards (the typical case with globally sorted shards),
            but stored explicitly so workers can validate the carry buffer.
    """

    shard_ids: Sequence[int]
    permutations: Sequence[Sequence[int]]
    batch_ends: Sequence[Sequence[int]]
    carry_in_sizes: Sequence[int]


class FFDBatchPlan(BaseModel):
    """Full per-epoch FFD plan for one rank, partitioned by DataLoader worker.

    Attributes:
        worker_plans: One :class:`FFDWorkerPlan` per DataLoader worker; the
            ``i``th entry is consumed by the worker with ``worker_info.id ==
            i``.
    """

    worker_plans: list[FFDWorkerPlan]


@dataclasses.dataclass(frozen=True)
class ShardBudgetParameters:
    """All scalar inputs needed to compute one epoch's FFDBatchPlan.

    Attributes:
        shard_dir: Directory containing the shard tars and metadata.
        structlog_path: Path to the structlog output file for the worker.
        token_budget: Maximum padded token cost per batch.
        max_seq_len: Per-protein hard truncation ceiling.
        seed: RNG seed; an epoch offset is added per epoch for diversity.
        n_threads: Threads inside the worker subprocess for parallel packing.
        world_size: Number of DDP processes.
        rank: This process's DDP rank.
        n_proteins_in_shard: Expected number of proteins per shard tar.
        noise_magnitude: Half-width of the uniform noise added to each
            protein's length before sorting; controls cross-epoch batch
            diversity vs. within-batch length variance.
        num_workers: Number of DataLoader worker processes per rank. The
            plan is partitioned across workers so each worker's FFD batches
            stay within its strided shard assignment.
    """

    shard_dir: Path
    structlog_path: Path
    token_budget: int
    max_seq_len: int
    seed: int
    n_threads: int
    world_size: int
    rank: int
    n_proteins_in_shard: int
    noise_magnitude: int
    num_workers: int


class ProteinShardDataset(torch.utils.data.IterableDataset[list[Protein]]):
    """Plan-driven streaming dataset backed by WebDataset tar shards.

    Each epoch a ShardBatchPlan is injected via set_plan before the DataLoader
    iterates. __iter__ assigns shards to DataLoader workers round-robin
    (worker w takes shard_order[w::num_workers]), streams each assigned shard
    sequentially via WebDataset, and cuts proteins into pre-computed batches
    using cut_stream_into_batches.

    Shards are created from the source JSONL on first construction if the
    shard metadata file does not yet exist. At most one tar file is open per
    DataLoader worker at any moment.

    Args:
        budget_parameters: Scalar batching and shard configuration.
        names: Protein entry names to include in the training split.
        dataset_jsonl: Path to the source JSONL protein database.
    """

    def __init__(
        self,
        budget_parameters: ShardBudgetParameters,
        names: list[str],
        dataset_jsonl: Path,
    ) -> None:
        self.names: list[str] = names
        self.dataset_jsonl: Path = dataset_jsonl
        self.n_proteins_in_shard: int = budget_parameters.n_proteins_in_shard
        self.shard_dir: Path = budget_parameters.shard_dir
        sidecar_file: str = "shard_sidecar.npz"
        manifest_file: str = "shard_metadata.json"
        lengths_file: str = "all_protein_lengths.npy"
        self.shard_sidecar_path: Path = self.shard_dir / sidecar_file
        self.shard_metadata_path: Path = self.shard_dir / manifest_file
        self.lengths_path: Path = self.shard_dir / lengths_file

        self.structlog_path: Path = budget_parameters.structlog_path
        self._log: FilteringBoundLogger = cast(
            FilteringBoundLogger,
            structlog.get_logger(),
        )
        self.token_budget: int = budget_parameters.token_budget
        self.max_seq_length: int = budget_parameters.max_seq_len

        # Construct the shards if they do not already exist.
        if not self.shard_metadata_path.exists():
            self._log.info(
                "shards_do_not_exist",
            )
            n_shards, all_lengths, shard_sizes_list = self.build_sorted_shards()
            _ = self.write_shard_metadata_sidecar(
                n_shards,
                all_lengths,
                shard_sizes_list,
            )

        # prefill the batch
        self._plan: FFDBatchPlan | None = None

    def build_sorted_shards(self) -> tuple[int, list[int], list[int]]:
        """Globally sort proteins descending by length and write to shard tars.

        Single JSONL pass collects ``(name, seq_len, byte_offset)`` for each
        included protein. Entries are sorted globally descending by
        ``seq_len`` and sliced into sequential chunks of
        ``n_proteins_in_shard``. Each chunk is written to
        ``shard_{id:05d}.tar`` by seeking to the recorded byte offsets in the
        source JSONL.

        Returns:
            Tuple of ``(n_shards, all_lengths, shard_sizes_list)`` where
            ``all_lengths`` contains the ``seq_len`` of every protein in
            global shard order (descending) and ``shard_sizes_list`` contains
            the protein count per shard.
        """
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        name_set = set(self.names)

        entries: list[tuple[str, int, int]] = []
        with self.dataset_jsonl.open("rb") as f:
            byte_pos = 0
            for raw_line in f:
                entry = ProteinEntry.model_validate_json(raw_line)
                if entry.name in name_set:
                    entries.append(
                        (
                            entry.name,
                            len(entry.seq),
                            byte_pos,
                        ),
                    )
                byte_pos += len(raw_line)

        # Global descending sort by seq_len.
        entries.sort(key=lambda e: -e[1])

        all_lengths: list[int] = []
        shard_sizes_list: list[int] = []
        shard_id: int = 0

        with self.dataset_jsonl.open("rb") as src:
            for shard_start in range(
                0,
                max(len(entries), 1),
                self.n_proteins_in_shard,
            ):
                shard_entries = entries[
                    shard_start : shard_start + self.n_proteins_in_shard
                ]
                if not shard_entries:
                    continue
                shard_path: Path = self.shard_dir / f"shard_{shard_id:05d}.tar"
                with TarWriter(str(shard_path)) as sink:
                    for local_idx, ent in enumerate(shard_entries):
                        _ = src.seek(ent[2])  # byte_pos
                        raw_line = src.readline()
                        _ = sink.write(
                            {
                                "__key__": f"{local_idx:06d}",
                                "json": raw_line,
                            },
                        )
                        all_lengths.append(ent[1])  # seq_len
                shard_sizes_list.append(len(shard_entries))
                shard_id += 1

        n_shards = shard_id
        self._log.info(
            "sorted_shards_built",
            n_shards=n_shards,
        )
        return n_shards, all_lengths, shard_sizes_list

    def write_shard_metadata_sidecar(
        self,
        n_shards: int,
        all_lengths: list[int],
        shard_sizes_list: list[int],
    ) -> ShardMetadata:
        """Persist shard layout to disk and return a ShardMetadata summary.

        Writes three files to ``self.shard_dir``:

        - ``all_protein_lengths.npy``: int16 array of every protein's seq_len
          in global shard order.
        - ``shard_sidecar.npz``: int32 arrays ``shard_starts`` and
          ``shard_sizes`` giving the global protein offset and count per shard.
        - ``shard_metadata.json``: ShardMetadata JSON including a SHA-256
          digest of the sorted names list for cache-validity checks.

        Args:
            n_shards: Total number of shard tars written by compute_shards.
            all_lengths: Sequence lengths in global shard order.
            shard_sizes_list: Protein count per shard.

        Returns:
            ShardMetadata instance also serialised to shard_metadata.json.
        """
        lengths_arr: npt.NDArray[np.int16] = np.array(
            all_lengths,
            dtype=np.int16,
        )
        np.save(self.lengths_path, lengths_arr)

        sizes_arr: npt.NDArray[np.int32] = np.array(
            shard_sizes_list,
            dtype=np.int32,
        )
        starts_arr: npt.NDArray[np.int32] = np.zeros(n_shards, dtype=np.int32)
        if n_shards > 1:
            _ = np.cumsum(sizes_arr[:-1], out=starts_arr[1:])
        np.savez(
            self.shard_sidecar_path,
            shard_starts=starts_arr,
            shard_sizes=sizes_arr,
        )

        # get a stable SHA-256 hex digest of the sorted names list.
        names_hash = hashlib.sha256(
            ProteinNamesManifest(root=sorted(self.names))
            .model_dump_json()
            .encode(),
        ).hexdigest()

        shard_metadata_manifest = ShardMetadata(
            names_hash=names_hash,
            token_budget=self.token_budget,
            shard_size=self.n_proteins_in_shard,
            n_shards=n_shards,
        )

        # Serialize directly to bytes using Rust core.
        json_bytes = shard_metadata_manifest.model_dump_json(indent=2).encode(
            "utf-8",
        )
        _ = (self.shard_metadata_path).write_bytes(json_bytes)
        self._log.info(
            "shard_metadata_written",
            shard_sidecar=self.shard_sidecar_path,
            shard_metadata_manifest=self.shard_metadata_path,
        )
        return shard_metadata_manifest

    def set_plan(self, plan: FFDBatchPlan) -> None:
        """Inject the FFDBatchPlan for the next epoch.

        Called by ShardDataLoader.__iter__ before the underlying DataLoader
        starts. Because persistent_workers=False, workers restart each epoch
        and pick up the updated plan via pickling of ProteinShardDataset._plan.

        Args:
            plan: Pre-computed plan from ShardDataLoader.compute_ffd_plan.
        """
        self._plan = plan

    def parse_protein(self, sample: dict[str, object]) -> Protein:
        """Parse one WebDataset sample dict into a Protein.

        Args:
            sample: Dict with key "json" containing the decoded protein dict
                (name, seq, coords) — produced by wds.WebDataset.decode("json").

        Returns:
            Protein with atom37 coords truncated to max_seq_length residues.
        """
        raw = sample["json"]
        entry = (
            ProteinEntry.model_validate_json(raw)
            if isinstance(raw, bytes | str | bytearray)
            else ProteinEntry.model_validate(raw)
        )
        np_example = make_np_example(entry.coords)
        center_positions(np_example)
        truncate_to_length(np_example, self.max_seq_length)
        n_res: int = len(np_example["atom_positions"])
        aatype = np.array(
            [
                restype_order.get(aa, restype_order["X"])
                for aa in entry.seq[:n_res]
            ],
            dtype=np.intp,
        )
        return Protein(
            atom_positions=np_example["atom_positions"].astype(np.float64),
            aatype=aatype,
            atom_mask=np_example["atom_mask"].astype(np.float64),
            residue_index=np_example["residue_index"].astype(np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )

    def load_shard(self, sid: int) -> list[Protein]:
        """Stream all proteins from one shard tar into a list.

        Args:
            sid: Shard index used to construct the tar filename.

        Returns:
            List of Protein objects parsed from the shard.
        """
        self._log.info(f"shard id: {sid}")
        url = str(self.shard_dir / f"shard_{sid:05d}.tar")
        ds: Iterable[dict[str, object]] = cast(
            Iterable[dict[str, object]],
            wds.DataPipeline(  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                wds.SimpleShardList(  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
                    [url],
                ),
                wds.tarfile_to_samples(),  # pyright: ignore[reportUnknownMemberType,reportAttributeAccessIssue]
            ),
        )
        return [self.parse_protein(s) for s in ds]

    def sort_by_permutation(
        self,
        proteins: list[Protein],
        permutation: Sequence[int],
    ) -> list[Protein]:
        """Reorder proteins according to a permutation.

        Args:
            proteins: Proteins in streaming order.
            permutation: Maps streaming position to sorted rank.

        Returns:
            Proteins reordered so that proteins[local_pos] is placed at
            sorted_rank.
        """
        out: list[Protein | None] = [None] * len(proteins)
        for local_pos, sorted_rank in enumerate(permutation):
            out[sorted_rank] = proteins[local_pos]
        return [p for p in out if p is not None]

    @override
    def __iter__(self) -> Iterator[list[Protein]]:
        """Yield FFD batches for this worker's assigned shards.

        For each shard in the worker's plan:
        1. Stream all proteins via WebDataset into a local list.
        2. Reorder them in memory using the per-shard permutation
           (``permutations[k][local_pos]`` is the sorted rank of the protein
           at streaming position ``local_pos`` within shard ``k``).
        3. Prepend the carry-over buffer from the previous shard.
        4. Cut at ``batch_ends[k]``; yield each completed batch.
        5. Stash any trailing proteins as the new carry-over buffer.

        Shard k+1 is read from disk in a background thread while batches
        from shard k are being yielded, hiding inter-shard I/O latency.

        If set_plan has not been called, yields nothing.
        """
        plan = self._plan
        if plan is None:
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id: int = worker_info.id if worker_info is not None else 0
        worker_plan = plan.worker_plans[worker_id]
        n_shards: int = len(worker_plan.shard_ids)

        if n_shards == 0:
            return

        carry_over: list[Protein] = []
        with ThreadPoolExecutor(max_workers=1) as loader:
            next_future: Future[list[Protein]] = loader.submit(
                self.load_shard,
                worker_plan.shard_ids[0],
            )
            for k in range(n_shards):
                shard_proteins: list[Protein] = next_future.result()
                # Start reading shard k+1 from disk while we yield from shard k.
                if k + 1 < n_shards:
                    next_future = loader.submit(
                        self.load_shard,
                        worker_plan.shard_ids[k + 1],
                    )

                sorted_proteins = self.sort_by_permutation(
                    shard_proteins,
                    worker_plan.permutations[k],
                )
                full: list[Protein] = carry_over + sorted_proteins
                prev_cut = 0
                for cut in worker_plan.batch_ends[k]:
                    yield full[prev_cut:cut]
                    prev_cut = cut
                carry_over = full[prev_cut:]

        if carry_over:
            yield carry_over


def batch_count_in_ffd_plan(plan: FFDBatchPlan) -> int:
    """Total batch count across all workers for single epoch's FFDBatchPlan."""
    return sum(len(be) for wp in plan.worker_plans for be in wp.batch_ends)


def plan_cache(
    budget: ShardBudgetParameters,
    sidecar_path: Path,
    cache_dir: Path,
) -> Path:
    """Return the JSON cache file path for a given budget.

    Computes a hex digest uniquely identifying a plan's inputs. Hashes all
    budget fields (excluding structlog_path, which does not affect plan
    content) plus the sidecar file's size and mtime so the key changes if the
    shard data is regenerated.

    Args:
        budget: Epoch-level packing parameters.
        sidecar_path: Path to the shard sidecar metadata file.
        cache_dir: Directory in which plan cache files are stored.

    Returns:
        Path to the ``<key>.json`` cache file.
    """
    h = hashlib.sha256()
    for part in (
        str(budget.shard_dir),
        str(budget.token_budget),
        str(budget.max_seq_len),
        str(budget.noise_magnitude),
        str(budget.seed),
        str(budget.world_size),
        str(budget.rank),
        str(budget.num_workers),
        str(budget.n_proteins_in_shard),
    ):
        h.update(part.encode())
    stat = sidecar_path.stat()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())
    return cache_dir / f"{h.hexdigest()[:24]}.json"


class ShardDataLoader(torch.utils.data.DataLoader[FeaturizedBatch]):
    """Plan-driven DataLoader wrapper for WebDataset shard streaming.

    Encapsulates ProteinShardDataset, DataLoader, and a plan prefetch queue.
    Each call to __iter__ dequeues the next pre-computed ShardBatchPlan,
    injects it into the dataset, schedules the following epoch's plan
    asynchronously, and delegates iteration to the underlying DataLoader.
    Workers restart each epoch (persistent_workers=False) and pick up the
    updated plan via pickling of ProteinShardDataset._plan.

    Args:
        dataset: Pre-constructed ProteinShardDataset to stream from.
        budget: Scalar batching and shard configuration shared with the
            dataset.
        tcfg: Training configuration; supplies num_workers,
            batch_prefetch_depth, epoch_prefetch_depth, and featurization
            parameters.
        distogram_res: Residue-level Cβ distogram module used by
            FeaturizeCollate.
        distogram_atom: Atom-level sparse distogram module used by
            FeaturizeCollate.
    """

    def __init__(
        self,
        *,
        dataset: ProteinShardDataset,
        budget: ShardBudgetParameters,
        tcfg: TrainConfig,
        distogram_res: Distogram,
        distogram_atom: Distogram,
    ) -> None:
        self.shard_dataset: ProteinShardDataset = dataset
        num_workers: int = tcfg.train_loader.num_workers
        self.prefetch_epochs: int = tcfg.train_loader.epoch_prefetch_depth
        collate = FeaturizeCollate(
            tcfg=tcfg,
            distogram_res=distogram_res,
            distogram_atom=distogram_atom,
        )
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            cast(
                torch.utils.data.Dataset[FeaturizedBatch],
                self.shard_dataset,
            ),
            batch_size=None,
            collate_fn=collate,
            num_workers=num_workers,
            timeout=60 if num_workers > 0 else 0,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=(
                tcfg.train_loader.batch_prefetch_depth
                if num_workers > 0
                else None
            ),
        )
        self.budget: ShardBudgetParameters = budget
        self.world_size: int = budget.world_size
        self.base_seed: int = budget.seed
        self.epoch: int = 0
        self._log: FilteringBoundLogger = cast(
            FilteringBoundLogger,
            structlog.get_logger(),
        )
        self.structlog_path: Path = budget.structlog_path
        self.protein_lengths_path: Path = self.shard_dataset.lengths_path
        self.shard_sidecar_path: Path = self.shard_dataset.shard_sidecar_path

        self.process_executor: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=ShardWorkerState.init_worker,
            initargs=(
                self.protein_lengths_path,
                self.shard_sidecar_path,
                self.structlog_path,
            ),
        )
        self.queue_watcher: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
        )
        self._watcher_future: Future[None] | None = None
        self.process_queue: queue.Queue[FFDBatchPlan] = queue.Queue(
            maxsize=self.prefetch_epochs,
        )
        self.plan_cache_dir: Path = budget.shard_dir / "plan_cache"
        self.plan_cache_dir.mkdir(parents=True, exist_ok=True)
        self._log.info(
            "process_queue_initialized",
            queue_depth=self.prefetch_epochs,
        )

        first_plan: FFDBatchPlan | None = None
        for offset in range(self.prefetch_epochs):
            epoch_budget = dataclasses.replace(
                budget,
                seed=self.base_seed + offset,
                num_workers=self.num_workers if self.num_workers > 0 else 1,
            )
            cache_path = plan_cache(
                epoch_budget,
                self.shard_sidecar_path,
                self.plan_cache_dir,
            )
            if cache_path.exists():
                plan = FFDBatchPlan.model_validate_json(cache_path.read_bytes())
                self._log.info(
                    "ffd_plan_cache_hit",
                )
            else:
                plan = self.process_executor.submit(
                    self.compute_ffd_plan,
                    epoch_budget,
                ).result()
                _ = cache_path.write_bytes(plan.model_dump_json().encode())
                self._log.info(
                    "ffd_plan_computed_and_cached",
                )
            self.process_queue.put(plan)
            if first_plan is None:
                first_plan = plan

        self._cached_len: int = (
            batch_count_in_ffd_plan(first_plan) if first_plan is not None else 0
        )
        self._log.info(
            "shard_prefetch_complete",
            prefetch_epochs=self.prefetch_epochs,
        )

    @staticmethod
    def ffd_pack(
        sorted_effective_lengths: list[int],
        token_budget: int,
    ) -> list[int]:
        """First-Fit-Decreasing pack with padded `(count + 1) * L_max²` budget.

        Walks a descending-sorted, max-seq-length-clamped length list once. For
        each protein, decides whether to extend the current open batch or
        close it and start a new one. L_max for the budget check is the first
        protein in the batch (never grows mid-batch because the list is sorted
        descending). Proteins whose own L² already exceeds the budget are
        emitted as solo batches, matching today's pack_shard policy.

        Args:
            sorted_effective_lengths: Protein lengths after applying
                ``min(L, max_seq_len)`` and sorting descending by the noisy
                sort key.
            token_budget: Maximum padded token cost per batch.

        Returns:
            List of cumulative batch-end positions (1-indexed element
            count at each cut). An empty input returns an empty list.
        """
        batch_ends: list[int] = []
        token_budget = token_budget * token_budget
        current_count = 0
        batch_max_sq = 0
        for i, length in enumerate(sorted_effective_lengths):
            cost_from_item = length * length
            if cost_from_item > token_budget:
                if current_count > 0:
                    batch_ends.append(i)
                    current_count = 0
                batch_ends.append(i + 1)
                batch_max_sq = 0
                continue
            if current_count == 0:
                batch_max_sq = cost_from_item
                current_count = 1
            elif (current_count + 1) * batch_max_sq > token_budget:
                batch_ends.append(i)
                batch_max_sq = cost_from_item
                current_count = 1
            else:
                current_count += 1
        if current_count > 0:
            batch_ends.append(len(sorted_effective_lengths))
        return batch_ends

    @staticmethod
    def compute_ffd_plan(
        budget: ShardBudgetParameters,
    ) -> FFDBatchPlan:
        """Compute one epoch's FFDBatchPlan inside the WorkerState subprocess.

        Shuffles all shard IDs with ``rng(budget.seed)``, takes this rank's
        strided slice, then partitions that slice across DataLoader workers.
        For each (worker, shard) pair: looks up the shard's lengths, adds
        uniform noise ``rng.uniform(-noise_magnitude, noise_magnitude)`` to
        each length, sorts descending by the noisy key, runs
        :meth:`ffd_pack` on the clamped sorted lengths. Per-shard FFD is
        equivalent to per-worker global FFD here because globally sorted
        shards have disjoint length ranges, so FFD batches never span
        shards (carry_in_sizes is always zero).

        Args:
            budget: Scalar parameters for this epoch's plan computation.

        Returns:
            FFDBatchPlan with one FFDWorkerPlan per DataLoader worker.

        Raises:
            ShardWorkerNotInitializedError: If the worker state has not been
                initialised (lengths, shard_starts, or shard_sizes is None).
        """
        with FatalOnError():
            log: FilteringBoundLogger = cast(
                FilteringBoundLogger,
                structlog.get_logger(),
            )
            ws = ShardWorkerState.get()
            if (
                ws.lengths is None
                or ws.shard_starts is None
                or ws.shard_sizes is None
            ):
                raise ShardWorkerNotInitializedError
            lengths = ws.lengths
            shard_starts = ws.shard_starts
            shard_sizes = ws.shard_sizes
            n_shards = len(shard_starts)

            rng = np.random.default_rng(budget.seed)
            all_ids: npt.NDArray[np.int32] = np.arange(n_shards, dtype=np.int32)
            rng.shuffle(all_ids)
            rank_ids: npt.NDArray[np.int32] = all_ids[
                budget.rank : len(all_ids) : budget.world_size
            ]

            log.info(
                "ffd_plan_start",
                seed=budget.seed,
                n_rank_shards=len(rank_ids),
                num_workers=budget.num_workers,
            )

            worker_plans: list[FFDWorkerPlan] = []
            for w in range(budget.num_workers):
                worker_shard_ids: list[int] = cast(
                    list[int],
                    rank_ids[w :: budget.num_workers].tolist(),
                )
                with ThreadPoolExecutor(
                    max_workers=budget.n_threads,
                ) as pool:
                    futures: list[Future[tuple[list[int], list[int]]]] = [
                        pool.submit(
                            ShardDataLoader.pack_one_shard,
                            sid,
                            lengths,
                            shard_starts,
                            shard_sizes,
                            budget,
                            int(rng.integers(0, np.iinfo(np.int64).max)),
                        )
                        for sid in worker_shard_ids
                    ]
                permutations, batch_ends = zip(
                    *(f.result() for f in futures),
                    strict=False,
                )
                worker_plans.append(
                    FFDWorkerPlan(
                        shard_ids=worker_shard_ids,
                        permutations=permutations,
                        batch_ends=batch_ends,
                        carry_in_sizes=[0] * len(worker_shard_ids),
                    ),
                )

            log.info(
                "ffd_plan_done",
                n_workers=len(worker_plans),
                n_batches=sum(
                    len(be) for wp in worker_plans for be in wp.batch_ends
                ),
            )
            return FFDBatchPlan(worker_plans=worker_plans)

    @staticmethod
    def pack_one_shard(
        shard_id: int,
        lengths: npt.NDArray[np.int16],
        shard_starts: npt.NDArray[np.int32],
        shard_sizes: npt.NDArray[np.int32],
        budget: ShardBudgetParameters,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        """Noisy-sort then FFD-pack one shard into variable-size batches.

        Adds uniform random noise to each protein's length before sorting
        descending, which breaks the strict length ordering and prevents
        every epoch from producing identical batch groupings. The noisy
        sorted lengths are then passed to ``ffd_pack``, which uses a
        First-Fit Decreasing bin-packing algorithm to group proteins into
        batches whose total padded token cost stays within
        ``budget.token_budget``.

        The return value contains two arrays: an inverse permutation that
        maps each protein's original streaming position within the shard
        to its rank in the sorted order, and a ``batch_ends`` cut array
        that encodes where each packed batch ends in sorted order.

        Args:
            shard_id: Global shard index into shard_starts/shard_sizes.
            lengths: Global int16 array of seq_len per protein in shard
                order.
            shard_starts: Start index into lengths for each shard.
            shard_sizes: Protein count for each shard.
            budget: Epoch-level packing parameters; ``max_seq_len``,
                ``token_budget``, and ``noise_magnitude`` are read from
                this object.
            seed: Per-shard RNG seed so threads do not contend on a
                shared RNG.

        Returns:
            Tuple ``(inverse_permutation, batch_ends)`` where
            ``inverse_permutation[local_pos]`` is the protein's sorted
            rank within this shard, and ``batch_ends`` is the FFD
            cumulative cut array.
        """
        start = int(cast(np.intp, shard_starts[shard_id]))
        size = int(cast(np.intp, shard_sizes[shard_id]))
        shard_lengths = lengths[start : start + size].astype(np.int32)
        rng = np.random.default_rng(seed)
        noise = rng.uniform(
            -budget.noise_magnitude,
            budget.noise_magnitude,
            size=size,
        )
        sort_key = shard_lengths.astype(np.float64) + noise
        # argsort of -key gives descending order indices.
        sorted_order: npt.NDArray[np.int32] = np.argsort(
            -sort_key,
            kind="stable",
        ).astype(np.int32)
        # inverse_permutation[streaming_pos] = sorted rank.
        inverse_permutation: npt.NDArray[np.int32] = np.empty_like(sorted_order)
        inverse_permutation[sorted_order] = np.arange(size, dtype=np.int32)
        sorted_effective = np.minimum(
            shard_lengths[sorted_order],
            budget.max_seq_len,
        )
        batch_ends = ShardDataLoader.ffd_pack(
            cast(list[int], sorted_effective.tolist()),
            budget.token_budget,
        )
        return cast(list[int], inverse_permutation.tolist()), batch_ends

    @override
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[list[Protein]]:
        """Dequeue next plan, inject into dataset, delegate to DataLoader."""
        self._log.info("checking watcher")
        if self._watcher_future is not None and self._watcher_future.done():
            self._watcher_future.result()  # re-raise exception from thread
        self._log.info("watcher checked")
        plan = self.process_queue.get()
        self._cached_len = batch_count_in_ffd_plan(plan)

        self.shard_dataset.set_plan(plan)

        seed = self.base_seed + self.epoch + self.prefetch_epochs
        epoch_budget = dataclasses.replace(
            self.budget,
            seed=seed,
            num_workers=self.num_workers if self.num_workers > 0 else 1,
        )
        cache_path = plan_cache(
            epoch_budget,
            self.shard_sidecar_path,
            self.plan_cache_dir,
        )

        def wait_and_enqueue() -> None:
            """Load from cache or compute, save, then enqueue the next plan."""
            if cache_path.exists():
                cached = FFDBatchPlan.model_validate_json(
                    cache_path.read_bytes(),
                )
                self._log.info("ffd_plan_cache_hit", seed=seed)
                self.process_queue.put(cached)
                return
            future: Future[FFDBatchPlan] = self.process_executor.submit(
                self.compute_ffd_plan,
                epoch_budget,
            )
            computed = future.result()
            _ = cache_path.write_bytes(computed.model_dump_json().encode())
            self._log.info("ffd_plan_computed_and_cached", seed=seed)
            self.process_queue.put(computed)

        self._watcher_future = self.queue_watcher.submit(wait_and_enqueue)
        self.epoch += 1
        return super().__iter__()

    @override
    def __len__(self) -> int:
        """Return exact number of batches this rank yields for current epoch.

        Returns:
            Sum of batches across all shards assigned to this rank, computed
            from the pre-built ShardBatchPlan and updated each epoch in
            __iter__.
        """
        return self._cached_len

    def __del__(self) -> None:
        """Shut down executor and watcher without blocking process exit.

        Uses ``hasattr`` guards because ``__del__`` is called even when
        ``__init__`` raised partway through. If ``super().__init__()`` or
        any line before the executor assignments throws, Python still
        garbage-collects the partially-constructed object and fires
        ``__del__``, but ``process_executor`` and ``queue_watcher`` were
        never set.
        """
        if hasattr(self, "process_executor"):
            self.process_executor.shutdown(wait=False)
        if hasattr(self, "queue_watcher"):
            self.queue_watcher.shutdown(wait=False)


def make_bucketed_data_loaders(
    *,
    cfg: TrainConfig,
    extra_train_args: TrainArgs,
) -> tuple[
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
    torch.utils.data.DataLoader[FeaturizedBatch],
]:
    """Build the train shard loader and val/test loaders; auto-detects DDP.

    When ``dist.is_initialized()`` is True the val and test loaders are built
    with ``DistributedSampler``; otherwise they behave identically to the
    single-GPU case. The train loader is always shard-based and rank-aware
    via ``ShardBudgetParameters``.

    Args:
        cfg: Training configuration; controls token budget, sequence length,
            cluster count, and other loader parameters.
        extra_train_args: Paths to the dataset JSONL, splits JSON, shard
            directory, and structured log file; also carries the debug_run
            flag.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    is_ddp: bool = dist.is_initialized()
    rank: int = dist.get_rank() if is_ddp else 0
    world_size: int = dist.get_world_size() if is_ddp else 1

    splits = DatasetSplitsManifest.model_validate_json(
        extra_train_args.keys_for_splits_json.read_bytes(),
    )

    dr = cfg.distogram_res
    da = cfg.distogram_atom
    distogram_res: Distogram = Distogram(
        n_bins=dr.n_bins - 1,
        min_dist=dr.min_dist,
        max_dist=dr.max_dist,
        overflow_bin=True,
    ).eval()
    distogram_atom: Distogram = Distogram(
        n_bins=da.n_bins,
        overflow_bin=False,
        min_dist=da.min_dist,
        max_dist=da.max_dist,
    ).eval()
    collate: FeaturizeCollate = FeaturizeCollate(
        tcfg=cfg,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
    )

    val_set = ProteinDataset(
        extra_train_args.dataset_jsonl,
        splits.validation,
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    test_set = ProteinDataset(
        extra_train_args.dataset_jsonl,
        splits.test,
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    train_names = (
        splits.train[:252] if extra_train_args.debug_run else splits.train
    )

    # budget = ShardBudgetParameters(
    #     shard_dir=extra_train_args.shard_dir,
    #     structlog_path=extra_train_args.structlog_jsonl,
    #     token_budget=cfg.train_loader.token_budget,
    #     max_seq_len=cfg.train_loader.max_seq_length,
    #     seed=cfg.train_loader.seed,
    #     n_threads=cfg.train_loader.n_threads,
    #     n_proteins_in_shard=len(train_names) // cfg.train_loader.n_shards,
    #     world_size=world_size,
    #     rank=rank,
    #     noise_magnitude=cfg.train_loader.noise_magnitude,
    #     num_workers=cfg.train_loader.num_workers,
    # )
    # train_set = ProteinShardDataset(  # this should take structlog_jsonl too
    #     budget_parameters=budget,
    #     names=train_names,
    #     dataset_jsonl=extra_train_args.dataset_jsonl,
    # )

    # train_loader = ShardDataLoader(
    #     dataset=train_set,
    #     budget=budget,
    #     tcfg=cfg,
    #     distogram_res=distogram_res,
    #     distogram_atom=distogram_atom,
    # )

    train_set = ProteinDataset(
        extra_train_args.dataset_jsonl,
        train_names,
        max_seq_length=cfg.test_loader.max_seq_length,
    )

    if is_ddp:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=train_sampler,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=val_sampler,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            sampler=test_sampler,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=cfg.test_loader.batch_size,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=cfg.test_loader.batch_size,
            shuffle=False,
            num_workers=cfg.test_loader.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )

    return (
        cast(torch.utils.data.DataLoader[FeaturizedBatch], train_loader),
        cast(torch.utils.data.DataLoader[FeaturizedBatch], val_loader),
        cast(torch.utils.data.DataLoader[FeaturizedBatch], test_loader),
    )
