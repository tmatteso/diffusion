"""Featurization utilities for converting raw atom data into model inputs.

Contains the Distogram module, sinusoidal positional encoding, per-item and
batch featurization functions, and classifier-free-guidance conditioning
dropout.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.atom_transformers import build_sparse_pairs
from beartype import beartype
from einops import rearrange, reduce, repeat
from helpers.atom_utils import (
    ATOM5_CA,
    ATOM5_ELEMENTS,
    ATOM5_NAMES,
    Protein,
    atom37_to_atom5,
    atom37_to_cb,
    rigid_group_atom_positions,
)
from helpers.batch_types import FeaturizedBatch, FeaturizedItem
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import NoiseScheduleParams, TrainConfig
from typing_extensions import override


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
        dist: Float[torch.Tensor, "... total_atom_count total_atom_count"] = (
            torch.sqrt(
                reduce(diff**2, "... n m d -> ... n m", "sum").clamp(min=1e-8),
            )
        )

        # ---- 2. Bin assignment & one-hot (n_bins[+1]) ----
        bin_idx: Int[torch.Tensor, "... total_atom_count total_atom_count"] = (
            torch.bucketize(dist, self.edges[1:])
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
            f_pair_mask = f_pair_mask & (dist <= self.max_dist)

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
    noise_params: NoiseScheduleParams,
    max_seq_len_in_batch: int,
    window_size: int,
) -> FeaturizedItem:
    """Featurize one protein into model-ready tensors.

    Converts atom37 coordinates to atom5, computes Cβ and atom-level
    distograms, tiles ALA reference geometry across residues, and
    samples a diffusion noise level from the lognormal schedule.

    Args:
        prot: Protein with atom37 coordinates, mask, sequence,
            and residue indices.
        c_beta_distogram_fn: Residue-level Cβ distogram function.
        atom_distogram_fn: Atom-level sparse distogram function.
        noise_params: Lognormal noise schedule parameters
            (P_std, P_mean, sigma_data).
        max_seq_len_in_batch: Padded sequence length; all tensors
            are zero-padded to this length.
        window_size: Residue-level neighbour window for GT atom-pair
            construction; must match model_params.window_size.

    Returns:
        FeaturizedItem with flat atom positions, masks, distogram
        labels, reference geometry, sequence indices, and sampled
        diffusion noise level.
    """
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
    N_res: int = atom37_positions.shape[0]

    pad = max_seq_len_in_batch - N_res
    if pad > 0:
        atom37_positions = F.pad(atom37_positions, (0, 0, 0, 0, 0, pad))
        atom37_mask = F.pad(atom37_mask, (0, 0, 0, pad))
        f_residue_idx = F.pad(f_residue_idx, (0, pad))
    N_res = max_seq_len_in_batch

    aa_indices: Int[torch.Tensor, "N_res"] = torch.full(
        (max_seq_len_in_batch,),
        -100,
        dtype=torch.long,
    )
    aa_indices[:N_res] = torch.tensor(prot.aatype, dtype=torch.long)

    ala_ref_pos: Float[torch.Tensor, "5 3"] = ref_pos_for_residue("ALA")
    ala_ref_elem: Float[torch.Tensor, "5 4"] = ATOM5_ELEMENTS.float()
    P_std, P_mean = noise_params.P_std, noise_params.P_mean
    sigma_data = noise_params.sigma_data

    # Noise schedule lognormal. ln(sigma) ~ N(Pmean, Pstd**2),
    # Pmean = -1.2, Pstd = 1.5, sigma_data = 16,
    # ~ N(0, I) * Pstd + Pmean == ~ N(Pmean, Pstd**2).
    # The reparameterization trick.
    ln_sigma: Float[torch.Tensor, ""] = torch.randn(()) * P_std + P_mean
    sigma: Float[torch.Tensor, ""] = torch.exp(ln_sigma)
    # sigma_data is a constant determined by the variance of the data
    # (default 16) t_hat is the sampled noise level. This according to AF3.
    t_hat: Float[torch.Tensor, ""] = sigma_data * sigma

    # t_normalized is drawn from uniform(0, 1) and broadcast to every
    # residue pair.
    t_scalar: Float[torch.Tensor, ""] = torch.rand(())
    t_template: Float[torch.Tensor, "N_res N_res"] = repeat(
        t_scalar,
        "-> n m",
        n=N_res,
        m=N_res,
    )

    atom5_pos: Float[torch.Tensor, "N_res 5 3"]
    atom5_mask: Float[torch.Tensor, "N_res 5"]
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

    c_beta_pos: Float[torch.Tensor, "B N_res 5 3"]
    c_beta_pos, _ = atom37_to_cb(
        rearrange(atom37_positions, "n a d -> 1 n a d"),
        rearrange(atom37_mask, "n a -> 1 n a"),
    )
    c_beta_pos = rearrange(c_beta_pos, "1 n d -> n d")

    gt_res_distogram: Float[torch.Tensor, "N_atom N_atom n_template_bins"]
    gt_res_distogram, _ = c_beta_distogram_fn(c_beta_pos, residue_mask)

    flat_pos: Float[torch.Tensor, "N_atom 3"] = rearrange(
        atom5_pos,
        "n a d -> (n a) d",
    )
    atom_mask_flat: Bool[torch.Tensor, "N_atom"] = rearrange(
        atom5_mask.bool(),
        "n a -> (n a)",
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
    ref_space_uid: Int[torch.Tensor, "B N_atom"] = (
        f_residue_idx.repeat_interleave(Natom)
    )
    # because we are only using one chain right now,
    # f_residue_idx == ref_space_uid.

    neighbor_idx: Float[torch.Tensor, "N, K"]
    valid_mask: Float[
        torch.Tensor,
        "N, K",
    ]  # True where the slot is a real neighbour (not padding)
    neighbor_idx, valid_mask = build_sparse_pairs(
        token_idx,
        window_size,
    )  # (N_atom, K)

    # atom_distogram_fn
    gt_atom_disto_dense: Float[torch.Tensor, "N_atom N_atom n_atom_bins"]
    gt_atom_mask_dense: Bool[torch.Tensor, "N_atom N_atom"]
    gt_atom_disto_dense, gt_atom_mask_dense = atom_distogram_fn(
        flat_pos,
        atom_mask_flat,
    )
    # Vectorised sparse gather: result[l, k] = dense[l, neighbor_idx[l, k]]
    # index must be the same shape as the output
    gt_atom_distogram_sparse: Float[torch.Tensor, "N_atom K n_atom_bins"] = (
        gt_atom_disto_dense.gather(
            1,
            repeat(
                neighbor_idx,
                "n k -> n k d",
                d=gt_atom_disto_dense.shape[-1],
            ),
        )
    )
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "N_atom K"] = (
        gt_atom_mask_dense.long().gather(1, neighbor_idx).bool() & valid_mask
    )
    del gt_atom_disto_dense, gt_atom_mask_dense

    return FeaturizedItem(
        r_gt=flat_pos,
        atom5_mask=atom_mask_flat,
        f_pseudo_beta_mask=residue_mask.long(),
        gt_res_distogram=gt_res_distogram.long(),
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
    max_seq_len_in_batch = max(prot.atom_positions.shape[0] for prot in batch)

    # ── Per-item featurization ────────────────────────────────────────────────
    items: list[FeaturizedItem] = [
        featurize_single_item(
            prot=batch[ix],
            c_beta_distogram_fn=distogram_res,
            atom_distogram_fn=distogram_atom,
            noise_params=tcfg.noise,
            max_seq_len_in_batch=max_seq_len_in_batch,
            window_size=tcfg.model.window_size,
        )
        for ix in range(B)
    ]

    # ── Stack per-item tensors into batched tensors ──────────────────
    packed = {
        f.name: torch.stack([getattr(it, f.name) for it in items])
        for f in dataclasses.fields(items[0])
    }
    # step 0:
    # Apply zero-centered noise to turn r_gt into r_input
    noise = torch.randn_like(packed["r_gt"])
    noise = noise - reduce(
        noise,
        "b n d -> b 1 d",
        "mean",
    )  # match sampling convention
    r_input: Float[torch.Tensor, "B N_atom 3"] = (
        packed["r_gt"] + rearrange(packed["t_hat"], "b -> b 1 1") * noise
    )
    return FeaturizedBatch(
        ref_pos=packed["ref_pos"],
        ref_element=packed["ref_element"],
        ref_space_uid=packed["ref_space_uid"],
        gt_res_distogram=packed["gt_res_distogram"],
        f_pseudo_beta_mask=packed["f_pseudo_beta_mask"],
        f_residue_idx=packed["f_residue_idx"],
        r_gt=packed["r_gt"],
        r_gt_noised=r_input,
        atom5_mask=packed["atom5_mask"],
        aa_indices=packed["aa_indices"],
        t_hat=packed["t_hat"],
        t_normalized=packed["t_normalized"],
        tok_idx=packed["tok_idx"],
        center_uid=packed["center_uid"],
        gt_atom_distogram_sparse=packed["gt_atom_distogram_sparse"],
        gt_atom_distogram_mask_sparse=packed["gt_atom_distogram_mask_sparse"],
    )


@jaxtyped(typechecker=beartype)
def apply_conditioning_dropout(
    batch: FeaturizedBatch,
    p_distogram: float,
    p_atom: float,
    p_seq: float,
    device: str,
) -> FeaturizedBatch:
    """Randomly zero out conditioning signals for classifier-free guidance.

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
        batch: Featurized batch whose conditioning tensors will be partially
            zeroed.
        p_distogram: Per-residue Bernoulli drop probability for the distogram
            and pseudo-β conditioning signals.
        p_atom: Per-residue Bernoulli drop probability for atom-level position
            conditioning (atom5_mask).
        p_seq: Per-residue Bernoulli drop probability for the sequence
            conditioning
            (aa_indices replaced with mask token 20).
        device: PyTorch device on which to allocate the Bernoulli sample
            tensors.

    Returns:
        A new FeaturizedBatch with the same fields as the input except
        ``gt_res_distogram``, ``f_pseudo_beta_mask``, ``atom5_mask``, and
        ``aa_indices`` are partially zeroed according to the sampled masks.
    """
    f_pseudo_beta: Bool[torch.Tensor, "B N_res"] = (
        batch.f_pseudo_beta_mask.bool()
    )
    B, N_res = f_pseudo_beta.shape

    drop_d: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(
            torch.full((B, N_res), p_distogram, device=device),
        ).bool()
        & f_pseudo_beta
    )
    drop_a: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_atom, device=device)).bool()
        & f_pseudo_beta
    )
    drop_s: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_seq, device=device)).bool()
        & f_pseudo_beta
    )

    # Distogram: zero rows AND columns for dropped residues
    # (matrix is symmetric).
    keep_d: Bool[torch.Tensor, "B N_res"] = ~drop_d
    disto_mask: Bool[torch.Tensor, "B N_res N_res"] = rearrange(
        keep_d,
        "b i -> b i 1",
    ) & rearrange(keep_d, "b j -> b 1 j")
    new_distogram = batch.gt_res_distogram * rearrange(
        disto_mask.long(),
        "b i j -> b i j 1",
    )
    new_pseudo_beta_mask = batch.f_pseudo_beta_mask * keep_d.long()

    # Atom mask: expand residue-level drop to atom level (5 atoms per residue)
    drop_a_expanded: Bool[torch.Tensor, "B N_atom"] = repeat(
        drop_a,
        "b n -> b (n a)",
        a=5,
    )
    new_atom5_mask: Bool[torch.Tensor, "B N_atom"] = (
        batch.atom5_mask & ~drop_a_expanded
    )

    # Sequence: replace dropped tokens with mask-token index 20 ("X")
    new_aa_indices: Int[torch.Tensor, "B N_res"] = batch.aa_indices.masked_fill(
        drop_s,
        20,
    )

    return dataclasses.replace(
        batch,
        gt_res_distogram=new_distogram,
        f_pseudo_beta_mask=new_pseudo_beta_mask,
        atom5_mask=new_atom5_mask,
        aa_indices=new_aa_indices,
    )
