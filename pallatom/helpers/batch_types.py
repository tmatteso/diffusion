"""Pure data containers shared across featurization and training.

Contains three frozen dataclasses — ``ProteinBatch``, ``FeaturizedItem``, and
``FeaturizedBatch`` — that carry tensors through the data-loading,
featurization, and model-forward pipeline.  All fields are runtime
shape-checked via jaxtyping and beartype.
"""

import dataclasses

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class FeaturizedItem:
    """Per-protein features produced by featurize_single_item before batching.

    Stores all tensors for a single protein after featurization but before the
    DataLoader collates them into a batch.  These tensors are later stacked
    into a ``FeaturizedBatch`` by ``featurize_batch``.

    Attributes:
        r_gt: Ground-truth atom positions in flat (atom-indexed) layout.
        atom5_mask: Boolean mask indicating which of the 5 backbone+Cβ atoms
            are present for each residue.
        f_pseudo_beta_mask: Boolean mask indicating residues that have a valid
            pseudo-β carbon, used to build the template Cβ distogram.
        gt_res_distogram: Discretised pairwise Cβ distance distribution between
            all residue pairs, binned into ``n_templ_bins`` distance categories.
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
    atom5_mask: Bool[torch.Tensor, "N_atom"]
    f_pseudo_beta_mask: Int[torch.Tensor, "N_res"]
    gt_res_distogram: Int[torch.Tensor, "N_res N_res n_templ_bins"]
    aa_indices: Int[torch.Tensor, "N_res"]
    ref_pos: Float[torch.Tensor, "N_atom 3"]
    ref_element: Float[torch.Tensor, "N_atom 4"]
    f_residue_idx: Int[torch.Tensor, "N_res"]
    t_hat: Float[torch.Tensor, ""]
    t_normalized: Float[torch.Tensor, "N_res N_res"]
    ref_space_uid: Int[torch.Tensor, "N_atom"]
    tok_idx: Int[torch.Tensor, "N_atom"]
    center_uid: Int[torch.Tensor, "N_atom"]
    gt_atom_distogram_sparse: Float[torch.Tensor, "N_atom K n_atom_bins"]
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
        gt_res_distogram: Discretised pairwise Cβ distance distribution binned
            into ``n_templ_bins`` categories, with batch dimension prepended.
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
    gt_res_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"]
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
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"]
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
            gt_res_distogram=self.gt_res_distogram.to(
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
            gt_atom_distogram_mask_sparse=self.gt_atom_distogram_mask_sparse.to(
                device,
                non_blocking=non_blocking,
            ),
        )
