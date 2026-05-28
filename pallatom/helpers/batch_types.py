"""Pure data containers shared across featurization, architecture, and training."""

import dataclasses

import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped


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
    f_residue_idx: Int[torch.Tensor, "N_res"]
    t_hat: Float[torch.Tensor, ""]
    t_template: Float[torch.Tensor, "N_res N_res"]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class FeaturizedBatch:
    """Model-ready batch produced by featurize_batch, noisy inputs and ground-truth labels."""

    ref_pos: Float[torch.Tensor, "B N_atom 3"]
    ref_element: Float[torch.Tensor, "B N_atom 4"]
    ref_space_uid: Int[torch.Tensor, "B N_atom"]
    gt_res_distogram: Int[torch.Tensor, "B N_res N_res n_templ_bins"]
    f_pseudo_beta_mask: Int[torch.Tensor, "B N_res"]
    f_residue_idx: Int[torch.Tensor, "B N_res"]
    r_gt: Float[torch.Tensor, "B N_atom 3"]
    atom5_mask: Bool[torch.Tensor, "B N_atom"]
    aa_indices: Int[torch.Tensor, "B N_res"]
    residue_mask: Bool[torch.Tensor, "B N_res"]
    t_hat: Float[torch.Tensor, "B"]
    t_normalized: Float[torch.Tensor, "B N_res N_res"]
    tok_idx: Int[torch.Tensor, "B N_atom"]
    center_uid: Int[torch.Tensor, "B N_res"]
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K n_atom_bins"]
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K"]
