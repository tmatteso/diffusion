"""Tests for ProteinBatch, FeaturizedItem, and FeaturizedBatch shape contracts."""

import pytest
import torch
from einops import repeat
from helpers.batch_types import FeaturizedBatch, FeaturizedItem, ProteinBatch
from jaxtyping import Float, TypeCheckError

B = 2
N_RES = 8
N_ATOM = N_RES * 5
N_TEMPL_BINS = 38
C_RES = 32
K = 16
N_ATOM_BINS = 16
AA_SEQ = "ACDEFGHI"


# ---------------------------------------------------------------------------
# ProteinBatch — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def atom_positions() -> Float[torch.Tensor, "B N_res 37 3"]:
    """Provide random atom positions (B, N_RES, 37, 3)."""
    return torch.randn(B, N_RES, 37, 3)


@pytest.fixture
def atom_mask() -> Float[torch.Tensor, "B N_res 37"]:
    """Provide an all-ones atom mask (B, N_RES, 37)."""
    return torch.ones(B, N_RES, 37)


@pytest.fixture
def residue_index() -> Float[torch.Tensor, "B N_res"]:
    """Provide sequential residue indices repeated over the batch dimension."""
    return repeat(torch.arange(N_RES).float(), "n -> b n", b=B)


@pytest.fixture
def protein_batch(
    atom_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom_mask: Float[torch.Tensor, "B N_res 37"],
    residue_index: Float[torch.Tensor, "B N_res"],
) -> ProteinBatch:
    """Provide a valid ProteinBatch with B=2, N_RES=8."""
    return ProteinBatch(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index,
        seq=[AA_SEQ, AA_SEQ],
    )


# ---------------------------------------------------------------------------
# ProteinBatch — valid construction
# ---------------------------------------------------------------------------


def test_protein_batch_constructs(protein_batch: ProteinBatch) -> None:
    """ProteinBatch constructs successfully from tensors with the correct shapes."""
    assert isinstance(protein_batch, ProteinBatch)


def test_protein_batch_atom_positions_shape(protein_batch: ProteinBatch) -> None:
    """ProteinBatch.atom_positions has shape (B, N_RES, 37, 3)."""
    assert protein_batch.atom_positions.shape == (B, N_RES, 37, 3)


def test_protein_batch_atom_mask_shape(protein_batch: ProteinBatch) -> None:
    """ProteinBatch.atom_mask has shape (B, N_RES, 37)."""
    assert protein_batch.atom_mask.shape == (B, N_RES, 37)


def test_protein_batch_residue_index_shape(protein_batch: ProteinBatch) -> None:
    """ProteinBatch.residue_index has shape (B, N_RES)."""
    assert protein_batch.residue_index.shape == (B, N_RES)


def test_protein_batch_seq_length(protein_batch: ProteinBatch) -> None:
    """ProteinBatch.seq has B entries, each of length N_RES."""
    assert len(protein_batch.seq) == B
    assert all(len(s) == N_RES for s in protein_batch.seq)


# ---------------------------------------------------------------------------
# ProteinBatch — shape enforcement
# ---------------------------------------------------------------------------


def test_protein_batch_rejects_3d_atom_positions() -> None:
    """ProteinBatch raises when atom_positions is 3-D (missing the coords dim)."""
    with pytest.raises((TypeCheckError, Exception)):
        ProteinBatch(
            atom_positions=torch.randn(B, N_RES, 37),
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=repeat(torch.arange(N_RES).float(), "n -> b n", b=B),
            seq=[AA_SEQ, AA_SEQ],
        )


def test_protein_batch_rejects_wrong_atom_count() -> None:
    """ProteinBatch raises when atom_positions has 36 atom slots instead of 37."""
    with pytest.raises((TypeCheckError, Exception)):
        ProteinBatch(
            atom_positions=torch.randn(B, N_RES, 36, 3),
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=repeat(torch.arange(N_RES).float(), "n -> b n", b=B),
            seq=[AA_SEQ, AA_SEQ],
        )


# ---------------------------------------------------------------------------
# FeaturizedItem — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_item() -> FeaturizedItem:
    """Provide a valid FeaturizedItem with N_RES=8, N_ATOM=40."""
    return FeaturizedItem(
        N_res=N_RES,
        flat_pos=torch.randn(N_ATOM, 3),
        atom_mask_flat=torch.ones(N_ATOM, dtype=torch.bool),
        residue_mask=torch.ones(N_RES, dtype=torch.bool),
        f_pseudo_beta=torch.zeros(N_RES, dtype=torch.long),
        gt_res_distogram=torch.zeros(N_RES, N_RES, N_TEMPL_BINS, dtype=torch.long),
        aa_indices=torch.zeros(N_RES, dtype=torch.long),
        ref_pos=torch.randn(N_ATOM, 3),
        ref_element=torch.zeros(N_ATOM, 4),
        f_residue_idx=torch.randn(N_RES, C_RES),
    )


# ---------------------------------------------------------------------------
# FeaturizedItem — valid construction
# ---------------------------------------------------------------------------


def test_featurized_item_constructs(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem constructs successfully from tensors with the correct shapes."""
    assert isinstance(featurized_item, FeaturizedItem)


def test_featurized_item_n_res(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.N_res stores the expected residue count."""
    assert featurized_item.N_res == N_RES


def test_featurized_item_flat_pos_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.flat_pos has shape (N_ATOM, 3)."""
    assert featurized_item.flat_pos.shape == (N_ATOM, 3)


def test_featurized_item_atom_mask_flat_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.atom_mask_flat has shape (N_ATOM,)."""
    assert featurized_item.atom_mask_flat.shape == (N_ATOM,)


def test_featurized_item_residue_mask_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.residue_mask has shape (N_RES,)."""
    assert featurized_item.residue_mask.shape == (N_RES,)


def test_featurized_item_gt_res_distogram_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.gt_res_distogram has shape (N_RES, N_RES, N_TEMPL_BINS)."""
    assert featurized_item.gt_res_distogram.shape == (N_RES, N_RES, N_TEMPL_BINS)


def test_featurized_item_ref_pos_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.ref_pos has shape (N_ATOM, 3)."""
    assert featurized_item.ref_pos.shape == (N_ATOM, 3)


def test_featurized_item_ref_element_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.ref_element has shape (N_ATOM, 4)."""
    assert featurized_item.ref_element.shape == (N_ATOM, 4)


def test_featurized_item_f_residue_idx_shape(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem.f_residue_idx has shape (N_RES, C_RES)."""
    assert featurized_item.f_residue_idx.shape == (N_RES, C_RES)


# ---------------------------------------------------------------------------
# FeaturizedItem — shape enforcement
# ---------------------------------------------------------------------------


def test_featurized_item_rejects_1d_flat_pos() -> None:
    """FeaturizedItem raises when flat_pos is 1-D instead of (N_ATOM, 3)."""
    with pytest.raises((TypeCheckError, Exception)):
        FeaturizedItem(
            N_res=N_RES,
            flat_pos=torch.randn(N_ATOM),
            atom_mask_flat=torch.ones(N_ATOM, dtype=torch.bool),
            residue_mask=torch.ones(N_RES, dtype=torch.bool),
            f_pseudo_beta=torch.zeros(N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(N_RES, N_RES, N_TEMPL_BINS, dtype=torch.long),
            aa_indices=torch.zeros(N_RES, dtype=torch.long),
            ref_pos=torch.randn(N_ATOM, 3),
            ref_element=torch.zeros(N_ATOM, 4),
            f_residue_idx=torch.randn(N_RES, C_RES),
        )


def test_featurized_item_rejects_wrong_ref_element_dim() -> None:
    """FeaturizedItem raises when ref_element last dim is 3 instead of 4."""
    with pytest.raises((TypeCheckError, Exception)):
        FeaturizedItem(
            N_res=N_RES,
            flat_pos=torch.randn(N_ATOM, 3),
            atom_mask_flat=torch.ones(N_ATOM, dtype=torch.bool),
            residue_mask=torch.ones(N_RES, dtype=torch.bool),
            f_pseudo_beta=torch.zeros(N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(N_RES, N_RES, N_TEMPL_BINS, dtype=torch.long),
            aa_indices=torch.zeros(N_RES, dtype=torch.long),
            ref_pos=torch.randn(N_ATOM, 3),
            ref_element=torch.zeros(N_ATOM, 3),
            f_residue_idx=torch.randn(N_RES, C_RES),
        )


# ---------------------------------------------------------------------------
# FeaturizedBatch — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_batch() -> FeaturizedBatch:
    """Provide a valid FeaturizedBatch with B=2, N_RES=8, N_ATOM=40, K=16."""
    return FeaturizedBatch(
        ref_pos=torch.randn(B, N_ATOM, 3),
        ref_element=torch.zeros(B, N_ATOM, 4),
        ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
        gt_res_distogram=torch.zeros(B, N_RES, N_RES, N_TEMPL_BINS, dtype=torch.long),
        f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
        f_residue_idx=torch.randn(B, N_RES, C_RES),
        r_input=torch.randn(B, N_ATOM, 3),
        r_gt=torch.randn(B, N_ATOM, 3),
        atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
        aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
        residue_mask=torch.ones(B, N_RES, dtype=torch.bool),
        t_hat=0.5,
        t_normalized=0.5,
        tok_idx=torch.zeros(B, N_ATOM, dtype=torch.long),
        center_uid=repeat(torch.arange(N_RES, dtype=torch.long), "n -> b n", b=B),
        gt_atom_distogram_sparse=torch.randn(B, N_ATOM, K, N_ATOM_BINS),
        gt_atom_distogram_mask_sparse=torch.ones(B, N_ATOM, K, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# FeaturizedBatch — valid construction
# ---------------------------------------------------------------------------


def test_featurized_batch_constructs(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch constructs successfully from tensors with the correct shapes."""
    assert isinstance(featurized_batch, FeaturizedBatch)


def test_featurized_batch_ref_pos_shape(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.ref_pos has shape (B, N_ATOM, 3)."""
    assert featurized_batch.ref_pos.shape == (B, N_ATOM, 3)


def test_featurized_batch_ref_element_shape(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.ref_element has shape (B, N_ATOM, 4)."""
    assert featurized_batch.ref_element.shape == (B, N_ATOM, 4)


def test_featurized_batch_gt_res_distogram_shape(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.gt_res_distogram has shape (B, N_RES, N_RES, N_TEMPL_BINS)."""
    assert featurized_batch.gt_res_distogram.shape == (B, N_RES, N_RES, N_TEMPL_BINS)


def test_featurized_batch_f_residue_idx_shape(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.f_residue_idx has shape (B, N_RES, C_RES)."""
    assert featurized_batch.f_residue_idx.shape == (B, N_RES, C_RES)


def test_featurized_batch_r_input_shape(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.r_input has shape (B, N_ATOM, 3)."""
    assert featurized_batch.r_input.shape == (B, N_ATOM, 3)


def test_featurized_batch_atom5_mask_shape(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.atom5_mask has shape (B, N_ATOM)."""
    assert featurized_batch.atom5_mask.shape == (B, N_ATOM)


def test_featurized_batch_gt_atom_distogram_sparse_shape(
    featurized_batch: FeaturizedBatch,
) -> None:
    """FeaturizedBatch.gt_atom_distogram_sparse has shape (B, N_ATOM, K, N_ATOM_BINS)."""
    assert featurized_batch.gt_atom_distogram_sparse.shape == (B, N_ATOM, K, N_ATOM_BINS)


def test_featurized_batch_t_hat_is_float(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.t_hat is a Python float."""
    assert isinstance(featurized_batch.t_hat, float)


def test_featurized_batch_t_normalized_in_unit_interval(featurized_batch: FeaturizedBatch) -> None:
    """FeaturizedBatch.t_normalized lies in [0, 1]."""
    assert 0.0 <= featurized_batch.t_normalized <= 1.0


# ---------------------------------------------------------------------------
# FeaturizedBatch — shape enforcement
# ---------------------------------------------------------------------------


def test_featurized_batch_rejects_unbatched_ref_pos() -> None:
    """FeaturizedBatch raises when ref_pos is 2-D (missing the batch dimension)."""
    with pytest.raises((TypeCheckError, Exception)):
        FeaturizedBatch(
            ref_pos=torch.randn(N_ATOM, 3),
            ref_element=torch.zeros(B, N_ATOM, 4),
            ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
            gt_res_distogram=torch.zeros(B, N_RES, N_RES, N_TEMPL_BINS, dtype=torch.long),
            f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
            f_residue_idx=torch.randn(B, N_RES, C_RES),
            r_input=torch.randn(B, N_ATOM, 3),
            r_gt=torch.randn(B, N_ATOM, 3),
            atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
            aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
            residue_mask=torch.ones(B, N_RES, dtype=torch.bool),
            t_hat=0.5,
            t_normalized=0.5,
            tok_idx=torch.zeros(B, N_ATOM, dtype=torch.long),
            center_uid=repeat(torch.arange(N_RES, dtype=torch.long), "n -> b n", b=B),
            gt_atom_distogram_sparse=torch.randn(B, N_ATOM, K, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.ones(B, N_ATOM, K, dtype=torch.bool),
        )


def test_featurized_batch_rejects_wrong_coords_dim() -> None:
    """FeaturizedBatch raises when r_input last dim is 4 instead of 3."""
    with pytest.raises((TypeCheckError, Exception)):
        FeaturizedBatch(
            ref_pos=torch.randn(B, N_ATOM, 3),
            ref_element=torch.zeros(B, N_ATOM, 4),
            ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
            gt_res_distogram=torch.zeros(B, N_RES, N_RES, N_TEMPL_BINS, dtype=torch.long),
            f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
            f_residue_idx=torch.randn(B, N_RES, C_RES),
            r_input=torch.randn(B, N_ATOM, 4),
            r_gt=torch.randn(B, N_ATOM, 3),
            atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
            aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
            residue_mask=torch.ones(B, N_RES, dtype=torch.bool),
            t_hat=0.5,
            t_normalized=0.5,
            tok_idx=torch.zeros(B, N_ATOM, dtype=torch.long),
            center_uid=repeat(torch.arange(N_RES, dtype=torch.long), "n -> b n", b=B),
            gt_atom_distogram_sparse=torch.randn(B, N_ATOM, K, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.ones(B, N_ATOM, K, dtype=torch.bool),
        )
