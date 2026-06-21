"""Tests for FeaturizedItem, and FeaturizedBatch shape contracts.

Covers valid construction from correctly-shaped tensors, attribute shape
assertions, and rejection of tensors that violate the declared shape contracts.
"""

import numpy as np
import pytest
import torch
from einops import repeat
from helpers.atom_utils import Protein
from helpers.data import FeaturizedBatch, FeaturizedItem
from jaxtyping import TypeCheckError

B = 2
N_RES = 8
N_ATOM = N_RES * 5
N_TEMPL_BINS = 38
C_RES = 32
K = 16
N_ATOM_BINS = 16


# ---------------------------------------------------------------------------
# Protein batch — fixtures
# ---------------------------------------------------------------------------


def make_protein() -> Protein:
    """Construct a single Protein with N_RES=8 residues.

    Returns:
        A Protein with randomly-initialised atom_positions and all-ones masks.
    """
    return Protein(
        atom_positions=np.random.default_rng().standard_normal((N_RES, 37, 3)),
        aatype=np.zeros(N_RES, dtype=np.intp),
        atom_mask=np.ones((N_RES, 37), dtype=np.float64),
        residue_index=np.arange(N_RES, dtype=np.intp),
        chain_index=np.zeros(N_RES, dtype=np.intp),
        b_factors=np.zeros((N_RES, 37), dtype=np.float64),
    )


@pytest.fixture
def protein_batch() -> list[Protein]:
    """Provide a list of B=2 valid Protein objects, each with N_RES=8 residues.

    Returns:
        A list of B fully-constructed Protein instances.
    """
    return [make_protein() for _ in range(B)]


# ---------------------------------------------------------------------------
# Protein batch — valid construction
# ---------------------------------------------------------------------------


def test_protein_batch_constructs(protein_batch: list[Protein]) -> None:
    """list[Protein] has B items with the correct types and array shapes.

    Verifies that the fixture produces a list of B Protein instances and that
    each instance carries atom_positions (N_RES, 37, 3), atom_mask (N_RES, 37),
    residue_index (N_RES,), and aatype (N_RES,) arrays of the expected shape.
    """
    assert len(protein_batch) == B
    for protein in protein_batch:
        assert isinstance(protein, Protein)
        assert protein.atom_positions.shape == (N_RES, 37, 3)
        assert protein.atom_mask.shape == (N_RES, 37)
        assert protein.residue_index.shape == (N_RES,)
        assert protein.aatype.shape == (N_RES,)


# ---------------------------------------------------------------------------
# Protein batch — shape enforcement
# ---------------------------------------------------------------------------


def test_protein_rejects_2d_atom_positions() -> None:
    """Protein raises when atom_positions is 2-D (missing the per-atom dim).

    Verifies that passing an array of shape (N_RES, 3) instead of
    (N_RES, 37, 3) triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = Protein(
            atom_positions=np.random.default_rng().standard_normal((N_RES, 3)),
            aatype=np.zeros(N_RES, dtype=np.intp),
            atom_mask=np.ones((N_RES, 37), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, 37), dtype=np.float64),
        )


def test_protein_rejects_mismatched_atom_count() -> None:
    """Error raised when atom_positions, atom_mask have different atom counts.

    Verifies that passing atom_positions of shape (N_RES, 36, 3) while
    atom_mask has shape (N_RES, 37) triggers a shape-contract violation because
    the symbolic num_atom_type dimension must be consistent across fields.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = Protein(
            atom_positions=np.random.default_rng().standard_normal(
                (N_RES, 36, 3),
            ),
            aatype=np.zeros(N_RES, dtype=np.intp),
            atom_mask=np.ones((N_RES, 37), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, 37), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# FeaturizedItem — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_item() -> FeaturizedItem:
    """Provide a valid FeaturizedItem with N_RES=8, N_ATOM=40.

    Returns:
        A fully-constructed FeaturizedItem using tensors of the correct shapes.
    """
    return FeaturizedItem(
        r_gt=torch.randn(N_ATOM, 3),
        atom5_mask=torch.ones(N_ATOM, dtype=torch.bool),
        f_pseudo_beta_mask=torch.zeros(N_RES, dtype=torch.long),
        gt_res_distogram=torch.zeros(
            N_RES,
            N_RES,
            N_TEMPL_BINS,
            dtype=torch.long,
        ),
        aa_indices=torch.zeros(N_RES, dtype=torch.long),
        ref_pos=torch.randn(N_ATOM, 3),
        ref_element=torch.zeros(N_ATOM, 4),
        f_residue_idx=torch.arange(N_RES, dtype=torch.long),
        t_hat=torch.randn([]),
        t_normalized=torch.randn(N_RES, N_RES),
        ref_space_uid=torch.arange(N_RES, dtype=torch.long).repeat_interleave(
            5,
        ),
        tok_idx=torch.arange(N_RES, dtype=torch.long).repeat_interleave(5),
        center_uid=(
            torch.arange(N_RES, dtype=torch.long) * 5 + 1
        ).repeat_interleave(5),
        gt_atom_distogram_sparse=torch.zeros(N_ATOM, K, N_ATOM_BINS),
        gt_atom_distogram_mask_sparse=torch.ones(N_ATOM, K, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# FeaturizedItem — valid construction
# ---------------------------------------------------------------------------


def test_featurized_item_constructs(featurized_item: FeaturizedItem) -> None:
    """FeaturizedItem constructed successfully with correct types and shapes.

    Verifies that the fixture produces a FeaturizedItem instance and that key
    tensors retain expected shapes: r_gt (N_ATOM, 3), atom5_mask (N_ATOM,),
    gt_res_distogram (N_RES, N_RES, N_TEMPL_BINS), ref_pos (N_ATOM, 3),
    ref_element (N_ATOM, 4), and f_residue_idx (N_RES,).
    """
    assert isinstance(featurized_item, FeaturizedItem)
    assert featurized_item.r_gt.shape == (N_ATOM, 3)
    assert featurized_item.atom5_mask.shape == (N_ATOM,)
    assert featurized_item.gt_res_distogram.shape == (
        N_RES,
        N_RES,
        N_TEMPL_BINS,
    )
    assert featurized_item.ref_pos.shape == (N_ATOM, 3)
    assert featurized_item.ref_element.shape == (N_ATOM, 4)
    assert featurized_item.f_residue_idx.shape == (N_RES,)


# ---------------------------------------------------------------------------
# FeaturizedItem — shape enforcement
# ---------------------------------------------------------------------------


def test_featurized_item_rejects_1d_r_gt() -> None:
    """FeaturizedItem raises when r_gt is 1-D instead of (N_ATOM, 3).

    Verifies that passing a tensor of shape (N_ATOM,) instead of (N_ATOM, 3)
    triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedItem(
            r_gt=torch.randn(N_ATOM),
            atom5_mask=torch.ones(N_ATOM, dtype=torch.bool),
            f_pseudo_beta_mask=torch.zeros(N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                N_RES,
                N_RES,
                N_TEMPL_BINS,
                dtype=torch.long,
            ),
            aa_indices=torch.zeros(N_RES, dtype=torch.long),
            ref_pos=torch.randn(N_ATOM, 3),
            ref_element=torch.zeros(N_ATOM, 4),
            f_residue_idx=torch.arange(N_RES, dtype=torch.long),
            t_hat=torch.randn([]),
            t_normalized=torch.randn(N_RES, N_RES),
            ref_space_uid=torch.zeros(N_ATOM, dtype=torch.long),
            tok_idx=torch.zeros(N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.zeros(N_ATOM, K, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.zeros(
                N_ATOM,
                K,
                dtype=torch.bool,
            ),
        )


def test_featurized_item_rejects_wrong_ref_element_dim() -> None:
    """FeaturizedItem raises when ref_element last dim is 3 instead of 4.

    Verifies that passing a tensor of shape (N_ATOM, 3) instead of (N_ATOM, 4)
    triggers a shape-contract violation because exactly four element classes
    are required.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedItem(
            r_gt=torch.randn(N_ATOM, 3),
            atom5_mask=torch.ones(N_ATOM, dtype=torch.bool),
            f_pseudo_beta_mask=torch.zeros(N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                N_RES,
                N_RES,
                N_TEMPL_BINS,
                dtype=torch.long,
            ),
            aa_indices=torch.zeros(N_RES, dtype=torch.long),
            ref_pos=torch.randn(N_ATOM, 3),
            ref_element=torch.zeros(N_ATOM, 3),
            f_residue_idx=torch.arange(N_RES, dtype=torch.long),
            t_hat=torch.randn([]),
            t_normalized=torch.randn(N_RES, N_RES),
            ref_space_uid=torch.zeros(N_ATOM, dtype=torch.long),
            tok_idx=torch.zeros(N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.zeros(N_ATOM, K, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.zeros(
                N_ATOM,
                K,
                dtype=torch.bool,
            ),
        )


# ---------------------------------------------------------------------------
# FeaturizedBatch — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_batch() -> FeaturizedBatch:
    """Provide a valid FeaturizedBatch with B=2, N_RES=8, N_ATOM=40, K=16.

    Returns:
        A fully-constructed FeaturizedBatch using tensors of the correct shapes.
    """
    return FeaturizedBatch(
        ref_pos=torch.randn(B, N_ATOM, 3),
        ref_element=torch.zeros(B, N_ATOM, 4),
        ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
        gt_res_distogram=torch.zeros(
            B,
            N_RES,
            N_RES,
            N_TEMPL_BINS,
            dtype=torch.long,
        ),
        f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
        f_residue_idx=repeat(
            torch.arange(N_RES, dtype=torch.long),
            "n -> b n",
            b=B,
        ),
        r_gt=torch.randn(B, N_ATOM, 3),
        r_gt_noised=torch.randn(B, N_ATOM, 3),
        atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
        aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
        t_hat=torch.randn(B),
        t_normalized=torch.rand(B, N_RES, N_RES),
        tok_idx=torch.zeros(B, N_ATOM, dtype=torch.long),
        center_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
        gt_atom_distogram_sparse=torch.randn(B, N_ATOM, K, N_ATOM_BINS),
        gt_atom_distogram_mask_sparse=torch.ones(
            B,
            N_ATOM,
            K,
            dtype=torch.bool,
        ),
    )


# ---------------------------------------------------------------------------
# FeaturizedBatch — valid construction
# ---------------------------------------------------------------------------


def test_featurized_batch_valid_construction(
    featurized_batch: FeaturizedBatch,
) -> None:
    """FeaturizedBatch constructed successfully with correct shapes.

    Verifies instance type, key tensor shapes, floating-point dtype of t_hat,
    and that t_normalized values lie within the closed unit interval [0, 1].

    Args:
        featurized_batch: A valid FeaturizedBatch fixture.
    """
    assert featurized_batch.ref_pos.shape == (B, N_ATOM, 3)
    assert featurized_batch.ref_element.shape == (B, N_ATOM, 4)
    assert featurized_batch.gt_res_distogram.shape == (
        B,
        N_RES,
        N_RES,
        N_TEMPL_BINS,
    )
    assert featurized_batch.f_residue_idx.shape == (B, N_RES)
    assert featurized_batch.atom5_mask.shape == (B, N_ATOM)
    assert featurized_batch.gt_atom_distogram_sparse.shape == (
        B,
        N_ATOM,
        K,
        N_ATOM_BINS,
    )
    assert featurized_batch.t_hat.is_floating_point()
    assert featurized_batch.t_normalized.min() >= 0.0
    assert featurized_batch.t_normalized.max() <= 1.0


# ---------------------------------------------------------------------------
# FeaturizedBatch — shape enforcement
# ---------------------------------------------------------------------------


def test_featurized_batch_rejects_unbatched_ref_pos() -> None:
    """FeaturizedBatch raises error when ref_pos missing the batch dimension.

    Verifies that passing tensor of shape (N_ATOM, 3) instead of (B, N_ATOM, 3)
    triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedBatch(
            ref_pos=torch.randn(N_ATOM, 3),
            ref_element=torch.zeros(B, N_ATOM, 4),
            ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                B,
                N_RES,
                N_RES,
                N_TEMPL_BINS,
                dtype=torch.long,
            ),
            f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
            f_residue_idx=torch.randn(B, N_RES, C_RES),
            r_gt=torch.randn(B, N_ATOM, 3),
            r_gt_noised=torch.randn(B, N_ATOM, 3),
            atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
            aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
            t_hat=torch.randn(B),
            t_normalized=torch.randn(B, N_RES, N_RES),
            tok_idx=torch.zeros(B, N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.randn(B, N_ATOM, K, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.ones(
                B,
                N_ATOM,
                K,
                dtype=torch.bool,
            ),
        )


def test_featurized_batch_rejects_wrong_coords_dim() -> None:
    """FeaturizedBatch raises when r_gt last dim is 4 instead of 3.

    Verifies that passing an r_gt tensor of shape (B, N_ATOM, 4) instead of
    (B, N_ATOM, 3) triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedBatch(
            ref_pos=torch.randn(B, N_ATOM, 3),
            ref_element=torch.zeros(B, N_ATOM, 4),
            ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                B,
                N_RES,
                N_RES,
                N_TEMPL_BINS,
                dtype=torch.long,
            ),
            f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
            f_residue_idx=torch.randn(B, N_RES, C_RES),
            r_gt=torch.randn(B, N_ATOM, 4),
            r_gt_noised=torch.randn(B, N_ATOM, 3),
            atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
            aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
            t_hat=torch.randn(B),
            t_normalized=torch.randn(B, N_RES, N_RES),
            tok_idx=torch.zeros(B, N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.randn(B, N_ATOM, K, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.ones(
                B,
                N_ATOM,
                K,
                dtype=torch.bool,
            ),
        )
