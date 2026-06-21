"""Tests for data types, featurization utilities, and dataset / data loading.

Covers FeaturizedItem and FeaturizedBatch shape contracts; Distogram shape,
one-hot, masking, symmetry, and bin-assignment properties; featurize_batch
output shapes and value contracts; apply_conditioning_dropout behaviour;
sinusoidal_encoding and ref_pos_for_residue correctness; ProteinDataset
length/indexing; and make_bucketed_data_loaders behaviour.
"""

import dataclasses
import json
import math
import pathlib
import pickle
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn as nn
from architecture.main_trunk import MainTrunk
from einops import rearrange, reduce, repeat
from helpers.atom_utils import RESTYPE_NUM_NO_X, Protein, restype_order
from helpers.data import (
    DatasetSplitsManifest,
    Distogram,
    FeaturizeCollate,
    FeaturizedBatch,
    FeaturizedItem,
    ProteinDataset,
    ProteinShardDataset,
    ShardBudgetParameters,
    ShardDataLoader,
    ShardMetadata,
    apply_conditioning_dropout,
    featurize_batch,
    featurize_single_item,
    make_bucketed_data_loaders,
    ref_pos_for_residue,
    sinusoidal_encoding,
)
from helpers.useful_objects import ModelSetup, manual_seed
from jaxtyping import Bool, Float, TypeCheckError
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from train.train_config import (
    LoaderConfig as EvalLoaderConfig,
)
from train.train_config import (
    TrainArgs,
    TrainConfig,
    TrainLoaderConfig,
)

# ---------------------------------------------------------------------------
# test_batch_types block — FeaturizedItem and FeaturizedBatch shape contracts
# ---------------------------------------------------------------------------

BT_B = 2
BT_N_RES = 8
BT_N_ATOM = BT_N_RES * 5
BT_N_TEMPL_BINS = 38
BT_C_RES = 32
BT_K = 16
BT_N_ATOM_BINS = 16


# ---------------------------------------------------------------------------
# Protein batch — fixtures
# ---------------------------------------------------------------------------


def make_protein() -> Protein:
    """Construct a single Protein with BT_N_RES=8 residues.

    Returns:
        A Protein with randomly-initialised atom_positions and all-ones masks.
    """
    return Protein(
        atom_positions=np.random.default_rng().standard_normal(
            (BT_N_RES, 37, 3),
        ),
        aatype=np.zeros(BT_N_RES, dtype=np.intp),
        atom_mask=np.ones((BT_N_RES, 37), dtype=np.float64),
        residue_index=np.arange(BT_N_RES, dtype=np.intp),
        chain_index=np.zeros(BT_N_RES, dtype=np.intp),
        b_factors=np.zeros((BT_N_RES, 37), dtype=np.float64),
    )


@pytest.fixture
def protein_batch_raw() -> list[Protein]:
    """Provide BT_B=2 valid Protein objects, each with BT_N_RES=8 residues.

    Returns:
        A list of BT_B fully-constructed Protein instances.
    """
    return [make_protein() for _ in range(BT_B)]


# ---------------------------------------------------------------------------
# Protein batch — valid construction
# ---------------------------------------------------------------------------


def test_protein_batch_constructs(protein_batch_raw: list[Protein]) -> None:
    """list[Protein] has BT_B items with the correct types and array shapes.

    Verifies that the fixture produces a list of BT_B Protein instances and that
    each instance carries atom_positions (BT_N_RES, 37, 3), atom_mask
    (BT_N_RES, 37), residue_index (BT_N_RES,), and aatype (BT_N_RES,) arrays
    of the expected shape.
    """
    assert len(protein_batch_raw) == BT_B
    for protein in protein_batch_raw:
        assert isinstance(protein, Protein)
        assert protein.atom_positions.shape == (BT_N_RES, 37, 3)
        assert protein.atom_mask.shape == (BT_N_RES, 37)
        assert protein.residue_index.shape == (BT_N_RES,)
        assert protein.aatype.shape == (BT_N_RES,)


# ---------------------------------------------------------------------------
# Protein batch — shape enforcement
# ---------------------------------------------------------------------------


def test_protein_rejects_2d_atom_positions() -> None:
    """Protein raises when atom_positions is 2-D (missing the per-atom dim).

    Verifies that passing an array of shape (BT_N_RES, 3) instead of
    (BT_N_RES, 37, 3) triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = Protein(
            atom_positions=np.random.default_rng().standard_normal(
                (BT_N_RES, 3),
            ),
            aatype=np.zeros(BT_N_RES, dtype=np.intp),
            atom_mask=np.ones((BT_N_RES, 37), dtype=np.float64),
            residue_index=np.arange(BT_N_RES, dtype=np.intp),
            chain_index=np.zeros(BT_N_RES, dtype=np.intp),
            b_factors=np.zeros((BT_N_RES, 37), dtype=np.float64),
        )


def test_protein_rejects_mismatched_atom_count() -> None:
    """Error raised when atom_positions, atom_mask have different atom counts.

    Verifies that passing atom_positions of shape (BT_N_RES, 36, 3) while
    atom_mask has shape (BT_N_RES, 37) triggers a shape-contract violation
    because the symbolic num_atom_type dimension must be consistent across
    fields.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = Protein(
            atom_positions=np.random.default_rng().standard_normal(
                (BT_N_RES, 36, 3),
            ),
            aatype=np.zeros(BT_N_RES, dtype=np.intp),
            atom_mask=np.ones((BT_N_RES, 37), dtype=np.float64),
            residue_index=np.arange(BT_N_RES, dtype=np.intp),
            chain_index=np.zeros(BT_N_RES, dtype=np.intp),
            b_factors=np.zeros((BT_N_RES, 37), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# FeaturizedItem — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_item_raw() -> FeaturizedItem:
    """Provide a valid FeaturizedItem with BT_N_RES=8, BT_N_ATOM=40.

    Returns:
        A fully-constructed FeaturizedItem using tensors of the correct shapes.
    """
    return FeaturizedItem(
        r_gt=torch.randn(BT_N_ATOM, 3),
        atom5_mask=torch.ones(BT_N_ATOM, dtype=torch.bool),
        f_pseudo_beta_mask=torch.zeros(BT_N_RES, dtype=torch.long),
        gt_res_distogram=torch.zeros(
            BT_N_RES,
            BT_N_RES,
            BT_N_TEMPL_BINS,
            dtype=torch.long,
        ),
        aa_indices=torch.zeros(BT_N_RES, dtype=torch.long),
        ref_pos=torch.randn(BT_N_ATOM, 3),
        ref_element=torch.zeros(BT_N_ATOM, 4),
        f_residue_idx=torch.arange(BT_N_RES, dtype=torch.long),
        t_hat=torch.randn([]),
        t_normalized=torch.randn(BT_N_RES, BT_N_RES),
        ref_space_uid=torch.arange(
            BT_N_RES,
            dtype=torch.long,
        ).repeat_interleave(5),
        tok_idx=torch.arange(BT_N_RES, dtype=torch.long).repeat_interleave(5),
        center_uid=(
            torch.arange(BT_N_RES, dtype=torch.long) * 5 + 1
        ).repeat_interleave(5),
        gt_atom_distogram_sparse=torch.zeros(BT_N_ATOM, BT_K, BT_N_ATOM_BINS),
        gt_atom_distogram_mask_sparse=torch.ones(
            BT_N_ATOM,
            BT_K,
            dtype=torch.bool,
        ),
    )


# ---------------------------------------------------------------------------
# FeaturizedItem — valid construction
# ---------------------------------------------------------------------------


def test_featurized_item_constructs(
    featurized_item_raw: FeaturizedItem,
) -> None:
    """FeaturizedItem constructed successfully with correct types and shapes.

    Verifies that the fixture produces a FeaturizedItem instance and that key
    tensors retain expected shapes: r_gt (BT_N_ATOM, 3), atom5_mask
    (BT_N_ATOM,), gt_res_distogram (BT_N_RES, BT_N_RES, BT_N_TEMPL_BINS),
    ref_pos (BT_N_ATOM, 3), ref_element (BT_N_ATOM, 4), and f_residue_idx
    (BT_N_RES,).
    """
    assert isinstance(featurized_item_raw, FeaturizedItem)
    assert featurized_item_raw.r_gt.shape == (BT_N_ATOM, 3)
    assert featurized_item_raw.atom5_mask.shape == (BT_N_ATOM,)
    assert featurized_item_raw.gt_res_distogram.shape == (
        BT_N_RES,
        BT_N_RES,
        BT_N_TEMPL_BINS,
    )
    assert featurized_item_raw.ref_pos.shape == (BT_N_ATOM, 3)
    assert featurized_item_raw.ref_element.shape == (BT_N_ATOM, 4)
    assert featurized_item_raw.f_residue_idx.shape == (BT_N_RES,)


# ---------------------------------------------------------------------------
# FeaturizedItem — shape enforcement
# ---------------------------------------------------------------------------


def test_featurized_item_rejects_1d_r_gt() -> None:
    """FeaturizedItem raises when r_gt is 1-D instead of (BT_N_ATOM, 3).

    Verifies that passing a tensor of shape (BT_N_ATOM,) instead of
    (BT_N_ATOM, 3) triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedItem(
            r_gt=torch.randn(BT_N_ATOM),
            atom5_mask=torch.ones(BT_N_ATOM, dtype=torch.bool),
            f_pseudo_beta_mask=torch.zeros(BT_N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                BT_N_RES,
                BT_N_RES,
                BT_N_TEMPL_BINS,
                dtype=torch.long,
            ),
            aa_indices=torch.zeros(BT_N_RES, dtype=torch.long),
            ref_pos=torch.randn(BT_N_ATOM, 3),
            ref_element=torch.zeros(BT_N_ATOM, 4),
            f_residue_idx=torch.arange(BT_N_RES, dtype=torch.long),
            t_hat=torch.randn([]),
            t_normalized=torch.randn(BT_N_RES, BT_N_RES),
            ref_space_uid=torch.zeros(BT_N_ATOM, dtype=torch.long),
            tok_idx=torch.zeros(BT_N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(BT_N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.zeros(
                BT_N_ATOM,
                BT_K,
                BT_N_ATOM_BINS,
            ),
            gt_atom_distogram_mask_sparse=torch.zeros(
                BT_N_ATOM,
                BT_K,
                dtype=torch.bool,
            ),
        )


def test_featurized_item_rejects_wrong_ref_element_dim() -> None:
    """FeaturizedItem raises when ref_element last dim is 3 instead of 4.

    Verifies that passing a tensor of shape (BT_N_ATOM, 3) instead of
    (BT_N_ATOM, 4) triggers a shape-contract violation because exactly four
    element classes are required.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedItem(
            r_gt=torch.randn(BT_N_ATOM, 3),
            atom5_mask=torch.ones(BT_N_ATOM, dtype=torch.bool),
            f_pseudo_beta_mask=torch.zeros(BT_N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                BT_N_RES,
                BT_N_RES,
                BT_N_TEMPL_BINS,
                dtype=torch.long,
            ),
            aa_indices=torch.zeros(BT_N_RES, dtype=torch.long),
            ref_pos=torch.randn(BT_N_ATOM, 3),
            ref_element=torch.zeros(BT_N_ATOM, 3),
            f_residue_idx=torch.arange(BT_N_RES, dtype=torch.long),
            t_hat=torch.randn([]),
            t_normalized=torch.randn(BT_N_RES, BT_N_RES),
            ref_space_uid=torch.zeros(BT_N_ATOM, dtype=torch.long),
            tok_idx=torch.zeros(BT_N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(BT_N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.zeros(
                BT_N_ATOM,
                BT_K,
                BT_N_ATOM_BINS,
            ),
            gt_atom_distogram_mask_sparse=torch.zeros(
                BT_N_ATOM,
                BT_K,
                dtype=torch.bool,
            ),
        )


# ---------------------------------------------------------------------------
# FeaturizedBatch — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_batch_raw() -> FeaturizedBatch:
    """Provide a valid FeaturizedBatch with BT_B=2, BT_N_RES=8, BT_N_ATOM=40.

    Returns:
        A fully-constructed FeaturizedBatch using tensors of the correct shapes.
    """
    return FeaturizedBatch(
        ref_pos=torch.randn(BT_B, BT_N_ATOM, 3),
        ref_element=torch.zeros(BT_B, BT_N_ATOM, 4),
        ref_space_uid=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
        gt_res_distogram=torch.zeros(
            BT_B,
            BT_N_RES,
            BT_N_RES,
            BT_N_TEMPL_BINS,
            dtype=torch.long,
        ),
        f_pseudo_beta_mask=torch.zeros(BT_B, BT_N_RES, dtype=torch.long),
        f_residue_idx=repeat(
            torch.arange(BT_N_RES, dtype=torch.long),
            "n -> b n",
            b=BT_B,
        ),
        r_gt=torch.randn(BT_B, BT_N_ATOM, 3),
        r_gt_noised=torch.randn(BT_B, BT_N_ATOM, 3),
        atom5_mask=torch.ones(BT_B, BT_N_ATOM, dtype=torch.bool),
        aa_indices=torch.zeros(BT_B, BT_N_RES, dtype=torch.long),
        t_hat=torch.randn(BT_B),
        t_normalized=torch.rand(BT_B, BT_N_RES, BT_N_RES),
        tok_idx=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
        center_uid=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
        gt_atom_distogram_sparse=torch.randn(
            BT_B,
            BT_N_ATOM,
            BT_K,
            BT_N_ATOM_BINS,
        ),
        gt_atom_distogram_mask_sparse=torch.ones(
            BT_B,
            BT_N_ATOM,
            BT_K,
            dtype=torch.bool,
        ),
    )


# ---------------------------------------------------------------------------
# FeaturizedBatch — valid construction
# ---------------------------------------------------------------------------


def test_featurized_batch_valid_construction(
    featurized_batch_raw: FeaturizedBatch,
) -> None:
    """FeaturizedBatch constructed successfully with correct shapes.

    Verifies instance type, key tensor shapes, floating-point dtype of t_hat,
    and that t_normalized values lie within the closed unit interval [0, 1].

    Args:
        featurized_batch_raw: A valid FeaturizedBatch fixture.
    """
    assert featurized_batch_raw.ref_pos.shape == (BT_B, BT_N_ATOM, 3)
    assert featurized_batch_raw.ref_element.shape == (BT_B, BT_N_ATOM, 4)
    assert featurized_batch_raw.gt_res_distogram.shape == (
        BT_B,
        BT_N_RES,
        BT_N_RES,
        BT_N_TEMPL_BINS,
    )
    assert featurized_batch_raw.f_residue_idx.shape == (BT_B, BT_N_RES)
    assert featurized_batch_raw.atom5_mask.shape == (BT_B, BT_N_ATOM)
    assert featurized_batch_raw.gt_atom_distogram_sparse.shape == (
        BT_B,
        BT_N_ATOM,
        BT_K,
        BT_N_ATOM_BINS,
    )
    assert featurized_batch_raw.t_hat.is_floating_point()
    assert featurized_batch_raw.t_normalized.min() >= 0.0
    assert featurized_batch_raw.t_normalized.max() <= 1.0


# ---------------------------------------------------------------------------
# FeaturizedBatch — shape enforcement
# ---------------------------------------------------------------------------


def test_featurized_batch_rejects_unbatched_ref_pos() -> None:
    """FeaturizedBatch raises error when ref_pos missing the batch dimension.

    Verifies that passing tensor of shape (BT_N_ATOM, 3) instead of
    (BT_B, BT_N_ATOM, 3) triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedBatch(
            ref_pos=torch.randn(BT_N_ATOM, 3),
            ref_element=torch.zeros(BT_B, BT_N_ATOM, 4),
            ref_space_uid=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                BT_B,
                BT_N_RES,
                BT_N_RES,
                BT_N_TEMPL_BINS,
                dtype=torch.long,
            ),
            f_pseudo_beta_mask=torch.zeros(BT_B, BT_N_RES, dtype=torch.long),
            f_residue_idx=torch.randn(BT_B, BT_N_RES, BT_C_RES),
            r_gt=torch.randn(BT_B, BT_N_ATOM, 3),
            r_gt_noised=torch.randn(BT_B, BT_N_ATOM, 3),
            atom5_mask=torch.ones(BT_B, BT_N_ATOM, dtype=torch.bool),
            aa_indices=torch.zeros(BT_B, BT_N_RES, dtype=torch.long),
            t_hat=torch.randn(BT_B),
            t_normalized=torch.randn(BT_B, BT_N_RES, BT_N_RES),
            tok_idx=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.randn(
                BT_B,
                BT_N_ATOM,
                BT_K,
                BT_N_ATOM_BINS,
            ),
            gt_atom_distogram_mask_sparse=torch.ones(
                BT_B,
                BT_N_ATOM,
                BT_K,
                dtype=torch.bool,
            ),
        )


def test_featurized_batch_rejects_wrong_coords_dim() -> None:
    """FeaturizedBatch raises when r_gt last dim is 4 instead of 3.

    Verifies that passing an r_gt tensor of shape (BT_B, BT_N_ATOM, 4) instead
    of (BT_B, BT_N_ATOM, 3) triggers a shape-contract violation.
    """
    with pytest.raises((TypeCheckError, Exception)):
        _ = FeaturizedBatch(
            ref_pos=torch.randn(BT_B, BT_N_ATOM, 3),
            ref_element=torch.zeros(BT_B, BT_N_ATOM, 4),
            ref_space_uid=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
            gt_res_distogram=torch.zeros(
                BT_B,
                BT_N_RES,
                BT_N_RES,
                BT_N_TEMPL_BINS,
                dtype=torch.long,
            ),
            f_pseudo_beta_mask=torch.zeros(BT_B, BT_N_RES, dtype=torch.long),
            f_residue_idx=torch.randn(BT_B, BT_N_RES, BT_C_RES),
            r_gt=torch.randn(BT_B, BT_N_ATOM, 4),
            r_gt_noised=torch.randn(BT_B, BT_N_ATOM, 3),
            atom5_mask=torch.ones(BT_B, BT_N_ATOM, dtype=torch.bool),
            aa_indices=torch.zeros(BT_B, BT_N_RES, dtype=torch.long),
            t_hat=torch.randn(BT_B),
            t_normalized=torch.randn(BT_B, BT_N_RES, BT_N_RES),
            tok_idx=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
            center_uid=torch.zeros(BT_B, BT_N_ATOM, dtype=torch.long),
            gt_atom_distogram_sparse=torch.randn(
                BT_B,
                BT_N_ATOM,
                BT_K,
                BT_N_ATOM_BINS,
            ),
            gt_atom_distogram_mask_sparse=torch.ones(
                BT_B,
                BT_N_ATOM,
                BT_K,
                dtype=torch.bool,
            ),
        )


# ---------------------------------------------------------------------------
# test_featurize block — Distogram, featurize_batch, and related utilities
# ---------------------------------------------------------------------------

_ = manual_seed(42)

FZ_B = 2
N_RES = 12
N_BINS = 16
C_RES = 32
MIN_DIST = 2.0
MAX_DIST = 22.0
AA_SEQ = "ACDEFGHIKLMN"  # length N_RES
TOLERANCE = 1e-6
X_TOKEN = RESTYPE_NUM_NO_X

# ---------------------------------------------------------------------------
# restype order and count — X as mask token
# ---------------------------------------------------------------------------


def test_restype_order_x_is_20() -> None:
    """Mask token 'X' is assigned index 20, one past the 20 canon amino acids.

    Verifies that the mask/unknown token always occupies a fixed slot distinct
    from the 20 canonical residue indices so downstream one-hot encodings are
    consistent.
    """
    assert restype_order["X"] == X_TOKEN


# ---------------------------------------------------------------------------
# Fixtures — models
# ---------------------------------------------------------------------------


@pytest.fixture
def disto() -> Distogram:
    """Provide a Distogram without overflow bin in eval mode.

    Uses N_BINS bins spanning [MIN_DIST, MAX_DIST]; pairs beyond max_dist are
    masked rather than captured in an overflow bin.
    """
    return Distogram(
        n_bins=N_BINS,
        overflow_bin=False,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
    ).eval()


@pytest.fixture
def disto_overflow() -> Distogram:
    """Provide a Distogram with overflow bin in eval mode.

    Uses N_BINS bins spanning [MIN_DIST, MAX_DIST] plus one extra overflow bin
    that captures pairs beyond max_dist; the pair mask is never gated by
    distance.
    """
    return Distogram(
        n_bins=N_BINS,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        overflow_bin=True,
    ).eval()


# ---------------------------------------------------------------------------
# Fixtures — tensors
# ---------------------------------------------------------------------------


@pytest.fixture
def coords() -> Float[torch.Tensor, "N_res 3"]:
    """Provide random single-chain coordinates (N_RES, 3).

    Drawn from a standard normal distribution; used as representative
    real-valued residue positions for Distogram forward-pass tests.
    """
    return torch.randn(N_RES, 3)


@pytest.fixture
def coords_batch() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random batched coordinates (2, N_RES, 3).

    Drawn from a standard normal distribution; shaped for tests that require a
    leading batch dimension.
    """
    return torch.randn(2, N_RES, 3)


@pytest.fixture
def mask() -> Bool[torch.Tensor, "N_res"]:
    """Provide an all-True residue mask (N_RES,).

    Indicates that every residue slot is valid; used as a baseline mask fixture
    before individual tests override specific entries.
    """
    return torch.ones(N_RES, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Distogram — overflow bin shape
# ---------------------------------------------------------------------------


def test_distogram_overflow_bin_output_shape(
    disto_overflow: Distogram,
    coords: Float[torch.Tensor, "N_res 3"],
) -> None:
    """Distogram with overflow bin outputs shape (N_RES, N_RES, N_BINS + 1).

    Verifies that enabling overflow_bin appends exactly one extra bin to the
    last dimension, giving N_BINS + 1 total bins.
    """
    with torch.no_grad():
        f, _ = disto_overflow(coords)
    assert f.shape == (N_RES, N_RES, N_BINS + 1)


# ---------------------------------------------------------------------------
# Distogram — one-hot property
# ---------------------------------------------------------------------------


def test_distogram_one_hot_sums_to_one(
    disto: Distogram,
    coords: Float[torch.Tensor, "N_res 3"],
) -> None:
    """Distogram bin probabilities sum to 1 for every residue pair.

    Verifies that the soft-one-hot output from the standard (no-overflow)
    Distogram is a valid probability distribution over bins for each (i, j)
    pair.
    """
    with torch.no_grad():
        f, _ = disto(coords)
    bin_sums = reduce(f, "i j b -> i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


def test_distogram_overflow_one_hot_sums_to_one(
    disto_overflow: Distogram,
    coords: Float[torch.Tensor, "N_res 3"],
) -> None:
    """Overflow-bin Distogram probabilities also sum to 1 for every pair.

    Verifies that adding overflow bin does not break the one-hot normalization
    property — the N_BINS + 1 bin outputs still form a valid probability
    distribution.
    """
    with torch.no_grad():
        f, _ = disto_overflow(coords)
    bin_sums = reduce(f, "i j b -> i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


# ---------------------------------------------------------------------------
# Distogram — pair mask
# ---------------------------------------------------------------------------


def test_distogram_mask_none_gives_all_true_within_range(
    disto: Distogram,
) -> None:
    """Without a mask, every residue pair within max_dist is unmasked.

    Verifies the default behaviour when no coords_mask is supplied: all pairs
    whose distance is below max_dist should appear as True in the returned mask.
    """
    c = torch.zeros(N_RES, 3)
    with torch.no_grad():
        _, m = disto(c)
    assert m.all()


def test_distogram_mask_zeros_out_invalid_residues(
    mask: Bool[torch.Tensor, "N_res"],
    disto: Distogram,
    coords: Float[torch.Tensor, "N_res 3"],
) -> None:
    """Distogram masks row 0 and column 0 when residue 0 is excluded from mask.

    Verifies that setting a residue as invalid (False in coords_mask) zeros out
    both its row and column in the pair mask, while leaving all other pairs
    intact.
    """
    mask[0] = False
    with torch.no_grad():
        _, m = disto(coords, mask)
    assert not m[0, :].any()
    assert not m[:, 0].any()
    assert m[1:, 1:].any()


def test_distogram_overflow_mask_ignores_distance_cutoff(
    disto_overflow: Distogram,
) -> None:
    """With overflow_bin=True the pair mask is not gated by a distance cutoff.

    Verifies that even when one residue is placed very far from all others, the
    overflow-bin Distogram marks all pairs as valid because distant pairs are
    captured by the overflow bin rather than being masked out.
    """
    c = torch.zeros(N_RES, 3)
    c[0] = 1000.0
    with torch.no_grad():
        _, m = disto_overflow(c)
    assert m.all()


def test_distogram_no_overflow_masks_distant_pairs(disto: Distogram) -> None:
    """Without overflow_bin, pairs beyond max_dist are set to False in the mask.

    Verifies that the standard Distogram (no overflow bin) applies a distance
    cutoff: pairs whose Euclidean distance exceeds max_dist are masked to False.
    """
    c = torch.zeros(N_RES, 3)
    c[0] = MAX_DIST * 2
    with torch.no_grad():
        _, m = disto(c)
    assert not m[0, 1:].any()


# ---------------------------------------------------------------------------
# Distogram — symmetry
# ---------------------------------------------------------------------------


def test_distogram_is_symmetric(
    disto: Distogram,
    coords: Float[torch.Tensor, "N_res 3"],
) -> None:
    """Distogram output is symmetric: f[i,j] == f[j,i] and m[i,j] == m[j,i].

    Verifies that both the bin-probability tensor and the pair-mask tensor are
    symmetric matrices, as required by the pairwise distance formulation.
    """
    with torch.no_grad():
        f, m = disto(coords)
    assert torch.allclose(f, rearrange(f, "i j b -> j i b"))
    assert torch.equal(m, rearrange(m, "i j -> j i"))


# ---------------------------------------------------------------------------
# Distogram — bin correctness
# ---------------------------------------------------------------------------


def test_distogram_close_pairs_land_in_first_bin(disto: Distogram) -> None:
    """Pairs at distance 0 (all at origin) are assigned to first distance bin.

    Verifies bin-assignment correctness at the lower boundary: when all
    residues share the same position, every pairwise distance is 0 which
    falls in bin 0.
    """
    c = torch.zeros(N_RES, 3)
    with torch.no_grad():
        f, _ = disto(c)
    assert f[..., 0].all()


def test_distogram_overflow_far_pairs_land_in_last_bin(
    disto_overflow: Distogram,
) -> None:
    """Pairs beyond max_dist land in the overflow bin when overflow_bin=True.

    Verifies that cross-group pairs whose distance vastly exceeds max_dist are
    assigned entirely to the last (overflow) bin index.
    """
    c = torch.zeros(N_RES, 3)
    c[N_RES // 2 :] = MAX_DIST * 10
    with torch.no_grad():
        f, _ = disto_overflow(c)
    cross = f[: N_RES // 2, N_RES // 2 :, :]
    assert cross[..., -1].all()


def test_distogram_exact_bin_for_known_interior_distance(
    disto: Distogram,
) -> None:
    """Distogram assigns a 9 Å pair to bin 5 given bin_width=(22-2)/16=1.25 Å.

    Verifies exact bin-index arithmetic: for a known distance of 9.0 Å,
    floor((9.0 - 2.0) / 1.25) = 5, so bin 5 should be 1.0 and all others 0.0.
    """
    c = torch.zeros(N_RES, 3)
    c[0, 0] = 9.0
    expected_bin = 5
    with torch.no_grad():
        f, _ = disto(c)
    assert math.isclose(f[0, 1, expected_bin].item(), 1.0)
    assert f[0, 1, :expected_bin].abs().max().item() < TOLERANCE
    assert f[0, 1, expected_bin + 1 :].abs().max().item() < TOLERANCE


# ---------------------------------------------------------------------------
# featurize_batch — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_protein() -> Protein:
    """Provide a single Protein with random coordinates of length N_RES."""
    rng = np.random.default_rng(42)
    return Protein(
        atom_positions=rng.standard_normal((N_RES, 37, 3)),
        aatype=np.array([restype_order[aa] for aa in AA_SEQ], dtype=np.intp),
        atom_mask=np.ones((N_RES, 37), dtype=np.float64),
        residue_index=np.arange(N_RES, dtype=np.intp),
        chain_index=np.zeros(N_RES, dtype=np.intp),
        b_factors=np.zeros((N_RES, 37), dtype=np.float64),
    )


@pytest.fixture
def featurize_protein_batch(single_protein: Protein) -> list[Protein]:
    """Provide FZ_B=2 identical Protein objects for batch featurization.

    Args:
        single_protein: A single Protein used to fill the batch.

    Returns:
        A list of FZ_B identical Protein instances.
    """
    return [single_protein for _ in range(FZ_B)]


@pytest.fixture
def tcfg() -> TrainConfig:
    """Provide a default TrainConfig with standard settings."""
    return TrainConfig()


@pytest.fixture
def c_beta_distogram_fn(tcfg: TrainConfig) -> Distogram:
    """Provide residue-level Cβ Distogram from tcfg.distogram_res."""
    dr = tcfg.distogram_res
    return Distogram(
        n_bins=dr.n_bins - 1,
        min_dist=dr.min_dist,
        max_dist=dr.max_dist,
        overflow_bin=True,
    ).eval()


@pytest.fixture
def atom_distogram_fn(tcfg: TrainConfig) -> Distogram:
    """Provide the atom-level Distogram configured from tcfg.distogram_atom."""
    da = tcfg.distogram_atom
    return Distogram(
        n_bins=da.n_bins,
        overflow_bin=False,
        min_dist=da.min_dist,
        max_dist=da.max_dist,
    ).eval()


@pytest.fixture
def model_params(
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> ModelSetup:
    """Provide ModelSetup bundling MainTrunk with training configuration."""
    trunk: MainTrunk = MainTrunk(
        model_params=tcfg.model,
        res_distogram_params=tcfg.distogram_res,
        atom_distogram_params=tcfg.distogram_atom,
        noise_params=tcfg.noise,
    ).eval()
    optimizer = Adam(trunk.parameters(), lr=tcfg.training.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg.training.num_epochs)
    return ModelSetup(
        model=trunk,
        tcfg=tcfg,
        distogram_res=c_beta_distogram_fn,
        distogram_atom=atom_distogram_fn,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=scheduler,
    )


@pytest.fixture
def featurized_item(
    single_protein: Protein,
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> FeaturizedItem:
    """FeaturizedItem produced by featurize_single_item on single_protein.

    Sets a fixed random seed so that the log-normal noise sample lands within
    [sigma_min, sigma_max], keeping t_normalized in [0, 1] regardless of prior
    RNG state.
    """
    _ = manual_seed(1)
    return featurize_single_item(
        prot=single_protein,
        c_beta_distogram_fn=c_beta_distogram_fn,
        atom_distogram_fn=atom_distogram_fn,
        noise_params=tcfg.noise,
        max_seq_len_in_batch=N_RES,
        window_size=tcfg.model.window_size,
    )


@pytest.fixture
def featurized_batch(
    featurize_protein_batch: list[Protein],
    model_params: ModelSetup,
) -> FeaturizedBatch:
    """FeaturizedBatch produced by featurize_batch on featurize_protein_batch.

    Sets a fixed random seed so that the log-normal noise sample lands within
    [sigma_min, sigma_max], keeping t_normalized in [0, 1] regardless of prior
    RNG state.
    """
    _ = manual_seed(1)
    return featurize_batch(
        batch=featurize_protein_batch,
        tcfg=model_params.tcfg,
        distogram_res=model_params.distogram_res,
        distogram_atom=model_params.distogram_atom,
    )


# ---------------------------------------------------------------------------
# featurize_batch — output shapes
# Batched layout for FZ_B=2, N_RES=12: tensors are (FZ_B, N_RES, *) or
# (FZ_B, N_ATOM, *)
# ---------------------------------------------------------------------------

N_ATOM = N_RES * 5  # 5 atoms per residue, no separators


def test_featurize_batch_output_shapes(
    featurized_batch: FeaturizedBatch,
) -> None:
    """featurize_batch produces correct shapes and values for all output fields.

    Batched layout for FZ_B=2, N_RES=12: tensors are (FZ_B, N_RES, *) or
    (FZ_B, N_ATOM, *). Checks shapes, finiteness, noise-level bounds, and
    index mapping invariants.
    """
    expected_shapes: dict[str, tuple[int, ...]] = {
        "ref_pos": (FZ_B, N_ATOM, 3),
        "ref_element": (FZ_B, N_ATOM, 4),
        "ref_space_uid": (FZ_B, N_ATOM),
        "gt_res_distogram": (FZ_B, N_RES, N_RES, 39),
        "f_pseudo_beta_mask": (FZ_B, N_RES),
        "tok_idx": (FZ_B, N_ATOM),
        "center_uid": (FZ_B, N_ATOM),
    }
    batch_dict: dict[str, object] = dataclasses.asdict(featurized_batch)
    assert {
        k: tuple(cast(torch.Tensor, batch_dict[k]).shape)
        for k in expected_shapes
    } == expected_shapes

    for field_name, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(
                val.float(),
            ).all(), f"non-finite in field '{field_name}'"

    assert all(
        [
            bool((featurized_batch.t_hat > 0.0).all()),
            bool((featurized_batch.t_normalized >= 0.0).all()),
            bool((featurized_batch.t_normalized <= 1.0).all()),
        ],
    )

    expected_res_idx = repeat(
        torch.arange(N_RES),
        "r -> b (r five)",
        b=FZ_B,
        five=5,
    )
    assert torch.equal(featurized_batch.ref_space_uid, expected_res_idx)
    assert torch.equal(featurized_batch.tok_idx, expected_res_idx)

    expected_ca = repeat(
        (torch.arange(N_RES) * 5 + 1).repeat_interleave(5),
        "a -> b a",
        b=FZ_B,
    )
    assert torch.equal(featurized_batch.center_uid, expected_ca)


# ---------------------------------------------------------------------------
# featurize_single_item — output shapes
# Unbatched layout for N_RES=12: tensors are (N_RES, *) or (N_ATOM, *)
# ---------------------------------------------------------------------------


def test_featurize_single_item_output_shapes(
    featurized_item: FeaturizedItem,
) -> None:
    """featurize_single_item produces correct shapes and values for all fields.

    Unbatched layout for N_RES=12: tensors are (N_RES, *) or (N_ATOM, *).
    Checks shapes, finiteness, noise-level bounds, and index mapping invariants.
    """
    expected_shapes: dict[str, tuple[int, ...]] = {
        "ref_pos": (N_ATOM, 3),
        "ref_element": (N_ATOM, 4),
        "ref_space_uid": (N_ATOM,),
        "gt_res_distogram": (N_RES, N_RES, 39),
        "f_pseudo_beta_mask": (N_RES,),
        "tok_idx": (N_ATOM,),
        "center_uid": (N_ATOM,),
    }
    item_dict: dict[str, object] = dataclasses.asdict(featurized_item)
    assert {
        k: tuple(cast(torch.Tensor, item_dict[k]).shape)
        for k in expected_shapes
    } == expected_shapes

    for field_name, val in item_dict.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(
                val.float(),
            ).all(), f"non-finite in field '{field_name}'"

    assert all(
        [
            bool(featurized_item.t_hat > 0.0),
            bool((featurized_item.t_normalized >= 0.0).all()),
            bool((featurized_item.t_normalized <= 1.0).all()),
        ],
    )

    expected_res_idx = torch.arange(N_RES).repeat_interleave(5)
    assert torch.equal(featurized_item.ref_space_uid, expected_res_idx)
    assert torch.equal(featurized_item.tok_idx, expected_res_idx)

    expected_ca = (torch.arange(N_RES) * 5 + 1).repeat_interleave(5)
    assert torch.equal(featurized_item.center_uid, expected_ca)


# ---------------------------------------------------------------------------
# featurize_batch — Protein type enforcement
# ---------------------------------------------------------------------------


def test_featurize_batch_rejects_wrong_atom_positions_rank() -> None:
    """Protein raises when atom_positions is 2-D instead of the required 3-D."""
    with pytest.raises((TypeError, Exception)):
        _ = Protein(
            atom_positions=np.random.default_rng().standard_normal(
                (N_RES, 37),
            ),
            aatype=np.array(
                [restype_order[aa] for aa in AA_SEQ],
                dtype=np.intp,
            ),
            atom_mask=np.ones((N_RES, 37), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, 37), dtype=np.float64),
        )


def test_featurize_batch_rejects_wrong_atom_count() -> None:
    """Protein raises when atom_positions has 36 atoms instead of 37."""
    with pytest.raises((TypeError, Exception)):
        _ = Protein(
            atom_positions=np.random.default_rng().standard_normal(
                (N_RES, 36, 3),
            ),
            aatype=np.array(
                [restype_order[aa] for aa in AA_SEQ],
                dtype=np.intp,
            ),
            atom_mask=np.ones((N_RES, 37), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, 37), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# FeaturizeCollate
# ---------------------------------------------------------------------------


def test_featurize_collate_returns_featurized_batch(
    featurize_protein_batch: list[Protein],
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> None:
    """FeaturizeCollate.__call__ returns a FeaturizedBatch."""
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_res=c_beta_distogram_fn,
        distogram_atom=atom_distogram_fn,
    )
    result = collate(featurize_protein_batch)
    assert isinstance(result, FeaturizedBatch)


def test_featurize_collate_is_picklable(
    featurize_protein_batch: list[Protein],
    tcfg: TrainConfig,
    c_beta_distogram_fn: Distogram,
    atom_distogram_fn: Distogram,
) -> None:
    """FeaturizeCollate survives a pickle round-trip for num_workers > 0."""
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_res=c_beta_distogram_fn,
        distogram_atom=atom_distogram_fn,
    )
    restored = cast(
        FeaturizeCollate,
        pickle.loads(pickle.dumps(collate)),  # noqa: S301
    )
    result = restored(featurize_protein_batch)
    assert isinstance(result, FeaturizedBatch)


# ---------------------------------------------------------------------------
# apply_conditioning_dropout
# ---------------------------------------------------------------------------


def test_conditioning_dropout_p1_distogram_zeroes_all(
    featurized_batch: FeaturizedBatch,
) -> None:
    """p_distogram=1.0 zeros entire distogram and mask for valid residues."""
    out = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=1.0,
        p_atom=0.0,
        p_seq=0.0,
        device="cpu",
    )
    assert out.gt_res_distogram.sum() == 0
    assert out.f_pseudo_beta_mask.sum() == 0


def test_conditioning_dropout_p1_atom_zeroes_all(
    featurized_batch: FeaturizedBatch,
) -> None:
    """p_atom=1.0 clears atom5_mask for all valid residues."""
    out = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=0.0,
        p_atom=1.0,
        p_seq=0.0,
        device="cpu",
    )
    assert not out.atom5_mask.any()


def test_conditioning_dropout_p1_seq_sets_all_to_mask_token(
    featurized_batch: FeaturizedBatch,
) -> None:
    """p_seq=1.0 replaces all valid amino-acid indices with mask token (20)."""
    out = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=0.0,
        p_atom=0.0,
        p_seq=1.0,
        device="cpu",
    )
    valid = featurized_batch.f_pseudo_beta_mask.bool()
    assert (out.aa_indices[valid] == X_TOKEN).all()


def test_conditioning_dropout_p0_is_noop(
    featurized_batch: FeaturizedBatch,
) -> None:
    """Dropout at 0 leaves distogram, atom mask, sequence unchanged."""
    out = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=0.0,
        p_atom=0.0,
        p_seq=0.0,
        device="cpu",
    )
    assert torch.equal(out.gt_res_distogram, featurized_batch.gt_res_distogram)
    assert torch.equal(out.atom5_mask, featurized_batch.atom5_mask)
    assert torch.equal(out.aa_indices, featurized_batch.aa_indices)


def test_conditioning_dropout_distogram_symmetric(
    featurized_batch: FeaturizedBatch,
) -> None:
    """Distogram dropout zeros both row and column for each dropped residue."""
    _ = manual_seed(0)
    out = apply_conditioning_dropout(
        featurized_batch,
        p_distogram=0.5,
        p_atom=0.0,
        p_seq=0.0,
        device="cpu",
    )
    row_sums = out.gt_res_distogram.sum(dim=(2, 3))
    col_sums = out.gt_res_distogram.sum(dim=(1, 3))
    assert torch.equal(row_sums == 0, col_sums == 0)


def test_conditioning_dropout_respects_residue_mask(
    featurized_batch: FeaturizedBatch,
) -> None:
    """Conditioning dropout never modifies padding residues."""
    batch_with_padding = dataclasses.replace(
        featurized_batch,
        f_pseudo_beta_mask=torch.zeros_like(
            featurized_batch.f_pseudo_beta_mask,
        ),
    )
    out = apply_conditioning_dropout(
        batch_with_padding,
        p_distogram=1.0,
        p_atom=1.0,
        p_seq=1.0,
        device="cpu",
    )
    assert torch.equal(out.aa_indices, batch_with_padding.aa_indices)


# ---------------------------------------------------------------------------
# sinusoidal_encoding, ref_pos_for_residue, featurize_single_item
# ---------------------------------------------------------------------------


def test_sinusoidal_encoding_output_shape_and_shape() -> None:
    """Maps (batch, N_res) indices to a finite (batch, N_res, dim) tensor."""
    positions = rearrange(torch.arange(N_RES).float(), "n -> 1 n")
    out = sinusoidal_encoding(positions, dim=32)
    assert out.shape == (1, N_RES, 32)
    assert torch.isfinite(out).all()


def test_ref_pos_for_residue_output_shape_and_finite() -> None:
    """ref_pos_for_residue returns finite (5, 3) reference position tensor."""
    pos = ref_pos_for_residue("ALA")
    assert pos.shape == (5, 3)
    assert torch.isfinite(pos).all()


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_distogram_forward_wrong_shape(disto: Distogram) -> None:
    """Wrong coords last dim (4 instead of 3) triggers TypeCheckError."""
    coords_bad = torch.zeros(N_RES, 4)
    with pytest.raises(TypeCheckError):
        _ = disto(coords_bad)


def test_sinusoidal_encoding_wrong_shape() -> None:
    """Wrong positions ndim (3-D instead of 2-D) triggers TypeCheckError."""
    positions_bad = torch.zeros(FZ_B, N_RES, 1)
    with pytest.raises(TypeCheckError):
        _ = sinusoidal_encoding(positions_bad)


# ---------------------------------------------------------------------------
# test_data block — ProteinDataset, ProteinShardDataset, data loaders
# ---------------------------------------------------------------------------

_N_RES_DATA = 6  # residues per synthetic entry
_MAX_SEQ = 8  # padded / truncated to this length
_ENTRY_NAMES = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES = ["1aa.A", "2bb.A", "3cc.A"]
_VAL_NAMES = ["4dd.A"]
_TEST_NAMES = ["5ee.A"]
_N_DEBUG = 252  # debug_run sampler uses SubsetRandomSampler(range(252))
PROT_1_LEN = 8
PROT_2_LEN = 16
PROT_3_LEN = 24
PROT_4_LEN = 32
PROT_5_LEN = 40
B = 5
MAX_SEQ_LENGTH = 128


def _make_coords(n: int) -> Mapping[str, list[list[float]]]:
    """Return synthetic backbone coordinates for N, CA, C, O atoms.

    Args:
        n: Number of residues to generate coordinates for.

    Returns:
        Mapping from atom name to a list of (n, 3) coordinate lists.
    """
    rng = np.random.default_rng()
    return {
        atom: rng.standard_normal((n, 3)).tolist()
        for atom in ("N", "CA", "C", "O")
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write JSONL file with synthetic protein entries and return its path.

    Writes one entry per name in _ENTRY_NAMES, each with _N_RES_DATA residues.
    """
    path = tmp_path / "proteins.jsonl"
    with path.open("w") as f:
        for name in _ENTRY_NAMES:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            _ = f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def splits_path(tmp_path: pathlib.Path) -> str:
    """Write splits JSON with train/val/test name lists and return its path.

    Uses _TRAIN_NAMES, _VAL_NAMES, and _TEST_NAMES as the split contents.
    """
    path = tmp_path / "splits.json"
    with path.open("w") as f:
        json.dump(
            {
                "train": _TRAIN_NAMES,
                "validation": _VAL_NAMES,
                "test": _TEST_NAMES,
            },
            f,
        )
    return str(path)


@pytest.fixture
def train_dataset(jsonl_path: str) -> ProteinDataset:
    """ProteinDataset loaded from the synthetic JSONL with training names.

    Loads only the entries listed in _TRAIN_NAMES with max_seq_length=_MAX_SEQ.
    """
    return ProteinDataset(jsonl_path, _TRAIN_NAMES, max_seq_length=_MAX_SEQ)


@pytest.fixture
def cfg() -> TrainConfig:
    """Minimal TrainConfig with batch_size=2 and max_seq_length=_MAX_SEQ.

    Both train and test loaders share the same batch_size and max_seq_length.
    """
    return TrainConfig(
        train_loader=TrainLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
        test_loader=EvalLoaderConfig(batch_size=2, max_seq_length=_MAX_SEQ),
    )


# ---------------------------------------------------------------------------
# ProteinDataset — length and indexing
# ---------------------------------------------------------------------------


def test_protein_dataset_len(train_dataset: ProteinDataset) -> None:
    """ProteinDataset length equals the number of listed training names.

    Verifies that len(dataset) matches len(_TRAIN_NAMES).
    """
    assert len(train_dataset) == len(_TRAIN_NAMES)


def test_protein_dataset_excludes_unlisted_names(jsonl_path: str) -> None:
    """ProteinDataset excludes entries whose names are not in provided list.

    Verifies that constructing dataset with only ["1aa.A"] yields single entry.
    """
    ds = ProteinDataset(jsonl_path, ["1aa.A"], max_seq_length=_MAX_SEQ)
    assert len(ds) == 1


def test_protein_dataset_empty_names(jsonl_path: str) -> None:
    """ProteinDataset with an empty names list has length 0.

    Verifies that passing an empty list to ProteinDataset yields a dataset of
    length zero.
    """
    ds = ProteinDataset(jsonl_path, [], max_seq_length=_MAX_SEQ)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# ProteinDataset — sample structure
# ---------------------------------------------------------------------------


def test_protein_dataset_sample_keys(train_dataset: ProteinDataset) -> None:
    """Checks that ProteinDataset sample is a Protein instance.

    Verifies the sample is a Protein instance and has the expected attributes.
    Also asserts ProteinDataset pads/truncates all protein fields to _MAX_SEQ.
    """
    sample = train_dataset[0]
    assert isinstance(sample, Protein)

    # aatype shape is (_MAX_SEQ,) — not one-hot encoded
    expected_shapes: dict[str, tuple[int, ...]] = {
        "atom_positions": (_MAX_SEQ, 37, 3),
        "atom_mask": (_MAX_SEQ, 37),
        "residue_index": (_MAX_SEQ,),
        "aatype": (_MAX_SEQ,),
    }
    for field, shape in expected_shapes.items():
        arr = cast(object, getattr(sample, field))
        assert isinstance(arr, np.ndarray)
        assert arr.shape == shape


def test_protein_dataset_float_fields_are_float64(
    train_dataset: ProteinDataset,
) -> None:
    """Float fields in a ProteinDataset sample have dtype float64.

    Verifies atom_positions and atom_mask are numpy float64 arrays.
    """
    sample = train_dataset[0]
    assert sample.atom_positions.dtype == np.float64
    assert sample.atom_mask.dtype == np.float64


def test_protein_dataset_aatype_is_integer_array(
    train_dataset: ProteinDataset,
) -> None:
    """The aatype field in a ProteinDataset sample is an integer numpy array.

    Verifies aatype is an ndarray with an integer dtype.
    """
    aatype = train_dataset[0].aatype
    assert isinstance(aatype, np.ndarray)
    assert np.issubdtype(aatype.dtype, np.integer)


def test_protein_dataset_aatype_truncated_to_max_seq_length(
    tmp_path: pathlib.Path,
) -> None:
    """ProteinDataset truncates aatype to max_seq_length entries.

    Verifies 100-residue entry is truncated to PROT_1_LEN entries in aatype.
    """
    path = pathlib.Path(tmp_path / "long.jsonl")
    with path.open("w", encoding="utf-8") as f:
        entry = {
            "name": "long.A",
            "seq": "A" * 100,
            "coords": _make_coords(100),
        }
        _ = f.write(json.dumps(entry) + "\n")
    ds = ProteinDataset(path, ["long.A"], max_seq_length=PROT_1_LEN)
    assert ds[0].aatype.shape[0] == PROT_1_LEN


def test_protein_dataset_all_items_accessible(
    train_dataset: ProteinDataset,
) -> None:
    """Every index in ProteinDataset is accessible and returns a Protein.

    Verifies __getitem__ succeeds for valid indices and returns a Protein.
    """
    for i in range(len(train_dataset)):
        assert isinstance(train_dataset[i], Protein)


# ---------------------------------------------------------------------------
# ProteinDataset — pickle / multiprocessing compatibility
# ---------------------------------------------------------------------------


def test_protein_dataset_picklable_before_open(
    train_dataset: ProteinDataset,
) -> None:
    """ProteinDataset can be pickled and restored before JSONL file is opened.

    Verifies the dataset length is preserved after a pickle round-trip with no
    prior access.
    """
    ds2 = cast(
        ProteinDataset,
        pickle.loads(pickle.dumps(train_dataset)),  # noqa: S301
    )
    assert len(ds2) == len(train_dataset)


def test_protein_dataset_picklable_after_open(
    train_dataset: ProteinDataset,
) -> None:
    """ProteinDataset can be pickled and restored after file handle open.

    Verifies atom_positions shape is correct after a pickle round-trip with the
    handle open.
    """
    _ = train_dataset[0]  # opens the file handle
    ds2 = cast(
        ProteinDataset,
        pickle.loads(pickle.dumps(train_dataset)),  # noqa: S301
    )
    assert ds2[0].atom_positions.shape == (_MAX_SEQ, 37, 3)


# ---------------------------------------------------------------------------
# make_data_loaders — debug_run=True
# ---------------------------------------------------------------------------

# The debug sampler is SubsetRandomSampler(range(8)), so iterating requires
# a dataset with ≥256 entries; len() only reads sampler size and works cheaply.


@pytest.fixture
def debug_jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write a JSONL file with _N_DEBUG protein entries for debug-mode testing.

    Each entry has _N_RES_DATA residues; names are zero-padded integers with
    the "x.A" suffix.
    """
    path = tmp_path / "debug_proteins.jsonl"
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    with path.open("w") as f:
        for name in names:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            _ = f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def debug_splits_path(tmp_path: pathlib.Path) -> str:
    """Write splits JSON assigning _N_DEBUG entries to train for debug-mode.

    Validation and test splits receive only first entry to satisfy schema.
    """
    names = [f"{i:04d}x.A" for i in range(_N_DEBUG)]
    path = tmp_path / "debug_splits.json"
    with path.open("w") as f:
        json.dump(
            {"train": names, "validation": names[:1], "test": names[:1]},
            f,
        )
    return str(path)


# ---------------------------------------------------------------------------
# Helpers for ProteinShardDataset tests
# ---------------------------------------------------------------------------


def _make_entry(name: str, seq_len: int) -> dict[str, object]:
    """Build a minimal JSONL entry with the given name and sequence length.

    Args:
        name: Protein entry identifier.
        seq_len: Number of residues (all set to alanine 'A').

    Returns:
        Dict with name, seq, and backbone coords for JSONL serialisation.
    """
    return {
        "name": name,
        "seq": "A" * seq_len,
        "coords": _make_coords(seq_len),
    }


def _write_jsonl(path: pathlib.Path, entries: list[dict[str, object]]) -> None:
    """Write entries as JSONL to path.

    Args:
        path: Destination file path.
        entries: List of dicts to serialise, one JSON object per line.
    """
    with path.open("w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in entries)


@pytest.fixture
def shard_budget(tmp_path: pathlib.Path) -> ShardBudgetParameters:
    """Minimal ShardBudgetParameters for ProteinShardDataset tests."""
    return ShardBudgetParameters(
        shard_dir=tmp_path / "shards",
        structlog_path=tmp_path / "train.jsonl",
        token_budget=512,
        max_seq_len=MAX_SEQ_LENGTH,
        seed=0,
        n_threads=1,
        world_size=1,
        rank=0,
        n_proteins_in_shard=100,
        noise_magnitude=0,
        num_workers=1,
    )


@pytest.fixture
def multi_shard_budget(tmp_path: pathlib.Path) -> ShardBudgetParameters:
    """ShardBudgetParameters with two workers and zero noise for plan tests.

    token_budget=250 is chosen so ffd_pack on four equal-length proteins
    produces two batches of two (3*L²>250 flushes at count 2).
    """
    return ShardBudgetParameters(
        shard_dir=tmp_path / "shards",
        structlog_path=tmp_path / "train.jsonl",
        token_budget=250,
        max_seq_len=MAX_SEQ_LENGTH,
        seed=0,
        n_threads=1,
        world_size=1,
        rank=0,
        n_proteins_in_shard=100,
        noise_magnitude=0,
        num_workers=2,
    )


# ---------------------------------------------------------------------------
# ProteinShardDataset
# ---------------------------------------------------------------------------


@pytest.fixture
def protein_shard_dataset_factory(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> Callable[[list[str], list[int]], ProteinShardDataset]:
    """Returns callable to build ProteinShardDataset from names and lengths."""

    def _build(
        protein_name_array: list[str],
        protein_len_array: list[int],
    ) -> ProteinShardDataset:
        entries: list[dict[str, object]] = [
            _make_entry(name, length)
            for name, length in zip(
                protein_name_array,
                protein_len_array,
                strict=True,
            )
        ]
        dataset_jsonl_path = tmp_path / "p.jsonl"
        _write_jsonl(dataset_jsonl_path, entries)
        return ProteinShardDataset(
            budget_parameters=shard_budget,
            names=protein_name_array,
            dataset_jsonl=dataset_jsonl_path,
        )

    return _build


# ---------------------------------------------------------------------------
# Fixtures for bucketed loader tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bucketed_jsonl(tmp_path: pathlib.Path) -> str:
    """Write a JSONL with 5 proteins of varying lengths.

    Proteins p1-p5 have lengths 8, 16, 24, 32, and 40 residues respectively.
    """
    entries = [
        _make_entry("p1", 8),
        _make_entry("p2", 16),
        _make_entry("p3", 24),
        _make_entry("p4", 32),
        _make_entry("p5", 40),
    ]
    path = tmp_path / "proteins.jsonl"
    _write_jsonl(path, entries)
    return str(path)


@pytest.fixture
def bucketed_splits(tmp_path: pathlib.Path) -> str:
    """Write a splits JSON using the 5-protein JSONL names."""
    path = tmp_path / "splits.json"
    with path.open("w") as f:
        json.dump(
            {
                "train": ["p1", "p2", "p3"],
                "validation": ["p4"],
                "test": ["p5"],
            },
            f,
        )
    return str(path)


@pytest.fixture
def bucketed_cfg() -> TrainConfig:
    """Minimal TrainConfig with token_budget=512 for bucketed loader tests."""
    return TrainConfig(
        train_loader=TrainLoaderConfig(
            batch_size=2,
            max_seq_length=_MAX_SEQ,
            token_budget=512,
            num_workers=1,
            epoch_prefetch_depth=1,
            n_shards=1,
        ),
        test_loader=EvalLoaderConfig(
            batch_size=2,
            max_seq_length=_MAX_SEQ,
        ),
    )


@pytest.fixture
def bucketed_train_args(
    bucketed_jsonl: str,
    bucketed_splits: str,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs pointing at the bucketed JSONL and splits fixtures."""
    return TrainArgs(
        dataset_jsonl=pathlib.Path(bucketed_jsonl),
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=pathlib.Path(bucketed_splits),
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=False,
    )


@pytest.fixture
def debug_train_args(
    debug_jsonl_path: str,
    debug_splits_path: str,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs fixture with debug_run=True."""
    return TrainArgs(
        dataset_jsonl=pathlib.Path(debug_jsonl_path),
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=pathlib.Path(debug_splits_path),
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=True,
    )


@pytest.fixture
def full_dataset_jsonl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a JSONL file with _N_FULL synthetic protein entries.

    Used to seed a shard cache before a debug_run=True call so the
    cache-poisoning scenario can be exercised.
    """
    path = tmp_path / "full_proteins.jsonl"
    names = [f"{i:04d}p.A" for i in range(_N_FULL)]
    with path.open("w") as f:
        for name in names:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_DATA],
                "coords": _make_coords(_N_RES_DATA),
            }
            _ = f.write(json.dumps(entry) + "\n")
    return path


@pytest.fixture
def full_dataset_splits(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a splits JSON assigning all _N_FULL entries to train."""
    names = [f"{i:04d}p.A" for i in range(_N_FULL)]
    path = tmp_path / "full_splits.json"
    with path.open("w") as f:
        json.dump(
            {"train": names, "validation": names[:1], "test": names[:1]},
            f,
        )
    return path


@pytest.fixture
def full_train_args(
    full_dataset_jsonl: pathlib.Path,
    full_dataset_splits: pathlib.Path,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs pointing at the _N_FULL dataset with debug_run=False."""
    return TrainArgs(
        dataset_jsonl=full_dataset_jsonl,
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=full_dataset_splits,
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=False,
    )


@pytest.fixture
def full_debug_train_args(
    full_dataset_jsonl: pathlib.Path,
    full_dataset_splits: pathlib.Path,
    tmp_path: pathlib.Path,
) -> TrainArgs:
    """TrainArgs pointing at the _N_FULL dataset with debug_run=True."""
    return TrainArgs(
        dataset_jsonl=full_dataset_jsonl,
        shard_dir=tmp_path / "shards",
        keys_for_splits_json=full_dataset_splits,
        config=tmp_path / "config.json",
        structlog_jsonl=tmp_path / "train.jsonl",
        ddp=False,
        debug_run=True,
    )


# ---------------------------------------------------------------------------
# make_bucketed_data_loaders tests
# ---------------------------------------------------------------------------


def test_bucketed_train_loader_yields_protein_batch(
    bucketed_cfg: TrainConfig,
    bucketed_train_args: TrainArgs,
) -> None:
    """Training loader yields a list of Protein objects."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    batch = cast(list[Protein], next(iter(train_loader)))
    assert isinstance(batch, list)
    assert all(isinstance(p, Protein) for p in batch)


def test_bucketed_debug_run_train_dataset_has_252_items(
    bucketed_cfg: TrainConfig,
    debug_train_args: TrainArgs,
) -> None:
    """Using debug_run=True yields a train dataset of _N_DEBUG items."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=debug_train_args,
    )
    assert (
        len(cast(ShardDataLoader, train_loader).shard_dataset.names) == _N_DEBUG
    )


_N_FULL = 300  # dataset larger than _N_DEBUG to expose cache-poisoning bug


def test_bucketed_debug_run_not_poisoned_by_prior_full_cache(
    bucketed_cfg: TrainConfig,
    full_train_args: TrainArgs,
    full_debug_train_args: TrainArgs,
) -> None:
    """Must yield _N_DEBUG items even when shard cache built by full run.

    A cache built from all _N_FULL proteins must not silently return _N_FULL
    items for subsequent debug_run=True call that caps train names at _N_DEBUG.
    """
    # Seed the shard cache with the full _N_FULL-protein dataset.
    _, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=full_train_args,
    )

    # A subsequent debug_run=True call must cap at _N_DEBUG despite warm cache.
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=full_debug_train_args,
    )
    assert (
        len(cast(ShardDataLoader, train_loader).shard_dataset.names) == _N_DEBUG
    )


# ---------------------------------------------------------------------------
# write_shard_metadata_sidecar tests
# ---------------------------------------------------------------------------


def test_write_shard_metadata_fields_match_inputs(
    protein_shard_dataset_factory: Callable[
        [list[str], list[int]],
        ProteinShardDataset,
    ],
) -> None:
    """ShardMetadata JSON fields match the values used at construction."""
    ds = protein_shard_dataset_factory(["p1", "p2"], [10, 20])
    meta = ShardMetadata.model_validate_json(ds.shard_metadata_path.read_text())
    assert meta.token_budget == ds.token_budget
    assert meta.shard_size == ds.n_proteins_in_shard
    assert meta.n_shards >= 1


def test_write_shard_metadata_names_hash_stable(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """The same name set produces an identical names_hash on both runs."""
    names = ["p1", "p2"]
    _write_jsonl(
        tmp_path / "p.jsonl",
        [_make_entry(n, 10) for n in names],
    )
    ds_a = ProteinShardDataset(
        budget_parameters=dataclasses.replace(
            shard_budget,
            shard_dir=tmp_path / "shards_a",
        ),
        names=names,
        dataset_jsonl=tmp_path / "p.jsonl",
    )
    ds_b = ProteinShardDataset(
        budget_parameters=dataclasses.replace(
            shard_budget,
            shard_dir=tmp_path / "shards_b",
        ),
        names=names,
        dataset_jsonl=tmp_path / "p.jsonl",
    )
    hash_a = ShardMetadata.model_validate_json(
        ds_a.shard_metadata_path.read_text(),
    ).names_hash
    hash_b = ShardMetadata.model_validate_json(
        ds_b.shard_metadata_path.read_text(),
    ).names_hash
    assert hash_a == hash_b


def test_write_shard_metadata_names_hash_changes_on_different_names(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """Different name sets produce different names_hash values."""
    _write_jsonl(
        tmp_path / "p.jsonl",
        [_make_entry("p1", 10), _make_entry("p2", 20)],
    )
    ds_a = ProteinShardDataset(
        budget_parameters=dataclasses.replace(
            shard_budget,
            shard_dir=tmp_path / "shards_a",
        ),
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
    )
    ds_b = ProteinShardDataset(
        budget_parameters=dataclasses.replace(
            shard_budget,
            shard_dir=tmp_path / "shards_b",
        ),
        names=["p1", "p2"],
        dataset_jsonl=tmp_path / "p.jsonl",
    )
    hash_a = ShardMetadata.model_validate_json(
        ds_a.shard_metadata_path.read_text(),
    ).names_hash
    hash_b = ShardMetadata.model_validate_json(
        ds_b.shard_metadata_path.read_text(),
    ).names_hash
    assert hash_a != hash_b


def test_build_sorted_shards_existing_metadata_prevents_rebuild(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> None:
    """A second init with the same shard_dir reuses cached metadata as-is."""
    _write_jsonl(tmp_path / "p.jsonl", [_make_entry("p1", 10)])
    first = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
    )
    mtime_before = first.shard_metadata_path.stat().st_mtime
    _ = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=tmp_path / "p.jsonl",
    )
    mtime_after = first.shard_metadata_path.stat().st_mtime
    assert mtime_before == mtime_after


# ---------------------------------------------------------------------------
# ShardDataLoader lifecycle tests
# ---------------------------------------------------------------------------


def test_shard_data_loader_epoch_increments_after_iter(
    bucketed_cfg: TrainConfig,
    bucketed_train_args: TrainArgs,
) -> None:
    """ShardDataLoader.epoch starts at 0 and increments with each __iter__."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    loader = cast(ShardDataLoader, train_loader)
    assert loader.epoch == 0
    _ = list(loader)
    assert loader.epoch == 1


def test_shard_data_loader_cached_len_is_positive(
    bucketed_cfg: TrainConfig,
    bucketed_train_args: TrainArgs,
) -> None:
    """__len__ returns a positive batch count immediately after construction."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    loader = cast(ShardDataLoader, train_loader)
    assert len(loader) > 0


def test_shard_data_loader_del_no_error(
    bucketed_cfg: TrainConfig,
    bucketed_train_args: TrainArgs,
) -> None:
    """Deleting a ShardDataLoader shuts down executors without raising."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    del train_loader


# ---------------------------------------------------------------------------
# parse_protein tests
# ---------------------------------------------------------------------------


def test_parse_protein_bytes_and_dict_give_identical_result(
    protein_shard_dataset_factory: Callable[
        [list[str], list[int]],
        ProteinShardDataset,
    ],
) -> None:
    """parse_protein accepts raw bytes and a decoded dict with equal results."""
    ds = protein_shard_dataset_factory(["p1"], [10])
    raw_bytes: bytes = json.dumps(_make_entry("p1", 10)).encode()
    raw_dict: dict[str, object] = cast(
        "dict[str, object]",
        json.loads(raw_bytes),
    )
    p_from_bytes = ds.parse_protein({"json": raw_bytes})
    p_from_dict = ds.parse_protein({"json": raw_dict})
    np.testing.assert_array_equal(
        p_from_bytes.atom_positions,
        p_from_dict.atom_positions,
    )


def test_parse_protein_coordinates_are_centered(
    protein_shard_dataset_factory: Callable[
        [list[str], list[int]],
        ProteinShardDataset,
    ],
) -> None:
    """After parsing, the masked CA-position centroid is near zero."""
    ds = protein_shard_dataset_factory(["p1"], [10])
    entry: dict[str, object] = _make_entry("p1", 10)
    protein = ds.parse_protein({"json": entry})
    ca_pos = protein.atom_positions[:, 1, :]  # (n_res, 3)
    ca_mask = protein.atom_mask[:, 1]  # (n_res,)
    masked_mean = cast(
        npt.NDArray[np.float64],
        np.sum(ca_mask[:, None] * ca_pos, axis=0) / np.sum(ca_mask),
    )
    np.testing.assert_allclose(masked_mean, np.zeros(3), atol=1e-6)


def test_parse_protein_unknown_aa_maps_to_restype_x(
    protein_shard_dataset_factory: Callable[
        [list[str], list[int]],
        ProteinShardDataset,
    ],
) -> None:
    """Amino acids not in restype_order are mapped to restype_order['X']."""
    ds = protein_shard_dataset_factory(["p1"], [3])
    entry: dict[str, object] = {
        "name": "p1",
        "seq": "ZZZ",
        "coords": _make_coords(3),
    }
    protein = ds.parse_protein({"json": entry})
    expected_idx = restype_order["X"]
    aatype_list = cast(list[int], protein.aatype.tolist())
    assert all(a == expected_idx for a in aatype_list)


# ---------------------------------------------------------------------------
# DatasetSplitsManifest tests
# ---------------------------------------------------------------------------


def test_dataset_splits_manifest_ignores_extra_fields() -> None:
    """DatasetSplitsManifest silently drops unknown JSON fields."""
    data: dict[str, object] = {
        "train": ["p1"],
        "validation": ["p2"],
        "test": ["p3"],
        "unknown_field": "ignored",
        "also_ignored": 42,
    }
    manifest = DatasetSplitsManifest.model_validate(data)
    assert manifest.train == ["p1"]
    assert manifest.validation == ["p2"]
    assert manifest.test == ["p3"]


def test_dataset_splits_manifest_cath_nodes_defaults_empty() -> None:
    """cath_nodes defaults to an empty dict when absent from the JSON."""
    data: dict[str, object] = {
        "train": ["p1"],
        "validation": ["p2"],
        "test": ["p3"],
    }
    manifest = DatasetSplitsManifest.model_validate(data)
    assert manifest.cath_nodes == {}


def test_dataset_splits_manifest_cath_nodes_populated() -> None:
    """cath_nodes is fully populated when present in the JSON."""
    data: dict[str, object] = {
        "train": ["p1"],
        "validation": ["p2"],
        "test": ["p3"],
        "cath_nodes": {"p1": ["1.20.5"], "p2": ["2.60.40"]},
    }
    manifest = DatasetSplitsManifest.model_validate(data)
    assert manifest.cath_nodes == {"p1": ["1.20.5"], "p2": ["2.60.40"]}


# ---------------------------------------------------------------------------
# FeaturizedItem and FeaturizedBatch — importability from helpers.data
# ---------------------------------------------------------------------------


def test_featurized_item_is_importable_from_data() -> None:
    """FeaturizedItem is a dataclass importable directly from helpers.data."""
    assert dataclasses.is_dataclass(FeaturizedItem)


def test_featurized_batch_is_importable_from_data() -> None:
    """FeaturizedBatch is a dataclass importable directly from helpers.data."""
    assert dataclasses.is_dataclass(FeaturizedBatch)


# ---------------------------------------------------------------------------
# Featurization utilities — importability from helpers.data
# ---------------------------------------------------------------------------


def test_distogram_is_importable_from_data() -> None:
    """Distogram is an nn.Module importable directly from helpers.data."""
    assert issubclass(Distogram, nn.Module)


def test_sinusoidal_encoding_is_importable_from_data() -> None:
    """sinusoidal_encoding is callable and importable from helpers.data."""
    assert callable(sinusoidal_encoding)


def test_ref_pos_for_residue_is_importable_from_data() -> None:
    """ref_pos_for_residue is callable and importable from helpers.data."""
    assert callable(ref_pos_for_residue)


def test_featurize_single_item_is_importable_from_data() -> None:
    """featurize_single_item is callable and importable from helpers.data."""
    assert callable(featurize_single_item)


def test_featurize_batch_is_importable_from_data() -> None:
    """featurize_batch is callable and importable from helpers.data."""
    assert callable(featurize_batch)


def test_apply_conditioning_dropout_is_importable_from_data() -> None:
    """apply_conditioning_dropout is importable from helpers.data."""
    assert callable(apply_conditioning_dropout)
