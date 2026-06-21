"""Tests for featurization utilities.

Covers Distogram shape, one-hot, masking, symmetry, and bin-assignment
properties; featurize_batch output shapes and value contracts;
apply_conditioning_dropout behaviour; sinusoidal_encoding and
ref_pos_for_residue correctness; and jaxtyping shape-contract
enforcement for all public featurization functions and dataclasses.
"""

import dataclasses
import math
from typing import cast

import numpy as np
import pytest
import torch
from architecture.main_trunk import MainTrunk
from einops import rearrange, reduce, repeat
from helpers.atom_utils import RESTYPE_NUM_NO_X, Protein, restype_order
from helpers.data import FeaturizedBatch, FeaturizedItem
from helpers.featurize import (
    Distogram,
    apply_conditioning_dropout,
    featurize_batch,
    featurize_single_item,
    ref_pos_for_residue,
    sinusoidal_encoding,
)
from helpers.useful_objects import ModelSetup, manual_seed
from jaxtyping import Bool, Float, TypeCheckError
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from train.train_config import TrainConfig

_ = manual_seed(42)

B = 2
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
    # Without a coords_mask, every pair within max_dist should be unmasked.
    # Place coords all at origin so all distances are 0 (< max_dist).
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
    # Row 0 and column 0 must all be False
    assert not m[0, :].any()
    assert not m[:, 0].any()
    # All other pairs still potentially valid
    assert m[1:, 1:].any()


def test_distogram_overflow_mask_ignores_distance_cutoff(
    disto_overflow: Distogram,
) -> None:
    """With overflow_bin=True the pair mask is not gated by a distance cutoff.

    Verifies that even when one residue is placed very far from all others, the
    overflow-bin Distogram marks all pairs as valid because distant pairs are
    captured by the overflow bin rather than being masked out.
    """
    # With overflow_bin=True the mask does NOT apply a distance cutoff —
    # pairs are valid as long as both atoms have valid coords.
    c = torch.zeros(N_RES, 3)
    c[0] = 1000.0  # very far from all others
    with torch.no_grad():
        _, m = disto_overflow(c)
    assert m.all()


def test_distogram_no_overflow_masks_distant_pairs(disto: Distogram) -> None:
    """Without overflow_bin, pairs beyond max_dist are set to False in the mask.

    Verifies that the standard Distogram (no overflow bin) applies a distance
    cutoff: pairs whose Euclidean distance exceeds max_dist are masked to False.
    """
    # Without overflow_bin, pairs beyond max_dist are masked out.
    c = torch.zeros(N_RES, 3)
    c[0] = MAX_DIST * 2  # residue 0 far from residue 1..N-1
    with torch.no_grad():
        _, m = disto(c)
    # Pair (0, 1..N-1) should be masked
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
    # All coords at origin → all distances are 0 → all pairs in bin 0.
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
    # Two groups of atoms separated by >> max_dist.
    c = torch.zeros(N_RES, 3)
    c[N_RES // 2 :] = MAX_DIST * 10
    with torch.no_grad():
        f, _ = disto_overflow(c)
    # Cross-group pairs must occupy the overflow bin (last bin).
    cross = f[: N_RES // 2, N_RES // 2 :, :]
    assert cross[..., -1].all()


def test_distogram_exact_bin_for_known_interior_distance(
    disto: Distogram,
) -> None:
    """Distogram assigns a 9 Å pair to bin 5 given bin_width=(22-2)/16=1.25 Å.

    Verifies exact bin-index arithmetic: for a known distance of 9.0 Å,
    floor((9.0 - 2.0) / 1.25) = 5, so bin 5 should be 1.0 and all others 0.0.
    """
    # bin_width = (MAX_DIST - MIN_DIST) / N_BINS = (22 - 2) / 16 = 1.25 Å
    # d = 9.0 Å → bin = floor((9.0 - 2.0) / 1.25) = floor(5.6) = 5
    c = torch.zeros(N_RES, 3)
    c[0, 0] = 9.0  # residue 0 at (9, 0, 0); all others at origin
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
def protein_batch(single_protein: Protein) -> list[Protein]:
    """Provide B=2 identical Protein objects for batch featurization tests."""
    return [single_protein for _ in range(B)]


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
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> FeaturizedBatch:
    """FeaturizedBatch produced by featurize_batch on protein_batch fixture.

    Sets a fixed random seed so that the log-normal noise sample lands within
    [sigma_min, sigma_max], keeping t_normalized in [0, 1] regardless of prior
    RNG state.
    """
    _ = manual_seed(1)
    return featurize_batch(
        batch=protein_batch,
        tcfg=model_params.tcfg,
        distogram_res=model_params.distogram_res,
        distogram_atom=model_params.distogram_atom,
    )


# ---------------------------------------------------------------------------
# featurize_batch — output shapes
# Batched layout for B=2, N_RES=12:  tensors are (B, N_RES, *) or (B, N_ATOM, *)
# ---------------------------------------------------------------------------

N_ATOM = N_RES * 5  # 5 atoms per residue, no separators


def test_featurize_batch_output_shapes(
    featurized_batch: FeaturizedBatch,
) -> None:
    """featurize_batch produces correct shapes and values for all output fields.

    Batched layout for B=2, N_RES=12: tensors are (B, N_RES, *) or
    (B, N_ATOM, *). Checks shapes, finiteness, noise-level bounds, and index
    mapping invariants.
    """
    # shapes — one comprehension + one assert instead of seven asserts
    expected_shapes: dict[str, tuple[int, ...]] = {
        "ref_pos": (B, N_ATOM, 3),
        "ref_element": (B, N_ATOM, 4),
        "ref_space_uid": (B, N_ATOM),
        "gt_res_distogram": (B, N_RES, N_RES, 39),
        "f_pseudo_beta_mask": (B, N_RES),
        "tok_idx": (B, N_ATOM),
        "center_uid": (B, N_ATOM),
    }
    batch_dict: dict[str, object] = dataclasses.asdict(featurized_batch)
    assert {
        k: tuple(cast(torch.Tensor, batch_dict[k]).shape)
        for k in expected_shapes
    } == expected_shapes

    # all tensor fields are finite
    for field_name, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(
                val.float(),
            ).all(), f"non-finite in field '{field_name}'"

    # noise schedule values — one assert instead of three
    assert all(
        [
            bool((featurized_batch.t_hat > 0.0).all()),
            bool((featurized_batch.t_normalized >= 0.0).all()),
            bool((featurized_batch.t_normalized <= 1.0).all()),
        ],
    )

    # index mapping: each residue r owns atoms [r*5, r*5+5)
    expected_res_idx = repeat(
        torch.arange(N_RES),
        "r -> b (r five)",
        b=B,
        five=5,
    )
    assert torch.equal(featurized_batch.ref_space_uid, expected_res_idx)
    assert torch.equal(featurized_batch.tok_idx, expected_res_idx)

    # center_uid points to C-alpha atom (index r*5+1) for every atom in
    # residue r.
    expected_ca = repeat(
        (torch.arange(N_RES) * 5 + 1).repeat_interleave(5),
        "a -> b a",
        b=B,
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

    # all tensor fields are finite
    for field_name, val in item_dict.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(
                val.float(),
            ).all(), f"non-finite in field '{field_name}'"

    # noise schedule values
    assert all(
        [
            bool(featurized_item.t_hat > 0.0),
            bool((featurized_item.t_normalized >= 0.0).all()),
            bool((featurized_item.t_normalized <= 1.0).all()),
        ],
    )

    # index mapping: each residue r owns atoms [r*5, r*5+5)
    expected_res_idx = torch.arange(N_RES).repeat_interleave(5)
    assert torch.equal(featurized_item.ref_space_uid, expected_res_idx)
    assert torch.equal(featurized_item.tok_idx, expected_res_idx)

    # center_uid points to C-alpha atom (index r*5+1) for each atom in res r
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
                (N_RES, 37),  # missing last dim
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
    # All valid residues should have their rows/cols zeroed
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
    # If row i is zeroed, column i must also be zeroed (and vice versa)
    row_sums = out.gt_res_distogram.sum(dim=(2, 3))  # (B, N_res)
    col_sums = out.gt_res_distogram.sum(dim=(1, 3))  # (B, N_res)
    assert torch.equal(row_sums == 0, col_sums == 0)


def test_conditioning_dropout_respects_residue_mask(
    featurized_batch: FeaturizedBatch,
) -> None:
    """Conditioning dropout never modifies padding residues."""
    # Padding residues (f_pseudo_beta_mask=0) must not be changed
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
    positions = rearrange(torch.arange(N_RES).float(), "n -> 1 n")  # (1, N_RES)
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
    coords_bad = torch.zeros(N_RES, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        _ = disto(coords_bad)


def test_sinusoidal_encoding_wrong_shape() -> None:
    """Wrong positions ndim (3-D instead of 2-D) triggers TypeCheckError."""
    positions_bad = torch.zeros(B, N_RES, 1)  # must be 2-D
    with pytest.raises(TypeCheckError):
        _ = sinusoidal_encoding(positions_bad)
