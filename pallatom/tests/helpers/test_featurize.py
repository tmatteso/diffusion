"""Tests for featurization utilities."""

import dataclasses
import math

import pytest
import torch
from architecture.main_trunk import MainTrunk
from einops import rearrange, reduce, repeat
from helpers.atom_utils import restype_num, restype_order
from helpers.featurize import (
    Distogram,
    FeaturizedBatch,
    FeaturizedItem,
    ProteinBatch,
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

manual_seed(42)

B = 2
N_RES = 12
N_BINS = 16
C_RES = 32
MIN_DIST = 2.0
MAX_DIST = 22.0
AA_SEQ = "ACDEFGHIKLMN"  # length N_RES


# ---------------------------------------------------------------------------
# restype order and count — X as mask token
# ---------------------------------------------------------------------------


def test_restype_order_x_is_20() -> None:
    """Mask token 'X' is assigned index 20, one past the 20 standard amino acids."""
    assert restype_order["X"] == 20


def test_restype_num_is_21() -> None:
    """restype_num equals 21 (20 standard amino acids + 1 unknown/mask token)."""
    assert restype_num == 21


# ---------------------------------------------------------------------------
# Fixtures — models
# ---------------------------------------------------------------------------


@pytest.fixture
def disto() -> Distogram:
    """Provide a Distogram without overflow bin in eval mode."""
    return Distogram(n_bins=N_BINS, overflow_bin=False, min_dist=MIN_DIST, max_dist=MAX_DIST).eval()


@pytest.fixture
def disto_overflow() -> Distogram:
    """Provide a Distogram with overflow bin in eval mode."""
    return Distogram(n_bins=N_BINS, min_dist=MIN_DIST, max_dist=MAX_DIST, overflow_bin=True).eval()


# ---------------------------------------------------------------------------
# Fixtures — tensors
# ---------------------------------------------------------------------------


@pytest.fixture
def coords() -> Float[torch.Tensor, "N_res 3"]:
    """Provide random single-chain coordinates (N_RES, 3)."""
    return torch.randn(N_RES, 3)


@pytest.fixture
def coords_batch() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random batched coordinates (2, N_RES, 3)."""
    return torch.randn(2, N_RES, 3)


@pytest.fixture
def mask() -> Bool[torch.Tensor, "N_res"]:
    """Provide an all-True residue mask (N_RES,)."""
    return torch.ones(N_RES, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Distogram — overflow bin shape
# ---------------------------------------------------------------------------


def test_distogram_overflow_bin_output_shape(
    disto_overflow: Distogram, coords: Float[torch.Tensor, "N_res 3"]
) -> None:
    """Distogram with overflow bin outputs shape (N_RES, N_RES, N_BINS + 1)."""
    with torch.no_grad():
        f, _ = disto_overflow(coords)
    assert f.shape == (N_RES, N_RES, N_BINS + 1)


# ---------------------------------------------------------------------------
# Distogram — one-hot property
# ---------------------------------------------------------------------------


def test_distogram_one_hot_sums_to_one(
    disto: Distogram, coords: Float[torch.Tensor, "N_res 3"]
) -> None:
    """Distogram bin probabilities sum to 1 for every residue pair."""
    with torch.no_grad():
        f, _ = disto(coords)
    bin_sums = reduce(f, "i j b -> i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


def test_distogram_overflow_one_hot_sums_to_one(
    disto_overflow: Distogram, coords: Float[torch.Tensor, "N_res 3"]
) -> None:
    """Overflow-bin Distogram probabilities also sum to 1 for every residue pair."""
    with torch.no_grad():
        f, _ = disto_overflow(coords)
    bin_sums = reduce(f, "i j b -> i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


# ---------------------------------------------------------------------------
# Distogram — pair mask
# ---------------------------------------------------------------------------


def test_distogram_mask_none_gives_all_true_within_range(disto: Distogram) -> None:
    """Without a mask, every residue pair within max_dist is unmasked."""
    # Without a coords_mask, every pair within max_dist should be unmasked.
    # Place coords all at origin so all distances are 0 (< max_dist).
    c = torch.zeros(N_RES, 3)
    with torch.no_grad():
        _, m = disto(c)
    assert m.all()


def test_distogram_mask_zeros_out_invalid_residues(
    disto: Distogram, coords: Float[torch.Tensor, "N_res 3"]
) -> None:
    """Distogram masks row 0 and column 0 when residue 0 is excluded from the mask."""
    mask = torch.ones(N_RES, dtype=torch.bool)
    mask[0] = False
    with torch.no_grad():
        _, m = disto(coords, mask)
    # Row 0 and column 0 must all be False
    assert not m[0, :].any()
    assert not m[:, 0].any()
    # All other pairs still potentially valid
    assert m[1:, 1:].any()


def test_distogram_overflow_mask_ignores_distance_cutoff(disto_overflow: Distogram) -> None:
    """With overflow_bin=True the pair mask is not gated by a distance cutoff."""
    # With overflow_bin=True the mask does NOT apply a distance cutoff —
    # pairs are valid as long as both atoms have valid coords.
    c = torch.zeros(N_RES, 3)
    c[0] = 1000.0  # very far from all others
    with torch.no_grad():
        _, m = disto_overflow(c)
    assert m.all()


def test_distogram_no_overflow_masks_distant_pairs(disto: Distogram) -> None:
    """Without overflow_bin, pairs beyond max_dist are set to False in the mask."""
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


def test_distogram_is_symmetric(disto: Distogram, coords: Float[torch.Tensor, "N_res 3"]) -> None:
    """Distogram output is symmetric: f[i,j] == f[j,i] and m[i,j] == m[j,i]."""
    with torch.no_grad():
        f, m = disto(coords)
    assert torch.allclose(f, rearrange(f, "i j b -> j i b"))
    assert torch.equal(m, rearrange(m, "i j -> j i"))


# ---------------------------------------------------------------------------
# Distogram — bin correctness
# ---------------------------------------------------------------------------


def test_distogram_close_pairs_land_in_first_bin(disto: Distogram) -> None:
    """Pairs at distance 0 (all at origin) are assigned to the first distance bin."""
    # All coords at origin → all distances are 0 → all pairs in bin 0.
    c = torch.zeros(N_RES, 3)
    with torch.no_grad():
        f, _ = disto(c)
    assert f[..., 0].all()


def test_distogram_overflow_far_pairs_land_in_last_bin(disto_overflow: Distogram) -> None:
    """Pairs far beyond max_dist land in the overflow (last) bin when overflow_bin=True."""
    # Two groups of atoms separated by >> max_dist.
    c = torch.zeros(N_RES, 3)
    c[N_RES // 2 :] = MAX_DIST * 10
    with torch.no_grad():
        f, _ = disto_overflow(c)
    # Cross-group pairs must occupy the overflow bin (last bin).
    cross = f[: N_RES // 2, N_RES // 2 :, :]
    assert cross[..., -1].all()


def test_distogram_exact_bin_for_known_interior_distance(disto: Distogram) -> None:
    """Distogram assigns a 9 Å pair to bin 5 given bin_width=(22-2)/16=1.25 Å."""
    # bin_width = (MAX_DIST - MIN_DIST) / N_BINS = (22 - 2) / 16 = 1.25 Å
    # d = 9.0 Å → bin = floor((9.0 - 2.0) / 1.25) = floor(5.6) = 5
    c = torch.zeros(N_RES, 3)
    c[0, 0] = 9.0  # residue 0 at (9, 0, 0); all others at origin
    expected_bin = 5
    with torch.no_grad():
        f, _ = disto(c)
    assert math.isclose(f[0, 1, expected_bin].item(), 1.0)
    assert f[0, 1, :expected_bin].abs().max().item() < 1e-6
    assert f[0, 1, expected_bin + 1 :].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# featurize_batch — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def protein_batch() -> ProteinBatch:
    """Provide a ProteinBatch with random coordinates for B=2 proteins of length N_RES."""
    return ProteinBatch(
        atom_positions=torch.randn(B, N_RES, 37, 3),
        atom_mask=torch.ones(B, N_RES, 37),
        residue_index=repeat(torch.arange(N_RES).float(), "n -> b n", b=B),
        seq=[AA_SEQ, AA_SEQ],
    )


@pytest.fixture
def tcfg() -> TrainConfig:
    """Provide a default TrainConfig with standard noise and distogram settings."""
    return TrainConfig()


@pytest.fixture
def c_beta_distogram_fn(tcfg: TrainConfig) -> Distogram:
    """Provide the residue-level Cβ Distogram configured from tcfg.distogram_res."""
    dr = tcfg.distogram_res
    return Distogram(
        n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True
    ).eval()


@pytest.fixture
def atom_distogram_fn(tcfg: TrainConfig) -> Distogram:
    """Provide the atom-level Distogram configured from tcfg.distogram_atom."""
    da = tcfg.distogram_atom
    return Distogram(
        n_bins=da.n_bins, overflow_bin=False, min_dist=da.min_dist, max_dist=da.max_dist
    ).eval()


@pytest.fixture
def model_params(
    tcfg: TrainConfig, c_beta_distogram_fn: Distogram, atom_distogram_fn: Distogram
) -> ModelSetup:
    """Provide a ModelSetup bundling a small CPU MainTrunk with the tcfg training configuration."""
    trunk = MainTrunk(
        f_ref_dim=tcfg.model.f_ref_dim,
        n_bins=tcfg.model.n_bins,
        n_atom_bins=tcfg.distogram_atom.n_bins,
        c_atom=tcfg.model.c_atom,
        c_pair=tcfg.model.c_pair,
        c_res=tcfg.model.c_res,
        c_atompair=tcfg.model.c_atompair,
        K_unit=tcfg.model.K_unit,
    ).eval()
    optimizer = Adam(trunk.parameters(), lr=tcfg.training.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg.training.num_epochs)
    return ModelSetup(
        model=trunk,
        tcfg=tcfg,
        distogram_res=c_beta_distogram_fn,
        distogram_atom=atom_distogram_fn,
        device="cpu",
        optimizer=optimizer,
        scheduler=scheduler,
    )


@pytest.fixture
def featurized_batch(
    protein_batch: ProteinBatch,
    model_params: ModelSetup,
) -> FeaturizedBatch:
    """Provide a FeaturizedBatch produced by featurize_batch on the protein_batch fixture.

    Sets a fixed random seed so that the log-normal noise sample lands within
    [sigma_min, sigma_max], keeping t_normalized in [0, 1] regardless of prior RNG state.
    """
    manual_seed(1)
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


def test_featurize_batch_ref_pos_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces ref_pos of shape (B, N_ATOM, 3)."""
    assert featurized_batch.ref_pos.shape == (B, N_ATOM, 3)


def test_featurize_batch_ref_element_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces ref_element of shape (B, N_ATOM, 4)."""
    assert featurized_batch.ref_element.shape == (B, N_ATOM, 4)


def test_featurize_batch_ref_space_uid_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces ref_space_uid of shape (B, N_ATOM)."""
    assert featurized_batch.ref_space_uid.shape == (B, N_ATOM)


def test_featurize_batch_distogram_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces gt_res_distogram of shape (B, N_RES, N_RES, 39)."""
    assert featurized_batch.gt_res_distogram.shape == (B, N_RES, N_RES, 39)


def test_featurize_batch_pseudo_beta_mask_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces f_pseudo_beta_mask of shape (B, N_RES)."""
    assert featurized_batch.f_pseudo_beta_mask.shape == (B, N_RES)


def test_featurize_batch_r_input_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces r_input of shape (B, N_ATOM, 3)."""
    assert featurized_batch.r_input.shape == (B, N_ATOM, 3)


def test_featurize_batch_tok_idx_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces tok_idx of shape (B, N_ATOM)."""
    assert featurized_batch.tok_idx.shape == (B, N_ATOM)


def test_featurize_batch_center_uid_shape(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch produces center_uid of shape (B, N_RES)."""
    assert featurized_batch.center_uid.shape == (B, N_RES)


# ---------------------------------------------------------------------------
# featurize_batch — output values
# ---------------------------------------------------------------------------


def test_featurize_batch_all_tensor_fields_finite(featurized_batch: FeaturizedBatch) -> None:
    """All tensor fields of the FeaturizedBatch are finite (no NaN or Inf)."""
    for field in dataclasses.fields(featurized_batch):
        val = getattr(featurized_batch, field.name)
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val.float()).all(), f"non-finite in field '{field.name}'"


def test_featurize_batch_t_hat_is_positive(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch samples a strictly positive noise level t_hat from the log-normal prior."""
    assert featurized_batch.t_hat > 0.0


def test_featurize_batch_t_normalized_in_unit_interval(featurized_batch: FeaturizedBatch) -> None:
    """featurize_batch normalises t_hat to the [0, 1] interval via the log-linear schedule."""
    assert 0.0 <= featurized_batch.t_normalized <= 1.0


def test_featurize_batch_ref_space_uid_all_zeros(featurized_batch: FeaturizedBatch) -> None:
    """ref_space_uid is 0 everywhere for single-chain proteins (no chain separators needed)."""
    # Single-chain proteins: ref_space_uid is 0 for all atoms in all batch items.
    assert (featurized_batch.ref_space_uid == 0).all()


def test_featurize_batch_tok_idx_maps_atoms_to_residues(featurized_batch: FeaturizedBatch) -> None:
    """tok_idx assigns atoms r*5 through r*5+4 to residue index r for every residue r."""
    for r in range(N_RES):
        assert (featurized_batch.tok_idx[:, r * 5 : (r + 1) * 5] == r).all()


def test_featurize_batch_center_uid_points_to_ca(featurized_batch: FeaturizedBatch) -> None:
    """center_uid[b, r] equals r*5+1, the C alpha atom index for residue r."""
    expected = torch.arange(N_RES) * 5 + 1  # (N_RES,)
    for b in range(B):
        assert torch.equal(featurized_batch.center_uid[b], expected)


def test_featurize_batch_returns_featurized_batch_instance(
    featurized_batch: FeaturizedBatch,
) -> None:
    """featurize_batch returns an instance of the FeaturizedBatch dataclass."""
    assert isinstance(featurized_batch, FeaturizedBatch)


# ---------------------------------------------------------------------------
# featurize_batch — ProteinBatch type enforcement
# ---------------------------------------------------------------------------


def test_featurize_batch_rejects_wrong_atom_positions_rank() -> None:
    """ProteinBatch raises when atom_positions is 3-D instead of the required 4-D."""
    with pytest.raises((TypeError, Exception)):
        ProteinBatch(
            atom_positions=torch.randn(B, N_RES, 37),  # missing last dim
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=repeat(torch.arange(N_RES).float(), "n -> b n", b=B),
            seq=[AA_SEQ, AA_SEQ],
        )


def test_featurize_batch_rejects_wrong_atom_count() -> None:
    """ProteinBatch raises when atom_positions has 36 atom slots instead of the required 37."""
    # atom_positions second-to-last dim must be exactly 37.
    with pytest.raises((TypeError, Exception)):
        ProteinBatch(
            atom_positions=torch.randn(B, N_RES, 36, 3),
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=repeat(torch.arange(N_RES).float(), "n -> b n", b=B),
            seq=[AA_SEQ, AA_SEQ],
        )


# ---------------------------------------------------------------------------
# apply_conditioning_dropout
# ---------------------------------------------------------------------------


def test_conditioning_dropout_p1_distogram_zeroes_all(featurized_batch: FeaturizedBatch) -> None:
    """p_distogram=1.0 zeros the entire distogram and pseudo-β mask for valid residues."""
    out = apply_conditioning_dropout(
        featurized_batch, p_distogram=1.0, p_atom=0.0, p_seq=0.0, device="cpu"
    )
    # All valid residues should have their rows/cols zeroed
    assert out.gt_res_distogram.sum() == 0
    assert out.f_pseudo_beta_mask.sum() == 0


def test_conditioning_dropout_p1_atom_zeroes_all(featurized_batch: FeaturizedBatch) -> None:
    """p_atom=1.0 clears atom5_mask for all valid residues."""
    out = apply_conditioning_dropout(
        featurized_batch, p_distogram=0.0, p_atom=1.0, p_seq=0.0, device="cpu"
    )
    assert not out.atom5_mask.any()


def test_conditioning_dropout_p1_seq_sets_all_to_mask_token(
    featurized_batch: FeaturizedBatch,
) -> None:
    """p_seq=1.0 replaces all valid amino-acid indices with the mask token (20)."""
    out = apply_conditioning_dropout(
        featurized_batch, p_distogram=0.0, p_atom=0.0, p_seq=1.0, device="cpu"
    )
    valid = featurized_batch.residue_mask
    assert (out.aa_indices[valid] == 20).all()


def test_conditioning_dropout_p0_is_noop(featurized_batch: FeaturizedBatch) -> None:
    """All dropout probabilities at 0 leaves distogram, atom mask, and sequence unchanged."""
    out = apply_conditioning_dropout(
        featurized_batch, p_distogram=0.0, p_atom=0.0, p_seq=0.0, device="cpu"
    )
    assert torch.equal(out.gt_res_distogram, featurized_batch.gt_res_distogram)
    assert torch.equal(out.atom5_mask, featurized_batch.atom5_mask)
    assert torch.equal(out.aa_indices, featurized_batch.aa_indices)


def test_conditioning_dropout_distogram_symmetric(featurized_batch: FeaturizedBatch) -> None:
    """Distogram dropout zeros both the row and column for each dropped residue."""
    manual_seed(0)
    out = apply_conditioning_dropout(
        featurized_batch, p_distogram=0.5, p_atom=0.0, p_seq=0.0, device="cpu"
    )
    # If row i is zeroed, column i must also be zeroed (and vice versa)
    row_sums = out.gt_res_distogram.sum(dim=(2, 3))  # (B, N_res)
    col_sums = out.gt_res_distogram.sum(dim=(1, 3))  # (B, N_res)
    assert torch.equal(row_sums == 0, col_sums == 0)


def test_conditioning_dropout_respects_residue_mask(featurized_batch: FeaturizedBatch) -> None:
    """Conditioning dropout never modifies padding residues (residue_mask=False)."""
    # Padding residues (residue_mask=False) must not be changed
    batch_with_padding = dataclasses.replace(
        featurized_batch,
        residue_mask=torch.zeros_like(featurized_batch.residue_mask, dtype=torch.bool),
    )
    out = apply_conditioning_dropout(
        batch_with_padding, p_distogram=1.0, p_atom=1.0, p_seq=1.0, device="cpu"
    )
    assert torch.equal(out.aa_indices, batch_with_padding.aa_indices)


# ---------------------------------------------------------------------------
# sinusoidal_encoding, ref_pos_for_residue, featurize_single_item, FeaturizedItem
# ---------------------------------------------------------------------------


def test_sinusoidal_encoding_output_shape() -> None:
    """sinusoidal_encoding maps (batch, N_res) positions to (batch, N_res, dim) encodings."""
    positions = rearrange(torch.arange(N_RES).float(), "n -> 1 n")  # (1, N_RES)
    out = sinusoidal_encoding(positions, dim=32)
    assert out.shape == (1, N_RES, 32)


def test_sinusoidal_encoding_output_finite() -> None:
    """sinusoidal_encoding produces finite values for standard residue indices."""
    positions = rearrange(torch.arange(N_RES).float(), "n -> 1 n")  # (1, N_RES)
    out = sinusoidal_encoding(positions, dim=32)
    assert torch.isfinite(out).all()


def testref_pos_for_residue_output_shape() -> None:
    """ref_pos_for_residue returns a (5, 3) tensor of reference atom positions."""
    pos = ref_pos_for_residue("ALA")
    assert pos.shape == (5, 3)


def testref_pos_for_residue_output_finite() -> None:
    """ref_pos_for_residue returns finite coordinates for a standard amino acid."""
    pos = ref_pos_for_residue("ALA")
    assert torch.isfinite(pos).all()


def test_featurize_single_item_returns_featurized_item(c_beta_distogram_fn: Distogram) -> None:
    """featurize_single_item returns a FeaturizedItem with the expected N_res."""
    atom37_positions = torch.randn(N_RES, 37, 3)
    atom37_mask = torch.ones(N_RES, 37)
    index = torch.arange(N_RES).float()
    ala_ref_pos = ref_pos_for_residue("ALA")
    ala_ref_elem = torch.zeros(5, 4)
    ala_ref_elem[:, 0] = 1.0  # C element for all atoms
    item = featurize_single_item(
        atom37_positions,
        atom37_mask,
        index,
        AA_SEQ,
        ala_ref_pos,
        ala_ref_elem,
        c_res=C_RES,
        c_beta_distogram_fn=c_beta_distogram_fn,
        device="cpu",
    )
    assert isinstance(item, FeaturizedItem)
    assert item.N_res == N_RES


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_distogram_forward_wrong_shape(disto: Distogram) -> None:
    """Wrong coords last dim (4 instead of 3) triggers TypeCheckError."""
    coords_bad = torch.zeros(N_RES, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        disto(coords_bad)


def test_sinusoidal_encoding_wrong_shape() -> None:
    """Wrong positions ndim (3-D instead of 2-D) triggers TypeCheckError."""
    positions_bad = torch.zeros(B, N_RES, 1)  # must be 2-D
    with pytest.raises(TypeCheckError):
        sinusoidal_encoding(positions_bad)


def testref_pos_for_residue_wrong_type() -> None:
    """Non-str resname triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        ref_pos_for_residue(42)  # type: ignore[arg-type]


def test_protein_batch_wrong_shape() -> None:
    """Wrong atom_positions last dim (4 instead of 3) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        ProteinBatch(
            atom_positions=torch.zeros(B, N_RES, 37, 4),  # last dim must be 3
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=repeat(torch.arange(N_RES, dtype=torch.float), "n -> b n", b=B),
            seq=[AA_SEQ] * B,
        )


def test_featurized_batch_wrong_shape() -> None:
    """Wrong ref_pos ndim (2-D instead of 3-D) triggers TypeCheckError."""
    n_atom = N_RES * 5
    k_local = 4
    with pytest.raises(TypeCheckError):
        FeaturizedBatch(
            ref_pos=torch.zeros(B, n_atom),  # must be 3-D
            ref_element=torch.zeros(B, n_atom, 4),
            ref_space_uid=torch.zeros(B, n_atom, dtype=torch.long),
            gt_res_distogram=torch.zeros(B, N_RES, N_RES, N_BINS, dtype=torch.long),
            f_pseudo_beta_mask=torch.zeros(B, N_RES, dtype=torch.long),
            f_residue_idx=torch.zeros(B, N_RES, C_RES),
            r_input=torch.zeros(B, n_atom, 3),
            r_gt=torch.zeros(B, n_atom, 3),
            atom5_mask=torch.zeros(B, n_atom, dtype=torch.bool),
            aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
            residue_mask=torch.zeros(B, N_RES, dtype=torch.bool),
            t_hat=1.0,
            t_normalized=0.5,
            tok_idx=torch.zeros(B, n_atom, dtype=torch.long),
            center_uid=torch.zeros(B, N_RES, dtype=torch.long),
            gt_atom_distogram_sparse=torch.zeros(B, n_atom, k_local, 5),
            gt_atom_distogram_mask_sparse=torch.zeros(B, n_atom, k_local, dtype=torch.bool),
        )


def test_featurized_item_wrong_shape() -> None:
    """Wrong flat_pos ndim (1-D instead of 2-D) triggers TypeCheckError."""
    n_atom = N_RES * 5
    with pytest.raises(TypeCheckError):
        FeaturizedItem(
            N_res=N_RES,
            flat_pos=torch.zeros(n_atom),  # must be 2-D "N_atom 3"
            atom_mask_flat=torch.zeros(n_atom, dtype=torch.bool),
            residue_mask=torch.zeros(N_RES, dtype=torch.bool),
            f_pseudo_beta=torch.zeros(N_RES, dtype=torch.long),
            gt_res_distogram=torch.zeros(N_RES, N_RES, N_BINS, dtype=torch.long),
            aa_indices=torch.zeros(N_RES, dtype=torch.long),
            ref_pos=torch.zeros(n_atom, 3),
            ref_element=torch.zeros(n_atom, 4),
            f_residue_idx=torch.zeros(N_RES, C_RES),
        )


def test_featurize_single_item_wrong_shape(disto: Distogram) -> None:
    """Wrong atom37_positions last dim (4 instead of 3) triggers TypeCheckError."""
    ala_ref_pos = ref_pos_for_residue("ALA")
    ala_ref_elem = torch.zeros(5, 4)
    positions_bad = torch.zeros(N_RES, 37, 4)  # last dim must be 3
    atom37_mask = torch.ones(N_RES, 37)
    index = torch.arange(N_RES, dtype=torch.float)
    with pytest.raises(TypeCheckError):
        featurize_single_item(
            positions_bad,
            atom37_mask,
            index,
            AA_SEQ,
            ala_ref_pos,
            ala_ref_elem,
            c_res=C_RES,
            c_beta_distogram_fn=disto,
            device="cpu",
        )


def test_featurize_batch_wrong_type() -> None:
    """Non-ProteinBatch first arg triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        featurize_batch("not a batch", None, None, None)  # type: ignore[arg-type]


def test_apply_conditioning_dropout_wrong_type() -> None:
    """Non-FeaturizedBatch first arg triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        apply_conditioning_dropout("not a batch", 0.5, 0.5, 0.5, "cpu")  # type: ignore[arg-type]
