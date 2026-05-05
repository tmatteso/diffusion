import dataclasses
import pytest
import torch
import torch.nn as nn
from einops import rearrange, reduce

from helpers.featurize import Distogram, FeaturizedBatch, ProteinBatch, ResidueIndexEmbedding, featurize_batch
from train.train_config import TrainConfig

torch.manual_seed(42)

B = 2
N_RES = 12
N_BINS = 16
C_RES = 32
MIN_DIST = 2.0
MAX_DIST = 22.0
AA_SEQ = "ACDEFGHIKLMN"   # length N_RES


# ---------------------------------------------------------------------------
# Fixtures — models
# ---------------------------------------------------------------------------

@pytest.fixture
def disto():
    return Distogram(n_bins=N_BINS, min_dist=MIN_DIST, max_dist=MAX_DIST).eval()


@pytest.fixture
def disto_overflow():
    return Distogram(n_bins=N_BINS, min_dist=MIN_DIST, max_dist=MAX_DIST, overflow_bin=True).eval()


@pytest.fixture
def emb():
    return ResidueIndexEmbedding(max_residues=256, c_res=C_RES).eval()


# ---------------------------------------------------------------------------
# Fixtures — tensors
# ---------------------------------------------------------------------------

@pytest.fixture
def coords():
    return torch.randn(N_RES, 3)


@pytest.fixture
def coords_batch():
    return torch.randn(2, N_RES, 3)


@pytest.fixture
def mask():
    return torch.ones(N_RES, dtype=torch.bool)


@pytest.fixture
def residue_indices():
    return torch.arange(N_RES)


# ---------------------------------------------------------------------------
# Distogram — output shapes
# ---------------------------------------------------------------------------

def test_distogram_output_shapes_unbatched(disto, coords):
    with torch.no_grad():
        f, m = disto(coords)
    assert f.shape == (N_RES, N_RES, N_BINS)
    assert m.shape == (N_RES, N_RES)


def test_distogram_output_shapes_batched(disto, coords_batch):
    with torch.no_grad():
        f, m = disto(coords_batch)
    assert f.shape == (2, N_RES, N_RES, N_BINS)
    assert m.shape == (2, N_RES, N_RES)


def test_distogram_overflow_bin_output_shape(disto_overflow, coords):
    with torch.no_grad():
        f, _ = disto_overflow(coords)
    assert f.shape == (N_RES, N_RES, N_BINS + 1)


# ---------------------------------------------------------------------------
# Distogram — output dtypes
# ---------------------------------------------------------------------------

def test_distogram_f_distogram_is_float(disto, coords):
    with torch.no_grad():
        f, _ = disto(coords)
    assert f.dtype == torch.float32


def test_distogram_f_pair_mask_is_bool(disto, coords):
    with torch.no_grad():
        _, m = disto(coords)
    assert m.dtype == torch.bool


# ---------------------------------------------------------------------------
# Distogram — one-hot property
# ---------------------------------------------------------------------------

def test_distogram_one_hot_sums_to_one(disto, coords):
    with torch.no_grad():
        f, _ = disto(coords)
    bin_sums = reduce(f, "i j b -> i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


def test_distogram_overflow_one_hot_sums_to_one(disto_overflow, coords):
    with torch.no_grad():
        f, _ = disto_overflow(coords)
    bin_sums = reduce(f, "i j b -> i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


# ---------------------------------------------------------------------------
# Distogram — pair mask
# ---------------------------------------------------------------------------

def test_distogram_mask_none_gives_all_true_within_range(disto, coords):
    # Without a coords_mask, every pair within max_dist should be unmasked.
    # Place coords all at origin so all distances are 0 (< max_dist).
    c = torch.zeros(N_RES, 3)
    with torch.no_grad():
        _, m = disto(c)
    assert m.all()


def test_distogram_mask_zeros_out_invalid_residues(disto, coords):
    mask = torch.ones(N_RES, dtype=torch.bool)
    mask[0] = False
    with torch.no_grad():
        _, m = disto(coords, mask)
    # Row 0 and column 0 must all be False
    assert not m[0, :].any()
    assert not m[:, 0].any()
    # All other pairs still potentially valid
    assert m[1:, 1:].any()


def test_distogram_overflow_mask_ignores_distance_cutoff(disto_overflow):
    # With overflow_bin=True the mask does NOT apply a distance cutoff —
    # pairs are valid as long as both atoms have valid coords.
    c = torch.zeros(N_RES, 3)
    c[0] = 1000.0   # very far from all others
    with torch.no_grad():
        _, m = disto_overflow(c)
    assert m.all()


def test_distogram_no_overflow_masks_distant_pairs(disto):
    # Without overflow_bin, pairs beyond max_dist are masked out.
    c = torch.zeros(N_RES, 3)
    c[0] = MAX_DIST * 2   # residue 0 far from residue 1..N-1
    with torch.no_grad():
        _, m = disto(c)
    # Pair (0, 1..N-1) should be masked
    assert not m[0, 1:].any()


# ---------------------------------------------------------------------------
# Distogram — symmetry
# ---------------------------------------------------------------------------

def test_distogram_is_symmetric(disto, coords):
    with torch.no_grad():
        f, m = disto(coords)
    assert torch.allclose(f, rearrange(f, "i j b -> j i b"))
    assert torch.equal(m, rearrange(m, "i j -> j i"))


# ---------------------------------------------------------------------------
# Distogram — bin correctness
# ---------------------------------------------------------------------------

def test_distogram_close_pairs_land_in_first_bin(disto):
    # All coords at origin → all distances are 0 → all pairs in bin 0.
    c = torch.zeros(N_RES, 3)
    with torch.no_grad():
        f, _ = disto(c)
    assert f[..., 0].all()


def test_distogram_overflow_far_pairs_land_in_last_bin(disto_overflow):
    # Two groups of atoms separated by >> max_dist.
    c = torch.zeros(N_RES, 3)
    c[N_RES // 2:] = MAX_DIST * 10
    with torch.no_grad():
        f, _ = disto_overflow(c)
    # Cross-group pairs must occupy the overflow bin (last bin).
    cross = f[:N_RES // 2, N_RES // 2:, :]
    assert cross[..., -1].all()


# ---------------------------------------------------------------------------
# ResidueIndexEmbedding — output shapes and dtype
# ---------------------------------------------------------------------------

def test_residue_index_embedding_output_shape(emb, residue_indices):
    with torch.no_grad():
        out = emb(residue_indices)
    assert out.shape == (N_RES, C_RES)


def test_residue_index_embedding_output_dtype(emb, residue_indices):
    with torch.no_grad():
        out = emb(residue_indices)
    assert out.dtype == torch.float32


def test_residue_index_embedding_output_finite(emb, residue_indices):
    with torch.no_grad():
        out = emb(residue_indices)
    assert torch.isfinite(out).all()


def test_residue_index_embedding_different_indices_give_different_output(emb):
    with torch.no_grad():
        out0 = emb(torch.tensor([0]))
        out1 = emb(torch.tensor([1]))
    assert not torch.allclose(out0, out1)


def test_residue_index_embedding_same_index_gives_same_output(emb):
    with torch.no_grad():
        out_a = emb(torch.tensor([5]))
        out_b = emb(torch.tensor([5]))
    assert torch.equal(out_a, out_b)


def test_residue_index_embedding_gradient_flows(emb, residue_indices):
    out = emb(residue_indices)
    reduce(out, "n c -> ", "sum").backward()
    assert emb.embed.weight.grad is not None
    assert torch.isfinite(emb.embed.weight.grad).all()


# ---------------------------------------------------------------------------
# featurize_batch — fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def protein_batch() -> ProteinBatch:
    return ProteinBatch(
        atom_positions=torch.randn(B, N_RES, 37, 3),
        atom_mask=torch.ones(B, N_RES, 37),
        residue_index=torch.arange(N_RES).float().unsqueeze(0).expand(B, -1).clone(),
        seq=[AA_SEQ, AA_SEQ],
    )


@pytest.fixture
def tcfg() -> TrainConfig:
    return TrainConfig()


@pytest.fixture
def c_beta_distogram_fn(tcfg: TrainConfig) -> Distogram:
    dr = tcfg.distogram_res
    return Distogram(n_bins=dr.n_bins, min_dist=dr.min_dist, max_dist=dr.max_dist, overflow_bin=True).eval()


@pytest.fixture
def atom_distogram_fn(tcfg: TrainConfig) -> Distogram:
    da = tcfg.distogram_atom
    return Distogram(n_bins=da.n_bins, min_dist=da.min_dist, max_dist=da.max_dist).eval()


@pytest.fixture
def index_embedding(tcfg: TrainConfig):
    return nn.Embedding(tcfg.model.max_residues, tcfg.model.c_res)


@pytest.fixture
def featurized_batch(protein_batch, tcfg, c_beta_distogram_fn, atom_distogram_fn, index_embedding) -> FeaturizedBatch:
    return featurize_batch(protein_batch, tcfg, c_beta_distogram_fn, atom_distogram_fn, index_embedding, device="cpu")


# ---------------------------------------------------------------------------
# featurize_batch — output shapes
# Batched layout for B=2, N_RES=12:  tensors are (B, N_RES, *) or (B, N_ATOM, *)
# ---------------------------------------------------------------------------

N_ATOM = N_RES * 5   # 5 atoms per residue, no separators


def test_featurize_batch_ref_pos_shape(featurized_batch):
    assert featurized_batch.ref_pos.shape == (B, N_ATOM, 3)


def test_featurize_batch_ref_element_shape(featurized_batch):
    assert featurized_batch.ref_element.shape == (B, N_ATOM, 4)


def test_featurize_batch_ref_space_uid_shape(featurized_batch):
    assert featurized_batch.ref_space_uid.shape == (B, N_ATOM)


def test_featurize_batch_distogram_shape(featurized_batch):
    assert featurized_batch.gt_res_distogram.shape == (B, N_RES, N_RES, 39)


def test_featurize_batch_pseudo_beta_mask_shape(featurized_batch):
    assert featurized_batch.f_pseudo_beta_mask.shape == (B, N_RES)


def test_featurize_batch_r_input_shape(featurized_batch):
    assert featurized_batch.r_input.shape == (B, N_ATOM, 3)


def test_featurize_batch_tok_idx_shape(featurized_batch):
    assert featurized_batch.tok_idx.shape == (B, N_ATOM)


def test_featurize_batch_center_uid_shape(featurized_batch):
    assert featurized_batch.center_uid.shape == (B, N_RES)


# ---------------------------------------------------------------------------
# featurize_batch — output values
# ---------------------------------------------------------------------------

def test_featurize_batch_all_tensor_fields_finite(featurized_batch):
    for field in dataclasses.fields(featurized_batch):
        val = getattr(featurized_batch, field.name)
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val.float()).all(), f"non-finite in field '{field.name}'"


def test_featurize_batch_t_hat_is_positive(featurized_batch):
    assert featurized_batch.t_hat > 0.0


def test_featurize_batch_t_normalized_in_unit_interval(featurized_batch):
    assert 0.0 <= featurized_batch.t_normalized <= 1.0


def test_featurize_batch_ref_space_uid_all_zeros(featurized_batch):
    # Single-chain proteins: ref_space_uid is 0 for all atoms in all batch items.
    assert (featurized_batch.ref_space_uid == 0).all()


def test_featurize_batch_tok_idx_maps_atoms_to_residues(featurized_batch):
    for r in range(N_RES):
        assert (featurized_batch.tok_idx[:, r * 5 : (r + 1) * 5] == r).all()


def test_featurize_batch_center_uid_points_to_ca(featurized_batch):
    expected = torch.arange(N_RES) * 5 + 1   # (N_RES,)
    for b in range(B):
        assert torch.equal(featurized_batch.center_uid[b], expected)


def test_featurize_batch_returns_featurized_batch_instance(featurized_batch):
    assert isinstance(featurized_batch, FeaturizedBatch)


# ---------------------------------------------------------------------------
# featurize_batch — ProteinBatch type enforcement
# ---------------------------------------------------------------------------

def test_featurize_batch_rejects_wrong_atom_positions_rank():
    with pytest.raises(Exception):
        ProteinBatch(
            atom_positions=torch.randn(B, N_RES, 37),   # missing last dim
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=torch.arange(N_RES).float().unsqueeze(0).expand(B, -1).clone(),
            seq=[AA_SEQ, AA_SEQ],
        )


def test_featurize_batch_rejects_wrong_atom_count():
    # atom_positions second-to-last dim must be exactly 37.
    with pytest.raises(Exception):
        ProteinBatch(
            atom_positions=torch.randn(B, N_RES, 36, 3),
            atom_mask=torch.ones(B, N_RES, 37),
            residue_index=torch.arange(N_RES).float().unsqueeze(0).expand(B, -1).clone(),
            seq=[AA_SEQ, AA_SEQ],
        )
