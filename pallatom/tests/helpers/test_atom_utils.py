import numpy as np
import pytest
import torch
from helpers.atom_utils import (
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_C,
    ATOM37_N,
    ATOM37_O,
    ATOM5_CB,
    PDB_MAX_CHAINS,
    Protein,
    _chain_end,
    atom37_to_atom5,
    atom37_to_cb,
    atom_types,
    center_positions,
    get_cb_coords,
    make_fixed_size,
    make_np_example,
    pseudo_cb,
    restype_num,
    to_pdb,
)

torch.manual_seed(42)

B = 2
N_RES = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atom37_positions():
    return torch.randn(B, N_RES, 37, 3)


@pytest.fixture
def atom37_mask():
    return torch.ones(B, N_RES, 37)


@pytest.fixture
def atom5_positions():
    return torch.randn(B, N_RES, 5, 3)


@pytest.fixture
def atom5_mask():
    return torch.ones(B, N_RES, 5)


@pytest.fixture
def n():
    return torch.randn(B, N_RES, 3)


@pytest.fixture
def ca():
    return torch.randn(B, N_RES, 3)


@pytest.fixture
def c():
    return torch.randn(B, N_RES, 3)


# ---------------------------------------------------------------------------
# pseudo_cb
# ---------------------------------------------------------------------------

def test_pseudo_cb_output_shape(n, ca, c):
    out = pseudo_cb(n, ca, c)
    assert out.shape == (B, N_RES, 3)


def test_pseudo_cb_output_finite(n, ca, c):
    out = pseudo_cb(n, ca, c)
    assert torch.isfinite(out).all()


def test_pseudo_cb_unbatched_shape():
    n_  = torch.randn(N_RES, 3)
    ca_ = torch.randn(N_RES, 3)
    c_  = torch.randn(N_RES, 3)
    out = pseudo_cb(n_, ca_, c_)
    assert out.shape == (N_RES, 3)


def test_pseudo_cb_output_not_equal_ca(n, ca, c):
    out = pseudo_cb(n, ca, c)
    assert not torch.allclose(out, ca)


def test_pseudo_cb_single_residue_finite():
    # Single residue: shape (3,) — zero batch dims
    out = pseudo_cb(torch.randn(3), torch.randn(3), torch.randn(3))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_pseudo_cb_collinear_backbone_finite():
    # Degenerate case: collinear N, CA, C — cross product is zero;
    # the epsilon guard in linalg.norm must prevent NaN in output.
    n_  = torch.zeros(N_RES, 3)
    ca_ = torch.zeros(N_RES, 3)
    c_  = torch.zeros(N_RES, 3)
    out = pseudo_cb(n_, ca_, c_)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# atom37_to_atom5
# ---------------------------------------------------------------------------

def test_atom37_to_atom5_output_shapes(atom37_positions, atom37_mask):
    pos5, mask5 = atom37_to_atom5(atom37_positions, atom37_mask)
    assert pos5.shape  == (B, N_RES, 5, 3)
    assert mask5.shape == (B, N_RES, 5)


def test_atom37_to_atom5_selects_correct_atoms(atom37_positions, atom37_mask):
    pos5, _ = atom37_to_atom5(atom37_positions, atom37_mask)
    assert torch.equal(pos5[:, :, 0, :], atom37_positions[:, :, ATOM37_N,  :])
    assert torch.equal(pos5[:, :, 1, :], atom37_positions[:, :, ATOM37_CA, :])
    assert torch.equal(pos5[:, :, 2, :], atom37_positions[:, :, ATOM37_C,  :])
    assert torch.equal(pos5[:, :, 3, :], atom37_positions[:, :, ATOM37_O,  :])
    assert torch.equal(pos5[:, :, 4, :], atom37_positions[:, :, ATOM37_CB, :])


def test_atom37_to_atom5_mask_preserved(atom37_positions):
    mask = torch.zeros(B, N_RES, 37)
    mask[:, :, [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O]] = 1.0
    _, mask5 = atom37_to_atom5(atom37_positions, mask)
    assert (mask5[:, :, :4] == 1.0).all()   # backbone present
    assert (mask5[:, :,  4] == 0.0).all()   # CB absent


def test_atom37_to_atom5_output_finite(atom37_positions, atom37_mask):
    pos5, mask5 = atom37_to_atom5(atom37_positions, atom37_mask)
    assert torch.isfinite(pos5).all()
    assert torch.isfinite(mask5).all()


# ---------------------------------------------------------------------------
# get_cb_coords
# ---------------------------------------------------------------------------

def test_get_cb_coords_output_shapes(atom5_positions, atom5_mask):
    cb, cb_present = get_cb_coords(atom5_positions, atom5_mask)
    assert cb.shape         == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool


def test_get_cb_coords_real_cb_used_when_present(atom5_positions, atom5_mask):
    cb, cb_present = get_cb_coords(atom5_positions, atom5_mask, fill_pseudo=True)
    assert torch.allclose(cb, atom5_positions[:, :, ATOM5_CB, :])
    assert cb_present.all()


def test_get_cb_coords_pseudo_when_cb_absent(atom5_positions):
    mask = torch.ones(B, N_RES, 5)
    mask[:, :, ATOM5_CB] = 0.0
    cb, cb_present = get_cb_coords(atom5_positions, mask, fill_pseudo=True)
    assert not cb_present.any()
    assert torch.isfinite(cb).all()
    assert not torch.allclose(cb, atom5_positions[:, :, ATOM5_CB, :])


def test_get_cb_coords_no_fill_pseudo_returns_raw_slot(atom5_positions):
    mask = torch.ones(B, N_RES, 5)
    mask[:, :, ATOM5_CB] = 0.0
    cb, cb_present = get_cb_coords(atom5_positions, mask, fill_pseudo=False)
    assert torch.equal(cb, atom5_positions[:, :, ATOM5_CB, :])
    assert not cb_present.any()


def test_get_cb_coords_mixed_residues(atom5_positions):
    mask = torch.ones(B, N_RES, 5)
    mask[:, 0, ATOM5_CB] = 0.0   # residue 0 has no CB (Gly-like)
    cb, cb_present = get_cb_coords(atom5_positions, mask, fill_pseudo=True)
    assert not cb_present[:, 0].any()
    assert cb_present[:, 1:].all()
    assert torch.allclose(cb[:, 1:, :], atom5_positions[:, 1:, ATOM5_CB, :])


def test_get_cb_coords_pseudo_beta_mask_dtype(atom5_positions, atom5_mask):
    _, cb_present = get_cb_coords(atom5_positions, atom5_mask)
    assert cb_present.dtype == torch.bool


# ---------------------------------------------------------------------------
# atom37_to_cb
# ---------------------------------------------------------------------------

def test_atom37_to_cb_output_shapes(atom37_positions, atom37_mask):
    cb, cb_present = atom37_to_cb(atom37_positions, atom37_mask)
    assert cb.shape         == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool


def test_atom37_to_cb_output_finite(atom37_positions, atom37_mask):
    cb, _ = atom37_to_cb(atom37_positions, atom37_mask)
    assert torch.isfinite(cb).all()


def test_atom37_to_cb_all_cb_present(atom37_positions, atom37_mask):
    _, cb_present = atom37_to_cb(atom37_positions, atom37_mask)
    assert cb_present.all()


def test_atom37_to_cb_glycine_gets_pseudo_cb(atom37_positions):
    mask = torch.ones(B, N_RES, 37)
    mask[:, :, ATOM37_CB] = 0.0
    cb, cb_present = atom37_to_cb(atom37_positions, mask)
    assert not cb_present.any()
    assert torch.isfinite(cb).all()


def test_atom37_to_cb_matches_manual_pipeline(atom37_positions, atom37_mask):
    cb_direct, mask_direct = atom37_to_cb(atom37_positions, atom37_mask)
    pos5, mask5 = atom37_to_atom5(atom37_positions, atom37_mask)
    cb_manual, mask_manual = get_cb_coords(pos5, mask5, fill_pseudo=True)
    assert torch.equal(cb_direct, cb_manual)
    assert torch.equal(mask_direct, mask_manual)


# ---------------------------------------------------------------------------
# Protein dataclass
# ---------------------------------------------------------------------------

N_ATOM_TYPE = 37


@pytest.fixture
def valid_protein():
    return Protein(
        atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float32),
        aatype=np.zeros(N_RES, dtype=np.int32),
        atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float32),
        residue_index=np.arange(N_RES, dtype=np.int32),
        chain_index=np.zeros(N_RES, dtype=np.int32),
        b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float32),
    )


def test_protein_accepts_valid_input(valid_protein):
    assert valid_protein.atom_positions.shape == (N_RES, N_ATOM_TYPE, 3)
    assert valid_protein.aatype.shape == (N_RES,)
    assert valid_protein.atom_mask.shape == (N_RES, N_ATOM_TYPE)
    assert valid_protein.residue_index.shape == (N_RES,)
    assert valid_protein.chain_index.shape == (N_RES,)
    assert valid_protein.b_factors.shape == (N_RES, N_ATOM_TYPE)


def test_protein_is_frozen(valid_protein):
    with pytest.raises(Exception):
        valid_protein.aatype = np.zeros(N_RES, dtype=np.int32)


def test_protein_rejects_wrong_atom_positions_rank():
    with pytest.raises(Exception):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float32),  # missing 3
            aatype=np.zeros(N_RES, dtype=np.int32),
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float32),
            residue_index=np.arange(N_RES, dtype=np.int32),
            chain_index=np.zeros(N_RES, dtype=np.int32),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float32),
        )


def test_protein_rejects_wrong_coord_dim():
    with pytest.raises(Exception):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 4), dtype=np.float32),  # 4 not 3
            aatype=np.zeros(N_RES, dtype=np.int32),
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float32),
            residue_index=np.arange(N_RES, dtype=np.int32),
            chain_index=np.zeros(N_RES, dtype=np.int32),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float32),
        )


def test_protein_rejects_float_aatype():
    with pytest.raises(Exception):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float32),
            aatype=np.zeros(N_RES, dtype=np.float32),   # float, not int
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float32),
            residue_index=np.arange(N_RES, dtype=np.int32),
            chain_index=np.zeros(N_RES, dtype=np.int32),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float32),
        )


def test_protein_rejects_inconsistent_num_res():
    with pytest.raises(Exception):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float32),
            aatype=np.zeros(N_RES + 1, dtype=np.int32),   # mismatched num_res
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float32),
            residue_index=np.arange(N_RES, dtype=np.int32),
            chain_index=np.zeros(N_RES, dtype=np.int32),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# make_np_example
# ---------------------------------------------------------------------------

NP_NUM_RES = 6


@pytest.fixture
def coords_dict():
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal((NP_NUM_RES, 3)) for name in ('N', 'CA', 'C', 'O')}


@pytest.fixture
def np_example(coords_dict):
    return make_np_example(coords_dict)


def test_make_np_example_output_keys(np_example):
    assert {'atom_positions', 'atom_mask', 'residue_index'} <= np_example.keys()


def test_make_np_example_atom_positions_shape(np_example):
    assert np_example['atom_positions'].shape == (NP_NUM_RES, 37, 3)


def test_make_np_example_atom_mask_shape(np_example):
    assert np_example['atom_mask'].shape == (NP_NUM_RES, 37)


def test_make_np_example_residue_index_is_arange(np_example):
    np.testing.assert_array_equal(np_example['residue_index'], np.arange(NP_NUM_RES))


def test_make_np_example_backbone_atoms_masked(np_example):
    bb_atom_types = {'N', 'CA', 'C', 'O'}
    for i, atom_type in enumerate(atom_types):
        if atom_type in bb_atom_types:
            assert (np_example['atom_mask'][:, i] == 1.0).all()


def test_make_np_example_nan_coords_zeroed(coords_dict):
    coords_dict['N'][0] = [float('nan'), float('nan'), float('nan')]
    batch = make_np_example(coords_dict)
    assert np.isfinite(batch['atom_positions']).all()


def test_make_np_example_nan_coords_zero_mask(coords_dict):
    coords_dict['N'][0] = [float('nan'), float('nan'), float('nan')]
    batch = make_np_example(coords_dict)
    n_idx = next(i for i, t in enumerate(atom_types) if t == 'N')
    assert batch['atom_mask'][0, n_idx] == 0.0


# ---------------------------------------------------------------------------
# make_fixed_size
# ---------------------------------------------------------------------------

NP_MAX_LEN = 10


@pytest.fixture
def short_np_example():
    return {
        'atom_positions': np.zeros((5, 37, 3)),
        'atom_mask': np.ones((5, 37)),
        'residue_index': np.arange(5),
    }


@pytest.fixture
def long_np_example():
    return {
        'atom_positions': np.random.randn(20, 37, 3),
        'residue_index': np.arange(20),
    }


@pytest.fixture
def exact_np_example():
    return {'residue_index': np.arange(NP_MAX_LEN)}


@pytest.fixture
def ones_np_example():
    return {'residue_index': np.ones(3)}


def test_make_fixed_size_pads_shorter_sequence(short_np_example):
    make_fixed_size(short_np_example, max_seq_length=NP_MAX_LEN)
    assert short_np_example['atom_positions'].shape[0] == NP_MAX_LEN
    assert short_np_example['atom_mask'].shape[0] == NP_MAX_LEN
    assert short_np_example['residue_index'].shape[0] == NP_MAX_LEN


def test_make_fixed_size_truncates_longer_sequence(long_np_example):
    make_fixed_size(long_np_example, max_seq_length=NP_MAX_LEN)
    assert long_np_example['atom_positions'].shape[0] == NP_MAX_LEN
    assert long_np_example['residue_index'].shape[0] == NP_MAX_LEN


def test_make_fixed_size_no_change_when_exact(exact_np_example):
    make_fixed_size(exact_np_example, max_seq_length=NP_MAX_LEN)
    assert exact_np_example['residue_index'].shape[0] == NP_MAX_LEN


def test_make_fixed_size_padded_values_are_zero(ones_np_example):
    make_fixed_size(ones_np_example, max_seq_length=6)
    assert (ones_np_example['residue_index'][3:] == 0.0).all()


# ---------------------------------------------------------------------------
# center_positions
# ---------------------------------------------------------------------------

@pytest.fixture
def full_mask_np_example():
    rng = np.random.default_rng(1)
    return {
        'atom_positions': rng.standard_normal((8, 37, 3)),
        'atom_mask': np.ones((8, 37)),
    }


@pytest.fixture
def ca_only_np_example():
    rng = np.random.default_rng(2)
    mask = np.zeros((5, 37))
    mask[:, 1] = 1.0
    return {
        'atom_positions': rng.standard_normal((5, 37, 3)),
        'atom_mask': mask,
    }


def test_center_positions_ca_center_at_origin(full_mask_np_example):
    center_positions(full_mask_np_example)
    ca_center = full_mask_np_example['atom_positions'][:, 1, :].mean(axis=0)
    np.testing.assert_allclose(ca_center, np.zeros(3), atol=1e-6)


def test_center_positions_masked_atoms_remain_zero(ca_only_np_example):
    center_positions(ca_only_np_example)
    np.testing.assert_array_equal(ca_only_np_example['atom_positions'][:, 0, :], 0.0)


def test_center_positions_modifies_in_place(full_mask_np_example):
    original = full_mask_np_example['atom_positions'].copy()
    center_positions(full_mask_np_example)
    assert not np.allclose(full_mask_np_example['atom_positions'], original)


# ---------------------------------------------------------------------------
# _chain_end
# ---------------------------------------------------------------------------

def test_chain_end_starts_with_ter():
    result = _chain_end(100, 'ALA', 'A', 10)
    assert result.startswith('TER')


def test_chain_end_contains_resname_and_chain():
    result = _chain_end(100, 'GLY', 'B', 42)
    assert 'GLY' in result
    assert 'B' in result


# ---------------------------------------------------------------------------
# to_pdb
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_protein():
    num_res = 5
    atom_mask = np.zeros((num_res, 37), dtype=np.float32)
    atom_mask[:, [0, 1, 2, 3]] = 1.0
    return Protein(
        atom_positions=np.random.default_rng(4).standard_normal((num_res, 37, 3)).astype(np.float32),
        aatype=np.zeros(num_res, dtype=np.int32),
        atom_mask=atom_mask,
        residue_index=np.arange(num_res, dtype=np.int32),
        chain_index=np.zeros(num_res, dtype=np.int32),
        b_factors=np.zeros((num_res, 37), dtype=np.float32),
    )


@pytest.fixture
def ca_only_protein():
    num_res = 3
    atom_mask = np.zeros((num_res, 37), dtype=np.float32)
    atom_mask[:, 1] = 1.0
    return Protein(
        atom_positions=np.random.default_rng(5).standard_normal((num_res, 37, 3)).astype(np.float32),
        aatype=np.zeros(num_res, dtype=np.int32),
        atom_mask=atom_mask,
        residue_index=np.arange(num_res, dtype=np.int32),
        chain_index=np.zeros(num_res, dtype=np.int32),
        b_factors=np.zeros((num_res, 37), dtype=np.float32),
    )


@pytest.fixture
def two_chain_protein():
    num_res = 4
    atom_mask = np.zeros((num_res, 37), dtype=np.float32)
    atom_mask[:, [0, 1, 2, 3]] = 1.0
    return Protein(
        atom_positions=np.random.default_rng(6).standard_normal((num_res, 37, 3)).astype(np.float32),
        aatype=np.zeros(num_res, dtype=np.int32),
        atom_mask=atom_mask,
        residue_index=np.arange(num_res, dtype=np.int32),
        chain_index=np.array([0, 0, 1, 1], dtype=np.int32),
        b_factors=np.zeros((num_res, 37), dtype=np.float32),
    )


def test_to_pdb_returns_string(simple_protein):
    assert isinstance(to_pdb(simple_protein), str)


def test_to_pdb_contains_model_endmdl_end(simple_protein):
    result = to_pdb(simple_protein)
    assert 'MODEL' in result
    assert 'ENDMDL' in result
    assert 'END' in result


def test_to_pdb_lines_padded_to_80(simple_protein):
    for line in to_pdb(simple_protein).splitlines():
        assert len(line) == 80


def test_to_pdb_contains_atom_records(simple_protein):
    assert 'ATOM' in to_pdb(simple_protein)


def test_to_pdb_skips_unmasked_atoms(ca_only_protein):
    atom_lines = [l for l in to_pdb(ca_only_protein).splitlines() if l.startswith('ATOM')]
    assert len(atom_lines) == ca_only_protein.atom_positions.shape[0]


def test_to_pdb_multichain_has_ter(two_chain_protein):
    assert 'TER' in to_pdb(two_chain_protein)


def test_to_pdb_raises_on_invalid_aatype():
    num_res = 3
    prot = Protein(
        atom_positions=np.zeros((num_res, 37, 3), dtype=np.float32),
        aatype=np.full(num_res, restype_num + 1, dtype=np.int32),
        atom_mask=np.ones((num_res, 37), dtype=np.float32),
        residue_index=np.arange(num_res, dtype=np.int32),
        chain_index=np.zeros(num_res, dtype=np.int32),
        b_factors=np.zeros((num_res, 37), dtype=np.float32),
    )
    with pytest.raises(ValueError, match='Invalid aatypes'):
        to_pdb(prot)


def test_to_pdb_raises_too_many_chains():
    num_res = 2
    prot = Protein(
        atom_positions=np.zeros((num_res, 37, 3), dtype=np.float32),
        aatype=np.zeros(num_res, dtype=np.int32),
        atom_mask=np.ones((num_res, 37), dtype=np.float32),
        residue_index=np.arange(num_res, dtype=np.int32),
        chain_index=np.array([0, PDB_MAX_CHAINS], dtype=np.int32),
        b_factors=np.zeros((num_res, 37), dtype=np.float32),
    )
    with pytest.raises(ValueError, match='chains'):
        to_pdb(prot)
