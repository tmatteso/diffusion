"""Tests for atom utility dataclasses and functions."""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pytest
import torch
from helpers.atom_utils import (
    ATOM5_CB,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
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
from jaxtyping import Float, TypeCheckError

torch.manual_seed(42)

B = 2
N_RES = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def atom37_positions() -> Float[torch.Tensor, "B N_res 37 3"]:
    """Provide random atom37 coordinate tensor (B, N_RES, 37, 3)."""
    return torch.randn(B, N_RES, 37, 3)


@pytest.fixture
def atom37_mask() -> Float[torch.Tensor, "B N_res 37"]:
    """Provide all-ones atom37 mask (B, N_RES, 37)."""
    return torch.ones(B, N_RES, 37)


@pytest.fixture
def atom5_positions() -> Float[torch.Tensor, "B N_res 5 3"]:
    """Provide random atom5 coordinate tensor (B, N_RES, 5, 3)."""
    return torch.randn(B, N_RES, 5, 3)


@pytest.fixture
def atom5_mask() -> Float[torch.Tensor, "B N_res 5"]:
    """Provide all-ones atom5 mask (B, N_RES, 5)."""
    return torch.ones(B, N_RES, 5)


@pytest.fixture
def n() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random backbone N atom positions (B, N_RES, 3)."""
    return torch.randn(B, N_RES, 3)


@pytest.fixture
def ca() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random C alpha atom positions (B, N_RES, 3)."""
    return torch.randn(B, N_RES, 3)


@pytest.fixture
def c() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random backbone C atom positions (B, N_RES, 3)."""
    return torch.randn(B, N_RES, 3)


# ---------------------------------------------------------------------------
# pseudo_cb
# ---------------------------------------------------------------------------


def test_pseudo_cb_output_shape(
    n: Float[torch.Tensor, "B N_res 3"],
    ca: Float[torch.Tensor, "B N_res 3"],
    c: Float[torch.Tensor, "B N_res 3"],
):
    """pseudo_cb output shape is (B, N_RES, 3)."""
    out = pseudo_cb(n, ca, c)
    assert out.shape == (B, N_RES, 3)


def test_pseudo_cb_output_finite(
    n: Float[torch.Tensor, "B N_res 3"],
    ca: Float[torch.Tensor, "B N_res 3"],
    c: Float[torch.Tensor, "B N_res 3"],
):
    """pseudo_cb output is finite for random backbone atoms."""
    out = pseudo_cb(n, ca, c)
    assert torch.isfinite(out).all()


def test_pseudo_cb_unbatched_shape():
    """pseudo_cb works on unbatched (N_RES, 3) inputs and returns (N_RES, 3)."""
    n_ = torch.randn(N_RES, 3)
    ca_ = torch.randn(N_RES, 3)
    c_ = torch.randn(N_RES, 3)
    out = pseudo_cb(n_, ca_, c_)
    assert out.shape == (N_RES, 3)


def test_pseudo_cb_output_not_equal_ca(
    n: Float[torch.Tensor, "B N_res 3"],
    ca: Float[torch.Tensor, "B N_res 3"],
    c: Float[torch.Tensor, "B N_res 3"],
):
    """pseudo_cb coordinates differ from C alpha coordinates."""
    out = pseudo_cb(n, ca, c)
    assert not torch.allclose(out, ca)


def test_pseudo_cb_single_residue_finite():
    """pseudo_cb handles a scalar (3,) input without batch or residue dimensions."""
    out = pseudo_cb(torch.randn(3), torch.randn(3), torch.randn(3))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_pseudo_cb_collinear_backbone_finite():
    """pseudo_cb stays finite when N, CA, C are collinear so the cross product is zero.

    The epsilon guard in linalg.norm must prevent NaN in the normalised output.
    """
    n_ = torch.zeros(N_RES, 3)
    ca_ = torch.zeros(N_RES, 3)
    c_ = torch.zeros(N_RES, 3)
    out = pseudo_cb(n_, ca_, c_)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# atom37_to_atom5
# ---------------------------------------------------------------------------


def test_atom37_to_atom5_output_shapes(atom37_positions: torch.Tensor, atom37_mask: torch.Tensor):
    """atom37_to_atom5 returns position and mask tensors with the correct shapes."""
    pos5, mask5 = atom37_to_atom5(atom37_positions, atom37_mask)
    assert pos5.shape == (B, N_RES, 5, 3)
    assert mask5.shape == (B, N_RES, 5)


def test_atom37_to_atom5_selects_correct_atoms(
    atom37_positions: torch.Tensor, atom37_mask: torch.Tensor
):
    """atom37_to_atom5 places N, CA, C, O, CB into the expected atom5 slots."""
    pos5, _ = atom37_to_atom5(atom37_positions, atom37_mask)
    assert torch.equal(pos5[:, :, 0, :], atom37_positions[:, :, ATOM37_N, :])
    assert torch.equal(pos5[:, :, 1, :], atom37_positions[:, :, ATOM37_CA, :])
    assert torch.equal(pos5[:, :, 2, :], atom37_positions[:, :, ATOM37_C, :])
    assert torch.equal(pos5[:, :, 3, :], atom37_positions[:, :, ATOM37_O, :])
    assert torch.equal(pos5[:, :, 4, :], atom37_positions[:, :, ATOM37_CB, :])


def test_atom37_to_atom5_mask_preserved(atom37_positions: torch.Tensor):
    """atom37_to_atom5 propagates atom37 mask entries to the correct atom5 slots."""
    mask = torch.zeros(B, N_RES, 37)
    mask[:, :, [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O]] = 1.0
    _, mask5 = atom37_to_atom5(atom37_positions, mask)
    assert (mask5[:, :, :4] == 1.0).all()  # backbone present
    assert (mask5[:, :, 4] == 0.0).all()  # CB absent


def test_atom37_to_atom5_output_finite(atom37_positions: torch.Tensor, atom37_mask: torch.Tensor):
    """atom37_to_atom5 produces finite position and mask values for random input."""
    pos5, mask5 = atom37_to_atom5(atom37_positions, atom37_mask)
    assert torch.isfinite(pos5).all()
    assert torch.isfinite(mask5).all()


# ---------------------------------------------------------------------------
# get_cb_coords
# ---------------------------------------------------------------------------


def test_get_cb_coords_output_shapes(atom5_positions: torch.Tensor, atom5_mask: torch.Tensor):
    """get_cb_coords returns Cβ positions (B, N_RES, 3) and a bool mask (B, N_RES)."""
    cb, cb_present = get_cb_coords(atom5_positions, atom5_mask)
    assert cb.shape == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool


def test_get_cb_coords_real_cb_used_when_present(
    atom5_positions: torch.Tensor, atom5_mask: torch.Tensor
):
    """get_cb_coords returns the real atom5 CB slot when CB is present in the mask."""
    cb, cb_present = get_cb_coords(atom5_positions, atom5_mask, fill_pseudo=True)
    assert torch.allclose(cb, atom5_positions[:, :, ATOM5_CB, :])
    assert cb_present.all()


def test_get_cb_coords_pseudo_when_cb_absent(atom5_positions: torch.Tensor):
    """get_cb_coords falls back to the pseudo-Cβ position when CB is absent and fill_pseudo=True."""
    mask = torch.ones(B, N_RES, 5)
    mask[:, :, ATOM5_CB] = 0.0
    cb, cb_present = get_cb_coords(atom5_positions, mask, fill_pseudo=True)
    assert not cb_present.any()
    assert torch.isfinite(cb).all()
    assert not torch.allclose(cb, atom5_positions[:, :, ATOM5_CB, :])


def test_get_cb_coords_no_fill_pseudo_returns_raw_slot(atom5_positions: torch.Tensor):
    """get_cb_coords returns the raw (zero) CB slot without pseudo fill when fill_pseudo=False."""
    mask = torch.ones(B, N_RES, 5)
    mask[:, :, ATOM5_CB] = 0.0
    cb, cb_present = get_cb_coords(atom5_positions, mask, fill_pseudo=False)
    assert torch.equal(cb, atom5_positions[:, :, ATOM5_CB, :])
    assert not cb_present.any()


def test_get_cb_coords_mixed_residues(atom5_positions: torch.Tensor):
    """get_cb_coords selects pseudo-Cβ for residue 0 only and real CB for all others."""
    mask = torch.ones(B, N_RES, 5)
    mask[:, 0, ATOM5_CB] = 0.0  # residue 0 has no CB (Gly-like)
    cb, cb_present = get_cb_coords(atom5_positions, mask, fill_pseudo=True)
    assert not cb_present[:, 0].any()
    assert cb_present[:, 1:].all()
    assert torch.allclose(cb[:, 1:, :], atom5_positions[:, 1:, ATOM5_CB, :])


def test_get_cb_coords_pseudo_beta_mask_dtype(
    atom5_positions: torch.Tensor, atom5_mask: torch.Tensor
):
    """get_cb_coords always returns a bool tensor for the CB-present mask."""
    _, cb_present = get_cb_coords(atom5_positions, atom5_mask)
    assert cb_present.dtype == torch.bool


# ---------------------------------------------------------------------------
# atom37_to_cb
# ---------------------------------------------------------------------------


def test_atom37_to_cb_output_shapes(atom37_positions: torch.Tensor, atom37_mask: torch.Tensor):
    """atom37_to_cb returns Cβ positions (B, N_RES, 3) and a bool presence mask."""
    cb, cb_present = atom37_to_cb(atom37_positions, atom37_mask)
    assert cb.shape == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool


def test_atom37_to_cb_output_finite(atom37_positions: torch.Tensor, atom37_mask: torch.Tensor):
    """atom37_to_cb produces finite Cβ coordinates from random backbone atoms."""
    cb, _ = atom37_to_cb(atom37_positions, atom37_mask)
    assert torch.isfinite(cb).all()


def test_atom37_to_cb_all_cb_present(atom37_positions: torch.Tensor, atom37_mask: torch.Tensor):
    """atom37_to_cb marks all residues as CB-present when CB is set in the mask."""
    _, cb_present = atom37_to_cb(atom37_positions, atom37_mask)
    assert cb_present.all()


def test_atom37_to_cb_glycine_gets_pseudo_cb(atom37_positions: torch.Tensor):
    """atom37_to_cb falls back to pseudo-Cβ for glycine-like residues missing CB in mask."""
    mask = torch.ones(B, N_RES, 37)
    mask[:, :, ATOM37_CB] = 0.0
    cb, cb_present = atom37_to_cb(atom37_positions, mask)
    assert not cb_present.any()
    assert torch.isfinite(cb).all()


def test_atom37_to_cb_matches_manual_pipeline(
    atom37_positions: torch.Tensor, atom37_mask: torch.Tensor
):
    """atom37_to_cb is equivalent to atom37_to_atom5 followed by get_cb_coords."""
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
def valid_protein() -> Protein:
    """Provide a minimal valid Protein with N_RES residues and all atoms unmasked."""
    return Protein(
        atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float64),
        aatype=np.zeros(N_RES, dtype=np.intp),
        atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
        residue_index=np.arange(N_RES, dtype=np.intp),
        chain_index=np.zeros(N_RES, dtype=np.intp),
        b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
    )


def test_protein_accepts_valid_input(valid_protein: Protein):
    """Protein stores all fields with the expected shapes for a valid construction."""
    assert valid_protein.atom_positions.shape == (N_RES, N_ATOM_TYPE, 3)
    assert valid_protein.aatype.shape == (N_RES,)
    assert valid_protein.atom_mask.shape == (N_RES, N_ATOM_TYPE)
    assert valid_protein.residue_index.shape == (N_RES,)
    assert valid_protein.chain_index.shape == (N_RES,)
    assert valid_protein.b_factors.shape == (N_RES, N_ATOM_TYPE)


def test_protein_rejects_wrong_atom_positions_rank():
    """Protein raises when atom_positions is 2-D instead of the required 3-D (N_res, 37, 3)."""
    with pytest.raises(TypeCheckError):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),  # missing 3
            aatype=np.zeros(N_RES, dtype=np.intp),
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
        )


def test_protein_rejects_wrong_coord_dim():
    """Protein raises when the coordinate dimension is 4 instead of the required 3."""
    with pytest.raises(TypeCheckError):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 4), dtype=np.float64),  # 4 not 3
            aatype=np.zeros(N_RES, dtype=np.intp),
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
        )


def test_protein_rejects_inconsistent_num_res():
    """Protein raises when aatype has a different residue count than atom_positions."""
    with pytest.raises(TypeCheckError):
        Protein(
            atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float64),
            aatype=np.zeros(N_RES + 1, dtype=np.intp),  # mismatched num_res
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# make_np_example
# ---------------------------------------------------------------------------

NP_NUM_RES = 6


@pytest.fixture
def coords_dict() -> Mapping[str, npt.NDArray[np.float64]]:
    """Provide a random backbone coordinate dict with N, CA, C, O keys."""
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal((NP_NUM_RES, 3)) for name in ("N", "CA", "C", "O")}


@pytest.fixture
def np_example(
    coords_dict: Mapping[str, npt.NDArray[np.float64]],
) -> Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Provide the numpy example dict built from the coords_dict fixture."""
    return make_np_example(coords_dict)


def test_make_np_example_output_keys(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example returns a dict with at least atom_positions, atom_mask, residue_index."""
    assert {"atom_positions", "atom_mask", "residue_index"} <= np_example.keys()


def test_make_np_example_atom_positions_shape(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example packs backbone coords into (NP_NUM_RES, 37, 3) atom_positions."""
    assert np_example["atom_positions"].shape == (NP_NUM_RES, 37, 3)


def test_make_np_example_atom_mask_shape(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example produces an atom_mask of shape (NP_NUM_RES, 37)."""
    assert np_example["atom_mask"].shape == (NP_NUM_RES, 37)


def test_make_np_example_residue_index_is_arange(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example sets residue_index to a 0-based integer range."""
    np.testing.assert_array_equal(np_example["residue_index"], np.arange(NP_NUM_RES))


def test_make_np_example_backbone_atoms_masked(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example sets atom_mask to 1.0 for all provided backbone atom types."""
    bb_atom_types = {"N", "CA", "C", "O"}
    for i, atom_type in enumerate(atom_types):
        if atom_type in bb_atom_types:
            assert (np_example["atom_mask"][:, i] == 1.0).all()


def test_make_np_example_nan_coords_zeroed(
    coords_dict: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """make_np_example replaces NaN coordinate entries with zero to keep positions finite."""
    coords_dict["N"][0] = [float("nan"), float("nan"), float("nan")]
    batch = make_np_example(coords_dict)
    assert np.isfinite(batch["atom_positions"]).all()


def test_make_np_example_nan_coords_zero_mask(
    coords_dict: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """make_np_example sets atom_mask to 0 for residues whose coordinates were NaN."""
    coords_dict["N"][0] = [float("nan"), float("nan"), float("nan")]
    batch = make_np_example(coords_dict)
    n_idx = next(i for i, t in enumerate(atom_types) if t == "N")
    assert batch["atom_mask"][0, n_idx] == 0.0


# ---------------------------------------------------------------------------
# make_fixed_size
# ---------------------------------------------------------------------------

NP_MAX_LEN = 10


@pytest.fixture
def short_np_example() -> Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Provide a numpy example dict with 5 residues (shorter than NP_MAX_LEN=10)."""
    return {
        "atom_positions": np.zeros((5, 37, 3)),
        "atom_mask": np.ones((5, 37)),
        "residue_index": np.arange(5),
    }


@pytest.fixture
def long_np_example() -> Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Provide a numpy example dict with 20 residues (longer than NP_MAX_LEN=10)."""
    return {
        "atom_positions": np.random.randn(20, 37, 3),
        "residue_index": np.arange(20),
    }


@pytest.fixture
def exact_np_example() -> Mapping[str, npt.NDArray[np.intp]]:
    """Provide a numpy example dict with exactly NP_MAX_LEN=10 residues."""
    return {"residue_index": np.arange(NP_MAX_LEN)}


@pytest.fixture
def ones_np_example() -> Mapping[str, npt.NDArray[np.float64]]:
    """Provide a numpy example dict with 3 residues all set to residue index 1."""
    return {"residue_index": np.ones(3)}


def test_make_fixed_size_pads_shorter_sequence(
    short_np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_fixed_size zero-pads all arrays to NP_MAX_LEN when the input is shorter."""
    make_fixed_size(short_np_example, max_seq_length=NP_MAX_LEN)
    assert short_np_example["atom_positions"].shape[0] == NP_MAX_LEN
    assert short_np_example["atom_mask"].shape[0] == NP_MAX_LEN
    assert short_np_example["residue_index"].shape[0] == NP_MAX_LEN


def test_make_fixed_size_truncates_longer_sequence(
    long_np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_fixed_size truncates all arrays to NP_MAX_LEN when the input exceeds it."""
    make_fixed_size(long_np_example, max_seq_length=NP_MAX_LEN)
    assert long_np_example["atom_positions"].shape[0] == NP_MAX_LEN
    assert long_np_example["residue_index"].shape[0] == NP_MAX_LEN


def test_make_fixed_size_no_change_when_exact(
    exact_np_example: Mapping[str, npt.NDArray[np.intp]]
) -> None:
    """make_fixed_size leaves arrays unchanged when their length equals max_seq_length."""
    make_fixed_size(exact_np_example, max_seq_length=NP_MAX_LEN)
    assert exact_np_example["residue_index"].shape[0] == NP_MAX_LEN


def test_make_fixed_size_padded_values_are_zero(
    ones_np_example: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """make_fixed_size fills the padding region with zeros, not the original values."""
    make_fixed_size(ones_np_example, max_seq_length=6)
    assert (ones_np_example["residue_index"][3:] == 0.0).all()


# ---------------------------------------------------------------------------
# center_positions
# ---------------------------------------------------------------------------


@pytest.fixture
def full_mask_np_example() -> Mapping[str, npt.NDArray[np.float64]]:
    """Provide a numpy example with 8 residues and all atoms unmasked."""
    rng = np.random.default_rng(1)
    return {
        "atom_positions": rng.standard_normal((8, 37, 3)),
        "atom_mask": np.ones((8, 37)),
    }


@pytest.fixture
def ca_only_np_example() -> Mapping[str, npt.NDArray[np.float64]]:
    """Provide a numpy example with 5 residues where only the CA atom is unmasked."""
    rng = np.random.default_rng(2)
    mask = np.zeros((5, 37))
    mask[:, 1] = 1.0
    return {
        "atom_positions": rng.standard_normal((5, 37, 3)),
        "atom_mask": mask,
    }


def test_center_positions_ca_center_at_origin(
    full_mask_np_example: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """center_positions translates the structure so the mean CA position is at the origin."""
    center_positions(full_mask_np_example)
    ca_center = full_mask_np_example["atom_positions"][:, 1, :].mean(axis=0)
    np.testing.assert_allclose(ca_center, np.zeros(3), atol=1e-6)


def test_center_positions_masked_atoms_remain_zero(
    ca_only_np_example: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """center_positions leaves zero-masked atom slots at zero after centering."""
    center_positions(ca_only_np_example)
    np.testing.assert_array_equal(ca_only_np_example["atom_positions"][:, 0, :], 0.0)


def test_center_positions_modifies_in_place(
    full_mask_np_example: Mapping[str, npt.NDArray[np.float64]]
) -> None:
    """center_positions mutates the input dict's atom_positions array rather than copying."""
    original = full_mask_np_example["atom_positions"].copy()
    center_positions(full_mask_np_example)
    assert not np.allclose(full_mask_np_example["atom_positions"], original)


# ---------------------------------------------------------------------------
# _chain_end
# ---------------------------------------------------------------------------


def test_chain_end_starts_with_ter():
    """_chain_end returns a PDB-formatted line beginning with the TER record type."""
    result = _chain_end(100, "ALA", "A", 10)
    assert result.startswith("TER")


def test_chain_end_contains_resname_and_chain():
    """_chain_end embeds the residue name and chain ID in the returned TER record."""
    result = _chain_end(100, "GLY", "B", 42)
    assert "GLY" in result
    assert "B" in result


# ---------------------------------------------------------------------------
# to_pdb
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_protein() -> Protein:
    """Provide a 5-residue Protein with backbone atoms (N, CA, C, O) present."""
    num_res = 5
    atom_mask = np.zeros((num_res, 37), dtype=np.float64)
    atom_mask[:, [0, 1, 2, 3]] = 1.0
    return Protein(
        atom_positions=np.random.default_rng(4)
        .standard_normal((num_res, 37, 3))
        .astype(np.float64),
        aatype=np.zeros(num_res, dtype=np.intp),
        atom_mask=atom_mask,
        residue_index=np.arange(num_res, dtype=np.intp),
        chain_index=np.zeros(num_res, dtype=np.intp),
        b_factors=np.zeros((num_res, 37), dtype=np.float64),
    )


@pytest.fixture
def ca_only_protein() -> Protein:
    """Provide a 3-residue Protein with only the CA atom present per residue."""
    num_res = 3
    atom_mask = np.zeros((num_res, 37), dtype=np.float64)
    atom_mask[:, 1] = 1.0
    return Protein(
        atom_positions=np.random.default_rng(5)
        .standard_normal((num_res, 37, 3))
        .astype(np.float64),
        aatype=np.zeros(num_res, dtype=np.intp),
        atom_mask=atom_mask,
        residue_index=np.arange(num_res, dtype=np.intp),
        chain_index=np.zeros(num_res, dtype=np.intp),
        b_factors=np.zeros((num_res, 37), dtype=np.float64),
    )


@pytest.fixture
def two_chain_protein() -> Protein:
    """Provide a 4-residue Protein split across two chains (indices 0 and 1)."""
    num_res = 4
    atom_mask = np.zeros((num_res, 37), dtype=np.float64)
    atom_mask[:, [0, 1, 2, 3]] = 1.0
    return Protein(
        atom_positions=np.random.default_rng(6)
        .standard_normal((num_res, 37, 3))
        .astype(np.float64),
        aatype=np.zeros(num_res, dtype=np.intp),
        atom_mask=atom_mask,
        residue_index=np.arange(num_res, dtype=np.intp),
        chain_index=np.array([0, 0, 1, 1], dtype=np.intp),
        b_factors=np.zeros((num_res, 37), dtype=np.float64),
    )


def test_to_pdb_returns_string(simple_protein: Protein):
    """to_pdb returns a plain str, not bytes or another type."""
    assert isinstance(to_pdb(simple_protein), str)


def test_to_pdb_contains_model_endmdl_end(simple_protein: Protein):
    """to_pdb output contains the mandatory MODEL, ENDMDL, and END PDB records."""
    result = to_pdb(simple_protein)
    assert "MODEL" in result
    assert "ENDMDL" in result
    assert "END" in result


def test_to_pdb_lines_padded_to_80(simple_protein: Protein):
    """to_pdb right-pads every output line to exactly 80 characters as the PDB spec requires."""
    for line in to_pdb(simple_protein).splitlines():
        assert len(line) == 80


def test_to_pdb_contains_atom_records(simple_protein: Protein):
    """to_pdb includes at least one ATOM record for a protein with masked atoms."""
    assert "ATOM" in to_pdb(simple_protein)


def test_to_pdb_skips_unmasked_atoms(ca_only_protein: Protein):
    """to_pdb emits exactly one ATOM line per residue when only CA is unmasked."""
    atom_lines = [line for line in to_pdb(ca_only_protein).splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) == ca_only_protein.atom_positions.shape[0]


def test_to_pdb_multichain_has_ter(two_chain_protein: Protein):
    """to_pdb inserts a TER record between chains in a multi-chain protein."""
    assert "TER" in to_pdb(two_chain_protein)


def test_to_pdb_raises_on_invalid_aatype():
    """to_pdb raises ValueError when aatype contains an out-of-range residue type index."""
    num_res = 3
    prot = Protein(
        atom_positions=np.zeros((num_res, 37, 3), dtype=np.float64),
        aatype=np.full(num_res, restype_num + 1, dtype=np.intp),
        atom_mask=np.ones((num_res, 37), dtype=np.float64),
        residue_index=np.arange(num_res, dtype=np.intp),
        chain_index=np.zeros(num_res, dtype=np.intp),
        b_factors=np.zeros((num_res, 37), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="Invalid aatypes"):
        to_pdb(prot)


def test_to_pdb_raises_too_many_chains():
    """to_pdb raises ValueError when chain_index references more chains than PDB allows."""
    num_res = 2
    prot = Protein(
        atom_positions=np.zeros((num_res, 37, 3), dtype=np.float64),
        aatype=np.zeros(num_res, dtype=np.intp),
        atom_mask=np.ones((num_res, 37), dtype=np.float64),
        residue_index=np.arange(num_res, dtype=np.intp),
        chain_index=np.array([0, PDB_MAX_CHAINS], dtype=np.intp),
        b_factors=np.zeros((num_res, 37), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="chains"):
        to_pdb(prot)
