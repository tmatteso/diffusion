"""Tests for atom utility dataclasses and functions.

Covers pseudo_cb, atom37_to_atom14, atom14_to_atom37, atom37_to_cb,
atom14_to_cb, fill_atom14_virtual_atoms, generic_ref_pos_and_element, the
atom14 lookup tables, the
Protein dataclass, make_np_example, make_fixed_size, center_positions,
chain_end, to_pdb, protein_from_pdb, truncate_to_length, and molecule-type
constants.
"""

import pathlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from einops import rearrange
from helpers.atom_utils import (
    ATOM14_ATOM_EXISTS,
    ATOM14_C,
    ATOM14_CA,
    ATOM14_CB,
    ATOM14_ELEMENTS,
    ATOM14_N,
    ATOM14_O,
    ATOM14_REF_POS,
    ATOM37_ATOM_EXISTS,
    ATOM37_CB,
    ATOMS_PER_RES,
    DNA_RESTYPE_3TO1,
    DNA_RESTYPE_ORDER,
    DNA_RESTYPES,
    GENERIC_ATOM14_ELEMENTS,
    GENERIC_ATOM14_REF_POS,
    MOL_TYPE_DNA,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    PDB_CHAIN_IDS,
    PDB_MAX_CHAINS,
    PDB_REC_ATOM,
    PDB_REC_END,
    PDB_REC_ENDMDL,
    PDB_REC_MODEL,
    PDB_REC_TER,
    RESIDX_ATOM14_TO_ATOM37,
    RESIDX_ATOM37_TO_ATOM14,
    RESTYPE_NUM,
    RESTYPE_NUM_NO_X,
    RESTYPES,
    RNA_RESTYPE_3TO1,
    RNA_RESTYPE_ORDER,
    RNA_RESTYPES,
    Protein,
    _build_atom14_tables,  # pyright: ignore[reportPrivateUsage]
    _cb_from_backbone,  # pyright: ignore[reportPrivateUsage]
    atom14_to_atom37,
    atom14_to_cb,
    atom37_to_atom14,
    atom37_to_cb,
    atom_order,
    atom_types,
    center_positions,
    chain_end,
    classify_mol_type,
    fill_atom14_virtual_atoms,
    generic_ref_pos_and_element,
    make_fixed_size,
    make_np_example,
    protein_from_pdb,
    pseudo_cb,
    restype_1to3,
    restype_order,
    rigid_group_atom_positions,
    to_pdb,
    truncate_to_length,
)
from helpers.errors import InvalidAAtypesError
from helpers.useful_objects import manual_seed
from jaxtyping import Float, Int, TypeCheckError

_ = manual_seed(42)

B = 2
N_RES = 10
PDB_ATOM_NAME_CUTOFF = 4
PDB_LINE_LEN = 80
PROT_LEN = 3
_CB_ATOM_NAME = "CB"
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def atom37_positions() -> Float[torch.Tensor, "B N_res 37 3"]:
    """Provide random atom37 coordinate tensor (B, N_RES, 37, 3).

    Returns:
        A randomly initialised float tensor of shape (B, N_RES, 37, 3).
    """
    return torch.randn(B, N_RES, 37, 3)


@pytest.fixture
def atom37_mask() -> Float[torch.Tensor, "B N_res 37"]:
    """Provide all-ones atom37 mask (B, N_RES, 37).

    Returns:
        A ones tensor of shape (B, N_RES, 37) indicating all atoms present.
    """
    return torch.ones(B, N_RES, 37)


@pytest.fixture
def atom14_positions() -> Float[torch.Tensor, "B N_res 14 3"]:
    """Provide random atom14 coordinate tensor (B, N_RES, 14, 3).

    Returns:
        A randomly initialised float tensor of shape (B, N_RES, 14, 3).
    """
    return torch.randn(B, N_RES, ATOMS_PER_RES, 3)


@pytest.fixture
def atom14_mask() -> Float[torch.Tensor, "B N_res 14"]:
    """Provide all-ones atom14 mask (B, N_RES, 14).

    Returns:
        A ones tensor of shape (B, N_RES, 14) indicating all 14 slots present.
    """
    return torch.ones(B, N_RES, ATOMS_PER_RES)


@pytest.fixture
def ala_aatype() -> Int[torch.Tensor, "B N_res"]:
    """Provide an all-alanine aatype tensor (B, N_RES).

    Used by tests that only need a valid, arbitrary residue type (e.g.
    shape/finiteness checks) rather than exercising residue-type-dependent
    behavior.

    Returns:
        An integer tensor of shape (B, N_RES) filled with restype_order["A"].
    """
    return torch.full((B, N_RES), restype_order["A"], dtype=torch.long)


@pytest.fixture
def n() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random backbone N atom positions (B, N_RES, 3).

    Returns:
        A randomly initialised float tensor of shape (B, N_RES, 3).
    """
    return torch.randn(B, N_RES, 3)


@pytest.fixture
def ca() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random C alpha atom positions (B, N_RES, 3).

    Returns:
        A randomly initialised float tensor of shape (B, N_RES, 3).
    """
    return torch.randn(B, N_RES, 3)


@pytest.fixture
def c() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random backbone C atom positions (B, N_RES, 3).

    Returns:
        A randomly initialised float tensor of shape (B, N_RES, 3).
    """
    return torch.randn(B, N_RES, 3)


# ---------------------------------------------------------------------------
# pseudo_cb
# ---------------------------------------------------------------------------


def test_pseudo_cb_output_shape(
    n: Float[torch.Tensor, "B N_res 3"],
    ca: Float[torch.Tensor, "B N_res 3"],
    c: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """pseudo_cb output shape is (B, N_RES, 3).

    Verifies that the pseudo-Cβ positions tensor has expected batched shape.
    """
    out = pseudo_cb(n, ca, c)
    assert out.shape == (B, N_RES, 3)


def test_pseudo_cb_output_finite(
    n: Float[torch.Tensor, "B N_res 3"],
    ca: Float[torch.Tensor, "B N_res 3"],
    c: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """pseudo_cb output is finite for random backbone atoms.

    Verifies that no NaN or Inf values appear in computed pseudo-Cβ positions.
    """
    out = pseudo_cb(n, ca, c)
    assert torch.isfinite(out).all()


def test_pseudo_cb_unbatched_shape() -> None:
    """pseudo_cb works on unbatched (N_RES, 3) inputs and returns (N_RES, 3).

    Verifies that the function does not require a leading batch dimension.
    """
    n_ = torch.randn(N_RES, 3)
    ca_ = torch.randn(N_RES, 3)
    c_ = torch.randn(N_RES, 3)
    out = pseudo_cb(n_, ca_, c_)
    assert out.shape == (N_RES, 3)


def test_pseudo_cb_output_not_equal_ca(
    n: Float[torch.Tensor, "B N_res 3"],
    ca: Float[torch.Tensor, "B N_res 3"],
    c: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """pseudo_cb coordinates differ from C alpha coordinates.

    Verifies that the virtual Cβ position is not simply a copy of the CA atom.
    """
    out = pseudo_cb(n, ca, c)
    assert not torch.allclose(out, ca)


def test_pseudo_cb_single_residue_finite() -> None:
    """pseudo_cb handles scalar (3,) input without batch or residue dimensions.

    Verifies that function accepts single-atom input and returns finite vector.
    """
    out = pseudo_cb(torch.randn(3), torch.randn(3), torch.randn(3))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_pseudo_cb_collinear_backbone_finite() -> None:
    """pseudo_cb finite when N, CA, C are collinear so cross product is zero.

    The epsilon guard in linalg.norm must prevent NaN in the normalised output.
    """
    n_ = torch.zeros(N_RES, 3)
    ca_ = torch.zeros(N_RES, 3)
    c_ = torch.zeros(N_RES, 3)
    out = pseudo_cb(n_, ca_, c_)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# atom37_to_atom14
# ---------------------------------------------------------------------------


def _real_atom_names(one_letter: str) -> frozenset[str]:
    """Independently derive a residue type's true atom37 names.

    Reads directly from rigid_group_atom_positions (not from any atom14
    lookup table), so tests built on this oracle stay independent of the
    atom37_to_atom14/atom14_to_atom37 gather implementation under test.

    Args:
        one_letter: Single-letter residue code, e.g. "A"; "X" has no entry
            and produces an empty set.

    Returns:
        The frozenset of atom37 atom names this residue type has.
    """
    resname = restype_1to3.get(one_letter, "")
    return frozenset(
        name for name, _, _ in rigid_group_atom_positions.get(resname, [])
    )


def _sentinel_atom37(
    one_letter: str,
) -> tuple[Float[torch.Tensor, "1 1 37 3"], Float[torch.Tensor, "1 1 37"]]:
    """Build single-residue atom37 tensors with per-atom sentinel coords.

    Every atom37 slot gets coordinate value (atom37_idx + 1), and the mask
    is 1 only at slots this residue type actually has — so later gather
    results can be checked by value without needing to know atom14's exact
    slot ordering.

    Args:
        one_letter: Single-letter residue code to build the mask for.

    Returns:
        Tuple of (atom37_positions, atom37_mask), each with a leading
        (1, 1, ...) batch/residue shape.
    """
    real_names = _real_atom_names(one_letter)
    positions = torch.zeros(1, 1, len(atom_types), 3)
    mask = torch.zeros(1, 1, len(atom_types))
    for name, idx in atom_order.items():
        positions[:, :, idx, :] = float(idx + 1)
        if name in real_names:
            mask[:, :, idx] = 1.0
    return positions, mask


def _aatype_row(one_letter: str) -> Int[torch.Tensor, "1 1"]:
    """Build a single-residue aatype tensor for one_letter's restype row."""
    return torch.full((1, 1), restype_order[one_letter], dtype=torch.long)


def test_atom37_to_atom14_output_shapes(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
    ala_aatype: Int[torch.Tensor, "B N_res"],
) -> None:
    """atom37_to_atom14 returns position and mask tensors with correct shapes.

    Verifies that the output position tensor is (B, N_RES, 14, 3) and the
    mask tensor is (B, N_RES, 14).
    """
    pos14, mask14 = atom37_to_atom14(atom37_positions, atom37_mask, ala_aatype)
    assert pos14.shape == (B, N_RES, ATOMS_PER_RES, 3)
    assert mask14.shape == (B, N_RES, ATOMS_PER_RES)


def test_atom37_to_atom14_output_finite(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
    ala_aatype: Int[torch.Tensor, "B N_res"],
) -> None:
    """atom37_to_atom14 produces finite position, mask values for random input.

    Verifies that no NaN or Inf values appear in the position or mask outputs.
    """
    pos14, mask14 = atom37_to_atom14(atom37_positions, atom37_mask, ala_aatype)
    assert torch.isfinite(pos14).all()
    assert torch.isfinite(mask14).all()


@pytest.mark.parametrize("one_letter", RESTYPES, ids=RESTYPES)
def test_atom37_to_atom14_canonical_backbone_slots(one_letter: str) -> None:
    """Slots 0-3 are always N, CA, C, O; slot 4 is CB iff the residue has one.

    This is the one part of atom14's slot layout that is NOT residue-type
    dependent, so it is checked directly by name rather than through the
    gather's own lookup tables.
    """
    positions, mask = _sentinel_atom37(one_letter)
    aatype = _aatype_row(one_letter)
    pos14, mask14 = atom37_to_atom14(positions, mask, aatype)
    real_names = _real_atom_names(one_letter)

    for slot, name in zip(
        (ATOM14_N, ATOM14_CA, ATOM14_C, ATOM14_O),
        ("N", "CA", "C", "O"),
        strict=True,
    ):
        if name in real_names:
            assert mask14[0, 0, slot] == 1.0
            assert pos14[0, 0, slot, 0] == float(atom_order[name] + 1)
        else:
            assert mask14[0, 0, slot] == 0.0

    cb_present = _CB_ATOM_NAME in real_names
    assert mask14[0, 0, ATOM14_CB] == (1.0 if cb_present else 0.0)
    if cb_present:
        assert pos14[0, 0, ATOM14_CB, 0] == float(
            atom_order[_CB_ATOM_NAME] + 1,
        )


@pytest.mark.parametrize("one_letter", RESTYPES, ids=RESTYPES)
def test_atom37_to_atom14_recovers_exact_atom_set(one_letter: str) -> None:
    """atom14's masked-in values are exactly this residue type's atom set.

    Verifies, for every residue type (including the empty "X" row), that the
    number of masked-in atom14 slots equals the residue's true atom count
    and that the set of masked-in sentinel values matches the independently
    derived oracle — without assuming any particular sidechain slot order.
    """
    positions, mask = _sentinel_atom37(one_letter)
    aatype = _aatype_row(one_letter)
    pos14, mask14 = atom37_to_atom14(positions, mask, aatype)
    real_names = _real_atom_names(one_letter)

    present = mask14[0, 0, :].bool()
    assert int(present.sum()) == len(real_names)
    got_values = {float(v.item()) for v in pos14[0, 0, present, 0]}
    expected_values = {float(atom_order[name] + 1) for name in real_names}
    assert got_values == expected_values


def test_atom37_to_atom14_gly_has_only_backbone() -> None:
    """Glycine's atom14 row has exactly 4 real atoms and no CB.

    Mandatory edge case: the one standard residue with no Cβ.
    """
    gly_atom_count = 4  # N, CA, C, O — Gly is the one residue with no CB
    assert int(ATOM14_ATOM_EXISTS[restype_order["G"]].sum()) == gly_atom_count
    assert not ATOM14_ATOM_EXISTS[restype_order["G"], ATOM14_CB]


def test_atom37_to_atom14_trp_fills_all_slots() -> None:
    """Tryptophan's atom14 row uses all 14 slots.

    Mandatory edge case: the largest standard residue, which exactly
    saturates the 14-slot budget with no padding left over.
    """
    assert int(ATOM14_ATOM_EXISTS[restype_order["W"]].sum()) == ATOMS_PER_RES


def test_atom37_to_atom14_unknown_x_has_no_atoms() -> None:
    """The "X" mask token has no entry in rigid_group_atom_positions.

    Its atom14 row must be entirely empty rather than falling back to some
    other residue's geometry.
    """
    assert int(ATOM14_ATOM_EXISTS[restype_order["X"]].sum()) == 0


def test_atom37_to_atom14_sentinel_aatype_does_not_raise(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """The -100 padding sentinel used by data.py is clamped, not rejected.

    Verifies atom37_to_atom14 does not raise and produces finite output when
    aatype contains the padding sentinel; it is the caller's own residue
    mask, not this function, that is responsible for zeroing the result at
    padded positions.
    """
    aatype = torch.full((B, N_RES), -100, dtype=torch.long)
    pos14, mask14 = atom37_to_atom14(atom37_positions, atom37_mask, aatype)
    assert torch.isfinite(pos14).all()
    assert torch.isfinite(mask14).all()


# ---------------------------------------------------------------------------
# atom14_to_atom37 — round trip, mask handling, and residue-type coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("one_letter", RESTYPES, ids=RESTYPES)
def test_atom14_to_atom37_roundtrip_recovers_real_atoms(
    one_letter: str,
) -> None:
    """atom37 -> atom14 -> atom37 recovers every real atom's coordinates.

    Covers all 21 residue types (20 standard + "X"), including the Gly
    (no CB) and Trp (all 14 slots used) edge cases, without assuming any
    particular atom14 slot ordering: only the recovered-value set at the
    residue's true atom37 indices is checked.
    """
    positions, mask = _sentinel_atom37(one_letter)
    aatype = _aatype_row(one_letter)
    real_names = _real_atom_names(one_letter)

    pos14, mask14 = atom37_to_atom14(positions, mask, aatype)
    roundtrip_pos, roundtrip_mask = atom14_to_atom37(pos14, mask14, aatype)

    for name in real_names:
        idx = atom_order[name]
        assert roundtrip_mask[0, 0, idx] == 1.0
        assert roundtrip_pos[0, 0, idx, 0] == float(idx + 1)

    absent_indices = [
        idx for name, idx in atom_order.items() if name not in real_names
    ]
    assert torch.allclose(
        roundtrip_mask[0, 0, absent_indices],
        torch.zeros(len(absent_indices)),
    )


def test_atom14_to_atom37_output_shapes(
    atom14_positions: Float[torch.Tensor, "B N_res 14 3"],
    atom14_mask: Float[torch.Tensor, "B N_res 14"],
    ala_aatype: Int[torch.Tensor, "B N_res"],
) -> None:
    """atom14_to_atom37 returns position and mask tensors with correct shapes.

    Verifies that the output position tensor is (B, N_RES, 37, 3) and the
    mask tensor is (B, N_RES, 37).
    """
    x_37, mask_37 = atom14_to_atom37(
        atom14_positions,
        atom14_mask,
        ala_aatype,
    )
    assert x_37.shape == (B, N_RES, 37, 3)
    assert mask_37.shape == (B, N_RES, 37)


def test_atom14_to_atom37_single_residue() -> None:
    """atom14_to_atom37 handles single-residue input without error."""
    coords_14 = torch.randn(1, 1, ATOMS_PER_RES, 3)
    mask_14 = torch.ones(1, 1, ATOMS_PER_RES)
    aatype = _aatype_row("A")
    x_37, mask_37 = atom14_to_atom37(coords_14, mask_14, aatype)
    assert x_37.shape == (1, 1, 37, 3)
    assert mask_37.shape == (1, 1, 37)


# ---------------------------------------------------------------------------
# atom37_to_atom14 / atom14_to_atom37 — type enforcement
# ---------------------------------------------------------------------------


def test_atom37_to_atom14_rejects_wrong_third_dimension(
    ala_aatype: Int[torch.Tensor, "B N_res"],
) -> None:
    """A jaxtyping TypeCheckError raised when atom37_positions' last dim≠3."""
    with pytest.raises(TypeCheckError):
        _ = atom37_to_atom14(
            torch.zeros((B, N_RES, 37, 4), dtype=torch.float64),
            torch.ones((B, N_RES, 37), dtype=torch.float64),
            ala_aatype,
        )  # 4 ≠ 3


def test_atom14_to_atom37_rejects_wrong_third_dimension(
    ala_aatype: Int[torch.Tensor, "B N_res"],
) -> None:
    """A jaxtyping TypeCheckError raised when coords_14's third dim ≠ 14."""
    with pytest.raises(TypeCheckError):
        _ = atom14_to_atom37(
            torch.zeros((B, N_RES, 4, 3), dtype=torch.float64),
            torch.ones((B, N_RES, 4), dtype=torch.float64),
            ala_aatype,
        )  # 4 ≠ 14


# ---------------------------------------------------------------------------
# atom37_to_cb
# ---------------------------------------------------------------------------


def test_atom37_to_cb_output_shapes(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """atom37_to_cb returns Cβ positions (B, N_RES, 3) and bool presence mask.

    Verifies that both the coordinate tensor and the presence mask have the
    expected shapes and that the mask dtype is torch.bool.
    """
    cb, cb_present = atom37_to_cb(atom37_positions, atom37_mask)
    assert cb.shape == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool


def test_atom37_to_cb_output_finite(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """atom37_to_cb produces finite Cβ coordinates from random backbone atoms.

    Verifies that no NaN or Inf values appear in returned Cβ position tensor.
    """
    cb, _ = atom37_to_cb(atom37_positions, atom37_mask)
    assert torch.isfinite(cb).all()


def test_atom37_to_cb_all_cb_present(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
) -> None:
    """atom37_to_cb marks all residues as CB-present when CB is set in mask.

    Verifies that the presence mask is all-True when the input atom37 mask has
    the CB slot set to 1.0 for every residue.
    """
    _, cb_present = atom37_to_cb(atom37_positions, atom37_mask)
    assert cb_present.all()


def test_atom37_to_cb_glycine_gets_pseudo_cb(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
) -> None:
    """atom37_to_cb falls back to pseudo-Cβ for glycine-like residues.

    Verifies that when CB is absent from the atom37 mask the presence mask is
    all-False and the returned positions remain finite.
    """
    mask = torch.ones(B, N_RES, 37)
    mask[:, :, ATOM37_CB] = 0.0
    cb, cb_present = atom37_to_cb(atom37_positions, mask)
    assert not cb_present.any()
    assert torch.isfinite(cb).all()


def test_atom37_to_cb_agrees_with_atom14_to_cb(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
    ala_aatype: Int[torch.Tensor, "B N_res"],
) -> None:
    """atom37_to_cb and atom14_to_cb agree on the same underlying structure.

    Converts the same atom37 structure through atom37_to_atom14 and checks
    that extracting Cβ directly from atom37 matches extracting it from the
    derived atom14 representation.
    """
    cb_from_37, mask_from_37 = atom37_to_cb(atom37_positions, atom37_mask)
    pos14, mask14 = atom37_to_atom14(
        atom37_positions,
        atom37_mask,
        ala_aatype,
    )
    cb_from_14, mask_from_14 = atom14_to_cb(pos14, mask14)
    assert torch.equal(mask_from_37, mask_from_14)
    assert torch.equal(cb_from_37, cb_from_14)


# ---------------------------------------------------------------------------
# atom14_to_cb
# ---------------------------------------------------------------------------


def test_atom14_to_cb_output_shapes(
    atom14_positions: Float[torch.Tensor, "B N_res 14 3"],
    atom14_mask: Float[torch.Tensor, "B N_res 14"],
) -> None:
    """atom14_to_cb returns Cβ positions (B, N_RES, 3) and bool presence mask.

    Verifies that both the coordinate tensor and the presence mask have the
    expected shapes and that the mask dtype is torch.bool.
    """
    cb, cb_present = atom14_to_cb(atom14_positions, atom14_mask)
    assert cb.shape == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool


def test_atom14_to_cb_output_finite(
    atom14_positions: Float[torch.Tensor, "B N_res 14 3"],
    atom14_mask: Float[torch.Tensor, "B N_res 14"],
) -> None:
    """atom14_to_cb produces finite Cβ coordinates from random backbone atoms.

    Verifies that no NaN or Inf values appear in the returned Cβ position
    tensor.
    """
    cb, _ = atom14_to_cb(atom14_positions, atom14_mask)
    assert torch.isfinite(cb).all()


def test_atom14_to_cb_real_cb_used_when_present(
    atom14_positions: Float[torch.Tensor, "B N_res 14 3"],
    atom14_mask: Float[torch.Tensor, "B N_res 14"],
) -> None:
    """atom14_to_cb returns the real atom14 CB slot when CB is present.

    Verifies that the returned coordinates equal the atom14 CB channel and
    that the presence mask is all-True when all residues have CB.
    """
    cb, cb_present = atom14_to_cb(atom14_positions, atom14_mask)
    assert torch.allclose(cb, atom14_positions[:, :, ATOM14_CB, :])
    assert cb_present.all()


def test_atom14_to_cb_glycine_gets_pseudo_cb(
    atom14_positions: Float[torch.Tensor, "B N_res 14 3"],
) -> None:
    """atom14_to_cb falls back to pseudo-Cβ when CB is absent (Gly-like).

    Verifies that the presence mask is all-False, the fallback coordinates
    are finite, and they differ from the raw CB slot values.
    """
    mask = torch.ones(B, N_RES, ATOMS_PER_RES)
    mask[:, :, ATOM14_CB] = 0.0
    cb, cb_present = atom14_to_cb(atom14_positions, mask)
    assert not cb_present.any()
    assert torch.isfinite(cb).all()
    assert not torch.allclose(cb, atom14_positions[:, :, ATOM14_CB, :])


def test_atom14_to_cb_mixed_residues(
    atom14_positions: Float[torch.Tensor, "B N_res 14 3"],
) -> None:
    """atom14_to_cb gets pseudo-Cβ for residue 0 only, real CB for the rest.

    Verifies per-residue branching when only the first residue is missing
    its CB atom.
    """
    mask = torch.ones(B, N_RES, ATOMS_PER_RES)
    mask[:, 0, ATOM14_CB] = 0.0  # residue 0 has no CB (Gly-like)
    cb, cb_present = atom14_to_cb(atom14_positions, mask)
    assert not cb_present[:, 0].any()
    assert cb_present[:, 1:].all()
    assert torch.allclose(cb[:, 1:, :], atom14_positions[:, 1:, ATOM14_CB, :])


# ---------------------------------------------------------------------------
# fill_atom14_virtual_atoms / generic_ref_pos_and_element
# ---------------------------------------------------------------------------


def _ideal_atom14(
    one_letter: str,
) -> tuple[
    Float[torch.Tensor, "1 1 14 3"],
    Float[torch.Tensor, "1 1 14"],
    Int[torch.Tensor, "1 1"],
]:
    """Return random atom14 coords, the type's real-atom mask, and aatype.

    Args:
        one_letter: One-letter residue code.

    Returns:
        Tuple of (positions, mask, aatype) for a single residue.
    """
    row = restype_order[one_letter]
    positions = torch.randn(1, 1, ATOMS_PER_RES, 3)
    mask = rearrange(ATOM14_ATOM_EXISTS[row].float(), "a -> 1 1 a")
    aatype = torch.tensor([[row]], dtype=torch.long)
    return positions, mask, aatype


@pytest.mark.parametrize("one_letter", ["A", "C", "G", "F", "W"])
def test_fill_atom14_virtual_atoms_mask_is_all_set(one_letter: str) -> None:
    """Every slot of a fully resolved residue is present after filling.

    This is what keeps the mask from revealing the residue type.

    Args:
        one_letter: One-letter residue code under test.
    """
    positions, mask, aatype = _ideal_atom14(one_letter)
    _, filled_mask = fill_atom14_virtual_atoms(positions, mask, aatype)
    assert bool((filled_mask == 1).all())


@pytest.mark.parametrize("one_letter", ["A", "C", "G", "F", "W"])
def test_fill_atom14_virtual_atoms_places_virtual_atoms_at_ca(
    one_letter: str,
) -> None:
    """Slots the type lacks move to C alpha; real slots keep their coordinates.

    Args:
        one_letter: One-letter residue code under test.
    """
    positions, mask, aatype = _ideal_atom14(one_letter)
    filled, _ = fill_atom14_virtual_atoms(positions, mask, aatype)
    exists = ATOM14_ATOM_EXISTS[restype_order[one_letter]]
    ca_pos = positions[0, 0, ATOM14_CA]
    assert torch.equal(filled[0, 0, exists], positions[0, 0, exists])
    assert bool((filled[0, 0, ~exists] == ca_pos).all())


def test_fill_atom14_virtual_atoms_keeps_unresolved_real_atoms_masked() -> None:
    """An unresolved real atom stays masked instead of becoming virtual."""
    positions, mask, aatype = _ideal_atom14("F")
    mask[0, 0, ATOM14_CB] = 0.0
    filled, filled_mask = fill_atom14_virtual_atoms(positions, mask, aatype)
    assert filled_mask[0, 0, ATOM14_CB] == 0
    assert torch.equal(
        filled[0, 0, ATOM14_CB],
        positions[0, 0, ATOM14_CB],
    )


def test_fill_atom14_virtual_atoms_missing_ca_masks_virtual_atoms() -> None:
    """Virtual atoms inherit the C alpha mask, so padding stays masked."""
    positions, _, aatype = _ideal_atom14("A")
    mask = torch.zeros(1, 1, ATOMS_PER_RES)
    _, filled_mask = fill_atom14_virtual_atoms(positions, mask, aatype)
    assert bool((filled_mask == 0).all())


def test_generic_ref_pos_and_element_output_shapes() -> None:
    """generic_ref_pos_and_element returns (N_res, 14, 3) and (N_res, 14, 4)."""
    n_res = 4
    ref_pos, ref_elem = generic_ref_pos_and_element(n_res, device="cpu")
    assert ref_pos.shape == (n_res, ATOMS_PER_RES, 3)
    assert ref_elem.shape == (n_res, ATOMS_PER_RES, 4)


def test_generic_ref_pos_and_element_identical_for_every_residue() -> None:
    """Every residue gets the same reference, so it carries no sequence."""
    ref_pos, ref_elem = generic_ref_pos_and_element(3, device="cpu")
    assert torch.equal(ref_pos[0], ref_pos[1])
    assert torch.equal(ref_pos[0], ref_pos[2])
    assert torch.equal(ref_elem[0], ref_elem[2])


def test_generic_atom14_reference_slots_are_distinct() -> None:
    """No two slots share a reference position.

    The atom encoder distinguishes atoms within a residue only through
    reference-position differences, so coincident slots would be
    interchangeable.
    """
    dist = torch.cdist(GENERIC_ATOM14_REF_POS, GENERIC_ATOM14_REF_POS)
    off_diagonal = ~torch.eye(ATOMS_PER_RES, dtype=torch.bool)
    assert bool((dist[off_diagonal] > 1.0).all())


def test_generic_atom14_reference_backbone_matches_alanine() -> None:
    """Slots 0-4 use Ala's ideal backbone + Cβ geometry and elements."""
    ala = restype_order["A"]
    assert torch.equal(
        GENERIC_ATOM14_REF_POS[: ATOM14_CB + 1],
        ATOM14_REF_POS[ala, : ATOM14_CB + 1],
    )
    assert torch.equal(
        GENERIC_ATOM14_ELEMENTS[: ATOM14_CB + 1],
        ATOM14_ELEMENTS[ala, : ATOM14_CB + 1],
    )


def test_generic_atom14_reference_sidechain_slots_are_unk() -> None:
    """Slots 5-13 are UNK so their element reveals nothing."""
    unk = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert bool((GENERIC_ATOM14_ELEMENTS[ATOM14_CB + 1 :] == unk).all())


# ---------------------------------------------------------------------------
# _build_atom14_tables / _cb_from_backbone — private helpers, direct tests
# ---------------------------------------------------------------------------


def test_build_atom14_tables_is_deterministic() -> None:
    """_build_atom14_tables returns identical tensors on every call.

    The module-level constants (ATOM14_ATOM_EXISTS, etc.) are built by
    calling this function once at import time; calling it again directly
    must reproduce exactly the same tables, since the construction depends
    only on the fixed rigid_group_atom_positions data.
    """
    rebuilt = _build_atom14_tables()
    originals = (
        ATOM14_ATOM_EXISTS,
        RESIDX_ATOM14_TO_ATOM37,
        RESIDX_ATOM37_TO_ATOM14,
        ATOM37_ATOM_EXISTS,
        ATOM14_REF_POS,
        ATOM14_ELEMENTS,
    )
    for rebuilt_tensor, original_tensor in zip(
        rebuilt,
        originals,
        strict=True,
    ):
        assert torch.equal(rebuilt_tensor, original_tensor)


def test_cb_from_backbone_real_cb_used_when_present() -> None:
    """_cb_from_backbone returns the real CB coordinate when present."""
    n_backbone = torch.zeros(1, 1, 3)
    ca_backbone = torch.ones(1, 1, 3)
    c_backbone = torch.full((1, 1, 3), 2.0)
    cb = torch.full((1, 1, 3), 9.0)
    cb_present = torch.ones(1, 1, dtype=torch.bool)
    cb_out, mask_out = _cb_from_backbone(
        n_backbone,
        ca_backbone,
        c_backbone,
        cb,
        cb_present,
    )
    assert torch.equal(cb_out, cb)
    assert torch.equal(mask_out, cb_present)


def test_cb_from_backbone_pseudo_when_cb_absent() -> None:
    """_cb_from_backbone falls back to the Gly-safe pseudo-Cβ when absent."""
    n_backbone = torch.zeros(1, 1, 3)
    ca_backbone = torch.ones(1, 1, 3)
    c_backbone = torch.full((1, 1, 3), 2.0)
    cb = torch.full((1, 1, 3), 9.0)
    cb_present = torch.zeros(1, 1, dtype=torch.bool)
    cb_out, mask_out = _cb_from_backbone(
        n_backbone,
        ca_backbone,
        c_backbone,
        cb,
        cb_present,
    )
    assert not mask_out.any()
    assert torch.isfinite(cb_out).all()
    assert not torch.allclose(cb_out, cb)


# ---------------------------------------------------------------------------
# Protein dataclass
# ---------------------------------------------------------------------------

N_ATOM_TYPE = 37


@pytest.fixture
def valid_protein() -> Protein:
    """Minimal valid Protein with N_RES residues and all atoms unmasked.

    Returns:
        A Protein with zero atom positions, sequential residue indices, and a
        fully-set atom mask.
    """
    return Protein(
        atom_positions=np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float64),
        aatype=np.zeros(N_RES, dtype=np.intp),
        atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
        residue_index=np.arange(N_RES, dtype=np.intp),
        chain_index=np.zeros(N_RES, dtype=np.intp),
        b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
    )


def test_protein_accepts_valid_input(valid_protein: Protein) -> None:
    """Protein stores fields with expected shapes for a valid construction.

    Verifies that each attribute of the constructed Protein has the correct
    NumPy array shape after construction.
    """
    assert valid_protein.atom_positions.shape == (N_RES, N_ATOM_TYPE, 3)
    assert valid_protein.aatype.shape == (N_RES,)
    assert valid_protein.atom_mask.shape == (N_RES, N_ATOM_TYPE)
    assert valid_protein.residue_index.shape == (N_RES,)
    assert valid_protein.chain_index.shape == (N_RES,)
    assert valid_protein.b_factors.shape == (N_RES, N_ATOM_TYPE)


def test_protein_rejects_wrong_atom_positions_rank() -> None:
    """Protein raises when atom_positions is 2-D instead of (N_res, 37, 3).

    Verifies jaxtyping enforces the rank-3 shape contract on atom_positions.
    """
    with pytest.raises(TypeCheckError):
        _ = Protein(
            atom_positions=np.zeros(
                (N_RES, N_ATOM_TYPE),
                dtype=np.float64,
            ),  # missing 3
            aatype=np.zeros(N_RES, dtype=np.intp),
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
        )


def test_protein_rejects_wrong_coord_dim() -> None:
    """Protein raises when coordinate dimension is 4 instead of the required 3.

    Verifies jaxtyping enforces last-axis size-3 constraint on atom_positions.
    """
    with pytest.raises(TypeCheckError):
        _ = Protein(
            atom_positions=np.zeros(
                (N_RES, N_ATOM_TYPE, 4),
                dtype=np.float64,
            ),  # 4 not 3
            aatype=np.zeros(N_RES, dtype=np.intp),
            atom_mask=np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
            residue_index=np.arange(N_RES, dtype=np.intp),
            chain_index=np.zeros(N_RES, dtype=np.intp),
            b_factors=np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
        )


def test_protein_rejects_inconsistent_num_res() -> None:
    """Protein raises error when aatype len not same as atom_positions.

    Verifies that jaxtyping enforces consistent N_res across Protein fields.
    """
    with pytest.raises(TypeCheckError):
        _ = Protein(
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
    """Provide a random backbone coordinate dict with N, CA, C, O keys.

    Returns:
        A dict mapping atom name to (NP_NUM_RES, 3) float64 array of random
        positions.
    """
    rng = np.random.default_rng(0)
    return {
        name: rng.standard_normal((NP_NUM_RES, 3))
        for name in ("N", "CA", "C", "O")
    }


@pytest.fixture
def np_example(
    coords_dict: Mapping[str, npt.NDArray[np.float64]],
) -> Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Provide the numpy example dict built from the coords_dict fixture."""
    return make_np_example(coords_dict)


def test_make_np_example_output_keys(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example returns atom_positions, atom_mask, residue_index."""
    assert {"atom_positions", "atom_mask", "residue_index"} <= np_example.keys()


def test_make_np_example_atom_positions_shape(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example packs backbone coords into (NP_NUM_RES, 37, 3)."""
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
    np.testing.assert_array_equal(
        np_example["residue_index"],
        np.arange(NP_NUM_RES),
    )


def test_make_np_example_backbone_atoms_masked(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """make_np_example sets atom_mask to 1.0 for backbone atom types."""
    bb_atom_types = {"N", "CA", "C", "O"}
    for i, atom_type in enumerate(atom_types):
        if atom_type in bb_atom_types:
            assert cast(
                npt.NDArray[np.float64],
                np_example["atom_mask"][:, i] == 1.0,
            ).all()


def test_make_np_example_nan_coords_zeroed(
    coords_dict: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """make_np_example replaces NaN coords with zero to keep coords finite."""
    coords_dict["N"][0] = [float("nan"), float("nan"), float("nan")]
    batch = make_np_example(coords_dict)
    assert np.isfinite(batch["atom_positions"]).all()


def test_make_np_example_nan_coords_zero_mask(
    coords_dict: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """make_np_example sets atom_mask to 0 for residues with NaN coords."""
    coords_dict["N"][0] = [float("nan"), float("nan"), float("nan")]
    batch = make_np_example(coords_dict)
    n_idx = atom_order["N"]
    assert batch["atom_mask"][0, n_idx] == 0.0


# ---------------------------------------------------------------------------
# make_fixed_size
# ---------------------------------------------------------------------------

NP_MAX_LEN = 10


@pytest.fixture
def short_np_example() -> (
    Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]
):
    """Provide a numpy example dict with less residues than NP_MAX_LEN=10."""
    return {
        "atom_positions": np.zeros((5, 37, 3)),
        "atom_mask": np.ones((5, 37)),
        "residue_index": np.arange(5),
    }


@pytest.fixture
def long_np_example() -> (
    Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]
):
    """Provide numpy example dict with more residues than NP_MAX_LEN=10."""
    rng = np.random.default_rng(3)
    return {
        "atom_positions": rng.standard_normal((20, 37, 3)),
        "residue_index": np.arange(20),
    }


@pytest.fixture
def exact_np_example() -> Mapping[str, npt.NDArray[np.intp]]:
    """Provide a numpy example dict with exactly NP_MAX_LEN=10 residues."""
    return {"residue_index": np.arange(NP_MAX_LEN)}


@pytest.fixture
def ones_np_example() -> Mapping[str, npt.NDArray[np.float64]]:
    """Provide a numpy example dict with 3 residues set to residue index 1."""
    return {"residue_index": np.ones(3)}


def test_make_fixed_size_pads_shorter_sequence(
    short_np_example: Mapping[
        str,
        npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ],
) -> None:
    """make_fixed_size zero-pads arrays to NP_MAX_LEN when input is shorter."""
    make_fixed_size(short_np_example, max_seq_length=NP_MAX_LEN)
    assert short_np_example["atom_positions"].shape[0] == NP_MAX_LEN
    assert short_np_example["atom_mask"].shape[0] == NP_MAX_LEN
    assert short_np_example["residue_index"].shape[0] == NP_MAX_LEN


def test_make_fixed_size_truncates_longer_sequence(
    long_np_example: Mapping[
        str,
        npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ],
) -> None:
    """make_fixed_size truncates arrays to NP_MAX_LEN when too long."""
    make_fixed_size(long_np_example, max_seq_length=NP_MAX_LEN)
    assert long_np_example["atom_positions"].shape[0] == NP_MAX_LEN
    assert long_np_example["residue_index"].shape[0] == NP_MAX_LEN


def test_make_fixed_size_no_change_when_exact(
    exact_np_example: Mapping[str, npt.NDArray[np.intp]],
) -> None:
    """make_fixed_size leaves arrays unchanged when length is NP_MAX_LEN."""
    make_fixed_size(exact_np_example, max_seq_length=NP_MAX_LEN)
    assert exact_np_example["residue_index"].shape[0] == NP_MAX_LEN


def test_make_fixed_size_padded_values_are_zero(
    ones_np_example: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """make_fixed_size fills pad with zeros, not the original values."""
    make_fixed_size(ones_np_example, max_seq_length=6)
    assert cast(
        npt.NDArray[np.float64],
        ones_np_example["residue_index"][3:] == 0.0,
    ).all()


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
    """Provide numpy example with 5 residues where only CA atom is unmasked."""
    rng = np.random.default_rng(2)
    mask = np.zeros((5, 37))
    mask[:, 1] = 1.0
    return {
        "atom_positions": rng.standard_normal((5, 37, 3)),
        "atom_mask": mask,
    }


def test_center_positions_ca_center_at_origin(
    full_mask_np_example: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """center_positions translates structure so mean CA position at origin."""
    center_positions(full_mask_np_example)
    ca_center = cast(
        npt.NDArray[np.float64],
        full_mask_np_example["atom_positions"][:, 1, :].mean(axis=0),
    )
    np.testing.assert_allclose(ca_center, np.zeros(3), atol=1e-6)


def test_center_positions_masked_atoms_remain_zero(
    ca_only_np_example: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """center_positions leaves zero-masked atom slots at zero after center."""
    center_positions(ca_only_np_example)
    np.testing.assert_array_equal(
        ca_only_np_example["atom_positions"][:, 0, :],
        0.0,
    )


# ---------------------------------------------------------------------------
# chain_end
# ---------------------------------------------------------------------------


def testchain_end_starts_with_ter() -> None:
    """chain_end returns PDB-formatted line beginning with TER record type."""
    result = chain_end(100, "ALA", "A", 10)
    assert result.startswith("TER")


def testchain_end_contains_resname_and_chain() -> None:
    """chain_end embeds residue name and chain ID in returned TER record."""
    res_name = restype_1to3["G"]
    chain_id = PDB_CHAIN_IDS[1]
    result = chain_end(100, res_name, chain_id, 42)
    assert res_name in result
    assert chain_id in result


# ---------------------------------------------------------------------------
# to_pdb
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_protein() -> Protein:
    """Provide 5-residue Protein with backbone atoms (N, CA, C, O) present."""
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
    """Provide 3-residue Protein with only the CA atom present per residue."""
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
    """Provide 4-residue Protein split across two chains (indices 0 and 1)."""
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


@pytest.fixture
def roundtrip_protein() -> Protein:
    """Provide a 5-residue Protein for PDB serialisation roundtrip tests.

    Positions are pre-rounded to 3 decimal places to match the PDB ATOM
    column width (``%8.3f``), and absent-atom positions are zero so every
    field can be compared directly after a ``to_pdb`` / ``protein_from_pdb``
    cycle without masking.
    """
    # atom_types ordering in this repo: N=0, CA=1, C=2, CB=3, O=4
    _bb_cb: list[int] = [0, 1, 2, 3, 4]  # N, CA, C, CB, O — non-GLY residues
    _bb: list[int] = [0, 1, 2, 4]  # N, CA, C, O    — GLY has no CB

    # ALA=0, ARG=1, SER=15, VAL=19, GLY=7  (indices into restypes list)
    _aatype = np.array([0, 1, 15, 19, 7], dtype=np.intp)
    _num_res = len(_aatype)

    _atom_mask = np.zeros((_num_res, N_ATOM_TYPE), dtype=np.float64)
    for _i in range(_num_res - 1):  # residues 0-3: non-GLY
        _atom_mask[_i, _bb_cb] = 1.0
    _atom_mask[_num_res - 1, _bb] = 1.0  # last residue: GLY, no CB

    _rng = np.random.default_rng(7)
    _atom_positions = np.zeros((_num_res, N_ATOM_TYPE, 3), dtype=np.float64)
    for _i in range(_num_res):
        _present = _bb_cb if _i < _num_res - 1 else _bb
        for _j in _present:
            _atom_positions[_i, _j] = np.round(_rng.standard_normal(3), 3)

    return Protein(
        atom_positions=_atom_positions,
        aatype=_aatype,
        atom_mask=_atom_mask,
        residue_index=np.arange(_num_res, dtype=np.intp),
        chain_index=np.zeros(_num_res, dtype=np.intp),
        b_factors=np.zeros((_num_res, N_ATOM_TYPE), dtype=np.float64),
    )


def test_to_pdb_returns_string(simple_protein: Protein) -> None:
    """to_pdb returns a plain str, not bytes or another type."""
    assert isinstance(to_pdb(simple_protein), str)


def test_to_pdb_contains_model_endmdl_end(simple_protein: Protein) -> None:
    """to_pdb output contains mandatory MODEL, ENDMDL, and END PDB records."""
    result = to_pdb(simple_protein)
    assert PDB_REC_MODEL in result
    assert PDB_REC_ENDMDL in result
    assert PDB_REC_END in result


def test_to_pdb_lines_padded_to_80(simple_protein: Protein) -> None:
    """to_pdb right-pads every output line to exactly 80 characters."""
    for line in to_pdb(simple_protein).splitlines():
        assert len(line) == PDB_LINE_LEN


def test_to_pdb_contains_atom_records(simple_protein: Protein) -> None:
    """to_pdb includes at least one ATOM for a protein with masked atoms."""
    assert PDB_REC_ATOM in to_pdb(simple_protein)


def test_to_pdb_skips_unmasked_atoms(ca_only_protein: Protein) -> None:
    """to_pdb emits one ATOM line per residue when only CA is unmasked."""
    atom_lines = [
        line
        for line in to_pdb(ca_only_protein).splitlines()
        if line.startswith(PDB_REC_ATOM)
    ]
    assert len(atom_lines) == ca_only_protein.atom_positions.shape[0]


def test_to_pdb_multichain_has_ter(two_chain_protein: Protein) -> None:
    """to_pdb inserts a TER record between chains in a multi-chain protein."""
    assert PDB_REC_TER in to_pdb(two_chain_protein)


def test_to_pdb_raises_on_invalid_aatype() -> None:
    """to_pdb raises InvalidAAtypesError when aatype non-canonical residue."""
    num_res = 3
    prot = Protein(
        atom_positions=np.zeros((num_res, 37, 3), dtype=np.float64),
        aatype=np.full(num_res, RESTYPE_NUM + 1, dtype=np.intp),
        atom_mask=np.ones((num_res, 37), dtype=np.float64),
        residue_index=np.arange(num_res, dtype=np.intp),
        chain_index=np.zeros(num_res, dtype=np.intp),
        b_factors=np.zeros((num_res, 37), dtype=np.float64),
    )
    with pytest.raises(InvalidAAtypesError):
        _ = to_pdb(prot)


def test_to_pdb_raises_too_many_chains() -> None:
    """to_pdb raises ValueError when more chain_indices than PDB allows."""
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
        _ = to_pdb(prot)


def test_to_pdb_protein_from_pdb_roundtrip(
    roundtrip_protein: Protein,
    tmp_path: pathlib.Path,
) -> None:
    """All Protein fields preserved through to_pdb / protein_from_pdb cycle."""
    pdb_path = tmp_path / "roundtrip.pdb"
    _ = pdb_path.write_text(to_pdb(roundtrip_protein))
    recovered = protein_from_pdb(pdb_path)

    assert np.array_equal(roundtrip_protein.aatype, recovered.aatype)
    assert np.array_equal(
        roundtrip_protein.residue_index,
        recovered.residue_index,
    )
    assert np.array_equal(roundtrip_protein.chain_index, recovered.chain_index)
    assert np.array_equal(roundtrip_protein.atom_mask, recovered.atom_mask)
    assert np.array_equal(
        roundtrip_protein.atom_positions,
        recovered.atom_positions,
    )
    assert np.all(
        cast(
            npt.NDArray[np.bool_],
            recovered.b_factors == float(MOL_TYPE_PROTEIN),
        ),
    )


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_protein_wrong_shape() -> None:
    """Wrong atom_positions last dim triggers TypeCheckError."""
    n_res = 5
    with pytest.raises(TypeCheckError):
        _ = Protein(
            atom_positions=np.zeros(
                (n_res, 37, 4),
                dtype=np.float64,
            ),  # last dim must be 3
            aatype=np.zeros(n_res, dtype=np.intp),
            atom_mask=np.zeros((n_res, 37), dtype=np.float64),
            residue_index=np.zeros(n_res, dtype=np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )


def test_pseudo_cb_wrong_shape() -> None:
    """Wrong last dim on n triggers TypeCheckError."""
    n_bad = torch.zeros(10, 4)  # last dim must be 3
    ca_good = torch.zeros(10, 3)
    c_good = torch.zeros(10, 3)
    with pytest.raises(TypeCheckError):
        _ = pseudo_cb(n_bad, ca_good, c_good)


def test_atom14_to_cb_wrong_shape() -> None:
    """Wrong last dim on atom14_positions triggers TypeCheckError."""
    positions_bad = torch.zeros(B, N_RES, ATOMS_PER_RES, 4)  # must be 3
    mask = torch.ones(B, N_RES, ATOMS_PER_RES)
    with pytest.raises(TypeCheckError):
        _ = atom14_to_cb(positions_bad, mask)


def test_atom37_to_cb_wrong_shape() -> None:
    """Wrong last dim on atom37_positions triggers TypeCheckError."""
    positions_bad = torch.zeros(B, N_RES, 37, 4)  # last dim must be 3
    mask = torch.ones(B, N_RES, 37)
    with pytest.raises(TypeCheckError):
        _ = atom37_to_cb(positions_bad, mask)


# ---------------------------------------------------------------------------
# Molecule-type and nucleotide constants
# ---------------------------------------------------------------------------


def test_dna_restype_constants() -> None:
    """DNA_RESTYPE_* constants have the expected keys, values, and ordering."""
    assert list(DNA_RESTYPES) == ["DA", "DC", "DG", "DT"]
    assert isinstance(DNA_RESTYPE_ORDER, MappingProxyType)
    assert dict(DNA_RESTYPE_ORDER) == {"DA": 0, "DC": 1, "DG": 2, "DT": 3}
    assert isinstance(DNA_RESTYPE_3TO1, MappingProxyType)
    assert dict(DNA_RESTYPE_3TO1) == {
        "DA": "a",
        "DC": "c",
        "DG": "g",
        "DT": "t",
    }


def test_rna_restype_constants() -> None:
    """RNA_RESTYPE_* constants have the expected keys, values, and ordering."""
    assert list(RNA_RESTYPES) == ["A", "C", "G", "U"]
    assert isinstance(RNA_RESTYPE_ORDER, MappingProxyType)
    assert dict(RNA_RESTYPE_ORDER) == {"A": 0, "C": 1, "G": 2, "U": 3}
    assert isinstance(RNA_RESTYPE_3TO1, MappingProxyType)
    assert dict(RNA_RESTYPE_3TO1) == {"A": "a", "C": "c", "G": "g", "U": "u"}


def test_dna_restype_mappings_are_immutable() -> None:
    """Verify that DNA mapping constants reject mutation at runtime."""
    with pytest.raises(TypeError, match=r"does not support item assignment"):
        DNA_RESTYPE_ORDER["DX"] = 99
    with pytest.raises(TypeError, match=r"does not support item assignment"):
        DNA_RESTYPE_3TO1["DX"] = "x"


def test_rna_restype_mappings_are_immutable() -> None:
    """Verify that RNA mapping constants reject mutation at runtime."""
    with pytest.raises(TypeError, match=r"does not support item assignment"):
        RNA_RESTYPE_ORDER["UX"] = 99
    with pytest.raises(TypeError, match=r"does not support item assignment"):
        RNA_RESTYPE_3TO1["UX"] = "x"


# ---------------------------------------------------------------------------
# classify_mol_type
# ---------------------------------------------------------------------------


def testclassify_mol_type_protein() -> None:
    """classify_mol_type returns MOL_TYPE_PROTEIN for amino-acid residues."""
    assert classify_mol_type("ALA", frozenset()) == MOL_TYPE_PROTEIN
    assert classify_mol_type("GLY", frozenset()) == MOL_TYPE_PROTEIN
    assert classify_mol_type("VAL", frozenset()) == MOL_TYPE_PROTEIN


def testclassify_mol_type_dna_canonical() -> None:
    """classify_mol_type returns MOL_TYPE_DNA for D-prefix nucleotide names."""
    assert classify_mol_type("DA", frozenset()) == MOL_TYPE_DNA
    assert classify_mol_type("DC", frozenset()) == MOL_TYPE_DNA
    assert classify_mol_type("DG", frozenset()) == MOL_TYPE_DNA
    assert classify_mol_type("DT", frozenset()) == MOL_TYPE_DNA


def testclassify_mol_type_dna_bare_t() -> None:
    """classify_mol_type returns MOL_TYPE_DNA for D-prefix-less thymine."""
    assert classify_mol_type("T", frozenset()) == MOL_TYPE_DNA
    assert classify_mol_type("T", frozenset({"C1'", "C2'"})) == MOL_TYPE_DNA


def testclassify_mol_type_rna_u() -> None:
    """classify_mol_type returns MOL_TYPE_RNA for U regardless of atoms."""
    assert classify_mol_type("U", frozenset()) == MOL_TYPE_RNA
    assert classify_mol_type("U", frozenset({"C1'"})) == MOL_TYPE_RNA


def testclassify_mol_type_rna_via_o2prime() -> None:
    """classify_mol_type returns MOL_TYPE_RNA for A/C/G when O2' atom."""
    assert classify_mol_type("A", frozenset({"O2'", "C1'"})) == MOL_TYPE_RNA
    assert classify_mol_type("C", frozenset({"O2'"})) == MOL_TYPE_RNA
    assert classify_mol_type("G", frozenset({"O2'"})) == MOL_TYPE_RNA


def testclassify_mol_type_dna_no_prefix_no_o2prime() -> None:
    """classify_mol_type returns MOL_TYPE_DNA for A/C/G when no O2' atom."""
    assert classify_mol_type("A", frozenset({"C1'", "C2'"})) == MOL_TYPE_DNA
    assert classify_mol_type("C", frozenset()) == MOL_TYPE_DNA
    assert classify_mol_type("G", frozenset({"P"})) == MOL_TYPE_DNA


def testclassify_mol_type_unknown_defaults_protein() -> None:
    """classify_mol_type returns MOL_TYPE_PROTEIN for unrecognised residues."""
    assert classify_mol_type("UNK", frozenset()) == MOL_TYPE_PROTEIN
    assert classify_mol_type("MSE", frozenset()) == MOL_TYPE_PROTEIN
    assert classify_mol_type("", frozenset()) == MOL_TYPE_PROTEIN


# ---------------------------------------------------------------------------
# protein_from_pdb helpers
# ---------------------------------------------------------------------------


def _pdb_atom_line(
    record: str,
    serial: int,
    atom_name: str,
    residue: tuple[str, str, int],
    coords: tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> str:
    """Build an 80-character PDB ATOM or HETATM record for test fixtures.

    Args:
        record: Record type string, e.g. "ATOM" or "HETATM".
        serial: Atom serial number.
        atom_name: PDB atom name (e.g. "N", "CA", "O2'").
        residue: Tuple of (resname, chain, resseq) identifying the residue.
        coords: Cartesian (x, y, z) coordinates in Angstroms.

    Returns:
        An 80-character string formatted as a PDB record.
    """
    resname, chain, resseq = residue
    x, y, z = coords
    atom_field = (
        f" {atom_name:<3}"
        if len(atom_name) < PDB_ATOM_NAME_CUTOFF
        else atom_name[:PDB_ATOM_NAME_CUTOFF]
    )
    return (
        f"{record:<6}{serial:>5} {atom_field} {resname:>3} "
        f"{chain}{resseq:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
    ).ljust(80)


# ---------------------------------------------------------------------------
# protein_from_pdb — integration tests
# ---------------------------------------------------------------------------


def test_protein_from_pdb_protein_only(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb sets b_factors to MOL_TYPE_PROTEIN for amino-acids."""
    pdb = tmp_path / "prot.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "CA", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 3, "N", ("GLY", "A", 2))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(
        cast(npt.NDArray[np.bool_], prot.b_factors == float(MOL_TYPE_PROTEIN)),
    )
    assert prot.aatype[0] == 0  # ALA → index 0 in restypes


def test_protein_from_pdb_dna_canonical(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb sets b_factors to MOL_TYPE_DNA for DA/DC/DG/DT."""
    pdb = tmp_path / "dna.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "P", ("DA", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "P", ("DT", "A", 2))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(
        cast(npt.NDArray[np.bool_], prot.b_factors == float(MOL_TYPE_DNA)),
    )
    assert np.all(cast(npt.NDArray[np.bool_], prot.aatype == RESTYPE_NUM_NO_X))


def test_protein_from_pdb_rna_canonical(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb classifies A/U residues as RNA when O2' atom."""
    pdb = tmp_path / "rna.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "O2'", ("A", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "O2'", ("U", "A", 2))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(
        cast(npt.NDArray[np.bool_], prot.b_factors == float(MOL_TYPE_RNA)),
    )
    assert np.all(cast(npt.NDArray[np.bool_], prot.aatype == RESTYPE_NUM_NO_X))


def test_protein_from_pdb_dna_no_prefix(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb classifies A/T residues as DNA when no O2' atom."""
    pdb = tmp_path / "dna_noprefix.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "C1'", ("A", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "C1'", ("T", "A", 2))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(
        cast(npt.NDArray[np.bool_], prot.b_factors == float(MOL_TYPE_DNA)),
    )


def test_protein_from_pdb_protein_dna_multichain(
    tmp_path: pathlib.Path,
) -> None:
    """protein_from_pdb works correctly for protein+DNA complex."""
    pdb = tmp_path / "complex.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "N", ("GLY", "A", 2))
        + "\n"
        + _pdb_atom_line("ATOM", 3, "P", ("DA", "B", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 4, "P", ("DT", "B", 2))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape == (4, 37, 3)
    np.testing.assert_array_equal(prot.chain_index, [0, 0, 1, 1])
    assert np.all(
        cast(
            npt.NDArray[np.bool_],
            prot.b_factors[:2] == float(MOL_TYPE_PROTEIN),
        ),
    )
    assert np.all(
        cast(npt.NDArray[np.bool_], prot.b_factors[2:] == float(MOL_TYPE_DNA)),
    )


def test_protein_from_pdb_multi_chain_protein(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb assigns distinct chain_index for each PDB chain ID."""
    pdb = tmp_path / "twochains.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "N", ("GLY", "A", 2))
        + "\n"
        + _pdb_atom_line("ATOM", 3, "N", ("ALA", "B", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 4, "N", ("GLY", "B", 2))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    np.testing.assert_array_equal(prot.chain_index, [0, 0, 1, 1])


def test_protein_from_pdb_residue_index(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb sets residue_index to the PDB RESSEQ numbers."""
    pdb = tmp_path / "resseq.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 5))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "N", ("ALA", "A", 10))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    np.testing.assert_array_equal(prot.residue_index, [5, 10])


def test_protein_from_pdb_no_atoms_raises(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb raises ValueError when PDB file has no ATOM records."""
    pdb = tmp_path / "empty.pdb"
    _ = pdb.write_text("REMARK empty structure\nHEADER test\n")
    with pytest.raises(ValueError, match="No ATOM records"):
        _ = protein_from_pdb(pdb)


def test_protein_from_pdb_ignores_water_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb excludes HOH HETATM records from the output Protein."""
    pdb = tmp_path / "water.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("HETATM", 2, "O", ("HOH", "A", 100))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape[0] == 1


def test_protein_from_pdb_ignores_ion_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb excludes ion HETATM records (MG, ZN, etc.)."""
    pdb = tmp_path / "ion.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("HETATM", 2, "MG", ("MG", "A", 200))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape[0] == 1


def test_protein_from_pdb_ignores_ligand_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb excludes ligand HETATM records (ATP, HEM, etc.)."""
    pdb = tmp_path / "ligand.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("HETATM", 2, "N1", ("ATP", "A", 300))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape[0] == 1


def test_protein_from_pdb_correct_residue_count_with_hetatm(
    tmp_path: pathlib.Path,
) -> None:
    """protein_from_pdb counts only ATOM-record when HETATM are present."""
    pdb = tmp_path / "mixed.pdb"
    _ = pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", ("ALA", "A", 1))
        + "\n"
        + _pdb_atom_line("ATOM", 2, "N", ("GLY", "A", 2))
        + "\n"
        + _pdb_atom_line("ATOM", 3, "N", ("SER", "A", 3))
        + "\n"
        + _pdb_atom_line("HETATM", 4, "O", ("HOH", "A", 100))
        + "\n"
        + _pdb_atom_line("HETATM", 5, "O", ("HOH", "A", 101))
        + "\n",
    )
    prot = protein_from_pdb(pdb)
    assert prot.atom_positions.shape[0] == PROT_LEN


def test_truncate_to_length_shortens_all_arrays() -> None:
    """truncate_to_length truncates all arrays to max_length along axis 0."""
    long_example: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((100, 37, 3)),
        "atom_mask": np.zeros((100, 37)),
        "residue_index": np.arange(100),
    }
    truncate_to_length(long_example, 50)
    assert long_example["atom_positions"].shape == (50, 37, 3)
    assert long_example["atom_mask"].shape == (50, 37)
    assert long_example["residue_index"].shape == (50,)


def test_truncate_to_length_noop_when_already_short() -> None:
    """truncate_to_length leaves arrays unchanged when shorter than max_len."""
    short_example: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((30, 37, 3)),
    }
    truncate_to_length(short_example, 50)
    assert short_example["atom_positions"].shape == (30, 37, 3)


def test_truncate_to_length_noop_when_exact() -> None:
    """truncate_to_length leaves arrays unchanged when exactly max_len."""
    exact_example: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((50, 37, 3)),
    }
    truncate_to_length(exact_example, 50)
    assert exact_example["atom_positions"].shape == (50, 37, 3)
