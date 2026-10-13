"""Tests for atom utility dataclasses and functions.

Covers pseudo_cb, atom37_to_atom14, atom14_to_atom37, atom37_to_cb,
atom14_to_cb, fill_atom14_virtual_atoms, generic_ref_pos_and_element, the
atom14 lookup tables, the Protein dataclass, make_np_example,
make_fixed_size, center_positions, chain_end, to_pdb, protein_from_pdb,
truncate_to_length, and molecule-type constants.
"""

import dataclasses
import enum
import functools
import pathlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from einops import einsum, rearrange, reduce, repeat
from helpers.atom_utils import (
    ATOM14_C,
    ATOM14_CA,
    ATOM14_CB,
    ATOM14_ELEMENTS,
    ATOM14_N,
    ATOM14_REF_POS,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
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
    RESTYPE_NUM,
    RESTYPE_NUM_NO_X,
    RNA_RESTYPE_3TO1,
    RNA_RESTYPE_ORDER,
    RNA_RESTYPES,
    Protein,
    _build_atom14_tables,  # pyright: ignore[reportPrivateUsage]
    _build_generic_atom14_reference,  # pyright: ignore[reportPrivateUsage]
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
from jaxtyping import Bool, Float, Int, TypeCheckError

_ = manual_seed(42)

B = 2
N_RES = 10
PDB_ATOM_NAME_CUTOFF = 4
PDB_LINE_LEN = 80
PROT_LEN = 3
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
# atom37_to_atom14 / atom14_to_atom37 — every residue type
# ---------------------------------------------------------------------------


class ResidueScenario(enum.Enum):
    """One single-residue conversion case per residue type.

    Each member's value is ``(aatype, resname, n_heavy_atoms)``. ``resname``
    is the three-letter key into ``rigid_group_atom_positions``, which the
    ``sentinel_residue`` fixture reads as an oracle independent of the
    atom14 lookup tables under test. ``n_heavy_atoms`` is a hand-counted
    second oracle: the number of heavy atoms the residue type has, which is
    also the number of atom14 slots it really occupies. Members are read
    via their properties rather than branched on.

    Members cover all 20 standard amino acids, including the edge cases
    glycine (4 atoms, the one residue without a Cβ) and tryptophan
    (14 atoms, exactly filling every slot), plus ``unknown`` (the "X"
    token, which has no geometry and so no atoms). The -100 padding index
    is covered by ``VirtualAtomScenario.padding_sentinel``.
    """

    alanine = (restype_order["A"], "ALA", 5)
    arginine = (restype_order["R"], "ARG", 11)
    asparagine = (restype_order["N"], "ASN", 8)
    aspartate = (restype_order["D"], "ASP", 8)
    cysteine = (restype_order["C"], "CYS", 6)
    glutamine = (restype_order["Q"], "GLN", 9)
    glutamate = (restype_order["E"], "GLU", 9)
    glycine = (restype_order["G"], "GLY", 4)
    histidine = (restype_order["H"], "HIS", 10)
    isoleucine = (restype_order["I"], "ILE", 8)
    leucine = (restype_order["L"], "LEU", 8)
    lysine = (restype_order["K"], "LYS", 9)
    methionine = (restype_order["M"], "MET", 8)
    phenylalanine = (restype_order["F"], "PHE", 11)
    proline = (restype_order["P"], "PRO", 7)
    serine = (restype_order["S"], "SER", 6)
    threonine = (restype_order["T"], "THR", 7)
    tryptophan = (restype_order["W"], "TRP", 14)
    tyrosine = (restype_order["Y"], "TYR", 12)
    valine = (restype_order["V"], "VAL", 7)
    unknown = (restype_order["X"], "UNK", 0)

    @property
    def aatype(self) -> int:
        """Residue type index passed to the conversions."""
        return self.value[0]

    @property
    def resname(self) -> str:
        """Three-letter name keying ``rigid_group_atom_positions``."""
        return self.value[1]

    @property
    def n_heavy_atoms(self) -> int:
        """Hand-counted number of heavy atoms this residue type has."""
        return self.value[2]


@dataclasses.dataclass(frozen=True)
class SentinelResidue:
    """A single residue whose atom37 coordinates encode their own slot.

    Every atom37 slot ``i`` holds coordinate value ``i + 1`` on all three
    axes, so a gathered value names the atom37 slot it came from without
    the test needing to know atom14's residue-specific sidechain order.

    Attributes:
        atom37_positions: Sentinel atom37 coordinates (1, 1, 37, 3).
        atom37_mask: 1 exactly at the atom37 slots this residue type has.
        aatype: Residue type index (1, 1).
        expected_atom_count: Hand-counted number of atom14 slots the
            residue type really occupies.
        expected_head_mask: Expected atom14 mask of the fixed slots 0-4
            (N, CA, C, O, CB): 1 where the residue type has that atom.
        expected_head_values: Expected sentinel value in slots 0-4, zero
            where the atom is absent.
        expected_sentinel_values: Ascending sentinel values of every atom
            the residue type has, i.e. the values atom14 must gather.
        atom14_reference: The residue type's row of the reference-position
            table from ``_build_atom14_tables`` (1, 1, 14, 3).
        atom14_reference_mask: The residue type's row of the atom-exists
            table from ``_build_atom14_tables``, as a float mask (1, 1, 14).
        expected_reference_atom37: Ideal coordinates of each atom straight
            from ``rigid_group_atom_positions``, laid out in atom37 slots
            and zero elsewhere (1, 1, 37, 3).
    """

    atom37_positions: Float[torch.Tensor, "1 1 37 3"]
    atom37_mask: Float[torch.Tensor, "1 1 37"]
    aatype: Int[torch.Tensor, "1 1"]
    expected_atom_count: int
    expected_head_mask: Float[torch.Tensor, "5"]
    expected_head_values: Float[torch.Tensor, "5"]
    expected_sentinel_values: Float[torch.Tensor, "n_atoms"]
    atom14_reference: Float[torch.Tensor, "1 1 14 3"]
    atom14_reference_mask: Float[torch.Tensor, "1 1 14"]
    expected_reference_atom37: Float[torch.Tensor, "1 1 37 3"]


@pytest.fixture
def sentinel_residue(request: pytest.FixtureRequest) -> SentinelResidue:
    """Build the sentinel-coordinate residue for one ``ResidueScenario``.

    Factory fixture: receives a ``ResidueScenario`` through
    ``@pytest.mark.parametrize(..., indirect=["sentinel_residue"])``.

    The atom set comes from ``rigid_group_atom_positions`` (the geometry
    the atom14 tables are built from, but read here by atom name), so the
    expectations do not depend on the gather tables under test. The fixed
    head slots 0-4 are N, CA, C, O, CB for every residue type, which is
    the one part of atom14's layout that can be written down by name.

    Args:
        request: Pytest request carrying the ``ResidueScenario``.

    Returns:
        SentinelResidue with inputs and expectations for the scenario.
    """
    scenario = cast(ResidueScenario, request.param)
    head_names = ("N", "CA", "C", "O", "CB")
    ideal_by_name = {
        name: pos
        for name, _, pos in rigid_group_atom_positions.get(
            scenario.resname,
            [],
        )
    }
    exists14, _, _, _, ref_pos14, _ = _build_atom14_tables()
    sentinel = torch.arange(1, len(atom_types) + 1, dtype=torch.float32)
    real_atoms37 = torch.tensor(
        [float(name in ideal_by_name) for name in atom_types],
    )
    expected_reference = torch.tensor(
        [ideal_by_name.get(name, [0.0, 0.0, 0.0]) for name in atom_types],
    )
    head_mask = torch.tensor(
        [float(name in ideal_by_name) for name in head_names],
    )
    head_sentinels = torch.tensor(
        [float(atom_order[name] + 1) for name in head_names],
    )
    return SentinelResidue(
        atom37_positions=repeat(sentinel, "a -> 1 1 a d", d=3),
        atom37_mask=rearrange(real_atoms37, "a -> 1 1 a"),
        aatype=torch.full((1, 1), scenario.aatype, dtype=torch.long),
        expected_atom_count=scenario.n_heavy_atoms,
        expected_head_mask=head_mask,
        expected_head_values=head_mask * head_sentinels,
        expected_sentinel_values=torch.tensor(
            sorted(float(atom_order[name] + 1) for name in ideal_by_name),
        ).float(),
        atom14_reference=rearrange(
            ref_pos14[scenario.aatype],
            "a d -> 1 1 a d",
        ),
        atom14_reference_mask=rearrange(
            exists14[scenario.aatype].float(),
            "a -> 1 1 a",
        ),
        expected_reference_atom37=rearrange(
            expected_reference,
            "a d -> 1 1 a d",
        ),
    )


@pytest.mark.parametrize(
    "sentinel_residue",
    [pytest.param(s, id=s.name) for s in ResidueScenario],
    indirect=True,
)
def test_atom37_to_atom14_gathers_exactly_the_residue_atoms(
    sentinel_residue: SentinelResidue,
) -> None:
    """atom37_to_atom14 keeps every atom the residue type has, and no other.

    Arrange: ``sentinel_residue`` marks every atom37 slot with its own
    index and masks in exactly this residue type's atoms.
    Act: convert to atom14.
    Assert: the number of masked-in atom14 slots equals the hand-counted
    ``n_heavy_atoms`` (4 for glycine, 14 for tryptophan, 0 for "X" and
    the masked-in sentinel values are exactly
    the residue type's atoms -- checked as a set, so no particular
    sidechain slot order is assumed.

    Args:
        sentinel_residue: Factory-fixture-built residue for this scenario.
    """
    pos14, mask14 = atom37_to_atom14(
        sentinel_residue.atom37_positions,
        sentinel_residue.atom37_mask,
        sentinel_residue.aatype,
    )
    present = mask14[0, 0].bool()

    assert int(present.sum()) == sentinel_residue.expected_atom_count
    assert torch.equal(
        pos14[0, 0, present, 0].sort().values,
        sentinel_residue.expected_sentinel_values,
    )


@pytest.mark.parametrize(
    "sentinel_residue",
    [pytest.param(s, id=s.name) for s in ResidueScenario],
    indirect=True,
)
def test_atom37_to_atom14_fixed_backbone_and_cb_slots(
    sentinel_residue: SentinelResidue,
) -> None:
    """Slots 0-4 hold N, CA, C, O, CB for every residue type that has them.

    This fixed head is what lets ``atom14_to_cb`` and ``ATOM14_CA`` work
    without an aatype, so it is checked by atom name for every type.

    Arrange: ``sentinel_residue`` with the expected head mask and values.
    Act: convert to atom14.
    Assert: the head mask matches (glycine's CB slot is empty, "X" and
    the padding sentinel have no atoms), and each present head slot holds
    the sentinel value of the atom37 slot with that atom's name.

    Args:
        sentinel_residue: Factory-fixture-built residue for this scenario.
    """
    pos14, mask14 = atom37_to_atom14(
        sentinel_residue.atom37_positions,
        sentinel_residue.atom37_mask,
        sentinel_residue.aatype,
    )
    head_mask = mask14[0, 0, : ATOM14_CB + 1]

    assert torch.equal(head_mask, sentinel_residue.expected_head_mask)
    assert torch.equal(
        pos14[0, 0, : ATOM14_CB + 1, 0] * head_mask,
        sentinel_residue.expected_head_values,
    )


@pytest.mark.parametrize(
    "sentinel_residue",
    [pytest.param(s, id=s.name) for s in ResidueScenario],
    indirect=True,
)
def test_atom14_to_atom37_round_trip_restores_atom37(
    sentinel_residue: SentinelResidue,
) -> None:
    """atom37 -> atom14 -> atom37 restores every real atom and its mask.

    Arrange: ``sentinel_residue`` whose atom37 mask covers exactly the
    residue type's atoms.
    Act: convert to atom14 and back.
    Assert: the restored mask equals the input mask, and the restored
    coordinates equal the input coordinates at every masked-in slot.

    Args:
        sentinel_residue: Factory-fixture-built residue for this scenario.
    """
    pos14, mask14 = atom37_to_atom14(
        sentinel_residue.atom37_positions,
        sentinel_residue.atom37_mask,
        sentinel_residue.aatype,
    )
    pos37, mask37 = atom14_to_atom37(pos14, mask14, sentinel_residue.aatype)
    keep = rearrange(sentinel_residue.atom37_mask, "b n a -> b n a 1")

    assert torch.equal(mask37, sentinel_residue.atom37_mask)
    assert torch.equal(
        pos37 * keep,
        sentinel_residue.atom37_positions * keep,
    )


@pytest.mark.parametrize(
    "sentinel_residue",
    [pytest.param(s, id=s.name) for s in ResidueScenario],
    indirect=True,
)
def test_atom14_reference_tables_match_rigid_group_geometry(
    sentinel_residue: SentinelResidue,
) -> None:
    """_build_atom14_tables places each atom's ideal coordinates in its slot.

    ``_build_atom14_tables`` builds ``ATOM14_REF_POS`` and
    ``ATOM14_ATOM_EXISTS`` at import time from
    ``rigid_group_atom_positions``; the generic reference conformer takes
    its backbone from them, so a slot mix-up would silently corrupt every
    residue's reference features.

    Arrange: ``sentinel_residue`` with the residue type's table rows and
    the ideal coordinates read by atom name from the rigid-group data.
    Act: scatter the table row into atom37 with ``atom14_to_atom37``.
    Assert: the result equals the rigid-group coordinates at every atom
    the residue type has, and is zero everywhere else.

    Args:
        sentinel_residue: Factory-fixture-built residue for this scenario.
    """
    pos37, mask37 = atom14_to_atom37(
        sentinel_residue.atom14_reference,
        sentinel_residue.atom14_reference_mask,
        sentinel_residue.aatype,
    )

    assert torch.equal(
        pos37 * rearrange(mask37, "b n a -> b n a 1"),
        sentinel_residue.expected_reference_atom37,
    )


class MalformedConversion(enum.Enum):
    """Shape-contract violations the atom37 <-> atom14 conversions reject.

    Each member's value is ``(conversion, positions_shape, mask_shape)``;
    the ``malformed_conversion_call`` fixture builds zero-filled inputs of
    those shapes with a matching (1, 1) aatype.

    Members:
        atom37_coords_not_xyz: atom37 coordinates with 4 components.
        atom14_slots_not_fourteen: atom14 input with 4 slots instead of 14.
    """

    atom37_coords_not_xyz = (atom37_to_atom14, (1, 1, 37, 4), (1, 1, 37))
    atom14_slots_not_fourteen = (atom14_to_atom37, (1, 1, 4, 3), (1, 1, 4))

    @property
    def conversion(self) -> Callable[..., object]:
        """Conversion function under test."""
        return self.value[0]

    @property
    def positions_shape(self) -> tuple[int, ...]:
        """Shape of the malformed positions argument."""
        return self.value[1]

    @property
    def mask_shape(self) -> tuple[int, ...]:
        """Shape of the mask argument."""
        return self.value[2]


@pytest.fixture
def malformed_conversion_call(
    request: pytest.FixtureRequest,
) -> Callable[[], object]:
    """Bind a conversion to inputs that violate its shape contract.

    Factory fixture: receives a ``MalformedConversion`` through
    ``@pytest.mark.parametrize(..., indirect=[...])``.

    Args:
        request: Pytest request carrying the ``MalformedConversion``.

    Returns:
        Zero-argument callable that invokes the conversion.
    """
    scenario = cast(MalformedConversion, request.param)
    return functools.partial(
        scenario.conversion,
        torch.zeros(scenario.positions_shape),
        torch.ones(scenario.mask_shape),
        torch.zeros((1, 1), dtype=torch.long),
    )


@pytest.mark.parametrize(
    "malformed_conversion_call",
    [pytest.param(s, id=s.name) for s in MalformedConversion],
    indirect=True,
)
def test_atom_conversions_reject_malformed_shapes(
    malformed_conversion_call: Callable[[], object],
) -> None:
    """The jaxtyping shape contract rejects malformed conversion inputs.

    Arrange: ``malformed_conversion_call`` binds the conversion to inputs
    whose coordinate or slot dimension is wrong.
    Act: call it.
    Assert: jaxtyping raises ``TypeCheckError`` instead of gathering
    garbage.

    Args:
        malformed_conversion_call: Factory-fixture-built bound call.
    """
    with pytest.raises(TypeCheckError):
        _ = malformed_conversion_call()


# ---------------------------------------------------------------------------
# atom14_to_cb / atom37_to_cb — real Cβ where present, pseudo-Cβ otherwise
# ---------------------------------------------------------------------------


class CbScenario(enum.Enum):
    """Which residues of a three-residue chain have a resolved Cβ.

    Each member's value is the per-residue Cβ presence. The
    ``backbone_with_cb`` fixture builds the same chain in atom14 and atom37
    and expects the real Cβ where present and the pseudo-Cβ from the
    backbone where absent.

    Members:
        all_present: Every residue has a Cβ; no fallback.
        all_absent: No residue has a Cβ (glycine-like); all fallback.
        first_absent: Only residue 0 lacks a Cβ, so the per-residue choice
            must not leak across residues.
    """

    all_present = (True, True, True)
    all_absent = (False, False, False)
    first_absent = (False, True, True)

    @property
    def cb_present(self) -> tuple[bool, ...]:
        """Whether each residue has a resolved Cβ."""
        return self.value


@dataclasses.dataclass(frozen=True)
class BackboneWithCb:
    """A three-residue chain in both atom layouts, with its expected Cβ.

    Attributes:
        atom14_positions: Chain in atom14 layout (1, 3, 14, 3).
        atom14_mask: atom14 mask; backbone present, CB per scenario.
        atom37_positions: The same chain in atom37 layout (1, 3, 37, 3).
        atom37_mask: atom37 mask; backbone present, CB per scenario.
        expected_cb: Real Cβ where present, pseudo-Cβ otherwise (1, 3, 3).
        expected_cb_present: Per-residue Cβ presence (1, 3).
    """

    atom14_positions: Float[torch.Tensor, "1 3 14 3"]
    atom14_mask: Float[torch.Tensor, "1 3 14"]
    atom37_positions: Float[torch.Tensor, "1 3 37 3"]
    atom37_mask: Float[torch.Tensor, "1 3 37"]
    expected_cb: Float[torch.Tensor, "1 3 3"]
    expected_cb_present: Bool[torch.Tensor, "1 3"]


@pytest.fixture
def backbone_with_cb(request: pytest.FixtureRequest) -> BackboneWithCb:
    """Build the three-residue chain for one ``CbScenario``.

    Factory fixture: receives a ``CbScenario`` through
    ``@pytest.mark.parametrize(..., indirect=["backbone_with_cb"])``.

    Every residue has N (1, 0, 0), CA at the origin, C (0, 1, 0),
    O (1, 1, 0) and a real CB at (0, 0, 1) -- far from the pseudo-Cβ that
    backbone implies -- shifted by 10 Å along x per residue so a residue
    mix-up shows. The expected fallback is ``pseudo_cb`` of that backbone,
    which is tested on its own above.

    Args:
        request: Pytest request carrying the ``CbScenario``.

    Returns:
        BackboneWithCb with both layouts and the expected Cβ.
    """
    scenario = cast(CbScenario, request.param)
    n_res = len(scenario.cb_present)
    head = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    shift = repeat(
        torch.arange(n_res, dtype=torch.float32) * 10.0,
        "n -> n 1 1",
    ) * torch.tensor([1.0, 0.0, 0.0])
    chain_head = repeat(head, "a d -> n a d", n=n_res)
    chain_head = chain_head + shift
    cb_present = torch.tensor(scenario.cb_present)
    head_mask = torch.ones(n_res, ATOM14_CB + 1)
    head_mask[:, ATOM14_CB] = cb_present.float()

    positions14 = torch.zeros(n_res, ATOMS_PER_RES, 3)
    positions14[:, : ATOM14_CB + 1] = chain_head
    mask14 = torch.zeros(n_res, ATOMS_PER_RES)
    mask14[:, : ATOM14_CB + 1] = head_mask

    atom37_slots = [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB]
    positions37 = torch.zeros(n_res, len(atom_types), 3)
    positions37[:, atom37_slots] = chain_head
    mask37 = torch.zeros(n_res, len(atom_types))
    mask37[:, atom37_slots] = head_mask

    fallback = pseudo_cb(
        chain_head[:, ATOM14_N],
        chain_head[:, ATOM14_CA],
        chain_head[:, ATOM14_C],
    )
    expected_cb = torch.where(
        rearrange(cb_present, "n -> n 1"),
        chain_head[:, ATOM14_CB],
        fallback,
    )
    return BackboneWithCb(
        atom14_positions=rearrange(positions14, "n a d -> 1 n a d"),
        atom14_mask=rearrange(mask14, "n a -> 1 n a"),
        atom37_positions=rearrange(positions37, "n a d -> 1 n a d"),
        atom37_mask=rearrange(mask37, "n a -> 1 n a"),
        expected_cb=rearrange(expected_cb, "n d -> 1 n d"),
        expected_cb_present=rearrange(cb_present, "n -> 1 n"),
    )


@pytest.mark.parametrize(
    "backbone_with_cb",
    [pytest.param(s, id=s.name) for s in CbScenario],
    indirect=True,
)
def test_atom14_to_cb_uses_real_cb_else_pseudo_cb(
    backbone_with_cb: BackboneWithCb,
) -> None:
    """atom14_to_cb reads slot 4 where CB is present, else the pseudo-Cβ.

    Arrange: ``backbone_with_cb`` chain in atom14 layout.
    Act: extract the Cβ.
    Assert: positions equal the real CB where present and the backbone's
    pseudo-Cβ elsewhere, residue by residue, and the presence mask is the
    scenario's (as a bool tensor).

    Args:
        backbone_with_cb: Factory-fixture-built chain for this scenario.
    """
    cb, cb_present = atom14_to_cb(
        backbone_with_cb.atom14_positions,
        backbone_with_cb.atom14_mask,
    )

    assert torch.allclose(cb, backbone_with_cb.expected_cb)
    assert torch.equal(cb_present, backbone_with_cb.expected_cb_present)


@pytest.mark.parametrize(
    "backbone_with_cb",
    [pytest.param(s, id=s.name) for s in CbScenario],
    indirect=True,
)
def test_atom37_to_cb_uses_real_cb_else_pseudo_cb(
    backbone_with_cb: BackboneWithCb,
) -> None:
    """atom37_to_cb gives the same Cβ as atom14_to_cb on the same chain.

    ``data.py`` takes the ground-truth distogram's Cβ from atom37 and the
    self-conditioning template's Cβ from atom14, so the two must agree.

    Arrange: ``backbone_with_cb`` chain in atom37 layout.
    Act: extract the Cβ.
    Assert: the same expectations as the atom14 test.

    Args:
        backbone_with_cb: Factory-fixture-built chain for this scenario.
    """
    cb, cb_present = atom37_to_cb(
        backbone_with_cb.atom37_positions,
        backbone_with_cb.atom37_mask,
    )

    assert torch.allclose(cb, backbone_with_cb.expected_cb)
    assert torch.equal(cb_present, backbone_with_cb.expected_cb_present)


@pytest.mark.parametrize(
    "backbone_with_cb",
    [pytest.param(s, id=s.name) for s in CbScenario],
    indirect=True,
)
def test_cb_from_backbone_selects_per_residue(
    backbone_with_cb: BackboneWithCb,
) -> None:
    """_cb_from_backbone picks real or pseudo-Cβ from an explicit presence.

    The shared core of ``atom14_to_cb`` and ``atom37_to_cb``: given the
    backbone, a candidate Cβ, and a presence mask, it must return the
    candidate where present and the backbone's pseudo-Cβ elsewhere, and
    pass the presence mask through unchanged.

    Arrange: ``backbone_with_cb`` chain; its atom14 head slots supply N,
    CA, C, and the candidate Cβ, and its expected presence is the mask.
    Act: call ``_cb_from_backbone`` directly.
    Assert: the same expectations as the atom14 and atom37 tests.

    Args:
        backbone_with_cb: Factory-fixture-built chain for this scenario.
    """
    head = backbone_with_cb.atom14_positions
    cb, cb_present = _cb_from_backbone(
        head[:, :, ATOM14_N],
        head[:, :, ATOM14_CA],
        head[:, :, ATOM14_C],
        head[:, :, ATOM14_CB],
        backbone_with_cb.expected_cb_present,
    )

    assert torch.allclose(cb, backbone_with_cb.expected_cb)
    assert torch.equal(cb_present, backbone_with_cb.expected_cb_present)


# ---------------------------------------------------------------------------
# fill_atom14_virtual_atoms
# ---------------------------------------------------------------------------


class VirtualAtomScenario(enum.Enum):
    """Single-residue cases for filling missing atom14 slots at C alpha.

    Each member's value is ``(aatype, unresolved_slots, n_real_slots,
    expected_mask)``. A residue type's real atoms always occupy its first
    ``n_real_slots`` atom14 slots; every later slot is one the type lacks
    and must become a virtual atom at C alpha. ``unresolved_slots`` are
    real slots masked out of the input, as for atoms missing from an
    experimental structure. ``expected_mask`` is the full 14-slot output
    mask, written out so a failure shows exactly which slot is wrong.

    Members:
        alanine_resolved: Slots 5-13 become virtual; all 14 present.
        glycine_resolved: Slots 4-13, including the Cβ slot, become
            virtual; all 14 present.
        tryptophan_resolved: All 14 slots are real; nothing changes.
        phenylalanine_missing_cb: An unresolved real atom stays masked out
            and keeps its coordinates instead of being pulled onto C alpha.
        alanine_missing_ca: Virtual atoms take the C alpha mask, so with
            C alpha unresolved they are masked out too.
        unknown_residue: "X" has no real atoms and no C alpha, so every
            slot moves onto the (masked) C alpha slot and stays masked.
        padding_sentinel: The -100 index ``data.py`` writes into padded
            positions is clamped onto row 0 (alanine). A padded position
            has nothing resolved, so alanine's 5 real slots keep their
            coordinates, the rest move onto C alpha, and nothing is
            present.
    """

    alanine_resolved = (restype_order["A"], (), 5, (1,) * 14)
    glycine_resolved = (restype_order["G"], (), 4, (1,) * 14)
    tryptophan_resolved = (restype_order["W"], (), 14, (1,) * 14)
    phenylalanine_missing_cb = (
        restype_order["F"],
        (ATOM14_CB,),
        11,
        (1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    )
    alanine_missing_ca = (
        restype_order["A"],
        (ATOM14_CA,),
        5,
        (1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    unknown_residue = (restype_order["X"], (), 0, (0,) * 14)
    padding_sentinel = (-100, (0, 1, 2, 3, 4), 5, (0,) * 14)

    @property
    def aatype(self) -> int:
        """Residue type index passed to fill_atom14_virtual_atoms."""
        return self.value[0]

    @property
    def unresolved_slots(self) -> tuple[int, ...]:
        """Real slots masked out of the input."""
        return self.value[1]

    @property
    def n_real_slots(self) -> int:
        """Number of leading slots the residue type really has."""
        return self.value[2]

    @property
    def expected_mask(self) -> tuple[int, ...]:
        """Expected 14-slot output mask."""
        return self.value[3]


@dataclasses.dataclass(frozen=True)
class VirtualAtomFill:
    """Input to and expected output of fill_atom14_virtual_atoms.

    Attributes:
        positions: Input atom14 coordinates (1, 1, 14, 3).
        mask: Input atom14 mask (1, 1, 14).
        aatype: Residue type index (1, 1).
        expected_positions: Expected output coordinates (1, 1, 14, 3).
        expected_mask: Expected output mask (1, 1, 14).
    """

    positions: Float[torch.Tensor, "1 1 14 3"]
    mask: Float[torch.Tensor, "1 1 14"]
    aatype: Int[torch.Tensor, "1 1"]
    expected_positions: Float[torch.Tensor, "1 1 14 3"]
    expected_mask: Float[torch.Tensor, "1 1 14"]


@pytest.fixture
def virtual_atom_fill(request: pytest.FixtureRequest) -> VirtualAtomFill:
    """Build the fill input and expectation for one ``VirtualAtomScenario``.

    Factory fixture: receives a ``VirtualAtomScenario`` through
    ``@pytest.mark.parametrize(..., indirect=["virtual_atom_fill"])``.

    Slot ``s`` starts at coordinate ``s + 1`` on all three axes, so C alpha
    (slot 1) sits at 2. The expected output keeps ``s + 1`` in the residue
    type's real slots -- resolved or not -- and has 2 (C alpha) in every
    slot the type lacks.

    Args:
        request: Pytest request carrying the ``VirtualAtomScenario``.

    Returns:
        VirtualAtomFill with inputs and expectations for the scenario.
    """
    scenario = cast(VirtualAtomScenario, request.param)
    slot_values = torch.arange(1, ATOMS_PER_RES + 1, dtype=torch.float32)
    is_real = torch.arange(ATOMS_PER_RES) < scenario.n_real_slots
    mask = is_real.float()
    mask[list(scenario.unresolved_slots)] = 0.0
    expected_values = torch.where(
        is_real,
        slot_values,
        slot_values[ATOM14_CA],
    )
    return VirtualAtomFill(
        positions=repeat(slot_values, "a -> 1 1 a d", d=3),
        mask=rearrange(mask, "a -> 1 1 a"),
        aatype=torch.full((1, 1), scenario.aatype, dtype=torch.long),
        expected_positions=repeat(expected_values, "a -> 1 1 a d", d=3),
        expected_mask=rearrange(
            torch.tensor(scenario.expected_mask, dtype=torch.float32),
            "a -> 1 1 a",
        ),
    )


@pytest.mark.parametrize(
    "virtual_atom_fill",
    [pytest.param(s, id=s.name) for s in VirtualAtomScenario],
    indirect=True,
)
def test_fill_atom14_virtual_atoms(virtual_atom_fill: VirtualAtomFill) -> None:
    """Missing slots become virtual atoms at C alpha with C alpha's mask.

    This is what keeps the atom14 mask from revealing the residue type
    the model is asked to generate: every resolved residue fills all 14
    slots, whatever its type.

    Arrange: ``virtual_atom_fill`` with slot-numbered coordinates, the
    scenario's input mask, and the expected output.
    Act: fill the virtual atoms.
    Assert: coordinates and mask equal the expectation exactly -- real
    slots untouched (even when unresolved), lacking slots at C alpha, and
    the mask all-set only when C alpha is resolved.

    Args:
        virtual_atom_fill: Factory-fixture-built case for this scenario.
    """
    positions, mask = fill_atom14_virtual_atoms(
        virtual_atom_fill.positions,
        virtual_atom_fill.mask,
        virtual_atom_fill.aatype,
    )

    assert torch.equal(positions, virtual_atom_fill.expected_positions)
    assert torch.equal(mask, virtual_atom_fill.expected_mask)


# ---------------------------------------------------------------------------
# generic_ref_pos_and_element — the sequence-independent reference
# ---------------------------------------------------------------------------


class ReferenceSlotGroup(enum.Enum):
    """Slot ranges of the generic atom14 reference with distinct origins.

    Each member's value is the ``(start, stop)`` slot range.

    Members:
        backbone_and_cb: Slots 0-4 copy alanine's ideal N, CA, C, O, CB
            coordinates and C/N/O element one-hots.
        pseudo_sidechain: Slots 5-13 are UNK-element pseudo-atoms spaced
            1.5 Å apart, continuing the C alpha -> Cβ bond past Cβ.
    """

    backbone_and_cb = (0, ATOM14_CB + 1)
    pseudo_sidechain = (ATOM14_CB + 1, ATOMS_PER_RES)

    @property
    def slots(self) -> slice:
        """Slice selecting this group's slots."""
        return slice(*self.value)


@dataclasses.dataclass(frozen=True)
class ReferenceSlots:
    """One slot group of the generic reference, actual and expected.

    Attributes:
        ref_pos: Generic reference coordinates in the group's slots.
        ref_elem: Generic reference element one-hots in the group's slots.
        expected_ref_pos: Expected coordinates.
        expected_ref_elem: Expected element one-hots.
    """

    ref_pos: Float[torch.Tensor, "S 3"]
    ref_elem: Float[torch.Tensor, "S 4"]
    expected_ref_pos: Float[torch.Tensor, "S 3"]
    expected_ref_elem: Float[torch.Tensor, "S 4"]


@pytest.fixture
def reference_slots(request: pytest.FixtureRequest) -> ReferenceSlots:
    """Slice the generic reference and its expectation for one slot group.

    Factory fixture: receives a ``ReferenceSlotGroup`` through
    ``@pytest.mark.parametrize(..., indirect=["reference_slots"])``.

    The actual slots come from ``_build_generic_atom14_reference`` (which
    builds ``GENERIC_ATOM14_*`` at import time); the expectation is
    assembled from the specification instead: alanine's ideal head from
    ``ATOM14_REF_POS`` / ``ATOM14_ELEMENTS`` (themselves checked against
    the rigid-group data above), then nine UNK pseudo-atoms at
    ``CB + k * 1.5 Å * unit(CB - CA)`` for ``k = 1..9``.

    Args:
        request: Pytest request carrying the ``ReferenceSlotGroup``.

    Returns:
        ReferenceSlots for the group.
    """
    group = cast(ReferenceSlotGroup, request.param)
    spacing = 1.5
    n_pseudo = ATOMS_PER_RES - (ATOM14_CB + 1)
    ala = restype_order["A"]
    head_pos = ATOM14_REF_POS[ala, : ATOM14_CB + 1]
    bond = head_pos[ATOM14_CB] - head_pos[ATOM14_CA]
    unit = bond / torch.sqrt(reduce(bond**2, "d -> ", "sum"))
    steps = torch.arange(1, n_pseudo + 1, dtype=torch.float32) * spacing
    pseudo_pos = head_pos[ATOM14_CB] + einsum(steps, unit, "k, d -> k d")
    expected_pos = torch.cat([head_pos, pseudo_pos])
    expected_elem = torch.cat(
        [
            ATOM14_ELEMENTS[ala, : ATOM14_CB + 1],
            repeat(torch.tensor([0.0, 0.0, 0.0, 1.0]), "e -> k e", k=n_pseudo),
        ],
    )
    ref_pos, ref_elem = _build_generic_atom14_reference()
    return ReferenceSlots(
        ref_pos=ref_pos[group.slots],
        ref_elem=ref_elem[group.slots],
        expected_ref_pos=expected_pos[group.slots],
        expected_ref_elem=expected_elem[group.slots],
    )


@pytest.mark.parametrize(
    "reference_slots",
    [pytest.param(g, id=g.name) for g in ReferenceSlotGroup],
    indirect=True,
)
def test_generic_atom14_reference_slot_layout(
    reference_slots: ReferenceSlots,
) -> None:
    """The generic reference has alanine's head and an UNK pseudo-sidechain.

    Arrange: ``reference_slots`` for one slot group.
    Act: (the reference is built at import time; the fixture reads it.)
    Assert: coordinates match the specification to float tolerance and
    element one-hots match exactly.

    Args:
        reference_slots: Factory-fixture-built slot group.
    """
    assert torch.allclose(
        reference_slots.ref_pos,
        reference_slots.expected_ref_pos,
        atol=1e-6,
    )
    assert torch.equal(
        reference_slots.ref_elem,
        reference_slots.expected_ref_elem,
    )


@dataclasses.dataclass(frozen=True)
class TiledReference:
    """generic_ref_pos_and_element output with its expected tiling.

    Attributes:
        ref_pos: Output coordinates (N_res, 14, 3).
        ref_elem: Output element one-hots (N_res, 14, 4).
        expected_ref_pos: ``GENERIC_ATOM14_REF_POS`` repeated per residue.
        expected_ref_elem: ``GENERIC_ATOM14_ELEMENTS`` repeated per residue.
    """

    ref_pos: Float[torch.Tensor, "N_res 14 3"]
    ref_elem: Float[torch.Tensor, "N_res 14 4"]
    expected_ref_pos: Float[torch.Tensor, "N_res 14 3"]
    expected_ref_elem: Float[torch.Tensor, "N_res 14 4"]


class ReferenceTiling(enum.Enum):
    """Chain lengths to tile the generic reference over.

    Each member's value is the residue count.

    Members:
        single_residue: The smallest chain.
        short_chain: A few residues, enough to compare rows.
        max_residues: The default ``max_residues`` (128) of ModelParams.
    """

    single_residue = 1
    short_chain = 3
    max_residues = 128

    @property
    def n_res(self) -> int:
        """Number of residues."""
        return self.value


@pytest.fixture
def tiled_reference(request: pytest.FixtureRequest) -> TiledReference:
    """Call generic_ref_pos_and_element for one ``ReferenceTiling``.

    Factory fixture: receives a ``ReferenceTiling`` through
    ``@pytest.mark.parametrize(..., indirect=["tiled_reference"])``.

    Args:
        request: Pytest request carrying the ``ReferenceTiling``.

    Returns:
        TiledReference with the output and its expected tiling.
    """
    n_res = cast(ReferenceTiling, request.param).n_res
    ref_pos, ref_elem = generic_ref_pos_and_element(n_res, device="cpu")
    return TiledReference(
        ref_pos=ref_pos,
        ref_elem=ref_elem,
        expected_ref_pos=repeat(
            GENERIC_ATOM14_REF_POS,
            "a d -> n a d",
            n=n_res,
        ),
        expected_ref_elem=repeat(
            GENERIC_ATOM14_ELEMENTS,
            "a e -> n a e",
            n=n_res,
        ),
    )


@pytest.mark.parametrize(
    "tiled_reference",
    [pytest.param(t, id=t.name) for t in ReferenceTiling],
    indirect=True,
)
def test_generic_ref_pos_and_element_tiles_one_reference(
    tiled_reference: TiledReference,
) -> None:
    """Every residue gets the identical reference, so it carries no sequence.

    Arrange: ``tiled_reference`` for a single residue, a short chain, and
    the default ``max_residues`` (128).
    Act: (the fixture calls generic_ref_pos_and_element.)
    Assert: both outputs equal the shared reference repeated once per
    residue -- which also pins their shapes.

    Args:
        tiled_reference: Factory-fixture-built output for the count.
    """
    assert torch.equal(
        tiled_reference.ref_pos,
        tiled_reference.expected_ref_pos,
    )
    assert torch.equal(
        tiled_reference.ref_elem,
        tiled_reference.expected_ref_elem,
    )


@pytest.fixture
def generic_reference_pairwise_distances() -> Float[torch.Tensor, "14 14"]:
    """Pairwise distances between the 14 generic reference slots.

    Not a factory: there is exactly one generic reference, so there is
    nothing to parametrize.

    Returns:
        Symmetric (14, 14) distance matrix with a zero diagonal.
    """
    return torch.cdist(GENERIC_ATOM14_REF_POS, GENERIC_ATOM14_REF_POS)


def test_generic_atom14_reference_slots_are_distinct(
    generic_reference_pairwise_distances: Float[torch.Tensor, "14 14"],
) -> None:
    """No two reference slots are within 1 Å of each other.

    The atom encoder tells atoms within a residue apart only through
    reference-position differences, so two coincident slots would be
    interchangeable to the model. The shortest real bond in the head is
    C=O at about 1.23 Å, so 1 Å is a safe floor. Not parametrized: this
    is a property of the one shared reference.

    Arrange: ``generic_reference_pairwise_distances``.
    Act: take the off-diagonal entries.
    Assert: all exceed 1 Å.

    Args:
        generic_reference_pairwise_distances: Slot-to-slot distances.
    """
    off_diagonal = ~torch.eye(ATOMS_PER_RES, dtype=torch.bool)

    assert bool(
        (generic_reference_pairwise_distances[off_diagonal] > 1.0).all(),
    )


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
