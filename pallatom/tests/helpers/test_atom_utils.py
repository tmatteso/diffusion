"""Tests for atom utility dataclasses and functions.

Covers the atom37/atom5 conversions and atom37_to_cb's pseudo-Cβ fallback,
the Protein dataclass's shape contract, make_np_example, make_fixed_size,
truncate_to_length, center_positions, chain_end, to_pdb, protein_from_pdb
with its molecule-type classification, and the residue-type constants.

Every input is deterministic and every scenario enum's value is a frozen
dataclass, so each case reads as one row of named fields.
"""

import dataclasses
import enum
import pathlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import ClassVar, cast

import einops
import numpy as np
import numpy.typing as npt
import pytest
import torch
from helpers.atom_utils import (
    ATOM5_C,
    ATOM5_CA,
    ATOM5_CB,
    ATOM5_N,
    ATOM5_O,
    ATOM5_TO_ATOM37,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
    DNA_RESTYPE_3TO1,
    DNA_RESTYPE_ORDER,
    DNA_RESTYPES,
    MOL_TYPE_DNA,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
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
    ResidueRepresentation,
    atom5_to_atom37,
    atom37_to_atom5,
    atom37_to_cb,
    atom_types,
    center_positions,
    chain_end,
    classify_mol_type,
    make_fixed_size,
    make_np_example,
    protein_from_pdb,
    restype_order,
    to_pdb,
    truncate_to_length,
)
from helpers.errors import InvalidAAtypesError
from jaxtyping import Float, TypeCheckError

type Array = npt.NDArray[np.float64] | npt.NDArray[np.intp]
type Example = dict[str, Array]
type AnyFloat = Float[torch.Tensor, "..."]
type Convert = Callable[[AnyFloat, AnyFloat], object]
type ProteinFactory = Callable[[tuple[int, ...], tuple[int, ...]], Protein]


class TensorCase(enum.Enum):
    """Backbone geometries the atom tensor factory can build.

    Members:
        standard: N, CA, C, O and CB at fixed points, all present.
        all_zeros: Every atom at the origin, all present.
        no_beta_carbon: As ``standard`` but with CB masked out, like Gly.
        half_mask: All zeros, with the first half of residues masked out.
        collinear: N, CA, C on the x axis and CB masked out, so the
            pseudo-Cβ cross product vanishes.
    """

    standard = "standard"
    all_zeros = "all_zeros"
    no_beta_carbon = "no_beta_carbon"
    half_mask = "half_mask"
    collinear = "collinear"


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """Inputs of the atom tensor factory, shared by every conversion case.

    Attributes:
        representation: Atom layout to build; atom37 unless overridden.
        content: Backbone geometry to build.
        batch: Batch size.
        n_res: Residues per item.
    """

    representation: ClassVar[ResidueRepresentation] = (
        ResidueRepresentation.atom37
    )
    content: TensorCase
    batch: int
    n_res: int


@dataclasses.dataclass(frozen=True)
class Atom5Spec(TensorSpec):
    """Factory inputs in the atom5 layout, plus whether to pass the mask.

    Attributes:
        representation: The atom5 layout.
        provide_mask: Pass the mask to ``atom5_to_atom37``, or omit it to
            exercise the all-present default.
    """

    representation: ClassVar[ResidueRepresentation] = (
        ResidueRepresentation.atom5
    )
    provide_mask: bool


@dataclasses.dataclass(frozen=True)
class CbSpec(TensorSpec):
    """Factory inputs plus what ``atom37_to_cb`` must report for them.

    Attributes:
        present_from: Residue index from which CB is present.
        distinct: Whether the fallback pseudo-Cβ must differ from both the
            stored CB slot and CA; false where the backbone is degenerate.
    """

    present_from: int
    distinct: bool


class Atom37ToAtom5Scenario(enum.Enum):
    """Inputs for ``atom37_to_atom5``.

    Members:
        batched_standard: Well-formed backbone across a batch of two.
        single_batch_zeros: All-zero coordinates in a batch of one.
        batched_missing_cb: CB masked out for every residue.
        batched_half_mask: First half of the residues masked out.
    """

    batched_standard = TensorSpec(TensorCase.standard, 2, 10)
    single_batch_zeros = TensorSpec(TensorCase.all_zeros, 1, 10)
    batched_missing_cb = TensorSpec(TensorCase.no_beta_carbon, 2, 10)
    batched_half_mask = TensorSpec(TensorCase.half_mask, 2, 10)


class Atom5ToAtom37Scenario(enum.Enum):
    """Inputs for ``atom5_to_atom37``; all batched of two unless noted.

    Members:
        standard: Well-formed backbone with its mask passed through.
        missing_cb: CB masked out for every residue.
        half_mask: First half of the residues masked out.
        single_batch: Well-formed backbone in a batch of one.
        single_residue: One residue in a batch of one, the smallest input.
        default_mask: Mask omitted, so it must be synthesized as all ones.
    """

    standard = Atom5Spec(TensorCase.standard, 2, 10, provide_mask=True)
    missing_cb = Atom5Spec(TensorCase.no_beta_carbon, 2, 10, provide_mask=True)
    half_mask = Atom5Spec(TensorCase.half_mask, 2, 10, provide_mask=True)
    single_batch = Atom5Spec(TensorCase.standard, 1, 10, provide_mask=True)
    single_residue = Atom5Spec(TensorCase.standard, 1, 1, provide_mask=True)
    default_mask = Atom5Spec(TensorCase.standard, 2, 10, provide_mask=False)


class Atom37ToCbScenario(enum.Enum):
    """Inputs for ``atom37_to_cb`` and where CB must be reported present.

    Members:
        all_cb: Every residue has CB; the fallback is never used.
        no_cb: No residue has CB; the fallback must give a finite point
            distinct from both the empty CB slot and CA.
        collinear: No CB and N, CA, C collinear; the fallback must still be
            finite and off CA.
        half_mask: The first half of residues lack CB. The backbone is all
            zeros, so the fallback coincides with CA by construction.
    """

    all_cb = CbSpec(TensorCase.standard, 2, 10, 0, distinct=False)
    no_cb = CbSpec(TensorCase.no_beta_carbon, 2, 10, 10, distinct=True)
    collinear = CbSpec(TensorCase.collinear, 2, 10, 10, distinct=True)
    half_mask = CbSpec(TensorCase.half_mask, 2, 10, 5, distinct=False)


@dataclasses.dataclass(frozen=True)
class AtomTensors:
    """Positions and mask built by the atom tensor factory.

    Attributes:
        positions: ``(B, N_res, n_atoms, 3)`` coordinates.
        mask: ``(B, N_res, n_atoms)`` presence mask.
    """

    positions: Float[torch.Tensor, "B N_res atoms 3"]
    mask: Float[torch.Tensor, "B N_res atoms"]


@dataclasses.dataclass(frozen=True)
class ProteinBreak:
    """A Protein field replaced by an array with one axis wrong.

    Attributes:
        field: Name of the Protein field to replace.
        malform: Maps the valid array to the malformed one.
    """

    field: str
    malform: Callable[[Array], Array]


class InvalidProteinShape(enum.Enum):
    """Protein shape violations, each mirroring a plausible parser bug.

    Protein's ``@jaxtyped`` check spans the whole dataclass, so each of the
    three field signatures is broken once for rank, the coordinate width
    once, and the residue and atom counts twice each, on two fields, so the
    consistency checks are not tied to whichever field is checked first.

    Members:
        no_coord_axis: ``atom_positions`` lost its xyz axis.
        extra_coord: ``atom_positions`` gained a fourth coordinate.
        res_drift: ``atom_positions`` has one residue fewer, as from
            insertion-code or altloc handling.
        atom_drift: ``atom_positions`` has one atom type fewer than the
            37-slot table.
        mask_no_atom_axis: ``atom_mask`` lost its atom axis.
        mask_atom_drift: ``atom_mask`` atom count drifted on its own.
        aatype_column_vector: ``aatype`` became a column vector.
        residue_index_drift: ``residue_index`` lost a residue on its own.
    """

    no_coord_axis = ProteinBreak("atom_positions", lambda a: a[..., 0])
    extra_coord = ProteinBreak("atom_positions", lambda a: a[..., [0, 1, 2, 2]])
    res_drift = ProteinBreak("atom_positions", lambda a: a[1:])
    atom_drift = ProteinBreak("atom_positions", lambda a: a[:, 1:])
    mask_no_atom_axis = ProteinBreak("atom_mask", lambda a: a[:, 0])
    mask_atom_drift = ProteinBreak("atom_mask", lambda a: a[:, 1:])
    aatype_column_vector = ProteinBreak("aatype", lambda a: a[:, None])
    residue_index_drift = ProteinBreak("residue_index", lambda a: a[1:])


@dataclasses.dataclass(frozen=True)
class CenterCase:
    """A ``center_positions`` input and its hand-derived CA centre.

    Positions count up across residue, atom and xyz, so residue ``r``'s CA
    sits at ``111 r + (3, 4, 5)``; averaging over residues gives the centre.

    Attributes:
        n_res: Residues to build.
        present_atoms: Atom37 slots marked present.
        ca_center: Masked mean of the CA positions before centring.
    """

    n_res: int
    present_atoms: tuple[int, ...]
    ca_center: tuple[float, float, float]


class CenterPositionsScenario(enum.Enum):
    """Inputs for ``center_positions``.

    Members:
        full_mask: Eight residues with every atom present; CA centre is
            ``111 * 3.5 + (3, 4, 5)``.
        ca_only: Five residues with only CA present, so every other slot
            must end at zero; CA centre is ``111 * 2 + (3, 4, 5)``.
    """

    full_mask = CenterCase(8, tuple(range(37)), (391.5, 392.5, 393.5))
    ca_only = CenterCase(5, (ATOM37_CA,), (225.0, 226.0, 227.0))


@dataclasses.dataclass(frozen=True)
class WrongShape:
    """An atom conversion and positions of a deliberately wrong shape.

    Attributes:
        convert: Conversion under test, called as ``convert(pos, mask)``.
        positions_shape: Invalid shape for the positions.
        mask_shape: Valid shape for the mask.
    """

    convert: Convert
    positions_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]


class WrongAtomShape(enum.Enum):
    """Conversions given positions whose shape breaks their contract.

    These are what would fail if ``@jaxtyped`` were removed from a
    conversion; the coverage gate only checks names are referenced.

    Members:
        atom37_width: ``atom37_to_atom5`` given 4-component points.
        atom5_count: ``atom5_to_atom37`` given 4 atoms instead of 5.
        cb_width: ``atom37_to_cb`` given 4-component points.
    """

    atom37_width = WrongShape(atom37_to_atom5, (2, 10, 37, 4), (2, 10, 37))
    atom5_count = WrongShape(atom5_to_atom37, (2, 10, 4, 3), (2, 10, 5))
    cb_width = WrongShape(atom37_to_cb, (2, 10, 37, 4), (2, 10, 37))


@dataclasses.dataclass(frozen=True)
class PdbCase:
    """ATOM records for a PDB file and what ``protein_from_pdb`` reads.

    Attributes:
        records: ``"atom resname <chain><resseq>"`` per ATOM line,
            ``;``-joined, e.g. ``"N ALA A1"``.
        mol_types: One letter per residue: ``p``rotein, ``d``na, ``r``na.
        chains: Expected ``chain_index`` per residue.
    """

    records: str
    mol_types: str
    chains: tuple[int, ...]


class PdbScenario(enum.Enum):
    """PDB contents spanning the molecule types the parser tells apart.

    Members:
        protein: Two amino acids on chain A.
        dna_prefixed: ``DA`` and ``DT``, unambiguous DNA names.
        rna_with_ribose: ``A`` and ``U`` with an ``O2'`` ribose oxygen.
        dna_bare_names: ``A`` and ``T`` without ``O2'``: DNA, no prefix.
        protein_dna: A protein residue on chain A, a DNA one on chain B.
        two_protein_chains: Two chains of two residues, indexed 0 and 1.
    """

    protein = PdbCase("N ALA A1;CA ALA A1;N GLY A2", "pp", (0, 0))
    dna_prefixed = PdbCase("P DA A1;P DT A2", "dd", (0, 0))
    rna_with_ribose = PdbCase("O2' A A1;O2' U A2", "rr", (0, 0))
    dna_bare_names = PdbCase("C1' A A1;C1' T A2", "dd", (0, 0))
    protein_dna = PdbCase("N ALA A1;P DA B1", "pd", (0, 1))
    two_protein_chains = PdbCase("N ALA A1;N GLY A2;N ALA B1", "ppp", (0, 0, 1))


@dataclasses.dataclass(frozen=True)
class PdbProteinCase:
    """A protein layout ``to_pdb`` writes and the records it must contain.

    Attributes:
        present_atoms: Atom37 slots present in every residue.
        chains: Chain index per residue.
        atom_lines: Expected number of ATOM records.
        ter_lines: Expected number of TER records, one per chain.
    """

    present_atoms: tuple[int, ...]
    chains: tuple[int, ...]
    atom_lines: int
    ter_lines: int


class ToPdbScenario(enum.Enum):
    """Protein layouts for ``to_pdb``.

    Members:
        backbone: Five residues, N CA C O present: 20 ATOM lines.
        ca_only: Three residues, only CA present: one line per residue.
        two_chains: Four residues over chains 0 and 1: a TER per chain.
    """

    backbone = PdbProteinCase((0, 1, 2, 4), (0, 0, 0, 0, 0), 20, 1)
    ca_only = PdbProteinCase((1,), (0, 0, 0), 3, 1)
    two_chains = PdbProteinCase((0, 1, 2, 4), (0, 0, 1, 1), 16, 2)


@pytest.fixture
def atom_spec(request: pytest.FixtureRequest) -> TensorSpec:
    """Return the spec of the scenario routed in via indirect param.

    Args:
        request: Pytest request carrying an atom scenario member.

    Returns:
        The member's ``TensorSpec`` (or subclass).
    """
    scenario = cast(
        Atom37ToAtom5Scenario | Atom5ToAtom37Scenario | Atom37ToCbScenario,
        request.param,
    )
    return scenario.value


@pytest.fixture
def atom_tensors(atom_spec: TensorSpec) -> AtomTensors:
    """Build positions and mask for the spec's geometry and layout.

    Atom37 slot indices are used for both layouts: in atom5 the first five
    slots are N, CA, C, CB, O, so ``ATOM37_*`` stays inside the five.

    Args:
        atom_spec: Geometry, batch size, residue count and layout.

    Returns:
        The built tensors.
    """
    n_atoms = atom_spec.representation.value
    standard = {
        ATOM37_N: (1.0, 0.0, 0.0),
        ATOM37_CA: (0.0, 0.0, 0.0),
        ATOM37_C: (0.0, 1.0, 0.0),
        ATOM37_O: (1.0, 1.0, 0.0),
    }
    geometry = {
        TensorCase.standard: standard | {ATOM37_CB: (0.0, 0.0, 1.0)},
        TensorCase.no_beta_carbon: standard,
        TensorCase.collinear: {
            ATOM37_N: (0.0, 0.0, 0.0),
            ATOM37_CA: (1.0, 0.0, 0.0),
            ATOM37_C: (2.0, 0.0, 0.0),
            ATOM37_O: (2.0, 1.0, 0.0),
        },
    }.get(atom_spec.content, {})
    per_residue = torch.zeros(n_atoms, 3)
    for slot, xyz in geometry.items():
        per_residue[slot] = torch.tensor(xyz)
    positions = einops.repeat(
        per_residue,
        "a d -> b n a d",
        b=atom_spec.batch,
        n=atom_spec.n_res,
    )
    mask = torch.ones(atom_spec.batch, atom_spec.n_res, n_atoms)
    if atom_spec.content is TensorCase.half_mask:
        mask[:, : atom_spec.n_res // 2] = 0
    if atom_spec.content in {TensorCase.no_beta_carbon, TensorCase.collinear}:
        mask[..., ATOM37_CB] = 0
    return AtomTensors(positions=positions, mask=mask)


@pytest.fixture
def valid_protein() -> Protein:
    """Return a valid 10-residue Protein in the atom37 layout.

    Returns:
        Protein of alanines with zero positions and every atom present.
    """
    n, a = 10, len(atom_types)
    return Protein(
        atom_positions=np.zeros((n, a, 3)),
        aatype=np.zeros(n, dtype=np.intp),
        atom_mask=np.ones((n, a)),
        residue_index=np.arange(n, dtype=np.intp),
        chain_index=np.zeros(n, dtype=np.intp),
        b_factors=np.zeros((n, a)),
    )


@pytest.fixture
def protein_break(request: pytest.FixtureRequest) -> ProteinBreak:
    """Return the break of the ``InvalidProteinShape`` routed in via param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        Field and malformation to apply.
    """
    return cast(InvalidProteinShape, request.param).value


@pytest.fixture
def coords_dict() -> dict[str, list[list[float]]]:
    """Return backbone coordinates for 6 residues, as JSONL entries hold them.

    Atom ``k`` of residue ``r`` (N, CA, C, O for ``k`` = 0..3) has
    coordinates ``(r + k, r + k + 1, r + k + 2)``.

    Returns:
        Mapping from atom name to a ``(6, 3)`` nested list.
    """
    return {
        name: [[float(r + k + d) for d in range(3)] for r in range(6)]
        for k, name in enumerate(("N", "CA", "C", "O"))
    }


@pytest.fixture
def make_example() -> Callable[[int], Example]:
    """Return a factory for ``n``-residue numpy feature mappings.

    A mapping rather than a dataclass because ``make_fixed_size`` and
    ``truncate_to_length`` take one and rewrite its entries in place.
    Residue ``r`` holds the value ``r + 1`` everywhere, so padding (0) and
    kept residues are told apart.

    Returns:
        ``make(n)`` with ``atom_positions``, ``atom_mask`` and
        ``residue_index``.
    """

    def make(n: int) -> Example:
        """Build the mapping.

        Args:
            n: Residue count.

        Returns:
            The feature mapping.
        """
        value = np.arange(1, n + 1, dtype=np.float64)
        return {
            "atom_positions": einops.repeat(value, "n -> n 37 3"),
            "atom_mask": einops.repeat(value, "n -> n 37"),
            "residue_index": np.arange(1, n + 1),
        }

    return make


@pytest.fixture
def center_scenario(request: pytest.FixtureRequest) -> CenterPositionsScenario:
    """Return the ``CenterPositionsScenario`` routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(CenterPositionsScenario, request.param)


@pytest.fixture
def np_example_to_center(center_scenario: CenterPositionsScenario) -> Example:
    """Build the input mapping for a ``center_positions`` scenario.

    A mapping because ``center_positions`` writes its result back to the
    ``atom_positions`` key.

    Args:
        center_scenario: Residue count and present atoms.

    Returns:
        Deterministic ``atom_positions`` and ``atom_mask``.
    """
    case = center_scenario.value
    mask = np.zeros((case.n_res, 37))
    mask[:, list(case.present_atoms)] = 1.0
    positions = einops.rearrange(
        np.arange(case.n_res * 37 * 3, dtype=np.float64),
        "(r a d) -> r a d",
        a=37,
        d=3,
    )
    return {"atom_positions": positions, "atom_mask": mask}


@pytest.fixture
def expected_centered_positions(
    center_scenario: CenterPositionsScenario,
    np_example_to_center: Example,
) -> npt.NDArray[np.float64]:
    """Shift the input by the scenario's CA centre and zero absent atoms.

    Args:
        center_scenario: Supplies the hand-derived CA centre.
        np_example_to_center: Input mapping.

    Returns:
        Positions ``center_positions`` must produce.
    """
    positions = cast(
        npt.NDArray[np.float64],
        np_example_to_center["atom_positions"],
    )
    mask = einops.rearrange(np_example_to_center["atom_mask"], "r a -> r a 1")
    centre = np.array(center_scenario.value.ca_center)
    return (positions - centre) * mask


@pytest.fixture
def make_protein() -> ProteinFactory:
    """Return a factory for alanine proteins with chosen atoms and chains.

    ``make(present_atoms, chains)`` builds one residue per chain entry with
    those atom37 slots present at deterministic positions.

    Returns:
        The factory.
    """

    def make(present: tuple[int, ...], chains: tuple[int, ...]) -> Protein:
        """Build the protein.

        Args:
            present: Atom37 slots present in every residue.
            chains: Chain index per residue.

        Returns:
            The protein.
        """
        n = len(chains)
        mask = np.zeros((n, 37))
        mask[:, list(present)] = 1.0
        positions = einops.rearrange(
            np.arange(n * 37 * 3, dtype=np.float64) / 10,
            "(r a d) -> r a d",
            a=37,
            d=3,
        )
        return Protein(
            atom_positions=positions,
            aatype=np.zeros(n, dtype=np.intp),
            atom_mask=mask,
            residue_index=np.arange(n, dtype=np.intp),
            chain_index=np.array(chains, dtype=np.intp),
            b_factors=np.zeros((n, 37)),
        )

    return make


@pytest.fixture
def roundtrip_protein() -> Protein:
    """Return a 5-residue protein whose fields survive a PDB round trip.

    Residues are A, R, S, V and G; the first four have N, CA, C, CB, O and
    glycine has no CB. Present positions are multiples of 0.001, the PDB
    column precision, and absent ones are zero.

    Returns:
        The protein.
    """
    aatype = np.array([restype_order[c] for c in "ARSVG"], dtype=np.intp)
    present = np.zeros((5, 37))
    present[:, [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_CB, ATOM37_O]] = 1.0
    present[4, ATOM37_CB] = 0.0
    grid = einops.rearrange(np.arange(555.0), "(r a d) -> r a d", a=37, d=3)
    mask = einops.rearrange(present, "r a -> r a 1")
    return Protein(
        atom_positions=np.round(grid * 0.123, 3) * mask,
        aatype=aatype,
        atom_mask=present,
        residue_index=np.arange(5, dtype=np.intp),
        chain_index=np.zeros(5, dtype=np.intp),
        b_factors=np.zeros((5, 37)),
    )


@pytest.fixture
def write_pdb(tmp_path: pathlib.Path) -> Callable[[str], pathlib.Path]:
    """Return a writer turning ``"atom resname <chain><resseq>;..."`` to PDB.

    Each entry becomes one 80-column ATOM record at ``(1, 2, 3)``; an entry
    starting with ``HETATM`` becomes a HETATM record instead. Atom names
    are written in the 3-character form, which every name used here fits.

    Args:
        tmp_path: Per-test directory.

    Returns:
        ``write(records)``, returning the file path.
    """

    def write(records: str) -> pathlib.Path:
        """Write the records to ``model.pdb``.

        Args:
            records: ``;``-separated ``[HETATM] atom resname <chain><n>``.

        Returns:
            The file path.
        """
        lines: list[str] = []
        for serial, entry in enumerate(records.split(";"), start=1):
            record, atom, resname, where = [PDB_REC_ATOM, *entry.split()][-4:]
            name = f" {atom:<3}"
            lines.append(
                (
                    f"{record:<6}{serial:>5} {name} {resname:>3} "
                    + f"{where[0]}{int(where[1:]):>4}    "
                    + f"{1.0:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00  0.00"
                ).ljust(80),
            )
        path = tmp_path / "model.pdb"
        _ = path.write_text("\n".join(lines) + "\n")
        return path

    return write


@pytest.fixture
def pdb_scenario(request: pytest.FixtureRequest) -> PdbScenario:
    """Return the ``PdbScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(PdbScenario, request.param)


@pytest.fixture
def to_pdb_scenario(request: pytest.FixtureRequest) -> ToPdbScenario:
    """Return the ``ToPdbScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(ToPdbScenario, request.param)


@pytest.fixture
def wrong_shape(request: pytest.FixtureRequest) -> WrongShape:
    """Return the case of the ``WrongAtomShape`` routed in via param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        Conversion and shapes.
    """
    return cast(WrongAtomShape, request.param).value


@pytest.mark.parametrize(
    "atom_spec",
    list(Atom37ToAtom5Scenario),
    ids=[case.name for case in Atom37ToAtom5Scenario],
    indirect=True,
)
def test_atom37_to_atom5_copies_the_five_backbone_slots(
    atom_spec: TensorSpec,
    atom_tensors: AtomTensors,
) -> None:
    """Each atom5 slot holds its atom37 atom's position and mask, same dtype.

    A CB absent in atom37 stays absent in atom5.

    Args:
        atom_spec: Batch and residue counts.
        atom_tensors: Atom37 positions and mask.
    """
    x = atom_tensors
    positions, mask = atom37_to_atom5(x.positions, x.mask)
    atom5 = [ATOM5_N, ATOM5_CA, ATOM5_C, ATOM5_O, ATOM5_CB]
    atom37 = [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB]
    n_atoms = ResidueRepresentation.atom5.value
    shape = (atom_spec.batch, atom_spec.n_res, n_atoms)
    assert (positions.shape, mask.shape) == ((*shape, 3), shape)
    assert (positions.dtype, mask.dtype) == (x.positions.dtype, x.mask.dtype)
    assert torch.equal(positions[:, :, atom5], x.positions[:, :, atom37])
    assert torch.equal(mask[:, :, atom5], x.mask[:, :, atom37])


@pytest.mark.parametrize(
    "atom_spec",
    list(Atom5ToAtom37Scenario),
    ids=[case.name for case in Atom5ToAtom37Scenario],
    indirect=True,
)
def test_atom5_to_atom37_scatters_into_backbone_slots_only(
    atom_spec: Atom5Spec,
    atom_tensors: AtomTensors,
) -> None:
    """Each atom5 slot lands at its atom37 index; every other slot is zero.

    With no mask passed, the backbone slots are marked present.

    Args:
        atom_spec: Sizes and whether to pass the mask.
        atom_tensors: Atom5 positions and mask.
    """
    x = atom_tensors
    mask_in = x.mask if atom_spec.provide_mask else None
    expected_mask = (
        x.mask if atom_spec.provide_mask else torch.ones_like(x.mask)
    )
    positions, mask = atom5_to_atom37(x.positions, mask_in)
    other = sorted(set(range(37)) - set(ATOM5_TO_ATOM37))
    n_atoms = ResidueRepresentation.atom37.value
    shape = (atom_spec.batch, atom_spec.n_res, n_atoms)
    assert (positions.shape, mask.shape) == ((*shape, 3), shape)
    assert (positions.dtype, mask.dtype) == (x.positions.dtype, x.mask.dtype)
    assert torch.equal(positions[:, :, ATOM5_TO_ATOM37], x.positions)
    assert torch.equal(mask[:, :, ATOM5_TO_ATOM37], expected_mask)
    assert torch.count_nonzero(positions[:, :, other]) == 0
    assert torch.count_nonzero(mask[:, :, other]) == 0


@pytest.mark.parametrize(
    "atom_spec",
    list(Atom37ToCbScenario),
    ids=[case.name for case in Atom37ToCbScenario],
    indirect=True,
)
def test_atom37_to_cb_uses_cb_when_present_and_a_pseudo_cb_otherwise(
    atom_spec: CbSpec,
    atom_tensors: AtomTensors,
) -> None:
    """Present CBs are copied; absent ones become a finite pseudo-Cβ.

    Where the backbone is non-degenerate, the pseudo-Cβ is a new point,
    neither the empty CB slot nor CA. Collinear N, CA, C make the cross
    product vanish; the epsilon in the normalisation must keep it finite.

    Args:
        atom_spec: Where CB is present and whether the fallback is distinct.
        atom_tensors: Atom37 positions and mask.
    """
    x = atom_tensors
    cb, present = atom37_to_cb(x.positions, x.mask)
    expected = torch.zeros(atom_spec.batch, atom_spec.n_res, dtype=torch.bool)
    expected[:, atom_spec.present_from :] = True
    absent = ~expected
    stored_cb = x.positions[:, :, ATOM37_CB]
    ca = x.positions[:, :, ATOM37_CA]
    moved_off_cb = not torch.allclose(cb[absent], stored_cb[absent])
    moved_off_ca = not torch.allclose(cb[absent], ca[absent])
    assert cb.shape == (atom_spec.batch, atom_spec.n_res, 3)
    assert torch.equal(present, expected)
    assert torch.equal(cb[expected], stored_cb[expected])
    assert torch.isfinite(cb[absent]).all()
    assert (moved_off_cb, moved_off_ca) == (atom_spec.distinct,) * 2


@pytest.mark.parametrize(
    "wrong_shape",
    list(WrongAtomShape),
    ids=[case.name for case in WrongAtomShape],
    indirect=True,
)
def test_atom_conversions_reject_wrong_shapes(wrong_shape: WrongShape) -> None:
    """Each conversion's contract rejects positions of the wrong shape.

    Args:
        wrong_shape: Conversion and shapes.
    """
    positions = torch.zeros(wrong_shape.positions_shape)
    mask = torch.ones(wrong_shape.mask_shape)
    with pytest.raises(TypeCheckError):
        _ = wrong_shape.convert(positions, mask)


@pytest.mark.parametrize(
    "protein_break",
    list(InvalidProteinShape),
    ids=[case.name for case in InvalidProteinShape],
    indirect=True,
)
def test_protein_rejects_invalid_shapes(
    valid_protein: Protein,
    protein_break: ProteinBreak,
) -> None:
    """Rebuilding a valid Protein with one malformed field raises.

    Every other field stays valid, so the error comes from the replaced
    one.

    Args:
        valid_protein: Protein whose other fields stay valid.
        protein_break: Field and malformation.
    """
    old = cast(Array, getattr(valid_protein, protein_break.field))
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            valid_protein,
            **{protein_break.field: protein_break.malform(old)},
        )


def test_restype_x_is_the_index_after_the_canonical_twenty(
    valid_protein: Protein,
) -> None:
    """The unknown token ``X`` sits at index 20, one past the canonicals.

    The sequence loss ignores targets at or above ``RESTYPE_NUM_NO_X``, so
    ``X`` must be exactly there, above every canonical residue such as the
    fixture's alanines. Not parametrized: it is one constant.

    Args:
        valid_protein: Protein of alanines, index 0.
    """
    aatype = torch.tensor(valid_protein.aatype)
    assert restype_order["X"] == RESTYPE_NUM_NO_X
    assert bool((aatype < restype_order["X"]).all())


def test_make_np_example_packs_backbone_into_atom37(
    coords_dict: dict[str, list[list[float]]],
) -> None:
    """Backbone coordinates land in their atom37 slots, marked present.

    All other slots are zero and absent, and ``residue_index`` counts from
    0. Not parametrized: the NaN case is the next test.

    Args:
        coords_dict: Six residues of N, CA, C, O coordinates.
    """
    example = {
        k: torch.tensor(v) for k, v in make_np_example(coords_dict).items()
    }
    names = ("N", "CA", "C", "O")
    backbone = [atom_types.index(name) for name in names]
    other = sorted(set(range(37)) - set(backbone))
    expected = torch.tensor([coords_dict[a] for a in names]).transpose(0, 1)
    assert torch.equal(example["atom_positions"][:, backbone], expected)
    assert torch.count_nonzero(example["atom_positions"][:, other]) == 0
    assert bool(example["atom_mask"][:, backbone].all())
    assert torch.count_nonzero(example["atom_mask"][:, other]) == 0
    assert torch.equal(example["residue_index"], torch.arange(6))


@pytest.mark.parametrize("atom", ["N", "CA", "O"])
def test_make_np_example_masks_and_zeroes_nan_coordinates(
    coords_dict: dict[str, list[list[float]]],
    atom: str,
) -> None:
    """A NaN coordinate is set to zero and its atom marked absent.

    Other atoms of the same residue stay present.

    Args:
        coords_dict: Coordinates; residue 0's ``atom`` is set to NaN here.
        atom: Backbone atom to corrupt.
    """
    coords_dict[atom][0] = [float("nan")] * 3
    example = {
        k: torch.tensor(v) for k, v in make_np_example(coords_dict).items()
    }
    slot = atom_types.index(atom)
    assert bool(torch.isfinite(example["atom_positions"]).all())
    assert torch.count_nonzero(example["atom_positions"][0, slot]) == 0
    assert example["atom_mask"][0, slot].item() == 0
    assert example["atom_mask"][0, atom_types.index("C")].item() == 1


@pytest.mark.parametrize(
    "n_res",
    [5, 10, 20],
    ids=["padded", "unchanged", "truncated"],
)
def test_make_fixed_size_pads_or_truncates_to_ten(
    make_example: Callable[[int], Example],
    n_res: int,
) -> None:
    """Every array ends with 10 residues: the first kept, the rest zero.

    Args:
        make_example: Feature-mapping factory; residue ``r`` holds ``r + 1``.
        n_res: Residues before resizing.
    """
    example = make_example(n_res)
    make_fixed_size(example, max_seq_length=10)
    kept = min(n_res, 10)
    out = {k: torch.tensor(v) for k, v in example.items()}
    reference = {k: torch.tensor(v) for k, v in make_example(kept).items()}
    assert {k: len(v) for k, v in out.items()} == dict.fromkeys(out, 10)
    assert all(torch.equal(out[k][:kept], reference[k]) for k in out)
    assert all(torch.count_nonzero(out[k][kept:]) == 0 for k in out)


@pytest.mark.parametrize(
    ("n_res", "expected"),
    [
        pytest.param(100, 50, id="truncated"),
        pytest.param(50, 50, id="exact"),
        pytest.param(30, 30, id="short_not_padded"),
    ],
)
def test_truncate_to_length_cuts_but_never_pads(
    make_example: Callable[[int], Example],
    n_res: int,
    expected: int,
) -> None:
    """Arrays longer than 50 residues are cut to 50; shorter ones are kept.

    Args:
        make_example: Feature-mapping factory.
        n_res: Residues before truncation.
        expected: Residues after.
    """
    example = make_example(n_res)
    truncate_to_length(example, 50)
    out = {k: torch.tensor(v) for k, v in example.items()}
    reference = {k: torch.tensor(v) for k, v in make_example(expected).items()}
    assert all(torch.equal(out[k], reference[k]) for k in out)


@pytest.mark.parametrize(
    "center_scenario",
    list(CenterPositionsScenario),
    ids=[case.name for case in CenterPositionsScenario],
    indirect=True,
)
def test_center_positions(
    np_example_to_center: Example,
    expected_centered_positions: npt.NDArray[np.float64],
) -> None:
    """``center_positions`` writes CA-centred, masked positions to the mapping.

    Exact values cover moving the CA centre to the origin, keeping absent
    atoms at zero, and updating the mapping the caller passed in.

    Args:
        np_example_to_center: Input mapping.
        expected_centered_positions: Positions it must hold afterwards.
    """
    center_positions(np_example_to_center)
    np.testing.assert_allclose(
        np_example_to_center["atom_positions"],
        expected_centered_positions,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        pytest.param(
            (100, "ALA", "A", 10),
            "TER     100      ALA A  10",
            id="a",
        ),
        pytest.param(
            (7, "GLY", "B", 1234),
            "TER       7      GLY B1234",
            id="b",
        ),
    ],
)
def test_chain_end_formats_a_ter_record(
    args: tuple[int, str, str, int],
    expected: str,
) -> None:
    """``chain_end`` lays out serial, residue, chain and number in columns.

    Args:
        args: ``(serial, resname, chain, resseq)``.
        expected: The exact TER line.
    """
    assert chain_end(*args) == expected


@pytest.mark.parametrize(
    "to_pdb_scenario",
    list(ToPdbScenario),
    ids=[case.name for case in ToPdbScenario],
    indirect=True,
)
def test_to_pdb_writes_one_atom_line_per_present_atom(
    to_pdb_scenario: ToPdbScenario,
    make_protein: ProteinFactory,
) -> None:
    """The PDB has a MODEL block, an ATOM per present atom, a TER per chain.

    Every line is padded to 80 columns, and absent atoms are skipped.

    Args:
        to_pdb_scenario: Protein layout and expected record counts.
        make_protein: Protein factory.
    """
    case = to_pdb_scenario.value
    lines = to_pdb(make_protein(case.present_atoms, case.chains)).splitlines()
    records = [line.split()[0] for line in lines if line.strip()]
    assert {len(line) for line in lines} == {80}
    assert records.count(PDB_REC_ATOM) == case.atom_lines
    assert records.count(PDB_REC_TER) == case.ter_lines
    assert records[0] == PDB_REC_MODEL
    assert records[-2:] == [PDB_REC_ENDMDL, PDB_REC_END]


@pytest.mark.parametrize(
    ("aatype", "chains", "error"),
    [
        pytest.param(RESTYPE_NUM + 1, (0, 0), InvalidAAtypesError, id="bad_aa"),
        pytest.param(0, (0, PDB_MAX_CHAINS), ValueError, id="too_many_chains"),
    ],
)
def test_to_pdb_rejects_unwritable_proteins(
    make_protein: ProteinFactory,
    aatype: int,
    chains: tuple[int, ...],
    error: type[Exception],
) -> None:
    """A residue type past the vocabulary, or too many chains, raises.

    Args:
        make_protein: Protein factory.
        aatype: Residue type for every residue.
        chains: Chain index per residue.
        error: Exception expected.
    """
    protein = dataclasses.replace(
        make_protein((1,), chains),
        aatype=np.full(len(chains), aatype, dtype=np.intp),
    )
    with pytest.raises(error):
        _ = to_pdb(protein)


def test_to_pdb_and_protein_from_pdb_round_trip(
    roundtrip_protein: Protein,
    tmp_path: pathlib.Path,
) -> None:
    """Writing and re-reading a protein preserves every field.

    ``b_factors`` come back as the protein molecule-type code. Not
    parametrized: the protein mixes residues with and without CB.

    Args:
        roundtrip_protein: Protein at PDB precision.
        tmp_path: Per-test directory.
    """
    path = tmp_path / "roundtrip.pdb"
    _ = path.write_text(to_pdb(roundtrip_protein))
    recovered = protein_from_pdb(path)
    every_field = {f.name for f in dataclasses.fields(Protein)}
    fields = sorted(every_field - {"b_factors"})
    same = {
        name: torch.equal(
            torch.tensor(cast(Array, getattr(recovered, name))),
            torch.tensor(cast(Array, getattr(roundtrip_protein, name))),
        )
        for name in fields
    }
    b_factors = torch.tensor(recovered.b_factors)
    assert same == dict.fromkeys(fields, True)
    assert bool((b_factors == MOL_TYPE_PROTEIN).all())


@pytest.mark.parametrize(
    "pdb_scenario",
    list(PdbScenario),
    ids=[case.name for case in PdbScenario],
    indirect=True,
)
def test_protein_from_pdb_classifies_molecules_and_chains(
    pdb_scenario: PdbScenario,
    write_pdb: Callable[[str], pathlib.Path],
) -> None:
    """Each residue's molecule type goes to ``b_factors``; chains are indexed.

    Nucleotides get the unknown residue type, ``RESTYPE_NUM_NO_X``.

    Args:
        pdb_scenario: Records and expected classification.
        write_pdb: PDB writer.
    """
    case = pdb_scenario.value
    codes = {"p": MOL_TYPE_PROTEIN, "d": MOL_TYPE_DNA, "r": MOL_TYPE_RNA}
    is_nucleotide = {"p": False, "d": True, "r": True}
    protein = protein_from_pdb(write_pdb(case.records))
    b_factors = torch.tensor(protein.b_factors[:, 0])
    aatype = torch.tensor(protein.aatype)
    nucleotide = torch.tensor([is_nucleotide[c] for c in case.mol_types])
    expected_types = torch.tensor([codes[c] for c in case.mol_types])
    assert protein.atom_positions.shape == (len(case.mol_types), 37, 3)
    assert torch.equal(b_factors, expected_types.double())
    assert torch.equal(
        torch.tensor(protein.chain_index),
        torch.tensor(case.chains),
    )
    assert bool((aatype[nucleotide] == RESTYPE_NUM_NO_X).all())


@pytest.mark.parametrize(
    "hetatm",
    ["HETATM O HOH A100", "HETATM MG MG A200", "HETATM N1 ATP A300"],
    ids=["water", "ion", "ligand"],
)
def test_protein_from_pdb_skips_hetatm_records(
    write_pdb: Callable[[str], pathlib.Path],
    hetatm: str,
) -> None:
    """HETATM residues (water, ions, ligands) are left out of the protein.

    Args:
        write_pdb: PDB writer.
        hetatm: The HETATM record added after two ATOM residues.
    """
    protein = protein_from_pdb(write_pdb(f"N ALA A1;N GLY A2;{hetatm}"))
    assert protein.atom_positions.shape[0] == len(["ALA", "GLY"])


def test_protein_from_pdb_keeps_pdb_residue_numbers(
    write_pdb: Callable[[str], pathlib.Path],
) -> None:
    """``residue_index`` holds the RESSEQ column, gaps included.

    Not parametrized: one gap shows the numbers are copied, not counted.

    Args:
        write_pdb: PDB writer.
    """
    protein = protein_from_pdb(write_pdb("N ALA A5;N ALA A10"))
    assert torch.equal(
        torch.tensor(protein.residue_index),
        torch.tensor([5, 10]),
    )


def test_protein_from_pdb_rejects_a_file_without_atoms(
    tmp_path: pathlib.Path,
) -> None:
    """A PDB with only REMARK and HEADER lines raises ``ValueError``.

    Not parametrized: any file without ATOM records takes this path.

    Args:
        tmp_path: Per-test directory.
    """
    path = tmp_path / "empty.pdb"
    _ = path.write_text("REMARK empty structure\nHEADER test\n")
    with pytest.raises(ValueError, match="No ATOM records"):
        _ = protein_from_pdb(path)


@pytest.mark.parametrize(
    ("resname", "atoms", "expected"),
    [
        pytest.param("ALA", (), MOL_TYPE_PROTEIN, id="alanine"),
        pytest.param("VAL", ("C1'",), MOL_TYPE_PROTEIN, id="valine"),
        pytest.param("DA", (), MOL_TYPE_DNA, id="dna_prefixed"),
        pytest.param("DT", ("O2'",), MOL_TYPE_DNA, id="dna_prefix_wins"),
        pytest.param("T", (), MOL_TYPE_DNA, id="bare_thymine"),
        pytest.param("U", (), MOL_TYPE_RNA, id="uracil"),
        pytest.param("A", ("O2'", "C1'"), MOL_TYPE_RNA, id="a_with_ribose"),
        pytest.param("G", ("O2'",), MOL_TYPE_RNA, id="g_with_ribose"),
        pytest.param("C", (), MOL_TYPE_DNA, id="c_without_ribose"),
        pytest.param("G", ("P",), MOL_TYPE_DNA, id="g_without_ribose"),
        pytest.param("MSE", (), MOL_TYPE_PROTEIN, id="unknown_is_protein"),
        pytest.param("", (), MOL_TYPE_PROTEIN, id="empty_is_protein"),
    ],
)
def test_classify_mol_type(
    resname: str,
    atoms: tuple[str, ...],
    expected: int,
) -> None:
    """Names decide when unambiguous; ``O2'`` decides for bare A, C, G.

    Args:
        resname: Residue name.
        atoms: Atom names seen in the residue.
        expected: Molecule-type code.
    """
    assert classify_mol_type(resname, frozenset(atoms)) == expected


@pytest.mark.parametrize(
    ("restypes", "order", "names"),
    [
        pytest.param(DNA_RESTYPES, DNA_RESTYPE_ORDER, "DA DC DG DT", id="dna"),
        pytest.param(RNA_RESTYPES, RNA_RESTYPE_ORDER, "A C G U", id="rna"),
    ],
)
def test_nucleotide_order_indexes_bases_canonically(
    restypes: list[str],
    order: Mapping[str, int],
    names: str,
) -> None:
    """The base list and its order mapping both follow the canonical order.

    Args:
        restypes: The library's base list.
        order: The library's base-to-index mapping.
        names: Expected bases, space-separated, in order.
    """
    expected = names.split()
    assert list(restypes) == expected
    assert dict(order) == {name: i for i, name in enumerate(expected)}


@pytest.mark.parametrize(
    ("three_to_one", "names", "letters"),
    [
        pytest.param(DNA_RESTYPE_3TO1, "DA DC DG DT", "acgt", id="dna"),
        pytest.param(RNA_RESTYPE_3TO1, "A C G U", "acgu", id="rna"),
    ],
)
def test_nucleotide_three_to_one_gives_lower_case_letters(
    three_to_one: Mapping[str, str],
    names: str,
    letters: str,
) -> None:
    """Each base maps to its lower-case one-letter code.

    Args:
        three_to_one: The library's mapping.
        names: Bases, space-separated.
        letters: Their codes, in order.
    """
    assert dict(three_to_one) == dict(zip(names.split(), letters, strict=True))


@pytest.mark.parametrize(
    "table",
    [DNA_RESTYPE_ORDER, DNA_RESTYPE_3TO1, RNA_RESTYPE_ORDER, RNA_RESTYPE_3TO1],
    ids=["dna_order", "dna_3to1", "rna_order", "rna_3to1"],
)
def test_nucleotide_tables_are_read_only(table: Mapping[str, object]) -> None:
    """Every exported nucleotide table rejects item assignment.

    Args:
        table: The library mapping.
    """
    writable = cast(dict[str, object], cast(object, table))
    assert isinstance(table, MappingProxyType)
    with pytest.raises(TypeError, match="does not support item assignment"):
        writable["XX"] = 0
