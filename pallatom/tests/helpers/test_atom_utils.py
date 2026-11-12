"""Tests for atom utility dataclasses and functions.

Covers the atom37/atom5 conversions and atom37_to_cb's pseudo-Cβ fallback,
the Protein dataclass's shape contract, make_np_example, make_fixed_size,
truncate_to_length, center_positions, chain_end, to_pdb, protein_from_pdb
with its molecule-type classification, and the residue-type constants.

Every input is deterministic and every scenario enum's value is a frozen
dataclass, so each case reads as one row of named fields.
"""

import abc
import dataclasses
import enum
import pathlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, cast, override

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
    ATOM5_NAMES,
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
from jaxtyping import Bool, Float, TypeCheckError


def feature_example(
    n: int,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Build an ``n``-residue numpy feature mapping.

    A mapping rather than a dataclass because ``make_fixed_size`` and
    ``truncate_to_length`` take one and rewrite its entries in place.
    Residue ``r`` holds the value ``r + 1`` everywhere, so padding (0) and
    kept residues are told apart.

    Args:
        n: Residue count.

    Returns:
        ``atom_positions``, ``atom_mask`` and ``residue_index``.
    """
    value = np.arange(1, n + 1, dtype=np.float64)
    return {
        "atom_positions": einops.repeat(value, "n -> n 37 3"),
        "atom_mask": einops.repeat(value, "n -> n 37"),
        "residue_index": np.arange(1, n + 1),
    }


def alanine_protein(
    present: tuple[int, ...],
    chains: tuple[int, ...],
) -> Protein:
    """Build one alanine per chain entry with chosen atom37 slots present.

    Args:
        present: Atom37 slots present in every residue.
        chains: Chain index per residue.

    Returns:
        The protein, at deterministic positions.
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


def write_pdb(path: pathlib.Path, records: str) -> pathlib.Path:
    """Write ``"atom resname <chain><resseq>;..."`` records as a PDB file.

    Each entry becomes one 80-column ATOM record at ``(1, 2, 3)``; an entry
    starting with ``HETATM`` becomes a HETATM record instead. Atom names
    are written in the 3-character form, which every name used here fits.

    Args:
        path: Destination file.
        records: ``;``-separated ``[HETATM] atom resname <chain><n>``.

    Returns:
        The path written.
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
    _ = path.write_text("\n".join(lines) + "\n")
    return path


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
class AtomSlot:
    """One backbone atom's name, its index constants, and dataset presence.

    Attributes:
        atom_name: PDB atom name.
        atom37_index: Index constant for this atom in the atom37 layout.
        atom5_index: Index constant for this atom in the atom5 layout.
        in_dataset: Whether dataset entries carry coordinates for this atom.
            Entries hold N, CA, C and O only, so CB is always built as a
            pseudo-Cβ from the backbone.
    """

    atom_name: str
    atom37_index: int
    atom5_index: int
    in_dataset: bool


class NamedAtom(enum.Enum):
    """Backbone atoms with the index constants that must agree with their name.

    Members:
        n: Backbone nitrogen.
        ca: Alpha carbon.
        c: Carbonyl carbon.
        o: Carbonyl oxygen.
        cb: Beta carbon, which dataset entries never include.
    """

    n = AtomSlot("N", ATOM37_N, ATOM5_N, in_dataset=True)
    ca = AtomSlot("CA", ATOM37_CA, ATOM5_CA, in_dataset=True)
    c = AtomSlot("C", ATOM37_C, ATOM5_C, in_dataset=True)
    o = AtomSlot("O", ATOM37_O, ATOM5_O, in_dataset=True)
    cb = AtomSlot("CB", ATOM37_CB, ATOM5_CB, in_dataset=False)


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

    def slot(self, atom: NamedAtom) -> int:
        """Index of ``atom`` in this spec's atom layout.

        Args:
            atom: Backbone atom to locate.

        Returns:
            The atom's index along the atom axis of the built tensors.
        """
        return atom.value.atom37_index


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

    @override
    def slot(self, atom: NamedAtom) -> int:
        """Index of ``atom`` in the atom5 layout.

        Args:
            atom: Backbone atom to locate.

        Returns:
            The atom's index along the atom axis of the built tensors.
        """
        return atom.value.atom5_index


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
class SlotContents:
    """Coordinates and mask values held by one atom slot across residues.

    Attributes:
        positions: The slot's coordinates for each residue (N_res, 3).
        mask: The slot's mask value for each residue (N_res,).
    """

    positions: Float[torch.Tensor, "N_res 3"]
    mask: Float[torch.Tensor, "N_res"]


@dataclasses.dataclass(frozen=True)
class ProteinBreak(abc.ABC):
    """A Protein field replaced by an array with one axis wrong.

    Attributes:
        field: Name of the Protein field to replace.
    """

    field: str

    @abc.abstractmethod
    def malform(
        self,
        array: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.intp]:
        """Map the field's valid array to the malformed one.

        Args:
            array: The field's valid value.

        Returns:
            The malformed replacement.
        """


@dataclasses.dataclass(frozen=True)
class SelectBreak(ProteinBreak):
    """Keep one index of an axis, removing that axis.

    Attributes:
        axis: Axis to select from.
        index: Position along ``axis`` to keep.
    """

    axis: int
    index: int

    @override
    def malform(
        self,
        array: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.intp]:
        """Select ``index`` along ``axis``.

        Args:
            array: The field's valid value.

        Returns:
            ``array`` with ``axis`` removed.
        """
        return cast(
            npt.NDArray[np.float64] | npt.NDArray[np.intp],
            np.take(array, self.index, axis=self.axis),
        )


@dataclasses.dataclass(frozen=True)
class DropFirstBreak(ProteinBreak):
    """Remove the first entry of an axis, shortening it by one.

    Attributes:
        axis: Axis to shorten.
    """

    axis: int

    @override
    def malform(
        self,
        array: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.intp]:
        """Drop index 0 along ``axis``.

        Args:
            array: The field's valid value.

        Returns:
            ``array`` with ``axis`` one shorter.
        """
        return np.delete(array, 0, axis=self.axis)


@dataclasses.dataclass(frozen=True)
class RepeatFirstBreak(ProteinBreak):
    """Append a copy of an axis's first entry, lengthening it by one.

    Attributes:
        axis: Axis to lengthen.
    """

    axis: int

    @override
    def malform(
        self,
        array: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.intp]:
        """Append index 0 along ``axis`` to the end.

        Args:
            array: The field's valid value.

        Returns:
            ``array`` with ``axis`` one longer.
        """
        return cast(
            npt.NDArray[np.float64] | npt.NDArray[np.intp],
            np.concatenate(
                [array, np.take(array, [0], axis=self.axis)],
                axis=self.axis,
            ),
        )


@dataclasses.dataclass(frozen=True)
class AddAxisBreak(ProteinBreak):
    """Insert a length-one axis.

    Attributes:
        axis: Position of the new axis.
    """

    axis: int

    @override
    def malform(
        self,
        array: npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.intp]:
        """Insert a new axis at ``axis``.

        Args:
            array: The field's valid value.

        Returns:
            ``array`` with one more axis.
        """
        return cast(
            npt.NDArray[np.float64] | npt.NDArray[np.intp],
            np.expand_dims(array, self.axis),
        )


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

    no_coord_axis = SelectBreak("atom_positions", axis=-1, index=0)
    extra_coord = RepeatFirstBreak("atom_positions", axis=-1)
    res_drift = DropFirstBreak("atom_positions", axis=0)
    atom_drift = DropFirstBreak("atom_positions", axis=1)
    mask_no_atom_axis = SelectBreak("atom_mask", axis=1, index=0)
    mask_atom_drift = DropFirstBreak("atom_mask", axis=1)
    aatype_column_vector = AddAxisBreak("aatype", axis=1)
    residue_index_drift = DropFirstBreak("residue_index", axis=0)


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
class WrongShape(abc.ABC):
    """An atom conversion and positions of a deliberately wrong shape.

    Attributes:
        positions_shape: Invalid shape for the positions.
        mask_shape: Valid shape for the mask.
    """

    positions_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]

    @abc.abstractmethod
    def convert(
        self,
        positions: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
    ) -> object:
        """Run the conversion under test.

        Args:
            positions: Positions of ``positions_shape``.
            mask: Mask of ``mask_shape``.

        Returns:
            The conversion's result; the test expects it to raise.
        """


@dataclasses.dataclass(frozen=True)
class Atom37ToAtom5Shape(WrongShape):
    """Calls ``atom37_to_atom5(positions, mask)``."""

    @override
    def convert(
        self,
        positions: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
    ) -> object:
        """Call ``atom37_to_atom5``.

        Args:
            positions: Positions of ``positions_shape``.
            mask: Mask of ``mask_shape``.

        Returns:
            The converted tensors.
        """
        return atom37_to_atom5(positions, mask)


@dataclasses.dataclass(frozen=True)
class Atom5ToAtom37Shape(WrongShape):
    """Calls ``atom5_to_atom37(positions, mask)``."""

    @override
    def convert(
        self,
        positions: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
    ) -> object:
        """Call ``atom5_to_atom37``.

        Args:
            positions: Positions of ``positions_shape``.
            mask: Mask of ``mask_shape``.

        Returns:
            The converted tensors.
        """
        return atom5_to_atom37(positions, mask)


@dataclasses.dataclass(frozen=True)
class Atom37ToCbShape(WrongShape):
    """Calls ``atom37_to_cb(positions, mask)``."""

    @override
    def convert(
        self,
        positions: Float[torch.Tensor, "..."],
        mask: Float[torch.Tensor, "..."],
    ) -> object:
        """Call ``atom37_to_cb``.

        Args:
            positions: Positions of ``positions_shape``.
            mask: Mask of ``mask_shape``.

        Returns:
            The converted tensors.
        """
        return atom37_to_cb(positions, mask)


class WrongAtomShape(enum.Enum):
    """Conversions given positions whose shape breaks their contract.

    These are what would fail if ``@jaxtyped`` were removed from a
    conversion; the coverage gate only checks names are referenced.

    Members:
        atom37_width: ``atom37_to_atom5`` given 4-component points.
        atom5_count: ``atom5_to_atom37`` given 4 atoms instead of 5.
        cb_width: ``atom37_to_cb`` given 4-component points.
    """

    atom37_width = Atom37ToAtom5Shape((2, 10, 37, 4), (2, 10, 37))
    atom5_count = Atom5ToAtom37Shape((2, 10, 4, 3), (2, 10, 5))
    cb_width = Atom37ToCbShape((2, 10, 37, 4), (2, 10, 37))


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


@dataclasses.dataclass(frozen=True)
class ResizeCase:
    """A residue count before resizing and how many survive it.

    Attributes:
        n_res: Residues before resizing.
        kept: Leading residues that must come through unchanged.
    """

    n_res: int
    kept: int


class FixedSizeScenario(enum.Enum):
    """Inputs to ``make_fixed_size`` with ``max_seq_length = 10``.

    Members:
        padded: 5 residues, padded with zeros to 10.
        unchanged: Exactly 10 residues.
        truncated: 20 residues, cut to 10.
    """

    padded = ResizeCase(n_res=5, kept=5)
    unchanged = ResizeCase(n_res=10, kept=10)
    truncated = ResizeCase(n_res=20, kept=10)


class TruncateScenario(enum.Enum):
    """Inputs to ``truncate_to_length`` with a length of 50.

    Members:
        truncated: 100 residues, cut to 50.
        exact: Exactly 50 residues.
        short_not_padded: 30 residues, left at 30.
    """

    truncated = ResizeCase(n_res=100, kept=50)
    exact = ResizeCase(n_res=50, kept=50)
    short_not_padded = ResizeCase(n_res=30, kept=30)


@dataclasses.dataclass(frozen=True)
class TerCase:
    """``chain_end`` arguments and the TER line they must produce.

    Attributes:
        serial: Atom serial number.
        resname: Three-letter residue name.
        chain: Chain identifier.
        resseq: Residue number.
        expected: The exact TER line.
    """

    serial: int
    resname: str
    chain: str
    resseq: int
    expected: str


class TerScenario(enum.Enum):
    """TER records with narrow and full-width columns.

    Members:
        a: A three-digit serial and two-digit residue number.
        b: A one-digit serial and four-digit residue number.
    """

    a = TerCase(100, "ALA", "A", 10, "TER     100      ALA A  10")
    b = TerCase(7, "GLY", "B", 1234, "TER       7      GLY B1234")


@dataclasses.dataclass(frozen=True)
class UnwritableCase:
    """A protein ``to_pdb`` must refuse, and the error it raises.

    Attributes:
        aatype: Residue type for every residue.
        chains: Chain index per residue.
        error: Exception expected.
    """

    aatype: int
    chains: tuple[int, ...]
    error: type[Exception]


class UnwritableScenario(enum.Enum):
    """Proteins the PDB format cannot hold.

    Members:
        bad_aa: A residue type past the vocabulary.
        too_many_chains: A chain index past the PDB chain alphabet.
    """

    bad_aa = UnwritableCase(RESTYPE_NUM + 1, (0, 0), InvalidAAtypesError)
    too_many_chains = UnwritableCase(0, (0, PDB_MAX_CHAINS), ValueError)


@pytest.fixture
def atom_tensors(
    atom_scenario: (
        Atom37ToAtom5Scenario | Atom5ToAtom37Scenario | Atom37ToCbScenario
    ),
) -> AtomTensors:
    """Build positions and mask for the scenario's geometry and layout.

    Each atom is placed through ``TensorSpec.slot``, so atom5 inputs use
    the atom5 order (N, CA, C, O, CB), which differs from atom37's.

    Args:
        atom_scenario: Geometry, batch size, residue count and layout; the
            requesting test parametrizes it.

    Returns:
        The built tensors.
    """
    spec: TensorSpec = atom_scenario.value
    n_atoms = spec.representation.value
    standard = {
        NamedAtom.n: (1.0, 0.0, 0.0),
        NamedAtom.ca: (0.0, 0.0, 0.0),
        NamedAtom.c: (0.0, 1.0, 0.0),
        NamedAtom.o: (1.0, 1.0, 0.0),
    }
    geometry = {
        TensorCase.standard: standard | {NamedAtom.cb: (0.0, 0.0, 1.0)},
        TensorCase.no_beta_carbon: standard,
        TensorCase.collinear: {
            NamedAtom.n: (0.0, 0.0, 0.0),
            NamedAtom.ca: (1.0, 0.0, 0.0),
            NamedAtom.c: (2.0, 0.0, 0.0),
            NamedAtom.o: (2.0, 1.0, 0.0),
        },
    }.get(spec.content, {})
    per_residue = torch.zeros(n_atoms, 3)
    for atom, xyz in geometry.items():
        per_residue[spec.slot(atom)] = torch.tensor(xyz)
    positions = einops.repeat(
        per_residue,
        "a d -> b n a d",
        b=spec.batch,
        n=spec.n_res,
    )
    mask = torch.ones(spec.batch, spec.n_res, n_atoms)
    if spec.content is TensorCase.half_mask:
        mask[:, : spec.n_res // 2] = 0
    if spec.content in {TensorCase.no_beta_carbon, TensorCase.collinear}:
        mask[..., spec.slot(NamedAtom.cb)] = 0
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
def coords_dict() -> dict[str, list[list[float]]]:
    """Return backbone coordinates for 6 residues, as JSONL entries hold them.

    Atom ``k`` of residue ``r`` (N, CA, C, O for ``k`` = 0..3) has
    coordinates ``(r + k, r + k + 1, r + k + 2)``. A mapping because
    ``make_np_example`` takes one, keyed by atom name.

    Returns:
        Mapping from atom name to a ``(6, 3)`` nested list.
    """
    return {
        name: [[float(r + k + d) for d in range(3)] for r in range(6)]
        for k, name in enumerate(("N", "CA", "C", "O"))
    }


@pytest.fixture
def np_example_to_center(
    center_scenario: CenterPositionsScenario,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Build the input mapping for a ``center_positions`` scenario.

    A mapping because ``center_positions`` writes its result back to the
    ``atom_positions`` key.

    Args:
        center_scenario: Residue count and present atoms; the requesting test
            parametrizes it.

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
    np_example_to_center: dict[
        str,
        npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ],
) -> npt.NDArray[np.float64]:
    """Shift the input by the scenario's CA centre and zero absent atoms.

    Args:
        center_scenario: Supplies the hand-derived CA centre; the
            requesting test parametrizes it.
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
def resize_input(
    resize_scenario: FixedSizeScenario | TruncateScenario,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Return the scenario's input mapping.

    A mapping because ``make_fixed_size`` and ``truncate_to_length`` take
    one and resize every array in it in place.

    Args:
        resize_scenario: Residue count before resizing; the requesting test
            parametrizes it.

    Returns:
        A fresh ``n_res``-residue mapping, which the test resizes in place.
    """
    return feature_example(resize_scenario.value.n_res)


@pytest.fixture
def resize_reference(
    resize_scenario: FixedSizeScenario | TruncateScenario,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Return the mapping of just the residues the resize must keep.

    A mapping so it can be compared key by key with ``resize_input``.

    Args:
        resize_scenario: Number of residues kept; the requesting test
            parametrizes it.

    Returns:
        A ``kept``-residue mapping.
    """
    return feature_example(resize_scenario.value.kept)


@pytest.fixture
def to_pdb_protein(to_pdb_scenario: ToPdbScenario) -> Protein:
    """Build the alanine protein a ``to_pdb`` scenario describes.

    Args:
        to_pdb_scenario: Present atoms and chains; the requesting test
            parametrizes it.

    Returns:
        The protein.
    """
    case = to_pdb_scenario.value
    return alanine_protein(case.present_atoms, case.chains)


@pytest.fixture
def unwritable_protein(unwritable_scenario: UnwritableScenario) -> Protein:
    """Build a protein with the scenario's residue type and chains.

    Args:
        unwritable_scenario: Residue type and chains to write; the
            requesting test parametrizes it.

    Returns:
        A protein ``to_pdb`` must refuse.
    """
    case = unwritable_scenario.value
    return dataclasses.replace(
        alanine_protein((1,), case.chains),
        aatype=np.full(len(case.chains), case.aatype, dtype=np.intp),
    )


@pytest.fixture
def scenario_pdb(
    pdb_scenario: PdbScenario,
    tmp_path: pathlib.Path,
) -> pathlib.Path:
    """Write the scenario's records to a PDB file.

    Args:
        pdb_scenario: Records to write; the requesting test
            parametrizes it.
        tmp_path: Per-test directory.

    Returns:
        The file path.
    """
    return write_pdb(tmp_path / "model.pdb", pdb_scenario.value.records)


@pytest.fixture
def hetatm_pdb(hetatm_record: str, tmp_path: pathlib.Path) -> pathlib.Path:
    """Write two ATOM residues followed by ``hetatm_record``.

    Args:
        hetatm_record: The HETATM record, in ``write_pdb``'s format; the
            requesting test parametrizes it.
        tmp_path: Per-test directory.

    Returns:
        The file path.
    """
    records = f"N ALA A1;N GLY A2;{hetatm_record}"
    return write_pdb(tmp_path / "model.pdb", records)


@pytest.fixture
def gapped_pdb(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write two residues numbered 5 and 10, leaving a gap.

    Args:
        tmp_path: Per-test directory.

    Returns:
        The file path.
    """
    return write_pdb(tmp_path / "model.pdb", "N ALA A5;N ALA A10")


@pytest.fixture
def atom5_mask_argument(
    atom_scenario: Atom5ToAtom37Scenario,
    atom_tensors: AtomTensors,
) -> Float[torch.Tensor, "B N_res 5"] | None:
    """Return the mask to pass to ``atom5_to_atom37``, or None to omit it.

    Args:
        atom_scenario: Whether to pass the mask; the requesting test
            parametrizes it.
        atom_tensors: Atom5 positions and mask.

    Returns:
        The built mask, or None to exercise the all-present default.
    """
    return atom_tensors.mask if atom_scenario.value.provide_mask else None


@pytest.fixture
def expected_backbone_mask(
    atom_scenario: Atom5ToAtom37Scenario,
    atom_tensors: AtomTensors,
) -> Float[torch.Tensor, "B N_res 5"]:
    """Return the mask the atom37 backbone slots must hold after scattering.

    Args:
        atom_scenario: Whether the mask is passed; the requesting test
            parametrizes it.
        atom_tensors: Atom5 positions and mask.

    Returns:
        The passed mask, or all ones when it is omitted.
    """
    mask = atom_tensors.mask
    return mask if atom_scenario.value.provide_mask else torch.ones_like(mask)


@pytest.fixture
def expected_cb_present(
    atom_scenario: Atom37ToCbScenario,
) -> Bool[torch.Tensor, "B N_res"]:
    """Mark CB present from the scenario's ``present_from`` residue on.

    Args:
        atom_scenario: Batch, residue count and first residue with CB; the
            requesting test parametrizes it.

    Returns:
        The expected ``present`` flags.
    """
    case = atom_scenario.value
    expected = torch.zeros(case.batch, case.n_res, dtype=torch.bool)
    expected[:, case.present_from :] = True
    return expected


@pytest.fixture
def wrong_shape_positions(
    wrong_atom_shape: WrongAtomShape,
) -> Float[torch.Tensor, "..."]:
    """Build zero positions of the scenario's invalid shape.

    Args:
        wrong_atom_shape: Supplies the shape; the requesting test
            parametrizes it.

    Returns:
        Positions the conversion must reject.
    """
    return torch.zeros(wrong_atom_shape.value.positions_shape)


@pytest.fixture
def wrong_shape_mask(
    wrong_atom_shape: WrongAtomShape,
) -> Float[torch.Tensor, "..."]:
    """Build an all-present mask of the scenario's valid mask shape.

    Args:
        wrong_atom_shape: Supplies the shape; the requesting test
            parametrizes it.

    Returns:
        A mask the conversion accepts.
    """
    return torch.ones(wrong_atom_shape.value.mask_shape)


@pytest.fixture
def expected_backbone_positions(
    coords_dict: dict[str, list[list[float]]],
) -> Float[torch.Tensor, "6 4 3"]:
    """Stack ``coords_dict`` into ``(residue, backbone atom, xyz)`` order.

    Args:
        coords_dict: Backbone coordinates keyed by atom name.

    Returns:
        N, CA, C and O coordinates for every residue.
    """
    stacked = torch.tensor([coords_dict[a] for a in ("N", "CA", "C", "O")])
    return einops.rearrange(stacked, "a r d -> r a d")


@pytest.fixture
def coords_with_nan(
    coords_dict: dict[str, list[list[float]]],
    atom: str,
) -> dict[str, list[list[float]]]:
    """Return ``coords_dict`` with residue 0's ``atom`` set to NaN.

    A mapping because ``make_np_example`` takes one, keyed by atom name.

    Args:
        coords_dict: Backbone coordinates keyed by atom name.
        atom: Atom whose first coordinate becomes NaN; the requesting test
            parametrizes it.

    Returns:
        A copy with that one coordinate replaced.
    """
    coords = {
        name: [list(xyz) for xyz in rows] for name, rows in coords_dict.items()
    }
    coords[atom][0] = [float("nan")] * 3
    return coords


@pytest.fixture
def expected_mol_types(
    pdb_scenario: PdbScenario,
) -> Float[torch.Tensor, " N_res"]:
    """Map the scenario's per-residue molecule letters to their codes.

    Args:
        pdb_scenario: Molecule letter per residue; the requesting test
            parametrizes it.

    Returns:
        ``MOL_TYPE_PROTEIN``, ``MOL_TYPE_DNA`` or ``MOL_TYPE_RNA`` for each
        residue, as stored in ``b_factors``.
    """
    codes = {"p": MOL_TYPE_PROTEIN, "d": MOL_TYPE_DNA, "r": MOL_TYPE_RNA}
    types = [codes[c] for c in pdb_scenario.value.mol_types]
    return torch.tensor(types, dtype=torch.float64)


@pytest.fixture
def nucleotide_residues(
    expected_mol_types: Float[torch.Tensor, " N_res"],
) -> Bool[torch.Tensor, " N_res"]:
    """Flag the residues whose molecule type is DNA or RNA.

    Args:
        expected_mol_types: Molecule code per residue.

    Returns:
        True where the residue is not protein.
    """
    return expected_mol_types != MOL_TYPE_PROTEIN


@pytest.fixture
def atomless_pdb(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a PDB file with header records but no ATOM records.

    Args:
        tmp_path: Per-test directory.

    Returns:
        The file path.
    """
    path = tmp_path / "empty.pdb"
    _ = path.write_text("REMARK empty structure\nHEADER test\n")
    return path


@pytest.mark.parametrize(
    "atom_scenario",
    list(Atom37ToAtom5Scenario),
    ids=[case.name for case in Atom37ToAtom5Scenario],
)
def test_atom37_to_atom5_copies_the_five_backbone_slots(
    atom_scenario: Atom37ToAtom5Scenario,
    atom_tensors: AtomTensors,
) -> None:
    """Each atom5 slot holds its atom37 atom's position and mask, same dtype.

    A CB absent in atom37 stays absent in atom5.

    Args:
        atom_scenario: Batch and residue counts.
        atom_tensors: Atom37 positions and mask.
    """
    atom_spec = atom_scenario.value
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
    "atom_scenario",
    list(Atom5ToAtom37Scenario),
    ids=[case.name for case in Atom5ToAtom37Scenario],
)
def test_atom5_to_atom37_scatters_into_backbone_slots_only(
    atom_scenario: Atom5ToAtom37Scenario,
    atom_tensors: AtomTensors,
    atom5_mask_argument: Float[torch.Tensor, "B N_res 5"] | None,
    expected_backbone_mask: Float[torch.Tensor, "B N_res 5"],
) -> None:
    """Each atom5 slot lands at its atom37 index; every other slot is zero.

    With no mask passed, the backbone slots are marked present.

    Args:
        atom_scenario: Sizes and whether to pass the mask.
        atom_tensors: Atom5 positions and mask.
        atom5_mask_argument: Mask to pass, or None.
        expected_backbone_mask: Mask the backbone slots must hold.
    """
    atom_spec = atom_scenario.value
    x = atom_tensors
    positions, mask = atom5_to_atom37(x.positions, atom5_mask_argument)
    other = sorted(set(range(37)) - set(ATOM5_TO_ATOM37))
    n_atoms = ResidueRepresentation.atom37.value
    shape = (atom_spec.batch, atom_spec.n_res, n_atoms)
    assert (positions.shape, mask.shape) == ((*shape, 3), shape)
    assert (positions.dtype, mask.dtype) == (x.positions.dtype, x.mask.dtype)
    assert torch.equal(positions[:, :, ATOM5_TO_ATOM37], x.positions)
    assert torch.equal(mask[:, :, ATOM5_TO_ATOM37], expected_backbone_mask)
    assert torch.count_nonzero(positions[:, :, other]) == 0
    assert torch.count_nonzero(mask[:, :, other]) == 0


@pytest.mark.parametrize(
    "atom_scenario",
    list(Atom37ToCbScenario),
    ids=[case.name for case in Atom37ToCbScenario],
)
def test_atom37_to_cb_uses_cb_when_present_and_a_pseudo_cb_otherwise(
    atom_scenario: Atom37ToCbScenario,
    atom_tensors: AtomTensors,
    expected_cb_present: Bool[torch.Tensor, "B N_res"],
) -> None:
    """Present CBs are copied; absent ones become a finite pseudo-Cβ.

    Where the backbone is non-degenerate, the pseudo-Cβ is a new point,
    neither the empty CB slot nor CA. Collinear N, CA, C make the cross
    product vanish; the pseudo-Cβ must still be finite.

    Args:
        atom_scenario: Where CB is present and whether the fallback is distinct.
        atom_tensors: Atom37 positions and mask.
        expected_cb_present: Residues that must report CB present.
    """
    atom_spec = atom_scenario.value
    x = atom_tensors
    cb, present = atom37_to_cb(x.positions, x.mask)
    absent = ~expected_cb_present
    stored_cb = x.positions[:, :, ATOM37_CB]
    ca = x.positions[:, :, ATOM37_CA]
    moved_off_cb = not torch.allclose(cb[absent], stored_cb[absent])
    moved_off_ca = not torch.allclose(cb[absent], ca[absent])
    assert cb.shape == (atom_spec.batch, atom_spec.n_res, 3)
    assert torch.equal(present, expected_cb_present)
    assert torch.equal(cb[expected_cb_present], stored_cb[expected_cb_present])
    assert torch.isfinite(cb[absent]).all()
    assert (moved_off_cb, moved_off_ca) == (atom_spec.distinct,) * 2


@pytest.mark.parametrize(
    "wrong_atom_shape",
    list(WrongAtomShape),
    ids=[case.name for case in WrongAtomShape],
)
def test_atom_conversions_reject_wrong_shapes(
    wrong_atom_shape: WrongAtomShape,
    wrong_shape_positions: Float[torch.Tensor, "..."],
    wrong_shape_mask: Float[torch.Tensor, "..."],
) -> None:
    """Each conversion's contract rejects positions of the wrong shape.

    Args:
        wrong_atom_shape: Conversion and shapes.
        wrong_shape_positions: Positions of the invalid shape.
        wrong_shape_mask: Mask of the valid shape.
    """
    with pytest.raises(TypeCheckError):
        _ = wrong_atom_shape.value.convert(
            wrong_shape_positions,
            wrong_shape_mask,
        )


@pytest.mark.parametrize(
    "invalid_protein_shape",
    list(InvalidProteinShape),
    ids=[case.name for case in InvalidProteinShape],
)
def test_protein_rejects_invalid_shapes(
    valid_protein: Protein,
    invalid_protein_shape: InvalidProteinShape,
) -> None:
    """Rebuilding a valid Protein with one malformed field raises.

    Every other field stays valid, so the error comes from the replaced
    one.

    Args:
        valid_protein: Protein whose other fields stay valid.
        invalid_protein_shape: Field and malformation.
    """
    protein_break = invalid_protein_shape.value
    old = cast(
        npt.NDArray[np.float64] | npt.NDArray[np.intp],
        getattr(valid_protein, protein_break.field),
    )
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
    expected_backbone_positions: Float[torch.Tensor, "6 4 3"],
) -> None:
    """Backbone coordinates land in their atom37 slots, marked present.

    All other slots are zero and absent, and ``residue_index`` counts from
    0. Not parametrized: the NaN case is the next test.

    Args:
        coords_dict: Backbone coordinates for 6 residues.
        expected_backbone_positions: Those coordinates in atom order.
    """
    example = {
        k: torch.tensor(v) for k, v in make_np_example(coords_dict).items()
    }
    backbone = [atom_types.index(name) for name in ("N", "CA", "C", "O")]
    other = sorted(set(range(37)) - set(backbone))
    positions = example["atom_positions"]
    assert torch.equal(positions[:, backbone], expected_backbone_positions)
    assert torch.count_nonzero(positions[:, other]) == 0
    assert bool(example["atom_mask"][:, backbone].all())
    assert torch.count_nonzero(example["atom_mask"][:, other]) == 0
    assert torch.equal(example["residue_index"], torch.arange(6))


@pytest.mark.parametrize("atom", ["N", "CA", "O"])
def test_make_np_example_masks_and_zeroes_nan_coordinates(
    coords_with_nan: dict[str, list[list[float]]],
    atom: str,
) -> None:
    """A NaN coordinate is set to zero and its atom marked absent.

    Other atoms of the same residue stay present.

    Args:
        coords_with_nan: Coordinates with one NaN atom.
        atom: The atom set to NaN.
    """
    example = {
        k: torch.tensor(v) for k, v in make_np_example(coords_with_nan).items()
    }
    slot = atom_types.index(atom)
    assert bool(torch.isfinite(example["atom_positions"]).all())
    assert torch.count_nonzero(example["atom_positions"][0, slot]) == 0
    assert example["atom_mask"][0, slot].item() == 0
    assert example["atom_mask"][0, atom_types.index("C")].item() == 1


@pytest.mark.parametrize(
    "resize_scenario",
    list(FixedSizeScenario),
    ids=[case.name for case in FixedSizeScenario],
)
def test_make_fixed_size_pads_or_truncates_to_ten(
    resize_scenario: FixedSizeScenario,
    resize_input: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
    resize_reference: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """Every array ends with 10 residues: the first kept, the rest zero.

    Args:
        resize_scenario: Residues before resizing and residues kept.
        resize_input: Mapping to resize; residue ``r`` holds ``r + 1``.
        resize_reference: Mapping of only the kept residues.
    """
    kept = resize_scenario.value.kept
    make_fixed_size(resize_input, max_seq_length=10)
    out = {k: torch.tensor(v) for k, v in resize_input.items()}
    reference = {k: torch.tensor(v) for k, v in resize_reference.items()}
    assert {k: len(v) for k, v in out.items()} == dict.fromkeys(out, 10)
    assert all(torch.equal(out[k][:kept], reference[k]) for k in out)
    assert all(torch.count_nonzero(out[k][kept:]) == 0 for k in out)


@pytest.mark.parametrize(
    "resize_scenario",
    list(TruncateScenario),
    ids=[case.name for case in TruncateScenario],
)
def test_truncate_to_length_cuts_but_never_pads(
    resize_input: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
    resize_reference: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """Arrays longer than 50 residues are cut to 50; shorter ones are kept.

    Args:
        resize_input: Mapping to truncate.
        resize_reference: Mapping of only the residues that must remain.
    """
    truncate_to_length(resize_input, 50)
    out = {k: torch.tensor(v) for k, v in resize_input.items()}
    reference = {k: torch.tensor(v) for k, v in resize_reference.items()}
    assert all(torch.equal(out[k], reference[k]) for k in out)


@pytest.mark.parametrize(
    "center_scenario",
    list(CenterPositionsScenario),
    ids=[case.name for case in CenterPositionsScenario],
)
def test_center_positions(
    np_example_to_center: dict[
        str,
        npt.NDArray[np.float64] | npt.NDArray[np.intp],
    ],
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
    "ter_scenario",
    list(TerScenario),
    ids=[case.name for case in TerScenario],
)
def test_chain_end_formats_a_ter_record(ter_scenario: TerScenario) -> None:
    """``chain_end`` lays out serial, residue, chain and number in columns.

    Args:
        ter_scenario: Arguments and the exact TER line.
    """
    case = ter_scenario.value
    line = chain_end(case.serial, case.resname, case.chain, case.resseq)
    assert line == case.expected


@pytest.mark.parametrize(
    "to_pdb_scenario",
    list(ToPdbScenario),
    ids=[case.name for case in ToPdbScenario],
)
def test_to_pdb_writes_one_atom_line_per_present_atom(
    to_pdb_scenario: ToPdbScenario,
    to_pdb_protein: Protein,
) -> None:
    """The PDB has a MODEL block, an ATOM per present atom, a TER per chain.

    Every line is padded to 80 columns, and absent atoms are skipped.

    Args:
        to_pdb_scenario: Protein layout and expected record counts.
        to_pdb_protein: Protein with that layout.
    """
    case = to_pdb_scenario.value
    lines = to_pdb(to_pdb_protein).splitlines()
    records = [line.split()[0] for line in lines if line.strip()]
    assert {len(line) for line in lines} == {80}
    assert records.count(PDB_REC_ATOM) == case.atom_lines
    assert records.count(PDB_REC_TER) == case.ter_lines
    assert records[0] == PDB_REC_MODEL
    assert records[-2:] == [PDB_REC_ENDMDL, PDB_REC_END]


@pytest.mark.parametrize(
    "unwritable_scenario",
    list(UnwritableScenario),
    ids=[case.name for case in UnwritableScenario],
)
def test_to_pdb_rejects_unwritable_proteins(
    unwritable_scenario: UnwritableScenario,
    unwritable_protein: Protein,
) -> None:
    """A residue type past the vocabulary, or too many chains, raises.

    Args:
        unwritable_scenario: Supplies the exception expected.
        unwritable_protein: Protein the PDB format cannot hold.
    """
    with pytest.raises(unwritable_scenario.value.error):
        _ = to_pdb(unwritable_protein)


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
            torch.tensor(
                cast(
                    npt.NDArray[np.float64] | npt.NDArray[np.intp],
                    getattr(recovered, name),
                ),
            ),
            torch.tensor(
                cast(
                    npt.NDArray[np.float64] | npt.NDArray[np.intp],
                    getattr(roundtrip_protein, name),
                ),
            ),
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
)
def test_protein_from_pdb_classifies_molecules_and_chains(
    pdb_scenario: PdbScenario,
    scenario_pdb: pathlib.Path,
    expected_mol_types: Float[torch.Tensor, " N_res"],
    nucleotide_residues: Bool[torch.Tensor, " N_res"],
) -> None:
    """Each residue's molecule type goes to ``b_factors``; chains are indexed.

    Nucleotides get the unknown residue type, ``RESTYPE_NUM_NO_X``.

    Args:
        pdb_scenario: Records and expected classification.
        scenario_pdb: The scenario's PDB file.
        expected_mol_types: Molecule code per residue.
        nucleotide_residues: True for DNA and RNA residues.
    """
    case = pdb_scenario.value
    protein = protein_from_pdb(scenario_pdb)
    b_factors = torch.tensor(protein.b_factors[:, 0])
    aatype = torch.tensor(protein.aatype)
    assert protein.atom_positions.shape == (len(case.mol_types), 37, 3)
    assert torch.equal(b_factors, expected_mol_types)
    assert torch.equal(
        torch.tensor(protein.chain_index),
        torch.tensor(case.chains),
    )
    assert bool((aatype[nucleotide_residues] == RESTYPE_NUM_NO_X).all())


@pytest.mark.parametrize(
    "hetatm_record",
    ["HETATM O HOH A100", "HETATM MG MG A200", "HETATM N1 ATP A300"],
    ids=["water", "ion", "ligand"],
)
def test_protein_from_pdb_skips_hetatm_records(
    hetatm_pdb: pathlib.Path,
) -> None:
    """HETATM residues (water, ions, ligands) are left out of the protein.

    Args:
        hetatm_pdb: Two ATOM residues, then one HETATM record.
    """
    protein = protein_from_pdb(hetatm_pdb)
    assert protein.atom_positions.shape[0] == len(["ALA", "GLY"])


def test_protein_from_pdb_keeps_pdb_residue_numbers(
    gapped_pdb: pathlib.Path,
) -> None:
    """``residue_index`` holds the RESSEQ column, gaps included.

    Not parametrized: one gap shows the numbers are copied, not counted.

    Args:
        gapped_pdb: Residues numbered 5 and 10.
    """
    protein = protein_from_pdb(gapped_pdb)
    assert torch.equal(
        torch.tensor(protein.residue_index),
        torch.tensor([5, 10]),
    )


def test_protein_from_pdb_rejects_a_file_without_atoms(
    atomless_pdb: pathlib.Path,
) -> None:
    """A PDB with only REMARK and HEADER lines raises ``ValueError``.

    Not parametrized: any file without ATOM records takes this path.

    Args:
        atomless_pdb: PDB file with no ATOM records.
    """
    with pytest.raises(ValueError, match="No ATOM records"):
        _ = protein_from_pdb(atomless_pdb)


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


@pytest.fixture
def dataset_residue_count() -> int:
    """Number of identical residues in the dataset-shaped entry.

    Three residues is the smallest entry with an interior residue; every
    residue is the same ideal alanine, so the count only checks that each
    residue is handled, not any residue-to-residue geometry.

    Returns:
        The residue count.
    """
    return 3


@pytest.fixture
def dataset_atom37(
    ideal_alanine_atom5: Float[torch.Tensor, "5 3"],
    dataset_residue_count: int,
) -> AtomTensors:
    """Run a dataset-shaped entry of ideal alanines through make_np_example.

    The entry carries coordinates for every atom whose ``in_dataset`` flag
    is set (N, CA, C and O) and none for CB, like the training data.

    Args:
        ideal_alanine_atom5: Ideal alanine positions in atom5 order.
        dataset_residue_count: Number of residues in the entry.

    Returns:
        Batched atom37 positions and mask, batch size 1.
    """
    coords = {
        atom.value.atom_name: [
            [
                float(axis)
                for axis in ideal_alanine_atom5[atom.value.atom5_index]
            ],
        ]
        * dataset_residue_count
        for atom in NamedAtom
        if atom.value.in_dataset
    }
    example = make_np_example(coords)
    positions = torch.as_tensor(example["atom_positions"], dtype=torch.float64)
    mask = torch.as_tensor(example["atom_mask"], dtype=torch.float64)
    return AtomTensors(
        positions=einops.rearrange(positions, "n a d -> 1 n a d"),
        mask=einops.rearrange(mask, "n a -> 1 n a"),
    )


@pytest.fixture
def expected_atom5_slot(
    named_atom: NamedAtom,
    ideal_alanine_atom5: Float[torch.Tensor, "5 3"],
    dataset_residue_count: int,
) -> SlotContents:
    """What the named atom's atom5 slot must hold after the dataset round trip.

    Atoms present in the dataset keep their ideal position with mask 1; CB,
    which the dataset never carries, stays at the origin with mask 0.

    Args:
        named_atom: The atom whose slot is checked; the requesting test
            parametrizes it.
        ideal_alanine_atom5: Ideal alanine positions in atom5 order.
        dataset_residue_count: Number of residues in the entry.

    Returns:
        The expected per-residue positions and mask of that slot.
    """
    present = float(named_atom.value.in_dataset)
    position = ideal_alanine_atom5[named_atom.value.atom5_index] * present
    return SlotContents(
        positions=einops.repeat(
            position.to(torch.float64),
            "d -> n d",
            n=dataset_residue_count,
        ),
        mask=torch.full(
            (dataset_residue_count,),
            present,
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize(
    "named_atom",
    list(NamedAtom),
    ids=[atom.name for atom in NamedAtom],
)
def test_atom_index_constants_match_atom_names(named_atom: NamedAtom) -> None:
    """Index constants agree with the ordered name lists they index into.

    Looks each atom up by name in ``atom_types`` and ``ATOM5_NAMES`` rather
    than through other constants, so constants that agree with each other but
    not with the data order fail here.

    Args:
        named_atom: The atom whose constants are being checked.
    """
    slot = named_atom.value

    assert atom_types[slot.atom37_index] == slot.atom_name
    assert ATOM5_NAMES[slot.atom5_index] == slot.atom_name
    assert ATOM5_TO_ATOM37[slot.atom5_index] == slot.atom37_index


@pytest.mark.parametrize(
    "named_atom",
    list(NamedAtom),
    ids=[atom.name for atom in NamedAtom],
)
def test_make_np_example_atoms_reach_named_atom5_slots(
    named_atom: NamedAtom,
    dataset_atom37: AtomTensors,
    expected_atom5_slot: SlotContents,
) -> None:
    """Dataset atoms land in the atom5 slot with their name, CB stays empty.

    The slot is found with ``ATOM5_NAMES.index`` rather than the index
    constant, so a constant pointing at the wrong slot cannot hide itself.

    Args:
        named_atom: The atom whose atom5 slot is being checked.
        dataset_atom37: The atom37 tensors built from a dataset-shaped entry.
        expected_atom5_slot: What that slot must hold for each residue.
    """
    atom5_positions, atom5_mask = atom37_to_atom5(
        dataset_atom37.positions,
        dataset_atom37.mask,
    )

    slot = ATOM5_NAMES.index(named_atom.value.atom_name)
    assert torch.equal(
        atom5_positions[0, :, slot],
        expected_atom5_slot.positions,
    )
    assert torch.equal(atom5_mask[0, :, slot], expected_atom5_slot.mask)


def test_atom37_to_cb_places_ideal_cb_for_dataset_entries(
    ideal_alanine_cb: Float[torch.Tensor, "3"],
    dataset_atom37: AtomTensors,
) -> None:
    """Dataset entries get a pseudo-Cβ at the ideal alanine Cβ position.

    Dataset entries have no CB, so this path produces every residue's Cβ in
    training. It is not parametrized because one ideal residue pins the
    geometry: the expected point comes from the residue's known CB, and being
    near it rules out returning the O atom instead.

    Args:
        ideal_alanine_cb: The ideal alanine's known Cβ.
        dataset_atom37: The atom37 tensors built from a dataset-shaped entry.
    """
    cb, cb_present = atom37_to_cb(dataset_atom37.positions, dataset_atom37.mask)

    assert not cb_present.any()
    assert torch.allclose(cb, ideal_alanine_cb.to(torch.float64), atol=0.01)
