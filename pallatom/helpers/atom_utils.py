"""Utility dataclasses and functions for atom-level coordinate data."""

import dataclasses
from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
from beartype import beartype
from einops import rearrange
from jaxtyping import Bool, Float, Int, jaxtyped

# Data utils

# atom5 slot order: N=0, CA=1, C=2, O=3, CB=4
ATOM5_NAMES = ["N", "CA", "C", "O", "CB"]

# Element encoding: N=0, C=1, O=2, UNK=3 (side-chain placeholder)
# Backbone slots: N→[1,0,0,0], CA→[0,1,0,0], C→[0,1,0,0], O→[0,0,1,0], CB→[0,0,0,1]
ATOM5_ELEMENTS = torch.tensor(
    [
        [1, 0, 0, 0],  # N
        [0, 1, 0, 0],  # CA (Carbon)
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0],  # O
        [0, 0, 0, 1],  # CB → UNK class
    ],
    dtype=torch.long,
)  # (5, 4)

# ── atom37 index constants (standard residue_constants ordering) ──────────────
# Matches OpenFold / AlphaFold2 residue_constants.atom_order
ATOM37_N = 0
ATOM37_CA = 1
ATOM37_C = 2
ATOM37_CB = 4
ATOM37_O = 3

# atom5 slot indices
ATOM5_N = 0
ATOM5_CA = 1
ATOM5_C = 2
ATOM5_CB = 4
ATOM5_O = 3


atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",  # mask token: unknown / conditioning-dropout placeholder
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 21.

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Molecule type encoding stored in Protein.b_factors (broadcast to all atom slots per residue).
# 0.0 = amino-acid residue, 1.0 = DNA nucleotide, 2.0 = RNA nucleotide.
MOL_TYPE_PROTEIN: Final[int] = 0
MOL_TYPE_DNA: Final[int] = 1
MOL_TYPE_RNA: Final[int] = 2

# DNA monomers — PDB ATOM residue names for deoxyribonucleotides.
# Mirrors OpenFold's restype_3to1 / restype_order naming convention.
DNA_RESTYPES: Final[tuple[str, ...]] = ("DA", "DC", "DG", "DT")
DNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]] = MappingProxyType(
    {restype: i for i, restype in enumerate(DNA_RESTYPES)}
)
DNA_RESTYPE_3TO1: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"DA": "a", "DC": "c", "DG": "g", "DT": "t"}
)

# RNA monomers — PDB ATOM residue names for ribonucleotides.
RNA_RESTYPES: Final[tuple[str, ...]] = ("A", "C", "G", "U")
RNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]] = MappingProxyType(
    {restype: i for i, restype in enumerate(RNA_RESTYPES)}
)
RNA_RESTYPE_3TO1: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"A": "a", "C": "c", "G": "g", "U": "u"}
)


# Format: (atom_name, rigid_group_idx, (x, y, z))
# rigid_group_idx: 0=backbone, 1=omega, 2=phi, 3=psi, 4=chi1, 5=chi2, 6=chi3, 7=chi4
# Taken directly from AlphaFold2 / OpenFold residue_constants.py
rigid_group_atom_positions = {
    "ALA": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
    ],
    "ARG": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, -0.000)),
        ("CD", 5, (0.564, 1.414, 0.000)),
        ("NE", 6, (0.539, 1.357, 0.000)),
        ("CZ", 7, (0.758, 1.093, 0.000)),
        ("NH1", 7, (0.206, 2.301, 0.000)),
        ("NH2", 7, (2.078, 0.978, -0.000)),
    ],
    "ASN": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, 0.000)),
        ("OD1", 5, (0.633, 1.068, 0.000)),
        ("ND2", 5, (0.553, -1.141, 0.000)),
    ],
    "ASP": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.593, 1.398, 0.000)),
        ("OD1", 5, (0.610, 1.091, 0.000)),
        ("OD2", 5, (0.592, -1.108, -0.000)),
    ],
    "CYS": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("SG", 4, (0.728, 1.653, 0.000)),
    ],
    "GLN": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, 0.000)),
        ("CD", 5, (0.587, 1.399, 0.000)),
        ("OE1", 6, (0.634, 1.060, 0.000)),
        ("NE2", 6, (0.560, -1.147, 0.000)),
    ],
    "GLU": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, 0.000)),
        ("CD", 5, (0.600, 1.397, 0.000)),
        ("OE1", 6, (0.607, 1.095, 0.000)),
        ("OE2", 6, (0.589, -1.104, -0.000)),
    ],
    "GLY": [
        ("N", 0, (-0.572, 1.337, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.517, 0.000, 0.000)),
        ("O", 3, (0.626, 1.062, 0.000)),
    ],
    "HIS": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.600, 1.370, 0.000)),
        ("ND1", 5, (0.557, 1.425, 0.000)),
        ("CD2", 5, (0.704, -1.200, 0.000)),
        ("CE1", 5, (2.056, 1.137, 0.000)),
        ("NE2", 5, (2.075, -0.245, 0.000)),
    ],
    "ILE": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.536, -0.793, -1.213)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG1", 4, (0.534, 1.437, 0.000)),
        ("CG2", 4, (0.540, -0.785, -1.199)),
        ("CD1", 5, (0.619, 1.391, 0.000)),
    ],
    "LEU": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, 0.000)),
        ("CD1", 5, (0.540, 1.432, 0.000)),
        ("CD2", 5, (0.537, -0.776, -1.203)),
    ],
    "LYS": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, 0.000)),
        ("CD", 5, (0.559, 1.417, 0.000)),
        ("CE", 6, (0.560, 1.416, 0.000)),
        ("NZ", 7, (0.554, 1.387, 0.000)),
    ],
    "MET": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.616, 1.390, 0.000)),
        ("SD", 5, (0.703, 1.695, 0.000)),
        ("CE", 6, (0.320, 1.786, 0.000)),
    ],
    "PHE": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.607, 1.377, 0.000)),
        ("CD1", 5, (0.709, 1.195, 0.000)),
        ("CD2", 5, (0.706, -1.196, 0.000)),
        ("CE1", 5, (2.102, 1.198, 0.000)),
        ("CE2", 5, (2.098, -1.201, 0.000)),
        ("CZ", 5, (2.794, 0.000, 0.000)),
    ],
    "PRO": [
        ("N", 0, (-0.566, 1.351, -0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.527, 0.000, 0.000)),
        ("CB", 0, (-0.546, -0.611, -1.293)),
        ("O", 3, (0.621, 1.066, 0.000)),
        ("CG", 4, (0.382, 1.445, 0.000)),
        ("CD", 5, (0.427, 1.440, 0.000)),
    ],
    "SER": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("OG", 4, (0.503, 1.325, 0.000)),
    ],
    "THR": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.516, -0.793, -1.215)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("OG1", 4, (0.504, 1.330, 0.000)),
        ("CG2", 4, (0.511, -0.801, -1.228)),
    ],
    "TRP": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.596, 1.374, 0.000)),
        ("CD1", 5, (0.673, 1.371, 0.000)),
        ("CD2", 5, (0.828, -1.069, 0.000)),
        ("NE1", 5, (2.028, 1.268, 0.000)),
        ("CE2", 5, (2.271, -0.590, 0.000)),
        ("CE3", 5, (0.994, -2.460, 0.000)),
        ("CZ2", 5, (3.015, -1.682, 0.000)),
        ("CZ3", 5, (2.377, -3.541, 0.000)),
        ("CH2", 5, (3.791, -3.050, 0.000)),
    ],
    "TYR": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.529, -0.774, -1.205)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG", 4, (0.607, 1.377, 0.000)),
        ("CD1", 5, (0.716, 1.195, 0.000)),
        ("CD2", 5, (0.713, -1.194, 0.000)),
        ("CE1", 5, (2.107, 1.200, 0.000)),
        ("CE2", 5, (2.104, -1.201, 0.000)),
        ("CZ", 5, (2.791, -0.001, 0.000)),
        ("OH", 5, (4.168, 0.002, 0.000)),
    ],
    "VAL": [
        ("N", 0, (-0.525, 1.363, 0.000)),
        ("CA", 0, (0.000, 0.000, 0.000)),
        ("C", 0, (1.526, 0.000, 0.000)),
        ("CB", 0, (-0.533, -0.795, -1.213)),
        ("O", 3, (0.627, 1.062, 0.000)),
        ("CG1", 4, (0.540, 1.429, 0.000)),
        ("CG2", 4, (0.533, -0.776, -1.203)),
    ],
}


def make_np_example(
    coords_dict: Mapping[str, npt.NDArray[np.float64]],
) -> Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]]:
    """Make a dictionary of non-batched numpy protein features."""
    bb_atom_types = ["N", "CA", "C", "O"]
    bb_idx = [i for i, atom_type in enumerate(atom_types) if atom_type in bb_atom_types]

    num_res = np.array(coords_dict["N"]).shape[0]
    atom_positions = np.zeros([num_res, 37, 3], dtype=float)

    for i, atom_type in enumerate(atom_types):
        if atom_type in bb_atom_types:
            atom_positions[:, i, :] = np.array(coords_dict[atom_type])

    # Mask nan / None coordinates.
    nan_pos = np.isnan(atom_positions)[..., 0]
    atom_positions[nan_pos] = 0.0
    atom_mask = np.zeros([num_res, 37])
    atom_mask[..., bb_idx] = 1
    atom_mask[nan_pos] = 0

    return {
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "residue_index": np.arange(num_res),
    }


def make_fixed_size(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
    max_seq_length: int = 500,
) -> None:
    """Pad features to fixed sequence length, i.e. currently axis=0."""
    for k, v in np_example.items():
        pad = max_seq_length - v.shape[0]
        if pad > 0:
            np_example[k] = np.pad(v, ((0, pad),) + ((0, 0),) * (len(v.shape) - 1))
        elif pad < 0:
            np_example[k] = v[:max_seq_length]


def truncate_to_length(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
    max_length: int,
) -> None:
    """Truncate features to at most max_length along axis 0. Does not pad.

    Args:
        np_example: Dict of numpy arrays all sharing the same axis-0 size.
        max_length: Maximum allowed length along axis 0.
    """
    for k, v in np_example.items():
        if v.shape[0] > max_length:
            np_example[k] = v[:max_length]


def center_positions(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
) -> None:
    """Center 'atom_positions' on CA center of mass."""
    atom_positions = np_example["atom_positions"]  # N, 37, 3
    atom_mask = np_example["atom_mask"]  # N, 37
    ca_positions = atom_positions[:, 1, :]  # N, 1, 3
    ca_mask = atom_mask[:, 1]  # N, 1

    ca_center = np.sum(ca_mask[..., None] * ca_positions, axis=0) / (
        np.sum(ca_mask, axis=0) + 1e-9
    )  # this is the masked mean of ca positions with a small epsilon. shape of 1, 1, 3
    atom_positions = (atom_positions - ca_center[None, ...]) * atom_mask[
        ..., None
    ]  # subtract the atom positions fromm this mean alpha carbon position.
    # N, 37, 3 - 1 expand to N, 1, 3  = N,37,3
    # / N, 37, expand to 3 -> N,37,3
    np_example["atom_positions"] = atom_positions


# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: Float[npt.NDArray[np.float64], "num_res num_atom_type 3"]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: Int[npt.NDArray[np.intp], "num_res"]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: Float[npt.NDArray[np.float64], "num_res num_atom_type"]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: Int[npt.NDArray[np.intp], "num_res"]

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: Int[npt.NDArray[np.intp], "num_res"]

    # typically this would be B-factors, or temperature factors, of each residue
    # (in sq. angstroms units), representing the displacement of the residue from its
    # ground truth mean value.
    b_factors: Float[npt.NDArray[np.float64], "num_res num_atom_type"]


def _classify_mol_type(resname: str, atom_names: frozenset[str]) -> int:
    """Return the molecule-type constant for a PDB residue.

    Uses residue name first; falls back to 2'-OH atom presence for the
    ambiguous single-letter nucleotides A/C/G, which appear in both
    D-prefix-less DNA and canonical RNA.

    Args:
        resname: Stripped residue name from the PDB ATOM record
            (e.g. "ALA", "DA", "A").
        atom_names: All atom names observed in this residue; used to
            detect the RNA 2'-hydroxyl oxygen "O2'".

    Returns:
        MOL_TYPE_PROTEIN (0), MOL_TYPE_DNA (1), or MOL_TYPE_RNA (2).
    """
    if resname in restype_3to1:
        return MOL_TYPE_PROTEIN
    if resname in DNA_RESTYPE_3TO1 or resname == "T":
        return MOL_TYPE_DNA
    if resname == "U":
        return MOL_TYPE_RNA
    if resname in RNA_RESTYPE_3TO1:  # A / C / G — ambiguous
        return MOL_TYPE_RNA if "O2'" in atom_names else MOL_TYPE_DNA
    return MOL_TYPE_PROTEIN  # unknown residue → treat as protein


def _parse_pdb_atoms(
    pdb_path: str,
) -> tuple[
    MutableMapping[tuple[str, int, str], MutableMapping[str, tuple[float, float, float, float]]],
    MutableMapping[tuple[str, int, str], str],
    MutableMapping[tuple[str, int, str], str],
]:
    """Read ATOM records from a PDB file and collect per-residue atom data.

    HETATM records are explicitly skipped. Alternate locations other than
    blank or "A" are also ignored.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        A 3-tuple of ``(residue_atoms, residue_name, residue_chain)``.
        ``residue_atoms`` maps ``(chain_id, resseq, icode)`` to a dict of
        ``atom_name → (x, y, z, b_factor)``.
        ``residue_name`` maps the same key to the residue name string.
        ``residue_chain`` maps the same key to the single-character chain ID.
    """
    residue_atoms: MutableMapping[
        tuple[str, int, str], MutableMapping[str, tuple[float, float, float, float]]
    ] = {}
    residue_name: MutableMapping[tuple[str, int, str], str] = {}
    residue_chain: MutableMapping[tuple[str, int, str], str] = {}

    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "HETATM":
                continue  # skip: water, ions, ligands, crystallographic reagents
            if rec != "ATOM":
                continue
            atom_name = line[12:16].strip()
            alt_loc = line[16]
            if alt_loc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            chain_id = line[21]
            resseq = int(line[22:26])
            icode = line[26]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfac = float(line[60:66]) if len(line) > 66 else 0.0
            key = (chain_id, resseq, icode)
            if key not in residue_atoms:
                residue_atoms[key] = {}
                residue_name[key] = resname
                residue_chain[key] = chain_id
            if atom_name not in residue_atoms[key]:
                residue_atoms[key][atom_name] = (x, y, z, bfac)

    return residue_atoms, residue_name, residue_chain


def protein_from_pdb(pdb_path: str) -> "Protein":
    """Parse a PDB file (ATOM records only) into a Protein using the atom37 layout.

    HETATM records (water, ions, ligands, crystallographic reagents) are
    explicitly detected and skipped; only ATOM records produce residues in
    the returned Protein. The b_factors field encodes molecule type rather
    than real PDB B-factors: 0.0 = protein, 1.0 = DNA, 2.0 = RNA.

    Args:
        pdb_path: Path to the PDB file to parse.

    Returns:
        A Protein with atom37-layout coordinates; b_factors encodes
        MOL_TYPE_PROTEIN / MOL_TYPE_DNA / MOL_TYPE_RNA per residue.

    Raises:
        ValueError: If the file contains no ATOM records.
    """
    _atom_type_idx = {name: i for i, name in enumerate(atom_types)}
    _seen_chains: dict[str, int] = {}

    _residue_atoms, _residue_name, _residue_chain = _parse_pdb_atoms(pdb_path)

    if not _residue_atoms:
        raise ValueError(f"No ATOM records found in {pdb_path}")

    _keys = sorted(_residue_atoms)
    N_res = len(_keys)
    _atom_positions = np.zeros((N_res, 37, 3), dtype=np.float64)
    _atom_mask = np.zeros((N_res, 37), dtype=np.float64)
    _b_factors = np.zeros((N_res, 37), dtype=np.float64)
    _aatype = np.zeros(N_res, dtype=np.intp)
    _residue_index = np.zeros(N_res, dtype=np.intp)
    _chain_index = np.zeros(N_res, dtype=np.intp)

    for _i, _key in enumerate(_keys):
        _cid = _residue_chain[_key]
        if _cid not in _seen_chains:
            _seen_chains[_cid] = len(_seen_chains)
        _chain_index[_i] = _seen_chains[_cid]
        _residue_index[_i] = _key[1]
        _one = restype_3to1.get(_residue_name[_key], "X")
        _aatype[_i] = restype_order.get(_one, 20)
        _mol = _classify_mol_type(
            _residue_name[_key],
            frozenset(_residue_atoms[_key].keys()),
        )
        _b_factors[_i, :] = float(_mol)
        for _aname, (_x, _y, _z, _) in _residue_atoms[_key].items():
            if _aname in _atom_type_idx:
                _idx = _atom_type_idx[_aname]
                _atom_positions[_i, _idx] = [_x, _y, _z]
                _atom_mask[_i, _idx] = 1.0

    return Protein(
        atom_positions=_atom_positions,
        aatype=_aatype,
        atom_mask=_atom_mask,
        residue_index=_residue_index,
        chain_index=_chain_index,
        b_factors=_b_factors,
    )


def _chain_end(atom_index: int, end_resname: str, chain_name: str, residue_index: int) -> str:
    chain_end = "TER"
    return f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}"


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Note: `Protein.b_factors` encodes molecule type (0=protein, 1=DNA, 2=RNA)
    rather than crystallographic B-factors. The PDB B-factor column will
    therefore contain 0.00, 1.00, or 2.00 for model-generated structures.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
        "X",
    ]

    def res_1to3(r: int) -> str:
        return restype_1to3.get(restypes[r], "UNK")

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append("MODEL     1")
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i], strict=False
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(
        _chain_end(atom_index, res_1to3(aatype[-1]), chain_ids[chain_index[-1]], residue_index[-1])
    )
    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


@jaxtyped(typechecker=beartype)
def atom37_to_atom5(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
) -> tuple[Float[torch.Tensor, "B N_res 5 3"], Float[torch.Tensor, "B N_res 5"]]:
    """Extract the 5 backbone+Cβ atoms from atom37 representation.

    Returns:
    -------
    atom5_positions : (B, N_res, 5, 3)
    atom5_mask      : (B, N_res, 5)
    """
    select = [ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O, ATOM37_CB]

    atom5_positions = atom37_positions[:, :, select, :]
    atom5_mask = atom37_mask[:, :, select]

    return atom5_positions, atom5_mask


@jaxtyped(typechecker=beartype)
def pseudo_cb(
    n: Float[torch.Tensor, "... 3"],
    ca: Float[torch.Tensor, "... 3"],
    c: Float[torch.Tensor, "... 3"],
) -> Float[torch.Tensor, "... 3"]:
    """Compute a virtual Cβ from backbone geometry (Gly-safe).

    Uses the standard ideal-geometry recipe:
      b = C alpha - N        (N→C alpha bond vector)
      d = C  - C alpha       (C alpha→C  bond vector)
      Cross them, then combine with ideal tetrahedral offsets.

    This matches the AlphaFold2 / ESMFold convention exactly.
    """
    b = ca - n
    d = c - ca
    bc = b / (torch.linalg.norm(b, dim=-1, keepdim=True) + 1e-8)
    dc = d / (torch.linalg.norm(d, dim=-1, keepdim=True) + 1e-8)
    n_vec = torch.cross(bc, dc, dim=-1)
    n_vec = n_vec / (torch.linalg.norm(n_vec, dim=-1, keepdim=True) + 1e-8)

    # Ideal Cβ placement constants (Hs or N→C→C alpha geometry)
    # values from Havel (1998) / AlphaFold2 supplementary
    m = -0.58273431
    s = 0.56802827
    c_ = 0.54463904

    return ca + m * n_vec + s * (bc + dc) + c_ * (bc - dc)


@jaxtyped(typechecker=beartype)
def get_cb_coords(
    atom5_positions: Float[torch.Tensor, "B N_res 5 3"],
    atom5_mask: Float[torch.Tensor, "B N_res 5"],
) -> tuple[Float[torch.Tensor, "B N_res 3"], Bool[torch.Tensor, "B N_res"]]:
    """Extract Cβ (slot 4) from atom5, replacing missing Cβ (Gly) with pseudo-Cβ.

    Parameters
    ----------
    atom5_positions : (B, N_res, 5, 3)
    atom5_mask      : (B, N_res, 5)  — 1 where atom is present
    fill_pseudo     : if True,

    Returns:
    -------
    cb               : (B, N_res, 3)  — real Cβ where available, pseudo-Cβ otherwise
    pseudo_beta_mask : (B, N_res)     — True where real Cβ present, False where pseudo-Cβ was used
    """
    n = atom5_positions[:, :, ATOM5_N, :]  # (B, N_res, 3)
    ca = atom5_positions[:, :, ATOM5_CA, :]
    c = atom5_positions[:, :, ATOM5_C, :]
    cb = atom5_positions[:, :, ATOM5_CB, :]  # zero / garbage where missing
    # compute pseudo-Cβ wherever Cβ is absent
    cb_present = atom5_mask[:, :, ATOM5_CB].bool()  # (B, N_res)
    p_cb = pseudo_cb(n, ca, c)  # (B, N_res, 3)
    cb = torch.where(rearrange(cb_present, "b n -> b n 1"), cb, p_cb)

    return cb, cb_present


# ── convenience wrapper ───────────────────────────────────────────────────────


@jaxtyped(typechecker=beartype)
def atom37_to_cb(
    atom37_positions: Float[torch.Tensor, "B N_res 37 3"],
    atom37_mask: Float[torch.Tensor, "B N_res 37"],
) -> tuple[Float[torch.Tensor, "B N_res 3"], Bool[torch.Tensor, "B N_res"]]:
    """Full pipeline: atom37 → atom5 → Cβ / pseudo-Cβ.

    Returns:
    -------
    cb               : (B, N_res, 3)  — real Cβ where available, pseudo-Cβ otherwise
    pseudo_beta_mask : (B, N_res)     — True where real Cβ present, False where pseudo-Cβ was used
    """
    atom5_pos, atom5_mask = atom37_to_atom5(atom37_positions, atom37_mask)
    return get_cb_coords(atom5_pos, atom5_mask)
