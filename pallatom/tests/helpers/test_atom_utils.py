"""Tests for atom utility dataclasses and functions.

Covers pseudo_cb, atom37_to_atom5, atom37_to_cb, get_cb_coords, the Protein
dataclass, make_np_example, make_fixed_size, center_positions, chain_end,
to_pdb, protein_from_pdb, truncate_to_length, and molecule-type constants.
"""

import enum
import pathlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

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
    ResidueRepresentation,
    atom37_to_atom5,
    atom37_to_cb,
    atom_types,
    center_positions,
    chain_end,
    classify_mol_type,
    make_fixed_size,
    make_np_example,
    protein_from_pdb,
    restype_1to3,
    to_pdb,
    truncate_to_length,
)
from helpers.errors import InvalidAAtypesError
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError

_ = manual_seed(42)

B = 2
N_RES = 10
PDB_ATOM_NAME_CUTOFF = 4
PDB_LINE_LEN = 80
PROT_LEN = 3
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TensorCase(enum.Enum):
    """Enumeration of named test scenarios for atom position and mask tensors.

    Each member selects a distinct structural configuration used to parametrize
    fixtures via ``indirect=True``. The string value doubles as human-readable
    label that appears in pytest output when a test fails.

    Members:
        standard: A well-formed backbone with valid positions and a fully
            populated mask — the happy-path baseline.
        all_zeros: All atom positions set to the origin; exercises behaviour
            when coordinates carry no geometric information.
        no_beta_carbon: The beta-carbon slot is absent (masked out) for every
            residue; validates that functions degrade gracefully without Cβ.
        half_mask: Exactly half the residues are unmasked; probes boundary
            handling and partial-visibility logic.
        collinear: All atoms placed on a single line; tests robustness against
            degenerate geometry where cross-products vanish.
    """

    standard = "standard"
    all_zeros = "all_zeros"
    no_beta_carbon = "no_beta_carbon"
    half_mask = "half_mask"
    collinear = "collinear"


class ExpectedBetaCarbon(enum.Enum):
    """Whether atom37_to_cb is expected to report Cβ as present.

    Members:
        present: All residues should have CB marked present in the output
            mask — the happy-path case where atom37 has the CB slot filled.
        absent: No residues should have CB present; the function must fall
            back to a finite pseudo-Cβ computed from backbone geometry.
        half_present: The first half of residues are fully masked (CB absent)
            and the second half are fully unmasked (CB present); validates
            that atom37_to_cb handles partial visibility correctly.
    """

    present = True
    absent = False
    half_present = "half_present"


@pytest.fixture
def atom_position_and_mask_factory(
    request: pytest.FixtureRequest,
) -> tuple[
    Float[torch.Tensor, "B N_res atom_rep 3"],
    Float[torch.Tensor, "B N_res atom_rep"],
]:
    """Construct atom position and mask tensors from indirect parametrize args.

    Receives a 4-tuple via ``request.param`` when used with
    ``indirect=["atom_position_and_mask_factory"]``.

    Args:
        request: Pytest fixture request whose ``.param`` attribute is a
            4-tuple of (residue_representation_type, tensor_content,
            batch_size, residue_number).

    Returns:
        Tuple of (atom_positions, atom_mask) with shapes
        ``(B, N_res, atom_rep, 3)`` and ``(B, N_res, atom_rep)``.
    """
    (
        residue_representation_type,
        tensor_content,
        batch_size,
        residue_number,
    ) = cast(
        tuple[ResidueRepresentation, TensorCase, int, int],
        request.param,
    )
    n_atoms_per_residue: int = residue_representation_type.value
    per_residue = torch.zeros(n_atoms_per_residue, 3)

    if tensor_content in (
        TensorCase.standard,
        TensorCase.no_beta_carbon,
    ):
        per_residue[ATOM37_N] = torch.tensor([1.0, 0.0, 0.0])
        per_residue[ATOM37_CA] = torch.tensor([0.0, 0.0, 0.0])
        per_residue[ATOM37_C] = torch.tensor([0.0, 1.0, 0.0])
        per_residue[ATOM37_O] = torch.tensor([1.0, 1.0, 0.0])
        if tensor_content == TensorCase.standard:
            per_residue[ATOM37_CB] = torch.tensor([0.0, 0.0, 1.0])
    elif tensor_content == TensorCase.collinear:
        # N, CA, C on the x-axis: cross(CA-N, C-CA) = [0,0,0]
        per_residue[ATOM37_N] = torch.tensor([0.0, 0.0, 0.0])
        per_residue[ATOM37_CA] = torch.tensor([1.0, 0.0, 0.0])
        per_residue[ATOM37_C] = torch.tensor([2.0, 0.0, 0.0])
        per_residue[ATOM37_O] = torch.tensor([2.0, 1.0, 0.0])
        # CB slot stays zero; mask will be zeroed below

    atom_positions = einops.repeat(
        per_residue,
        "atom_rep three -> B N_res atom_rep three",
        B=batch_size,
        N_res=residue_number,
    )

    atom_mask = torch.ones(batch_size, residue_number, n_atoms_per_residue)
    if tensor_content == TensorCase.half_mask:
        atom_mask[:, : residue_number // 2, :] = 0
    if tensor_content in (TensorCase.no_beta_carbon, TensorCase.collinear):
        atom_mask[:, :, ATOM37_CB] = 0

    return atom_positions, atom_mask

@pytest.mark.parametrize(
    ("atom_position_and_mask_factory", "expected_pos_shape"),
    [
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.standard, 2, 10),
            (2, 10, 5, 3),
            id="batched-standard",
        ),
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.all_zeros, 1, 10),
            (1, 10, 5, 3),
            id="single-batch-zeros",
        ),
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.no_beta_carbon, 2, 10),
            (2, 10, 5, 3),
            id="batched-missing-cb",
        ),
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.half_mask, 2, 10),
            (2, 10, 5, 3),
            id="batched-half-mask",
        ),
    ],
    indirect=["atom_position_and_mask_factory"],
)
def test_atom37_to_atom5(
    atom_position_and_mask_factory: tuple[
        Float[torch.Tensor, "B N_res 37 3"],
        Float[torch.Tensor, "B N_res 37"],
    ],
    expected_pos_shape: tuple[int, ...],
) -> None:
    """Verify atom37_to_atom5 behavior.

    1. That it returns position and masks with correct shapes and data types.
    2. That it places N, CA, C, O, CB into the expected atom5 slots.
    3. That it propagates atom37 mask entries to correct atom5 slots.
    4. That it verifies that backbone atoms present in atom37 are present in
        atom5, and that absent CB atoms produce a zero mask in the CB slot.

    Args:
        atom_position_and_mask_factory: Fixture-produced (atom37_positions,
            atom37_mask) pair parametrized via indirect.
        expected_pos_shape: The expected shape returned by atom37_to_atom5()
            for the appropriate input. The expected shape of the mask is
            the same except for lacking the final dimension.
    """
    atom37_positions, atom37_mask = atom_position_and_mask_factory

    atom5_positions, atom5_mask = atom37_to_atom5(
        atom37_positions, atom37_mask,
    )

    # 1. Shape and dtype
    assert atom5_positions.shape == expected_pos_shape
    assert atom5_mask.shape == expected_pos_shape[:-1]
    assert atom5_positions.dtype == atom37_positions.dtype
    assert atom5_mask.dtype == atom37_mask.dtype

    atom5_indices: list[int] = [ATOM5_N, ATOM5_CA, ATOM5_C, ATOM5_O, ATOM5_CB]
    atom37_backbone_indices: list[int] = [
        ATOM37_N,
        ATOM37_CA,
        ATOM37_C,
        ATOM37_O,
        ATOM37_CB,
    ]

    # 2. Slot placement: each atom5 channel equals the corresponding atom37
    for a5, a37 in zip(atom5_indices, atom37_backbone_indices, strict=True):
        assert torch.equal(
            atom5_positions[:, :, a5, :],
            atom37_positions[:, :, a37, :],
        )

    # 3 & 4. Mask propagation; the batched-missing-cb case exercises that
    # atom5_mask[:, :, ATOM5_CB] is all-zero when atom37 CB slot is zero.
    for a5, a37 in zip(atom5_indices, atom37_backbone_indices, strict=True):
        assert torch.equal(
            atom5_mask[:, :, a5], atom37_mask[:, :, a37],
        )


@pytest.mark.parametrize(
    ("atom_position_and_mask_factory", "expect_cb_present"),
    [
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.standard, 2, 10),
            ExpectedBetaCarbon.present,
            id="standard-all-cb-present",
        ),
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.no_beta_carbon, 2, 10),
            ExpectedBetaCarbon.absent,
            id="no-beta-carbon-pseudo-cb",
        ),
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.collinear, 2, 10),
            ExpectedBetaCarbon.absent,
            id="collinear-no-cb-pseudo-cb-finite",
        ),
        pytest.param(
            (ResidueRepresentation.atom37, TensorCase.half_mask, 2, 10),
            ExpectedBetaCarbon.half_present,
            id="half-mask-mixed-cb-presence",
        ),
    ],
    indirect=["atom_position_and_mask_factory"],
)
def test_atom37_to_cb(
    atom_position_and_mask_factory: tuple[
        Float[torch.Tensor, "B N_res 37 3"],
        Float[torch.Tensor, "B N_res 37"],
    ],
    expect_cb_present: ExpectedBetaCarbon,
) -> None:
    """Verify atom37_to_cb output shapes, CB presence, and pseudo-CB fallback.

    1. Returns Cβ positions (B, N_RES, 3) and a bool presence mask (B, N_RES)
       for all input cases.
    2. Marks all residues CB-present when the atom37 mask has CB set
       (TensorCase.standard).
    3. Falls back to finite pseudo-Cβ positions and marks every residue
       CB-absent when CB is zeroed in the mask (TensorCase.no_beta_carbon,
       matching glycine-like residues).
    4. Verifies that the virtual Cβ position is not simply a copy of CA atom.
    5. Verifies that pseudo_cb is finite when N, CA, C are collinear
        (cross product is zero). The epsilon guard in linalg.norm must prevent
        NaN in the normalised output.

    Args:
        atom_position_and_mask_factory: Fixture-produced (atom37_positions,
            atom37_mask) pair parametrized via indirect.
        expect_cb_present: ExpectedBetaCarbon.present when parametrized input
            has CB in the mask; ExpectedBetaCarbon.absent for no-beta-carbon
            glycine case.
    """
    atom37_positions, atom37_mask = atom_position_and_mask_factory
    cb, cb_present = atom37_to_cb(atom37_positions, atom37_mask)

    assert cb.shape == (B, N_RES, 3)
    assert cb_present.shape == (B, N_RES)
    assert cb_present.dtype == torch.bool

    if expect_cb_present is ExpectedBetaCarbon.present:
        assert cb_present.all()
        assert torch.allclose(cb, atom37_positions[:, :, ATOM37_CB, :])
    elif expect_cb_present is ExpectedBetaCarbon.absent:
        assert not cb_present.any()
        assert torch.isfinite(cb).all()
        assert not torch.allclose(cb, atom37_positions[:, :, ATOM37_CB, :])
        # assertion 4: pseudo-Cβ is a distinct point, not simply CA
        assert not torch.allclose(
            cb, atom37_positions[:, :, ATOM37_CA, :],
        )
    else:  # half_present: first half masked out, second half unmasked
        assert not cb_present[:, : N_RES // 2].any()
        assert cb_present[:, N_RES // 2 :].all()
        assert torch.isfinite(cb).all()
        assert torch.allclose(
            cb[:, N_RES // 2 :, :],
            atom37_positions[:, N_RES // 2 :, ATOM37_CB, :],
        )


# ---------------------------------------------------------------------------
# Protein dataclass
# ---------------------------------------------------------------------------

N_ATOM_TYPE = len(atom_types) # 37



@dataclass(frozen=True)
class ProteinFieldOverride:
    """Describes a single Protein constructor field to break for negative tests.

    Attributes:
        field_name: Name of the Protein constructor keyword to override.
        shape: The deliberately wrong shape to allocate for that field.
    """

    field_name: str
    shape: tuple[int, ...]


@pytest.fixture
def invalid_protein_factory(
    request: pytest.FixtureRequest,
) -> Callable[[], Protein]:
    """Build a thunk that constructs a Protein with one field shape broken.

    Receives a ProteinFieldOverride via ``request.param`` when used with
    ``indirect=["invalid_protein_factory"]``.

    Args:
        request: Pytest fixture request whose ``.param`` attribute is a
            ProteinFieldOverride naming the field and shape to break.

    Returns:
        A zero-argument callable that constructs a Protein using the
        override; invoking it raises TypeCheckError.
    """
    override = cast(ProteinFieldOverride, request.param)
    int_fields = frozenset({"aatype", "residue_index", "chain_index"})
    dtype = np.intp if override.field_name in int_fields else np.float64
    kwargs: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((N_RES, N_ATOM_TYPE, 3), dtype=np.float64),
        "aatype": np.zeros(N_RES, dtype=np.intp),
        "atom_mask": np.ones((N_RES, N_ATOM_TYPE), dtype=np.float64),
        "residue_index": np.arange(N_RES, dtype=np.intp),
        "chain_index": np.zeros(N_RES, dtype=np.intp),
        "b_factors": np.zeros((N_RES, N_ATOM_TYPE), dtype=np.float64),
    }
    kwargs[override.field_name] = np.zeros(override.shape, dtype=dtype)
    return lambda: Protein(**kwargs)


@pytest.mark.parametrize(
    "invalid_protein_factory",
    [
        pytest.param(
            ProteinFieldOverride("atom_positions", (N_RES, N_ATOM_TYPE)),
            id="atom_positions_missing_coord_axis",
        ),
        pytest.param(
            ProteinFieldOverride("atom_positions", (N_RES, N_ATOM_TYPE, 4)),
            id="atom_positions_extra_coord_component",
        ),
        pytest.param(
            ProteinFieldOverride(
                "atom_positions", (N_RES + 1, N_ATOM_TYPE, 3),
            ),
            id="atom_positions_residue_count_drift",
        ),
        pytest.param(
            ProteinFieldOverride(
                "atom_positions", (N_RES, N_ATOM_TYPE + 1, 3),
            ),
            id="atom_positions_atom_table_drift",
        ),
        pytest.param(
            ProteinFieldOverride("atom_mask", (N_RES,)),
            id="atom_mask_missing_atom_axis",
        ),
        pytest.param(
            ProteinFieldOverride("atom_mask", (N_RES, N_ATOM_TYPE + 1)),
            id="atom_mask_atom_table_drift_independent_of_positions",
        ),
        pytest.param(
            ProteinFieldOverride("aatype", (N_RES, 1)),
            id="aatype_column_vector_bug",
        ),
        pytest.param(
            ProteinFieldOverride("residue_index", (N_RES + 1,)),
            id="residue_index_count_drift_independent_of_aatype",
        ),
    ],
    indirect=["invalid_protein_factory"],
)
def test_protein_rejects_invalid_shapes(
    invalid_protein_factory: Callable[[], Protein],
) -> None:
    """Protein raises TypeCheckError for shape-contract violations.

    Protein is validated by a single ``jaxtyped(typechecker=beartype)``
    check spanning the whole dataclass, so every field sharing a shape
    signature exercises the identical enforcement code path. Rather than
    breaking all six fields exhaustively, this covers each of the three
    distinct signatures exactly once for rank ((num_res, num_atom_type, 3)
    via atom_positions, (num_res, num_atom_type) via atom_mask, (num_res,)
    via aatype), once for the atom_positions-only fixed coordinate-width
    contract, and twice each (on two different fields) for the num_res and
    num_atom_type named-dimension consistency checks, to confirm those
    checks are not anchored to whichever field happens to be checked first.
    ``chain_index`` and ``b_factors`` are intentionally not parametrized
    here: they share the exact (num_res,) and (num_res, num_atom_type)
    signatures already exercised by ``residue_index`` and ``atom_mask``
    above, so adding them would restate the same mechanism rather than
    cover new behavior.

    Each case is also chosen to mirror a plausible bug in
    ``protein_from_pdb``'s array-assembly loop (dropped coordinate axis,
    a stray extra column, residue-count drift from insertion-code/altloc
    handling, or an atom-type table that has grown out of sync with a
    hardcoded dimension) rather than an arbitrary shape permutation.

    Args:
        invalid_protein_factory: Fixture-produced thunk parametrized via
            indirect to construct a Protein with one field shape broken.
    """
    with pytest.raises(TypeCheckError):
        _ = invalid_protein_factory()


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
    n_idx = next(i for i, t in enumerate(atom_types) if t == ATOM5_NAMES[0])
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


def test_center_positions_modifies_in_place(
    full_mask_np_example: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """center_positions mutates input dict's atom_positions array."""
    original = full_mask_np_example["atom_positions"].copy()
    center_positions(full_mask_np_example)
    assert not np.allclose(full_mask_np_example["atom_positions"], original)


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
    _num_res = _aatype.shape[0]

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
