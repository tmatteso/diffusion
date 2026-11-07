"""Hypothesis strategies and helpers for coordinate-geometry property tests.

Coordinates are float64 by default so tolerances reflect the algorithm under
test rather than float32 rounding. Generated clouds deliberately include
degenerate shapes, such as coincident, collinear and coplanar atoms.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from einops import einsum
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import SearchStrategy
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray

MAX_ATOMS = 12
MAX_BATCH = 3
MAX_COORD = 100.0
MIN_QUATERNION_SQ_NORM = 1e-3

_COORDS = st.floats(
    -MAX_COORD,
    MAX_COORD,
    allow_subnormal=False,
    width=64,
)
_QUATERNION_PART = st.floats(-1.0, 1.0, allow_subnormal=False, width=64)


@dataclass(frozen=True)
class CloudPair:
    """Two unrelated point clouds with the same atom count.

    Attributes:
        mobile: Coordinates to align, shape (N, 3).
        target: Reference coordinates, shape (N, 3).
    """

    mobile: Float[torch.Tensor, "N 3"]
    target: Float[torch.Tensor, "N 3"]


@dataclass(frozen=True)
class MaskedCloudPair:
    """A cloud pair with a mask that selects at least one atom.

    Attributes:
        pair: The coordinates.
        mask: True for atoms that count, shape (N,).
    """

    pair: CloudPair
    mask: Bool[torch.Tensor, "N"]


@dataclass(frozen=True)
class ShuffledCloudPair:
    """A cloud pair with a reordering of its atoms.

    Attributes:
        pair: The coordinates.
        order: A permutation of the atom indices, shape (N,).
    """

    pair: CloudPair
    order: Int[torch.Tensor, "N"]


def _as_tensor(values: NDArray[np.float64]) -> Float[torch.Tensor, "..."]:
    """Share a generated array's memory with a tensor."""
    return torch.from_numpy(values)  # pyright: ignore[reportUnknownMemberType]


def _tensors(
    shape: tuple[int, ...],
) -> SearchStrategy[Float[torch.Tensor, "..."]]:
    """Return a strategy for float64 coordinate tensors of a fixed shape.

    Args:
        shape: The tensor shape.

    Returns:
        A strategy producing tensors with values in +/- MAX_COORD.
    """
    return arrays(np.float64, shape, elements=_COORDS).map(_as_tensor)


def _pairs_of_size(atoms: int) -> SearchStrategy[CloudPair]:
    """Return a strategy for cloud pairs with a fixed atom count.

    Args:
        atoms: The number of atoms in each cloud.

    Returns:
        A strategy producing independent mobile and target clouds.
    """
    shape = (atoms, 3)
    return st.builds(CloudPair, _tensors(shape), _tensors(shape))


def _masked_of_size(atoms: int) -> SearchStrategy[MaskedCloudPair]:
    """Return a strategy for masked cloud pairs with a fixed atom count.

    Args:
        atoms: The number of atoms in each cloud.

    Returns:
        A strategy whose masks select at least one atom.
    """
    masks = (
        st.lists(st.booleans(), min_size=atoms, max_size=atoms)
        .filter(any)
        .map(torch.tensor)
    )
    return st.builds(MaskedCloudPair, _pairs_of_size(atoms), masks)


def _shuffled_of_size(atoms: int) -> SearchStrategy[ShuffledCloudPair]:
    """Return a strategy for shuffled cloud pairs with a fixed atom count.

    Args:
        atoms: The number of atoms in each cloud.

    Returns:
        A strategy pairing clouds with a permutation of their atoms.
    """
    orders = st.permutations(range(atoms)).map(torch.tensor)
    return st.builds(ShuffledCloudPair, _pairs_of_size(atoms), orders)


def _atom_counts(min_atoms: int) -> SearchStrategy[int]:
    """Return a strategy for atom counts.

    Args:
        min_atoms: The smallest count to generate.

    Returns:
        A strategy producing counts up to MAX_ATOMS.
    """
    return st.integers(min_atoms, MAX_ATOMS)


def point_clouds(
    min_atoms: int = 1,
) -> SearchStrategy[Float[torch.Tensor, "N 3"]]:
    """Return a strategy for single point clouds of shape (N, 3).

    Args:
        min_atoms: The smallest atom count to generate.

    Returns:
        A strategy producing clouds of varying size.
    """
    return _atom_counts(min_atoms).map(lambda n: (n, 3)).flatmap(_tensors)


def batched_clouds(
    min_atoms: int = 1,
) -> SearchStrategy[Float[torch.Tensor, "B N 3"]]:
    """Return a strategy for batched point clouds of shape (B, N, 3).

    Args:
        min_atoms: The smallest atom count to generate.

    Returns:
        A strategy producing batches of varying size.
    """
    return st.tuples(
        st.integers(1, MAX_BATCH),
        _atom_counts(min_atoms),
        st.just(3),
    ).flatmap(_tensors)


def cloud_pairs(min_atoms: int = 1) -> SearchStrategy[CloudPair]:
    """Return a strategy for cloud pairs of matching size.

    Args:
        min_atoms: The smallest atom count to generate.

    Returns:
        A strategy producing independent mobile and target clouds.
    """
    return _atom_counts(min_atoms).flatmap(_pairs_of_size)


def masked_cloud_pairs() -> SearchStrategy[MaskedCloudPair]:
    """Return a strategy for cloud pairs with a non-empty atom mask."""
    return _atom_counts(1).flatmap(_masked_of_size)


def shuffled_cloud_pairs() -> SearchStrategy[ShuffledCloudPair]:
    """Return a strategy for cloud pairs with an atom permutation."""
    return _atom_counts(1).flatmap(_shuffled_of_size)


def translations() -> SearchStrategy[Float[torch.Tensor, "1 3"]]:
    """Return a strategy for rigid translations of shape (1, 3)."""
    return _tensors((1, 3))


def _rotation_from_quaternion(
    quaternion: tuple[float, float, float, float],
) -> Float[torch.Tensor, "3 3"]:
    """Convert a non-zero quaternion into a rotation matrix.

    Args:
        quaternion: The (w, x, y, z) components, in any scale.

    Returns:
        The float64 rotation matrix.
    """
    scale = math.sqrt(sum(part * part for part in quaternion))
    w, x, y, z = (part / scale for part in quaternion)
    return torch.tensor(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=torch.float64,
    )


def rotations() -> SearchStrategy[Float[torch.Tensor, "3 3"]]:
    """Return a strategy for proper rotation matrices of shape (3, 3)."""
    return (
        st.tuples(
            _QUATERNION_PART,
            _QUATERNION_PART,
            _QUATERNION_PART,
            _QUATERNION_PART,
        )
        .filter(lambda q: sum(p * p for p in q) > MIN_QUATERNION_SQ_NORM)
        .map(_rotation_from_quaternion)
    )


def rotate(
    points: Float[torch.Tensor, "... n 3"],
    rotation: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "... n 3"]:
    """Rotate row-vector coordinates, as kabsch_align does.

    Args:
        points: The coordinates.
        rotation: The rotation matrix.

    Returns:
        The rotated coordinates.
    """
    return einsum(points, rotation.mH, "... n i, i j -> ... n j")
