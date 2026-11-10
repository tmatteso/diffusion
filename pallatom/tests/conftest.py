"""Fixtures shared by more than one test module under ``pallatom/tests``."""

from collections.abc import Generator

import pytest
import torch
import torch.distributed as dist
from helpers.atom_utils import ATOM5_CB, ATOM5_NAMES, rigid_group_atom_positions
from jaxtyping import Float


@pytest.fixture
def ideal_alanine_atom5() -> Float[torch.Tensor, "5 3"]:
    """Ideal alanine backbone and Cβ positions, one row per atom5 slot.

    Rows follow ``ATOM5_NAMES``. N, CA, C and CB share alanine's backbone
    rigid frame, so CB sits where pseudo-Cβ construction must put it. O is
    defined in a different rigid group, so its row only serves as a point
    distinct from the others.

    Returns:
        Positions of shape (5, 3) in atom5 order.
    """
    by_name = {
        atom_name: position
        for atom_name, _, position in rigid_group_atom_positions["ALA"]
    }
    return torch.tensor([by_name[atom_name] for atom_name in ATOM5_NAMES])


@pytest.fixture
def ideal_alanine_cb(
    ideal_alanine_atom5: Float[torch.Tensor, "5 3"],
) -> Float[torch.Tensor, "3"]:
    """The known Cβ of the ideal alanine, the target for pseudo-Cβ placement.

    Args:
        ideal_alanine_atom5: Ideal alanine positions in atom5 order.

    Returns:
        The Cβ position, shape (3,).
    """
    return ideal_alanine_atom5[ATOM5_CB]


@pytest.fixture
def single_process_gloo_group() -> Generator[None]:
    """Init a one-rank gloo process group for the test, then destroy it.

    Lets DDP wrapping and collective calls run on CPU in one process. The
    group lives only for the requesting test, so no test sees another's
    distributed state. ``HashStore`` is a private ``torch.distributed``
    C-extension symbol that is not officially re-exported; reading it with
    ``getattr`` sidesteps the static private-import check, whose result
    differs across torch's per-platform packaging.

    Yields:
        None: The group is active while the test runs.
    """
    dist.init_process_group(
        backend="gloo",
        store=getattr(  # noqa: B009  # pyright: ignore[reportAny]
            dist,
            "HashStore",
        )(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
