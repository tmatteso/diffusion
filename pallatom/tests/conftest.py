"""Fixtures shared by more than one test module under ``pallatom/tests``."""

from collections.abc import Generator

import pytest
import torch.distributed as dist


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
