"""Fixtures shared by more than one test module under ``pallatom/tests``."""

from collections.abc import Callable, Generator

import pytest
import torch
import torch.distributed as dist
from einops import rearrange
from jaxtyping import Float

type PatternFactory = Callable[
    [tuple[int, ...], float],
    Float[torch.Tensor, "..."],
]


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


@pytest.fixture
def make_pattern() -> PatternFactory:
    """Return a factory for deterministic sine-pattern tensors.

    ``make(shape, freq)`` fills ``shape`` with ``sin(freq * k)`` over the
    flat element index ``k``. Distinct frequencies give inputs that are
    unrelated to each other, with values in ``[-1, 1]``, so tests get
    varied, reproducible tensors without a random generator.

    Consecutive elements are linearly related, ``sin(a + 2f) =
    2 cos(f) sin(a + f) - sin(a)``, so a ``(..., 3)`` tensor from this
    factory holds coplanar points. Use it for features, not for
    coordinates whose geometry the test depends on.

    Returns:
        The factory.
    """

    def make(shape: tuple[int, ...], freq: float) -> Float[torch.Tensor, "..."]:
        """Build one sine-pattern tensor.

        Args:
            shape: Output shape.
            freq: Angular step between consecutive elements.

        Returns:
            Float32 tensor of ``shape``.
        """
        names = [f"d{axis}" for axis in range(len(shape))]
        axes = " ".join(names)
        flat = torch.sin(freq * torch.arange(torch.Size(shape).numel()))
        return rearrange(
            flat.float(),
            f"({axes}) -> {axes}",
            **dict(zip(names, shape, strict=True)),
        )

    return make
