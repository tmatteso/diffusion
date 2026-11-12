"""Deterministic tensor patterns shared by more than one test module.

These are plain module-level functions rather than fixtures: a fixture
must return data, never a callable, and these patterns are needed at
several shapes within one module. Each test module's fixtures call them
to build their own named inputs.
"""

import torch
from einops import rearrange
from jaxtyping import Float


def sine_pattern(
    shape: tuple[int, ...],
    freq: float,
) -> Float[torch.Tensor, "..."]:
    """Fill ``shape`` with ``sin(freq * k)`` over the flat element index.

    Distinct frequencies give inputs that are unrelated to each other, with
    values in ``[-1, 1]``, so tests get varied, reproducible tensors
    without a random generator.

    Consecutive elements are linearly related, ``sin(a + 2f) =
    2 cos(f) sin(a + f) - sin(a)``, so a ``(..., 3)`` tensor from this
    function holds coplanar points. Use it for features, not for
    coordinates whose geometry the test depends on.

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


def chiral_cloud(b: int, n: int) -> Float[torch.Tensor, "B N 3"]:
    """Build ``b`` deterministic, non-planar, chiral clouds of ``n`` atoms.

    Atom ``k``, counted across the whole batch, sits at
    ``(sin 0.37k, sin(0.91k + 1), cos 1.73k)``. Unrelated frequencies per
    axis keep the points off any common plane, which ``sine_pattern``
    cannot do; use this function whenever a test depends on the geometry
    of the coordinates.

    Args:
        b: Number of clouds.
        n: Atoms per cloud.

    Returns:
        ``(b, n, 3)`` coordinates in ``[-1, 1]``.
    """
    k = torch.arange(b * n, dtype=torch.float32)
    axes = torch.stack(
        [torch.sin(0.37 * k), torch.sin(0.91 * k + 1), torch.cos(1.73 * k)],
    )
    return rearrange(axes, "d (b n) -> b n d", b=b)
