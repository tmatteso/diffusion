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
