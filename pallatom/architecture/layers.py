"""Shared primitive layers used across architecture modules.

Provides TypedLinear, LinearNoBias, TypedEmbedding, TypedSequential,
TypedModuleList, and LayerNorm — all with typed ``__call__`` overrides so
call-site return types propagate without ``Any``.
"""

from collections.abc import Iterator
from typing import overload, override

import torch
import torch.nn as nn
from jaxtyping import Float, Int


class TypedLinear(nn.Linear):
    """nn.Linear with typed __call__ so downstream types don't degrade to Any.

    Subclass this or use directly where a typed linear projection is needed.
    """

    @override
    def __call__(
        self,
        input: Float[  # pylint: disable=redefined-builtin
            torch.Tensor,
            "... in_features",
        ],
    ) -> Float[torch.Tensor, "... out_features"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(input)


class LinearNoBias(TypedLinear):
    """Linear layer with no bias term.

    Identical to ``nn.Linear(bias=False)`` but avoids passing ``bias=False``
    at every call site.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            in_features,
            out_features,
            bias=False,
        )


class TypedEmbedding(nn.Embedding):
    """nn.Embedding with typed __call__ so downstream types don't become Any.

    Use wherever an integer index tensor needs to be embedded with a concrete
    return type at the call site.
    """

    @override
    def __call__(
        self,
        input: Int[torch.Tensor, "..."],  # pylint: disable=redefined-builtin
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(input)


class TypedSequential(nn.Sequential):
    """nn.Sequential with typed __call__ so downstream types don't become Any.

    Use when chaining layers whose input and output feature dimensions are
    homogeneous across the sequence.
    """

    @override
    def __call__(
        self,
        input: Float[  # pylint: disable=redefined-builtin
            torch.Tensor,
            "... in_features",
        ],
    ) -> Float[torch.Tensor, "... out_features"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(input)  # pyright: ignore[reportUnknownMemberType]


class TypedModuleList[TypedModule: nn.Module](nn.ModuleList):
    """nn.ModuleList with typed __iter__ and __getitem__ to keep element types.

    Generic over ``TypedModule`` so iteration and indexing preserve the
    concrete element type rather than falling back to ``nn.Module``.
    """

    @override
    def __iter__(
        self,
    ) -> Iterator[TypedModule]:  # pylint: disable=undefined-variable
        """Iterate over submodules with their concrete type preserved.

        Returns:
            Iterator over elements typed as ``TypedModule`` rather than
            ``nn.Module``.
        """
        return super().__iter__()  # pyright: ignore[reportReturnType]

    @overload
    def __getitem__(self, index: slice) -> "TypedModuleList[TypedModule]": ...
    @overload
    def __getitem__(
        self,
        index: int,
    ) -> TypedModule: ...  # pylint: disable=undefined-variable
    @override
    def __getitem__(
        self,
        index: int | slice,
    ) -> "TypedModule | TypedModuleList[TypedModule]":
        """Index into the list with the concrete element type preserved.

        Args:
            index: An integer returns the element at that position; a slice
                returns a ``TypedModuleList`` of the same element type.

        Returns:
            The selected element(s) typed as ``TypedModule`` or
            ``TypedModuleList[TypedModule]``.
        """
        return super().__getitem__(index)  # pyright: ignore[reportReturnType]


class LayerNorm(nn.LayerNorm):
    """LayerNorm with typed __call__ so downstream types don't degrade to Any.

    Drop-in replacement for ``nn.LayerNorm`` with a concrete return type at
    call sites.
    """

    @override
    def __call__(
        self,
        input: Float[  # pylint: disable=redefined-builtin
            torch.Tensor,
            "... normalized_shape",
        ],
    ) -> Float[torch.Tensor, "... normalized_shape"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(input)
