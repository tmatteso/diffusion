"""Shared primitive layers used across architecture modules."""

from collections.abc import Iterator
from typing import Generic, TypeVar, overload

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from typing_extensions import override

TypedModule = TypeVar("TypedModule", bound=nn.Module)


class TypedLinear(nn.Linear):
    """nn.Linear with a typed __call__ so downstream types don't degrade to Any."""

    @override
    def __call__(
        self, input: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(input)


class LinearNoBias(TypedLinear):
    """Linear layer with no bias term."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialise a bias-free linear projection.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
        """
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            in_features, out_features, bias=False
        )


class TypedEmbedding(nn.Embedding):
    """nn.Embedding with a typed __call__ so downstream types don't degrade to Any."""

    @override
    def __call__(self, input: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... embedding_dim"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(input)


class TypedSequential(nn.Sequential):
    """nn.Sequential with a typed __call__ so downstream types don't degrade to Any."""

    @override
    def __call__(
        self, input: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(input)  # pyright: ignore[reportUnknownMemberType]


class TypedModuleList(nn.ModuleList, Generic[TypedModule]):
    """nn.ModuleList with typed __iter__ and __getitem__ so element types aren't erased."""

    @override
    def __iter__(self) -> Iterator[TypedModule]:
        """Iterate over submodules with their concrete type preserved."""
        return super().__iter__()  # pyright: ignore[reportReturnType]

    @overload
    def __getitem__(self, index: slice) -> "TypedModuleList[TypedModule]": ...
    @overload
    def __getitem__(self, index: int) -> TypedModule: ...
    @override
    def __getitem__(self, index: int | slice) -> "TypedModule | TypedModuleList[TypedModule]":
        """Index into the list with the concrete element type preserved."""
        return super().__getitem__(index)  # pyright: ignore[reportReturnType]


class LayerNorm(nn.LayerNorm):
    """LayerNorm with a typed __call__ so downstream types don't degrade to Any."""

    @override
    def __call__(
        self, input: Float[torch.Tensor, "... normalized_shape"]
    ) -> Float[torch.Tensor, "... normalized_shape"]:
        """Call forward; typed override so call-site return types are not Any."""
        return self.forward(input)
