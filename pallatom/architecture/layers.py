"""Shared primitive layers used across architecture modules."""

import torch.nn as nn


class LinearNoBias(nn.Linear):
    """Linear layer with no bias term."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialise a bias-free linear projection.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
        """
        super().__init__(in_features, out_features, bias=False)  # type: ignore[reportUnknownMemberType]
