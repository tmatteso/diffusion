"""Custom exception types for architecture validation."""


class InvalidPairHeadDimensionError(ValueError):
    """Raised when c_pair is not divisible by n_heads."""

    def __init__(self, c_pair: int, n_heads: int) -> None:
        super().__init__(
            f"c_pair ({c_pair}) must be divisible by n_heads ({n_heads})",
        )


class LossComputationError(ValueError):
    """Loss computation returned None.

    This usually indicates an invalid training state
    (e.g. NaN/Inf propagation).
    """


class NoDenoiserBlockError(ValueError):
    """r_denoised_blocks must be non-empty."""


class BlockCountMismatchError(ValueError):
    """Raised when structure and sequence decoder block counts differ."""

    def __init__(self, struct: int, seq: int) -> None:
        super().__init__(
            f"{struct} structure decoder blocks != {seq} seq decoder blocks.",
        )
