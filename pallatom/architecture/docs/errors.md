# `errors.py` — architecture exception types

[← back to architecture overview](../README.md)

Custom exception types raised by validation checks elsewhere in
`architecture/`. No pseudocode diagrams apply — these are plain error
conditions, not algorithm steps.

## `InvalidPairHeadDimensionError`

Raised when `c_pair` is not evenly divisible by `n_heads`. Checked in
`PairformerBlock.__init__` ([pairformer_stack.md](pairformer_stack.md)) and
in `TriangleAttentionStartingNodeWithBias` /
`TriangleAttentionEndingNodeWithBias` ([pair_update.md](pair_update.md)),
since per-head attention requires the channel dimension to split evenly
across heads.

## `LossComputationError`

Raised by every loss function in [`losses.py`](losses.md) when the computed
loss contains NaN — a signal of an invalid training state (e.g. NaN/Inf
propagation upstream) rather than a normal error path.

## `NoDenoiserBlockError`

Raised by `med_loss` ([losses.md](losses.md#med_loss--l_med)) when
`r_denoised_blocks` is empty; the intermediate loss has nothing to average
over.

## `BlockCountMismatchError`

Raised by `med_loss` when the structure decoder block count and sequence
decoder block count disagree — the two lists must line up block-for-block.
