# `layers.py` — shared primitive layers

[← back to architecture overview](../README.md)

Shared building-block layers used throughout `architecture/`. Every class here
is a thin, typed wrapper around a standard `torch.nn` primitive; none of them
correspond to a numbered algorithm step in the AF3 or Pallatom papers, so
there are no pseudocode diagrams for this file.

The wrappers exist for one reason: plain `nn.Linear`, `nn.Sequential`,
`nn.ModuleList`, and `nn.LayerNorm` all type their `__call__`/`forward`
signatures loosely enough that downstream call sites degrade to `Any` under
basedpyright. Each wrapper below overrides `__call__` with a concrete
`jaxtyping` signature so shape contracts propagate through the rest of the
codebase.

## `TypedLinear`

`nn.Linear` with a typed `__call__`: `Float[Tensor, "... in_features"] ->
Float[Tensor, "... out_features"]`. Base class for `LinearNoBias`.

## `LinearNoBias`

`TypedLinear` with `bias=False` baked in, so call sites don't need to repeat
`bias=False` everywhere. This is the standard projection layer used across
`main_trunk.py`, `node_update.py`, `pair_update.py`, `pairformer_stack.py`,
`template_embedder.py`, and `atom_transformers.py` — AF3/Pallatom project
almost exclusively with bias-free linear layers.

## `TypedEmbedding`

`nn.Embedding` with a typed `__call__`: integer index tensor in, `Float[...,
"... embedding_dim"]` out.

## `TypedSequential`

`nn.Sequential` with a typed `__call__`, used for the small ReLU-gated MLP
stacks (e.g. the distogram heads in `main_trunk.py`, `mlp_p` in
`atom_transformers.py`).

## `TypedModuleList[TypedModule]`

`nn.ModuleList` generic over its element type, so iterating or indexing (e.g.
`self.node_updates[k]` in `MainTrunk`) returns the concrete module type
instead of `nn.Module`.

## `LayerNorm`

`nn.LayerNorm` with a typed `__call__`. Used everywhere normalisation is
needed; see `node_update.py`'s `AdaLN` for the conditioned variant.
