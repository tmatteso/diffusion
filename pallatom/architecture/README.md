# `pallatom/architecture` — model overview

Neural network architecture for the Pallatom all-atom diffusion model: an
AlphaFold-3-style trunk (Algorithm 2) that denoises a noisy all-atom
structure and predicts an amino-acid sequence, conditioned on a template
distogram and iteratively refined over `K_unit` decoder blocks.

For the complete per-tensor shape reference (every named dimension, every
input/intermediate/output tensor of `MainTrunk`), see the
[MainTrunk tensor reference](../CLAUDE.md#maintrunk-tensor-reference) table
in `pallatom/CLAUDE.md`.

## Pipeline

```
FeaturizedBatch
      │
      ▼
┌─────────────────────────── MainTrunk.embed_inputs (steps 1-8) ───────────┐
│  TimeFourierEmbedding + RelativePositionEncoding  →  s_init, z_ij        │
│  TemplateEmbedder(f_distogram, z_ij, t)           →  z_ij += template    │
│  AtomFeatureEncoder(ref_pos, ref_element, ...)    →  s_i, q_skip, c_skip,│
│                                                       p_skip, c_l        │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────── decoder loop, repeated K_unit times (steps 9-17) ──────┐
│  NodeUpdate(s_i, t_i, z_ij)             → refines s_i                    │
│  AtomAttentionDecoder(...)              → r_update (per-atom Δposition)  │
│  r_denoised = EDM-combine(r_input, Σ r_update)                           │
│  PairUpdate(z_ij, r_center)             → refines z_ij from new coords   │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
 distogram heads (residue + atom) · sequence head
      │
      ▼
 PredictedOutputs(r_denoised, seq_logits, residue/atom distogram logits, …)
```

Losses over `PredictedOutputs` are computed in [`losses.py`](docs/losses.md);
the weighted sum that combines them into `total_loss` is assembled in
`pallatom/train/train_loop.py` (see
[Combining the losses](docs/losses.md#combining-the-losses-total_loss)).

## Module map

| Module | Contents |
|---|---|
| [`docs/main_trunk.md`](docs/main_trunk.md) | `MainTrunk` (Algorithm 2) — top-level denoiser; `TimeFourierEmbedding`; `RelativePositionEncoding` |
| [`docs/template_embedder.md`](docs/template_embedder.md) | `TemplateEmbedder` — folds the template distogram into `z_ij` |
| [`docs/atom_transformers.md`](docs/atom_transformers.md) | `AtomFeatureEncoder`, `AtomAttentionDecoder`, `AtomTransformer`, `DiffusionTransformer`, `ConditionedTransitionBlock` — the per-atom encode/decode path |
| [`docs/node_update.md`](docs/node_update.md) | `NodeUpdate` (Algorithm 6), `AttentionPairBias`, `AdaLN` — residue single-embedding refinement |
| [`docs/pair_update.md`](docs/pair_update.md) | `PairUpdate` (Algorithm 7), triangle attention (starting/ending node), `Transition`, structured dropout |
| [`docs/pairformer_stack.md`](docs/pairformer_stack.md) | `PairformerStack`/`PairformerBlock`, triangle multiplication (incoming/outgoing) — used inside `TemplateEmbedder` |
| [`docs/losses.md`](docs/losses.md) | Kabsch-aligned MSE, intermediate (`L_med`), smooth lDDT, residue/atom distogram CE, sequence CE, and the `total_loss` combination |
| [`docs/layers.md`](docs/layers.md) | Shared typed primitives (`LinearNoBias`, `LayerNorm`, `TypedModuleList`, …) — no algorithm diagrams |
| [`docs/errors.md`](docs/errors.md) | Exception types raised by validation checks across the modules above |

`__init__.py` re-exports nothing beyond the package docstring and isn't
documented separately.

## Where the pseudocode images come from

Every `assets/*.png` is a screenshot of either an AlphaFold 3 algorithm box
(`af3_*.png`) or a Pallatom-paper algorithm box (`pallatom_*.png`), placed in
this directory's `assets/` and embedded directly in the relevant module doc
under `docs/`. `pallatom_complete_loss.png` and `pallatom_l_not_loss.png` are
the exception: they describe the *combination* of losses into `total_loss`
rather than any single loss function, so [`docs/losses.md`](docs/losses.md)
attaches them to a dedicated section instead of to one function.
