# pallatom/helpers

Helper utilities for data processing, featurization, and alignment. This
package provides modules for atom-level featurization, sequence and
structure alignment, batched dataset sampling, cluster indexing, data
loading, and other shared utilities used throughout the pallatom training
and inference pipelines.

## Modules

- **[`alignment.md`](alignment.md)** — `alignment.py`: rigid structure
  alignment via the Kabsch algorithm (`kabsch_rotation`, `kabsch_align`,
  `rmsd`, `kabsch_rmsd`, `apply_transform`), plus `masked_com` and
  `centre_random_augment`, the mask-aware centering and random
  rotation/translation augmentation used both during training featurization
  and at every step of EDM sampling.
- **[`data.md`](data.md)** — `data.py`: `FeaturizedItem`/`FeaturizedBatch`,
  the `Distogram` module, and the full training data pipeline —
  `featurize_single_item` (documented in depth), `featurize_batch`,
  `ProteinDataset`, and `make_bucketed_data_loaders`.
- **`atom_utils.py`** — the `Protein` dataclass, atom-type/residue-type
  constants, PDB parsing (`protein_from_pdb`) and writing (`to_pdb`), and
  tensor conversions between the atom37, atom5, and Cβ coordinate
  representations (`atom37_to_atom5`, `atom5_to_atom37`, `atom37_to_cb`,
  `pseudo_cb`).
- **`context_managers.py`** — distributed-training and logging context
  managers: `DistProcessGroup` (process-group lifecycle and peer-failure
  propagation), `FatalOnError`, `StructlogConfig`, `ShardWorkerState`,
  `DDPNoSync` (gradient-sync suppression during accumulation), and
  `StepContext` (eval-mode/autograd toggling).
- **`useful_objects.py`** — frozen dataclasses bundling mutable training
  objects and accumulating metrics: `ModelSetup`, `LossMetrics`,
  `ThroughputStatistics`, `ComponentNorms`, `EpochMetrics`, `StepProgress`,
  and the `TensorAccumulatorMixin` that gives the metrics dataclasses
  field-wise `+=`/`*`/`weighted_avg` arithmetic.
- **`errors.py`** — exception types raised by `atom_utils.py`'s validation
  checks: `NoAtomRecordsError`, `InvalidAAtypesError`, `TooManyChainsError`.

## How the pieces fit together

A training step pulls a raw `Protein` (built by `atom_utils.py`, either
from a PDB file or a JSONL dataset entry) through `data.py`'s
`featurize_single_item`, which centers/augments it with `alignment.py`'s
`centre_random_augment`, noises it to a sampled EDM noise level, and builds
every ground-truth tensor (`FeaturizedItem`) `MainTrunk` needs. Many such
items are stacked into a `FeaturizedBatch` by `featurize_batch` and handed
to the training loop, which uses `useful_objects.py`'s bundles to carry
model/optimizer state and accumulate metrics, and `context_managers.py`'s
context managers to handle DDP, logging, and eval-mode plumbing around each
step. See [`pallatom/train/README.md`](../train/README.md) for how the
training loop consumes all of this, and
[`pallatom/sample/README.md`](../sample/README.md) for how sampling reuses
`alignment.py`'s `centre_random_augment`/`masked_com` and `atom_utils.py`'s
coordinate conversions outside of training.
