# Design: Consolidate batch_types + featurize → data

**Date:** 2026-06-21
**Status:** Approved

## Goal

Merge `pallatom/helpers/batch_types.py` and `pallatom/helpers/featurize.py` into
`pallatom/helpers/data.py`, and consolidate their tests into
`pallatom/tests/helpers/test_data.py`. Delete the four now-redundant files.

## Motivation

Three tightly-coupled files (`batch_types`, `featurize`, `data`) have a linear
dependency chain with no real module boundary — `FeaturizedItem`/`FeaturizedBatch`
exist solely to be produced by `featurize` and consumed by `data`'s loaders. A
single `data.py` is easier to navigate and eliminates the multi-hop import chain.

## Circular Import — Root Cause and Fix

`data.py` currently imports `TrainArgs` from `helpers/useful_objects.py`, and
`useful_objects.py` imports `Distogram` from `helpers/featurize.py`. Absorbing
`featurize` into `data` would create:

```
data.py → useful_objects.py → data.py   # circular
```

**Fix:** Move `TrainArgs` (lines 333–354 of `useful_objects.py`) to
`train/train_config.py`. `TrainArgs` is a pure path/flag CLI dataclass with zero
domain dependencies; it logically belongs alongside `TrainConfig`. After the move,
`data.py` imports `TrainArgs` from `train.train_config`, which has no dependency on
`data.py`, breaking the cycle entirely.

## Section Ordering in the Merged `data.py`

```
module docstring (updated to cover all three origins)
stdlib imports
third-party imports
first-party imports (consolidated from all three files)

── Pure data types (from batch_types.py) ──
FeaturizedItem
FeaturizedBatch

── Featurization utilities (from featurize.py) ──
Distogram
sinusoidal_encoding
ref_pos_for_residue
featurize_single_item
featurize_batch
apply_conditioning_dropout

── Dataset / DataLoader (existing data.py) ──
ProteinEntry, ProteinNamesManifest, DatasetSplitsManifest
ClusterMetadataEntry, ShardMetadata, ShardBatchPlan, ShardBudgetParameters
ProteinDataset
ProteinShardDataset
ShardDataLoader
identity_collate
make_bucketed_data_loaders
```

## External Import Sites to Update

| File | Old import | New import |
|------|-----------|-----------|
| `helpers/useful_objects.py` | `from helpers.featurize import Distogram` | `from helpers.data import Distogram` |
| `helpers/data.py` | `from helpers.useful_objects import TrainArgs` | `from train.train_config import TrainArgs` |
| `architecture/losses.py` | `from helpers.batch_types import FeaturizedBatch` | `from helpers.data import FeaturizedBatch` |
| `architecture/main_trunk.py` | `from helpers.batch_types import FeaturizedBatch` | `from helpers.data import FeaturizedBatch` |
| `sample/sampling.py` | `from helpers.batch_types import FeaturizedBatch` | `from helpers.data import FeaturizedBatch` |
| `sample/sampling.py` | `from helpers.featurize import Distogram, ref_pos_for_residue` | `from helpers.data import Distogram, ref_pos_for_residue` |
| `train/train_loop.py` | `from helpers.batch_types import FeaturizedBatch` | `from helpers.data import FeaturizedBatch` |
| `train/train_loop.py` | `from helpers.featurize import (...)` | `from helpers.data import (...)` |
| `train/train_loop.py` | `from helpers.useful_objects import TrainArgs` | `from train.train_config import TrainArgs` |
| `tests/helpers/test_data.py` | `from helpers.useful_objects import TrainArgs` | `from train.train_config import TrainArgs` |
| `tests/architecture/test_losses.py` | `from helpers.batch_types import ...` | `from helpers.data import ...` |
| `tests/architecture/test_losses.py` | `from helpers.featurize import (...)` | `from helpers.data import (...)` |
| `tests/architecture/test_main_trunk.py` | `from helpers.batch_types import FeaturizedBatch` | `from helpers.data import FeaturizedBatch` |
| `tests/architecture/test_main_trunk.py` | `from helpers.featurize import sinusoidal_encoding` | `from helpers.data import sinusoidal_encoding` |
| `tests/sample/test_sampling.py` | `from helpers.batch_types import FeaturizedBatch` | `from helpers.data import FeaturizedBatch` |
| `tests/sample/test_sampling.py` | `from helpers.featurize import Distogram` | `from helpers.data import Distogram` |
| `tests/train/test_train_loop.py` | `from helpers.featurize import Distogram` | `from helpers.data import Distogram` |

## Test Consolidation

Merge in this order into `test_data.py`:
1. Content of `test_batch_types.py` — prepended before existing content
2. Content of `test_featurize.py` — inserted after batch_types section
3. Existing `test_data.py` content — unchanged at end

Deduplication: `test_featurize.py` and `test_batch_types.py` both define a
`protein_batch` fixture (different implementations). Rename `test_featurize.py`'s
`protein_batch` fixture to `featurize_protein_batch` and update its dependents
(`featurized_batch` fixture in that file).

Update all `from helpers.batch_types import ...` and
`from helpers.featurize import ...` within the merged test file to
`from helpers.data import ...`.

## Files Deleted After Merge

- `pallatom/helpers/batch_types.py`
- `pallatom/helpers/featurize.py`
- `pallatom/tests/helpers/test_batch_types.py`
- `pallatom/tests/helpers/test_featurize.py`

## Verification

After the merge, run:

```bash
pre-commit run --all-files
pytest pallatom/tests/helpers/test_data.py -x
pytest pallatom/tests/ -x
```

All hooks and tests must pass before the task is considered complete.
