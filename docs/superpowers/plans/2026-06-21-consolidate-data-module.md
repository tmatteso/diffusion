# Consolidate data module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `helpers/batch_types.py` and `helpers/featurize.py` into `helpers/data.py`, and merge their tests into `tests/helpers/test_data.py`, eliminating 4 files.

**Architecture:** Move `TrainArgs` to `train/train_config.py` first (breaks the circular import that would form once `featurize` is absorbed into `data`), then absorb `batch_types` and `featurize` content into `data.py` in two atomic commits (append content + update all importers + delete source file). Tests are merged last.

**Tech Stack:** Python 3.10, pytest, basedpyright, ruff, pre-commit

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `pallatom/train/train_config.py` | Modify | Add `import dataclasses` + `from pathlib import Path`; append `TrainArgs` class |
| `pallatom/helpers/useful_objects.py` | Modify | Remove `TrainArgs` class; change `from helpers.featurize import Distogram` → `from helpers.data import Distogram` |
| `pallatom/helpers/data.py` | Modify | Change `TrainArgs` import source; insert `FeaturizedItem`/`FeaturizedBatch`; insert all of featurize.py; extend import block in two stages |
| `pallatom/helpers/batch_types.py` | **Delete** | Absorbed into data.py |
| `pallatom/helpers/featurize.py` | **Delete** | Absorbed into data.py |
| `pallatom/architecture/losses.py` | Modify | `helpers.batch_types` → `helpers.data` |
| `pallatom/architecture/main_trunk.py` | Modify | `helpers.batch_types` → `helpers.data` |
| `pallatom/sample/sampling.py` | Modify | `helpers.batch_types` + `helpers.featurize` → `helpers.data` |
| `pallatom/train/train_loop.py` | Modify | `helpers.batch_types` + `helpers.featurize` → `helpers.data`; `TrainArgs` from `train.train_config` |
| `pallatom/tests/helpers/test_data.py` | Modify | Absorb test_batch_types + test_featurize content; update imports; rename constants and fixtures |
| `pallatom/tests/helpers/test_batch_types.py` | **Delete** | Absorbed into test_data.py |
| `pallatom/tests/helpers/test_featurize.py` | **Delete** | Absorbed into test_data.py |
| `pallatom/tests/architecture/test_losses.py` | Modify | `helpers.batch_types` + `helpers.featurize` → `helpers.data` |
| `pallatom/tests/architecture/test_main_trunk.py` | Modify | `helpers.batch_types` + `helpers.featurize` → `helpers.data` |
| `pallatom/tests/sample/test_sampling.py` | Modify | `helpers.batch_types` + `helpers.featurize` → `helpers.data` |
| `pallatom/tests/train/test_train_loop.py` | Modify | `helpers.featurize` → `helpers.data` |

---

## Task 1: Verify baseline

**Files:** (read-only)

- [ ] **Step 1: Run the full test suite to confirm a clean baseline**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -x -q 2>&1 | tail -20
```

Expected: all tests pass. If any fail, stop and fix them before proceeding — the tasks below assume a green baseline.

---

## Task 2: Move `TrainArgs` to `train/train_config.py`

**Files:**
- Modify: `pallatom/train/train_config.py`
- Modify: `pallatom/helpers/useful_objects.py`
- Modify: `pallatom/helpers/data.py:39`
- Modify: `pallatom/train/train_loop.py:49-57`
- Modify: `pallatom/tests/helpers/test_data.py:28`

- [ ] **Step 1: Add `dataclasses` and `Path` imports to `train/train_config.py`**

The file currently only imports `import pathlib`, `from typing import ClassVar`, and pydantic symbols. `TrainArgs` needs `dataclasses` and `Path`.

Change the import block at the top of `pallatom/train/train_config.py` to:

```python
import dataclasses
import pathlib
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
```

- [ ] **Step 2: Append `TrainArgs` to the end of `train/train_config.py`**

Add this block after the final class in the file (after `TrainConfig`):

```python


@dataclasses.dataclass
class TrainArgs:
    """Parsed command-line arguments for the training entry point.

    Attributes:
        dataset_jsonl: Path to the proteins JSONL dataset file.
        shard_dir: Directory containing the pre-built shard tars.
        keys_for_splits_json: Path to the train/val/test splits JSON.
        config: Path to TrainConfig JSON.
        structlog_jsonl: Path to write structured JSON log lines.
        ddp: If True, use DistributedDataParallel training.
        debug_run: If True, restrict to 252 proteins for fast iteration.
    """

    dataset_jsonl: Path
    shard_dir: Path
    keys_for_splits_json: Path
    config: Path
    structlog_jsonl: Path
    ddp: bool
    debug_run: bool
```

- [ ] **Step 3: Remove `TrainArgs` from `helpers/useful_objects.py`**

Delete the entire `TrainArgs` dataclass from `pallatom/helpers/useful_objects.py` — the block starting with `@dataclasses.dataclass` above `class TrainArgs` and ending after `debug_run: bool`. After deletion, the file ends with the `StepProgress` class.

Also check whether `from pathlib import Path` is still needed in `useful_objects.py` after removing `TrainArgs`. If `Path` appears nowhere else in the file, remove that import line too.

- [ ] **Step 4: Update `helpers/data.py` — change `TrainArgs` import source**

Find the line:

```python
from helpers.useful_objects import TrainArgs
```

Change it to:

```python
from train.train_config import TrainArgs
```

- [ ] **Step 5: Update `train/train_loop.py` — move `TrainArgs` out of the `useful_objects` block**

Find the import block that looks like:

```python
from helpers.useful_objects import (
    ComponentNorms,
    EpochMetrics,
    LossMetrics,
    ModelSetup,
    StepProgress,
    ThroughputStatistics,
    TrainArgs,
)
```

Remove `TrainArgs,` from that block. Then add `TrainArgs` to the existing `from train.train_config import (...)` block in the file (or as a new standalone line if no such block exists yet), keeping imports alphabetically sorted within the group.

- [ ] **Step 6: Update `tests/helpers/test_data.py` — change `TrainArgs` import source**

Find the line:

```python
from helpers.useful_objects import TrainArgs
```

Change it to:

```python
from train.train_config import TrainArgs
```

- [ ] **Step 7: Run pre-commit and tests to verify**

```bash
cd /workspaces/diffusion && pre-commit run --all-files && python -m pytest pallatom/tests/ -x -q 2>&1 | tail -20
```

Expected: all hooks pass, all tests pass.

- [ ] **Step 8: Commit**

```bash
git add pallatom/train/train_config.py pallatom/helpers/useful_objects.py pallatom/helpers/data.py pallatom/train/train_loop.py pallatom/tests/helpers/test_data.py
git commit -m "refactor: move TrainArgs from useful_objects to train_config"
```

---

## Task 3: Absorb `batch_types.py` into `data.py`

**Files:**
- Modify: `pallatom/helpers/data.py`
- Modify: `pallatom/helpers/featurize.py:26`
- Modify: `pallatom/architecture/losses.py:18`
- Modify: `pallatom/architecture/main_trunk.py:43`
- Modify: `pallatom/sample/sampling.py:36`
- Modify: `pallatom/train/train_loop.py:35`
- Modify: `pallatom/tests/architecture/test_losses.py:27`
- Modify: `pallatom/tests/architecture/test_main_trunk.py:32`
- Modify: `pallatom/tests/sample/test_sampling.py:15`
- Delete: `pallatom/helpers/batch_types.py`

- [ ] **Step 1: Add two new imports to `data.py`**

`batch_types.py` uses `beartype` and the jaxtyping dtype classes (`Bool`, `Float`, `Int`, `jaxtyped`), which are not yet in `data.py`. Add exactly these two lines to `data.py`'s existing import block in the correct alphabetical positions within the third-party group:

```python
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
```

Do not add any other imports at this stage — adding unused imports would fail ruff's `F401` check.

- [ ] **Step 2: Insert `FeaturizedItem` and `FeaturizedBatch` into `data.py`**

Open `pallatom/helpers/batch_types.py`. Copy both class definitions verbatim: everything from the first `@jaxtyped(typechecker=beartype)` decorator through the closing line of `FeaturizedBatch.to()`.

In `pallatom/helpers/data.py`, insert this block after the import block and before `class ProteinEntry`. The content order in the file becomes:

```
[module docstring]
[import block — now includes beartype and jaxtyping]

class FeaturizedItem:      ← inserted from batch_types.py
    ...

class FeaturizedBatch:     ← inserted from batch_types.py
    ...

class ProteinEntry:        ← existing, unchanged
    ...
```

- [ ] **Step 3: Update `helpers/featurize.py` — import types from `helpers.data`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch, FeaturizedItem
```

Change it to:

```python
from helpers.data import FeaturizedBatch, FeaturizedItem
```

- [ ] **Step 4: Update `architecture/losses.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch
```

Change it to:

```python
from helpers.data import FeaturizedBatch
```

- [ ] **Step 5: Update `architecture/main_trunk.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch
```

Change it to:

```python
from helpers.data import FeaturizedBatch
```

- [ ] **Step 6: Update `sample/sampling.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch
```

Change it to:

```python
from helpers.data import FeaturizedBatch
```

- [ ] **Step 7: Update `train/train_loop.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch
```

Change it to:

```python
from helpers.data import FeaturizedBatch
```

- [ ] **Step 8: Update `tests/architecture/test_losses.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch, FeaturizedItem
```

Change it to:

```python
from helpers.data import FeaturizedBatch, FeaturizedItem
```

- [ ] **Step 9: Update `tests/architecture/test_main_trunk.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch
```

Change it to:

```python
from helpers.data import FeaturizedBatch
```

- [ ] **Step 10: Update `tests/sample/test_sampling.py`**

Find the line:

```python
from helpers.batch_types import FeaturizedBatch
```

Change it to:

```python
from helpers.data import FeaturizedBatch
```

- [ ] **Step 11: Delete `batch_types.py`**

```bash
git rm pallatom/helpers/batch_types.py
```

- [ ] **Step 12: Run pre-commit and tests**

```bash
cd /workspaces/diffusion && pre-commit run --all-files && python -m pytest pallatom/tests/ -x -q 2>&1 | tail -20
```

Expected: all hooks pass, all tests pass.

- [ ] **Step 13: Commit**

```bash
git add -u pallatom/
git commit -m "refactor: absorb batch_types into data and update all import sites"
```

---

## Task 4: Absorb `featurize.py` into `data.py`

**Files:**
- Modify: `pallatom/helpers/data.py`
- Modify: `pallatom/helpers/useful_objects.py:17`
- Modify: `pallatom/sample/sampling.py`
- Modify: `pallatom/train/train_loop.py`
- Modify: `pallatom/tests/architecture/test_losses.py`
- Modify: `pallatom/tests/architecture/test_main_trunk.py`
- Modify: `pallatom/tests/sample/test_sampling.py`
- Modify: `pallatom/tests/train/test_train_loop.py`
- Delete: `pallatom/helpers/featurize.py`

- [ ] **Step 1: Extend `data.py`'s import block with featurize's dependencies**

`featurize.py` uses several imports not yet in `data.py`. Add these to the appropriate positions in `data.py`'s import block:

```python
# stdlib (add to existing stdlib block):
import math

# third-party (add to existing third-party block):
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

# first-party (add to existing first-party block):
from architecture.atom_transformers import WINDOW_SIZE, build_sparse_pairs
```

Also expand the `from helpers.atom_utils import (...)` block to include the symbols used by featurize.py that aren't already there:

```python
from helpers.atom_utils import (
    ATOM5_CA,
    ATOM5_ELEMENTS,
    ATOM5_NAMES,
    Protein,
    atom37_to_atom5,
    atom37_to_cb,
    center_positions,       # already present
    make_fixed_size,        # already present
    make_np_example,        # already present
    restype_order,          # already present
    rigid_group_atom_positions,
    truncate_to_length,     # already present
)
```

And add `NoiseScheduleParams` to the `train.train_config` import (already imports `TrainConfig` and `TrainArgs`):

```python
from train.train_config import NoiseScheduleParams, TrainArgs, TrainConfig
```

Also add `typing_extensions.override` if not already present (`featurize.py` imports it for `Distogram`):

```python
from typing_extensions import Self, override
```

(`Self` is already there from data.py; just ensure `override` is included.)

Do not add any import that is not actually used in the pasted content — ruff will flag unused imports.

- [ ] **Step 2: Insert featurize content into `data.py`**

Open `pallatom/helpers/featurize.py`. Copy all code after the module docstring and imports — starting from `class Distogram` through the end of `apply_conditioning_dropout`.

In `pallatom/helpers/data.py`, insert this block after `FeaturizedBatch` and before `class ProteinEntry`. The content order becomes:

```
[module docstring]
[import block]

class FeaturizedItem: ...        ← from Task 3
class FeaturizedBatch: ...       ← from Task 3

class Distogram: ...             ← inserted now
def sinusoidal_encoding(...): ...
def ref_pos_for_residue(...): ...
def featurize_single_item(...): ...
def featurize_batch(...): ...
def apply_conditioning_dropout(...): ...

class ProteinEntry: ...          ← existing, unchanged
...
```

- [ ] **Step 3: Update `helpers/useful_objects.py`**

Find the line:

```python
from helpers.featurize import Distogram
```

Change it to:

```python
from helpers.data import Distogram
```

- [ ] **Step 4: Update `sample/sampling.py`**

Find the line:

```python
from helpers.featurize import Distogram, ref_pos_for_residue
```

Change it to:

```python
from helpers.data import Distogram, ref_pos_for_residue
```

- [ ] **Step 5: Update `train/train_loop.py`**

Find the block:

```python
from helpers.featurize import (
    Distogram,
    apply_conditioning_dropout,
    featurize_batch,
)
```

Change it to:

```python
from helpers.data import (
    Distogram,
    apply_conditioning_dropout,
    featurize_batch,
)
```

Then merge the `from helpers.data import FeaturizedBatch` line (added in Task 3, Step 7) into this block so there is a single `from helpers.data import (...)` block:

```python
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    apply_conditioning_dropout,
    featurize_batch,
)
```

- [ ] **Step 6: Update `tests/architecture/test_losses.py`**

Find the block:

```python
from helpers.featurize import (
    Distogram,
    apply_conditioning_dropout,
    featurize_single_item,
)
```

Change it to:

```python
from helpers.data import (
    Distogram,
    apply_conditioning_dropout,
    featurize_single_item,
)
```

- [ ] **Step 7: Update `tests/architecture/test_main_trunk.py`**

Find the line:

```python
from helpers.featurize import sinusoidal_encoding
```

Change it to:

```python
from helpers.data import sinusoidal_encoding
```

- [ ] **Step 8: Update `tests/sample/test_sampling.py`**

Find the line:

```python
from helpers.featurize import Distogram
```

Change it to:

```python
from helpers.data import Distogram
```

- [ ] **Step 9: Update `tests/train/test_train_loop.py`**

Find the line:

```python
from helpers.featurize import Distogram
```

Change it to:

```python
from helpers.data import Distogram
```

- [ ] **Step 10: Delete `featurize.py`**

```bash
git rm pallatom/helpers/featurize.py
```

- [ ] **Step 11: Run pre-commit and tests**

```bash
cd /workspaces/diffusion && pre-commit run --all-files && python -m pytest pallatom/tests/ -x -q 2>&1 | tail -20
```

Expected: all hooks pass, all tests pass.

- [ ] **Step 12: Commit**

```bash
git add -u pallatom/
git commit -m "refactor: absorb featurize into data and update all import sites"
```

---

## Task 5: Merge test files into `test_data.py`

**Context — two categories of conflict to resolve before merging:**

**1. Fixture name collisions.** `test_batch_types.py` and `test_featurize.py` each define fixtures named `protein_batch`, `featurized_item`, and `featurized_batch`. Rename the three from the `test_batch_types.py` section by appending `_raw` (they construct objects directly, without featurization logic).

**2. Module-level constant collisions.** All three files share a flat Python namespace. The constants `B`, `N_RES`, and `N_ATOM` are defined in both `test_batch_types.py` and `test_featurize.py` with different values, and `B` also conflicts with `test_data.py`'s `B = 5`. Without renaming, pytest would see only the last-assigned value of each, silently breaking earlier tests.

Rename with file-of-origin prefixes:

| Original (test_batch_types.py) | Renamed | Original (test_featurize.py) | Renamed |
|-------------------------------|---------|------------------------------|---------|
| `B = 2` | `BT_B = 2` | `B = 2` | `FZ_B = 2` |
| `N_RES = 8` | `BT_N_RES = 8` | `N_RES = 12` | keep as `N_RES` (no conflict with test_data) |
| `N_ATOM = N_RES * 5` | `BT_N_ATOM = BT_N_RES * 5` | `N_ATOM = N_RES * 5` | keep as `N_ATOM` (no conflict with test_data) |
| `N_TEMPL_BINS = 38` | `BT_N_TEMPL_BINS = 38` | — | — |
| `C_RES = 32` | `BT_C_RES = 32` | `C_RES = 32` | keep (same value, test_data doesn't use it) |
| `K = 16` | `BT_K = 16` | — | — |
| `N_ATOM_BINS = 16` | `BT_N_ATOM_BINS = 16` | — | — |

After all renames, `test_data.py`'s `B = 5` is the only final assignment to `B`, which is what test_data tests expect.

**Files:**
- Modify: `pallatom/tests/helpers/test_data.py`
- Delete: `pallatom/tests/helpers/test_batch_types.py`
- Delete: `pallatom/tests/helpers/test_featurize.py`

- [ ] **Step 1: Replace `test_data.py`'s module docstring**

```python
"""Tests for data types, featurization utilities, and dataset / data loading.

Covers FeaturizedItem and FeaturizedBatch shape contracts; Distogram shape,
one-hot, masking, symmetry, and bin-assignment properties; featurize_batch
output shapes and value contracts; apply_conditioning_dropout behaviour;
sinusoidal_encoding and ref_pos_for_residue correctness; ProteinDataset
length/indexing; and make_bucketed_data_loaders behaviour.
"""
```

- [ ] **Step 2: Replace `test_data.py`'s import block with the merged import block**

Replace the existing import block with the deduplicated union of all three files' imports:

```python
import dataclasses
import json
import math
import pathlib
import pickle
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from architecture.main_trunk import MainTrunk
from beartype import beartype
from einops import rearrange, reduce, repeat
from helpers.atom_utils import RESTYPE_NUM_NO_X, Protein, restype_order
from helpers.data import (
    DatasetSplitsManifest,
    Distogram,
    FeaturizedBatch,
    FeaturizedItem,
    ProteinDataset,
    ProteinShardDataset,
    ShardBudgetParameters,
    ShardDataLoader,
    ShardMetadata,
    apply_conditioning_dropout,
    featurize_batch,
    featurize_single_item,
    make_bucketed_data_loaders,
    ref_pos_for_residue,
    sinusoidal_encoding,
)
from helpers.useful_objects import ModelSetup, manual_seed
from jaxtyping import Bool, Float, TypeCheckError
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from train.train_config import (
    LoaderConfig as EvalLoaderConfig,
    TrainArgs,
    TrainConfig,
    TrainLoaderConfig,
)
```

- [ ] **Step 3: Insert the `test_batch_types.py` block with renamed constants and fixtures**

Open `pallatom/tests/helpers/test_batch_types.py`. Copy all content after the module docstring and import block (starting from `B = 2` through the end of the file).

Paste this block into `test_data.py` after the import block and before `test_data.py`'s own constants (`_N_RES_DATA = 6`, etc.).

Apply these renames throughout the pasted block using editor search-and-replace scoped to this block only (do not change the rest of the file):

**Constants:**

| Find (whole word) | Replace |
|-------------------|---------|
| `B` | `BT_B` |
| `N_RES` | `BT_N_RES` |
| `N_ATOM` | `BT_N_ATOM` |
| `N_TEMPL_BINS` | `BT_N_TEMPL_BINS` |
| `C_RES` | `BT_C_RES` |
| `K` | `BT_K` |
| `N_ATOM_BINS` | `BT_N_ATOM_BINS` |

**Fixtures and their injecting test functions:**

| Find | Replace |
|------|---------|
| `def protein_batch(` | `def protein_batch_raw(` |
| `def featurized_item(` | `def featurized_item_raw(` |
| `def featurized_batch(` | `def featurized_batch_raw(` |
| `def test_protein_batch_constructs(protein_batch:` | `def test_protein_batch_constructs(protein_batch_raw:` |
| `def test_featurized_item_constructs(featurized_item:` | `def test_featurized_item_constructs(featurized_item_raw:` |
| `def test_featurized_batch_valid_construction(\n    featurized_batch:` | `def test_featurized_batch_valid_construction(\n    featurized_batch_raw:` |

- [ ] **Step 4: Insert the `test_featurize.py` block with renamed `B` and `protein_batch`**

Open `pallatom/tests/helpers/test_featurize.py`. Copy all content after the module docstring and import block (starting from `_ = manual_seed(42)` through the end of the file).

Paste this block into `test_data.py` after the `test_batch_types.py` block and before `test_data.py`'s own constants.

Apply these renames throughout the pasted block only:

**Constants:**

| Find (whole word) | Replace |
|-------------------|---------|
| `B` | `FZ_B` |

(`N_RES = 12` and `N_ATOM = N_RES * 5` keep their names — they don't conflict with `test_data.py`.)

**Fixtures:**

| Find | Replace |
|------|---------|
| `def protein_batch(single_protein` | `def featurize_protein_batch(single_protein` |

Update the `featurized_batch` fixture that depends on it:

```python
# Before:
@pytest.fixture
def featurized_batch(
    protein_batch: list[Protein],
    model_params: ModelSetup,
) -> FeaturizedBatch:
    ...
    return featurize_batch(
        batch=protein_batch,
        ...
    )

# After:
@pytest.fixture
def featurized_batch(
    featurize_protein_batch: list[Protein],
    model_params: ModelSetup,
) -> FeaturizedBatch:
    ...
    return featurize_batch(
        batch=featurize_protein_batch,
        ...
    )
```

- [ ] **Step 5: Delete the two old test files**

```bash
git rm pallatom/tests/helpers/test_batch_types.py pallatom/tests/helpers/test_featurize.py
```

- [ ] **Step 6: Run pre-commit and the full test suite**

```bash
cd /workspaces/diffusion && pre-commit run --all-files && python -m pytest pallatom/tests/ -x -q 2>&1 | tail -20
```

Expected: all hooks pass, all tests pass.

- [ ] **Step 7: Commit**

```bash
git add -u pallatom/tests/helpers/
git commit -m "refactor: merge test_batch_types and test_featurize into test_data"
```

---

## Task 6: Final verification

- [ ] **Step 1: Confirm no stale references to the deleted modules**

```bash
grep -r "helpers\.batch_types\|helpers\.featurize\|from helpers.batch_types\|from helpers.featurize" \
  /workspaces/diffusion/pallatom --include="*.py" | grep -v __pycache__
```

Expected: no output (zero matches).

- [ ] **Step 2: Confirm the deleted files are gone**

```bash
ls /workspaces/diffusion/pallatom/helpers/batch_types.py \
   /workspaces/diffusion/pallatom/helpers/featurize.py 2>&1
```

Expected: `No such file or directory` for both.

- [ ] **Step 3: Run the full test suite one final time**

```bash
cd /workspaces/diffusion && pre-commit run --all-files && python -m pytest pallatom/tests/ -v 2>&1 | tail -30
```

Expected: all hooks pass, all tests pass.
