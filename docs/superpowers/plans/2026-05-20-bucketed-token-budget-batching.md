# Bucketed Token-Budget Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-size padded `ProteinDataset` with a bucketed token-budget batching system that packs 512-residue batches with near-zero padding, and adds pack-rate / throughput metrics to `train_step`.

**Architecture:** `ClusterIndex` partitions proteins into 64 length buckets and caches them as per-cluster JSONL files. `ClusteredProteinDataset` wraps `ClusterIndex` with flat-integer `__getitem__` returning variable-length items. `BucketedBatchSampler` uses a `ProcessPoolExecutor` queue to precompute sortish-sampled batch plans one epoch ahead. The training DataLoader receives `batch_sampler=` (not `batch_size=`); collate pads dynamically to within-batch max length.

**Tech Stack:** Python 3.10, PyTorch, jaxtyping + beartype, einops, pytest, pydantic, `concurrent.futures.ProcessPoolExecutor`

**Spec:** `docs/superpowers/specs/2026-05-20-bucketed-token-budget-batching-design.md`

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `pallatom/helpers/cluster_index.py` | **NEW** | `ClusterIndex` — partitions source JSONL into 65 cluster files, builds byte-offset arrays |
| `pallatom/helpers/bucketed_sampler.py` | **NEW** | `_compute_batch_plan` (picklable), `BucketedBatchSampler` with process-pool prefetch queue |
| `pallatom/helpers/data.py` | **MOD** | Add `ClusteredProteinDataset`, `_to_protein_batch_dynamic`, `make_bucketed_data_loaders`, `make_ddp_bucketed_data_loaders` |
| `pallatom/helpers/atom_utils.py` | **MOD** | Add `truncate_to_length` |
| `pallatom/train/train_config.py` | **MOD** | Add `token_budget` field to `LoaderConfig` |
| `pallatom/train/train_loop.py` | **MOD** | Add timing + pack metrics to `train_step`; swap loaders; fix `set_epoch` cast |
| `pallatom/tests/helpers/test_cluster_index.py` | **NEW** | Tests for `ClusterIndex` |
| `pallatom/tests/helpers/test_bucketed_sampler.py` | **NEW** | Tests for `_compute_batch_plan` and `BucketedBatchSampler` |
| `pallatom/tests/helpers/test_data.py` | **MOD** | Add tests for `ClusteredProteinDataset` and `_to_protein_batch_dynamic` |
| `pallatom/tests/train/test_train_loop.py` | **MOD** | Add `protein_batch` fixture, `EXPECTED_STEP_KEYS`, and three new `train_step` tests |

---

## Task 1: Pack-rate and throughput metrics in `train_step`

**Files:**
- Modify: `pallatom/train/train_loop.py`
- Test: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Write the failing tests**

Add to `pallatom/tests/train/test_train_loop.py`, after the existing `EXPECTED_EVAL_KEYS` constant and before the `_ListDataset` class:

```python
from einops import rearrange  # already imported — no change needed
from torch.optim import Adam  # already imported — no change needed

EXPECTED_STEP_KEYS = frozenset(
    {
        "total loss",
        "Kabsch aligned MSE loss",
        "Cross Entropy loss",
        "smooth lddt",
        "Residue Distogram loss",
        "Atom Distogram loss",
        "Intermediate loss",
        "pack_rate",
        "residues_per_sec",
        "atoms_per_sec",
    }
)
```

Then add a new `protein_batch` fixture and three test functions after the existing `tcfg_save` fixture:

```python
@pytest.fixture
def protein_batch() -> ProteinBatch:
    """Provide a single-item ProteinBatch with _N_KEEP residues for train_step tests."""
    return ProteinBatch(
        atom_positions=torch.randn(1, _N_KEEP, 37, 3),
        atom_mask=torch.ones(1, _N_KEEP, 37),
        residue_index=rearrange(torch.arange(_N_KEEP, dtype=torch.float32), "n -> 1 n"),
        seq=[("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP]],
    )


def test_train_step_returns_expected_keys(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """train_step return dict contains exactly EXPECTED_STEP_KEYS."""
    optimizer = Adam(model.parameters(), lr=1e-4)
    metrics, _ = train_step(
        protein_batch, model, tcfg, distogram_res, distogram_atom, optimizer, "cpu"
    )
    assert set(metrics.keys()) == EXPECTED_STEP_KEYS


def test_train_step_pack_rate_in_range(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """pack_rate is in (0, 1]."""
    optimizer = Adam(model.parameters(), lr=1e-4)
    metrics, _ = train_step(
        protein_batch, model, tcfg, distogram_res, distogram_atom, optimizer, "cpu"
    )
    assert 0.0 < metrics["pack_rate"] <= 1.0


def test_train_step_throughput_metrics_positive(
    protein_batch: ProteinBatch,
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> None:
    """residues_per_sec and atoms_per_sec are positive floats."""
    optimizer = Adam(model.parameters(), lr=1e-4)
    metrics, _ = train_step(
        protein_batch, model, tcfg, distogram_res, distogram_atom, optimizer, "cpu"
    )
    assert metrics["residues_per_sec"] > 0.0
    assert metrics["atoms_per_sec"] > 0.0
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/test_train_loop.py::test_train_step_returns_expected_keys pallatom/tests/train/test_train_loop.py::test_train_step_pack_rate_in_range pallatom/tests/train/test_train_loop.py::test_train_step_throughput_metrics_positive -v
```

Expected: **FAILED** — `KeyError: 'pack_rate'` (keys not yet in return dict).

- [ ] **Step 3: Implement timing and metrics in `train_step`**

In `pallatom/train/train_loop.py`, add `import time` after the existing `import math` line (near the top of the file).

Then in `train_step`, wrap the forward + loss + backward block in a timer and append the three metrics. The current return at lines 383–391 reads:

```python
    return {
        "total loss": total_loss.item(),
        "Kabsch aligned MSE loss": Kabsch_aligned_MSE_loss.item(),
        "Cross Entropy loss": CE_loss.item(),
        "smooth lddt": lddt_loss.item(),
        "Residue Distogram loss": residue_distogram_loss.item(),
        "Atom Distogram loss": atom_distogram_loss.item(),
        "Intermediate loss": intermediate_med_loss.item(),
    }, grad_norm
```

Replace it with:

```python
    t1 = time.perf_counter()
    step_time = t1 - t0

    b_size, n_res = featurized_batch.residue_mask.shape
    actual_residues = int(featurized_batch.residue_mask.sum().item())
    actual_atoms = int(featurized_batch.atom5_mask.sum().item())

    return {
        "total loss": total_loss.item(),
        "Kabsch aligned MSE loss": Kabsch_aligned_MSE_loss.item(),
        "Cross Entropy loss": CE_loss.item(),
        "smooth lddt": lddt_loss.item(),
        "Residue Distogram loss": residue_distogram_loss.item(),
        "Atom Distogram loss": atom_distogram_loss.item(),
        "Intermediate loss": intermediate_med_loss.item(),
        "pack_rate": actual_residues / (b_size * n_res),
        "residues_per_sec": actual_residues / step_time,
        "atoms_per_sec": actual_atoms / step_time,
    }, grad_norm
```

And insert `t0 = time.perf_counter()` immediately before the `model(featurized_batch)` call (line 306 area). The block to wrap is from the model call through `optimizer.step()`.

Also add the three new keys to `epoch_metrics` in the `train()` function (around line 533) and in `train_ddp()` (around line 612). Each currently reads:

```python
        epoch_metrics: dict[str, float] = dict.fromkeys(
            [
                "total loss",
                "Kabsch aligned MSE loss",
                "Cross Entropy loss",
                "smooth lddt",
                "Residue Distogram loss",
                "Atom Distogram loss",
                "Intermediate loss",
            ],
            0.0,
        )
```

Change to:

```python
        epoch_metrics: dict[str, float] = dict.fromkeys(
            [
                "total loss",
                "Kabsch aligned MSE loss",
                "Cross Entropy loss",
                "smooth lddt",
                "Residue Distogram loss",
                "Atom Distogram loss",
                "Intermediate loss",
                "pack_rate",
                "residues_per_sec",
                "atoms_per_sec",
            ],
            0.0,
        )
```

Apply this change in **both** `train()` and `train_ddp()`.

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/test_train_loop.py::test_train_step_returns_expected_keys pallatom/tests/train/test_train_loop.py::test_train_step_pack_rate_in_range pallatom/tests/train/test_train_loop.py::test_train_step_throughput_metrics_positive -v
```

Expected: **PASSED** (all three).

- [ ] **Step 5: Run full test suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: add pack_rate and throughput metrics to train_step"
```

---

## Task 2: `truncate_to_length` in `atom_utils.py`

**Files:**
- Modify: `pallatom/helpers/atom_utils.py`
- Test: `pallatom/tests/helpers/test_atom_utils.py`

- [ ] **Step 1: Write the failing tests**

Add to `pallatom/tests/helpers/test_atom_utils.py`, after the existing imports, add `truncate_to_length` to the import line for `atom_utils`:

```python
from helpers.atom_utils import (
    ...
    make_np_example,
    truncate_to_length,   # add this
    ...
)
```

Then add these test functions (append to the file):

```python
def test_truncate_to_length_shortens_all_arrays() -> None:
    """truncate_to_length truncates all arrays to max_length along axis 0."""
    np_example: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((100, 37, 3)),
        "atom_mask": np.zeros((100, 37)),
        "residue_index": np.arange(100),
    }
    truncate_to_length(np_example, 50)
    assert np_example["atom_positions"].shape == (50, 37, 3)
    assert np_example["atom_mask"].shape == (50, 37)
    assert np_example["residue_index"].shape == (50,)


def test_truncate_to_length_noop_when_already_short() -> None:
    """truncate_to_length leaves arrays unchanged when they are shorter than max_length."""
    np_example: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((30, 37, 3)),
    }
    truncate_to_length(np_example, 50)
    assert np_example["atom_positions"].shape == (30, 37, 3)


def test_truncate_to_length_noop_when_exact() -> None:
    """truncate_to_length leaves arrays unchanged when they are exactly max_length."""
    np_example: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]] = {
        "atom_positions": np.zeros((50, 37, 3)),
    }
    truncate_to_length(np_example, 50)
    assert np_example["atom_positions"].shape == (50, 37, 3)
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_atom_utils.py::test_truncate_to_length_shortens_all_arrays pallatom/tests/helpers/test_atom_utils.py::test_truncate_to_length_noop_when_already_short pallatom/tests/helpers/test_atom_utils.py::test_truncate_to_length_noop_when_exact -v
```

Expected: **FAILED** — `ImportError: cannot import name 'truncate_to_length'`.

- [ ] **Step 3: Implement `truncate_to_length`**

In `pallatom/helpers/atom_utils.py`, add the following function directly after `make_fixed_size` (around line 419):

```python
def truncate_to_length(
    np_example: Mapping[str, npt.NDArray[np.float64] | npt.NDArray[np.intp]],
    max_length: int,
) -> None:
    """Truncate features to at most max_length along axis 0. Does not pad.

    Args:
        np_example: Dict of numpy arrays all sharing the same axis-0 size.
        max_length: Maximum allowed length along axis 0.
    """
    for k, v in np_example.items():
        if v.shape[0] > max_length:
            np_example[k] = v[:max_length]  # type: ignore[index]
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_atom_utils.py::test_truncate_to_length_shortens_all_arrays pallatom/tests/helpers/test_atom_utils.py::test_truncate_to_length_noop_when_already_short pallatom/tests/helpers/test_atom_utils.py::test_truncate_to_length_noop_when_exact -v
```

Expected: **PASSED**.

- [ ] **Step 5: Commit**

```bash
git add pallatom/helpers/atom_utils.py pallatom/tests/helpers/test_atom_utils.py
git commit -m "feat: add truncate_to_length to atom_utils"
```

---

## Task 3: `ClusterIndex`

**Files:**
- Create: `pallatom/helpers/cluster_index.py`
- Create: `pallatom/tests/helpers/test_cluster_index.py`

- [ ] **Step 1: Create the test file**

Create `pallatom/tests/helpers/test_cluster_index.py`:

```python
"""Tests for ClusterIndex."""

import json
import pathlib

import pytest
from helpers.cluster_index import ClusterIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coords(n: int) -> dict[str, list[list[float]]]:
    return {atom: [[0.0, 0.0, 0.0]] * n for atom in ("N", "CA", "C", "O")}


def _write_jsonl(path: pathlib.Path, entries: list[dict[str, object]]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# assign_cluster
# ---------------------------------------------------------------------------


def test_assign_cluster_first_bin(tmp_path: pathlib.Path) -> None:
    """Length 1 maps to cluster 0 (first bin [1-8])."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.assign_cluster(1) == 0


def test_assign_cluster_last_regular_bin(tmp_path: pathlib.Path) -> None:
    """Length 512 maps to cluster 63 (last regular bin [505-512])."""
    seq = "A" * 512
    _write_jsonl(
        tmp_path / "p.jsonl",
        [{"name": "p1", "seq": seq, "coords": _make_coords(512)}],
    )
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.assign_cluster(512) == 63


def test_assign_cluster_overflow(tmp_path: pathlib.Path) -> None:
    """Length > 512 maps to overflow cluster 64."""
    seq = "A" * 513
    _write_jsonl(
        tmp_path / "p.jsonl",
        [{"name": "p1", "seq": seq, "coords": _make_coords(513)}],
    )
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.assign_cluster(513) == 64


# ---------------------------------------------------------------------------
# cluster_rep_len
# ---------------------------------------------------------------------------


def test_cluster_rep_len_regular(tmp_path: pathlib.Path) -> None:
    """cluster_rep_len[0] == 8, cluster_rep_len[63] == 512."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.cluster_rep_len[0] == 8
    assert idx.cluster_rep_len[63] == 512


def test_cluster_rep_len_overflow(tmp_path: pathlib.Path) -> None:
    """cluster_rep_len[64] == token_budget + 1 == 513."""
    _write_jsonl(tmp_path / "p.jsonl", [{"name": "p1", "seq": "A", "coords": _make_coords(1)}])
    idx = ClusterIndex(tmp_path / "p.jsonl", ["p1"], token_budget=512, n_clusters=64)
    assert idx.cluster_rep_len[64] == 513


# ---------------------------------------------------------------------------
# file creation and flat index
# ---------------------------------------------------------------------------


def test_cluster_index_creates_cluster_dir(tmp_path: pathlib.Path) -> None:
    """ClusterIndex creates the cluster directory and 65 JSONL files."""
    entries = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p2", "seq": "A" * 16, "coords": _make_coords(16)},
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2"])
    cluster_dir = tmp_path / "proteins_clusters"
    assert cluster_dir.is_dir()
    for k in range(65):
        assert (cluster_dir / f"cluster_{k:03d}.jsonl").exists()


def test_cluster_index_flat_to_cluster_correct(tmp_path: pathlib.Path) -> None:
    """flat_to_cluster assigns each protein to the expected bucket."""
    entries = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},   # → cluster 0
        {"name": "p2", "seq": "A" * 9, "coords": _make_coords(9)},   # → cluster 1
        {"name": "p3", "seq": "A" * 513, "coords": _make_coords(513)},  # → overflow 64
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2", "p3"])
    cluster_ids = sorted(
        idx.flat_to_cluster[i] for i in range(len(idx))
    )
    assert cluster_ids == [0, 1, 64]


def test_cluster_index_len(tmp_path: pathlib.Path) -> None:
    """__len__ returns the total number of included proteins."""
    entries = [
        {"name": f"p{i}", "seq": "A" * (i + 1), "coords": _make_coords(i + 1)}
        for i in range(10)
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx = ClusterIndex(
        tmp_path / "proteins.jsonl", [f"p{i}" for i in range(10)]
    )
    assert len(idx) == 10


def test_cluster_index_name_filter(tmp_path: pathlib.Path) -> None:
    """Only names in the provided list are included."""
    entries = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p2", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p3", "seq": "A" * 8, "coords": _make_coords(8)},
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p3"])
    assert len(idx) == 2


# ---------------------------------------------------------------------------
# cache hit
# ---------------------------------------------------------------------------


def test_cluster_index_cache_hit_same_result(tmp_path: pathlib.Path) -> None:
    """Second construction from the same path reads cache and produces identical arrays."""
    entries = [
        {"name": "p1", "seq": "A" * 8, "coords": _make_coords(8)},
        {"name": "p2", "seq": "A" * 16, "coords": _make_coords(16)},
    ]
    _write_jsonl(tmp_path / "proteins.jsonl", entries)
    idx1 = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2"])
    idx2 = ClusterIndex(tmp_path / "proteins.jsonl", ["p1", "p2"])
    assert idx1.flat_to_cluster == idx2.flat_to_cluster
    assert idx1.flat_to_local == idx2.flat_to_local
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_cluster_index.py -v
```

Expected: **ERROR** — `ModuleNotFoundError: No module named 'helpers.cluster_index'`.

- [ ] **Step 3: Implement `ClusterIndex`**

Create `pallatom/helpers/cluster_index.py`:

```python
"""ClusterIndex: partitions a protein JSONL into 64 length-based cluster files."""

import json
from collections.abc import Mapping
from pathlib import Path


class ClusterIndex:
    """Partitions proteins from a source JSONL into per-length-cluster JSONL files.

    On construction, checks whether cluster files already exist. If not, scans the
    source JSONL, assigns each included protein to one of 64 regular clusters
    (width = token_budget // n_clusters residues each) or an overflow cluster, writes
    per-cluster JSONL files, and builds byte-offset arrays. If the files exist, reads
    them to rebuild the offset arrays without re-parsing protein data.

    The cluster directory is derived from the source path:
        <source_stem>_clusters/ sibling to <source>.jsonl

    Args:
        jsonl_path:   Path to the source JSONL protein dataset.
        names:        Names (entry["name"]) of proteins to include.
        token_budget: Maximum residues per batch; defines the upper edge of the last
                      regular cluster. Default 512.
        n_clusters:   Number of regular length clusters. Default 64.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        names: list[str],
        token_budget: int = 512,
        n_clusters: int = 64,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        self._names = names
        self._token_budget = token_budget
        self._n_clusters = n_clusters
        self._cluster_dir = self._jsonl_path.parent / (self._jsonl_path.stem + "_clusters")

        # Representative lengths: regular clusters have rep_len = bin_width * (k+1).
        # Overflow cluster (index n_clusters) has rep_len = token_budget + 1 so that
        # the greedy-packing overflow check `rep_len > token_budget` fires correctly.
        bin_width = token_budget // n_clusters
        self.cluster_rep_len: list[int] = [
            bin_width * (k + 1) for k in range(n_clusters)
        ] + [token_budget + 1]

        self.flat_to_cluster: list[int] = []
        self.flat_to_local: list[int] = []
        self.cluster_offsets: list[list[int]] = []

        if self._cache_exists():
            self._load_from_cache()
        else:
            self._build_and_cache()

    # ------------------------------------------------------------------
    # Public helpers

    def assign_cluster(self, seq_len: int) -> int:
        """Return cluster id (0..n_clusters-1 regular, n_clusters overflow) for seq_len.

        Args:
            seq_len: Number of residues in the protein.

        Returns:
            Cluster id in [0, n_clusters].
        """
        if seq_len > self._token_budget:
            return self._n_clusters
        bin_width = self._token_budget // self._n_clusters
        return min((seq_len - 1) // bin_width, self._n_clusters - 1)

    def cluster_file(self, k: int) -> Path:
        """Return the path to cluster k's JSONL file.

        Args:
            k: Cluster id (0..n_clusters).

        Returns:
            Path to cluster_k.jsonl inside the cluster directory.
        """
        return self._cluster_dir / f"cluster_{k:03d}.jsonl"

    def __len__(self) -> int:
        """Return the total number of included proteins across all clusters."""
        return len(self.flat_to_cluster)

    # ------------------------------------------------------------------
    # Cache management

    def _cache_exists(self) -> bool:
        return all(self.cluster_file(k).exists() for k in range(self._n_clusters + 1))

    def _build_and_cache(self) -> None:
        """Partition source JSONL by length, write cluster files, build offset arrays."""
        self._cluster_dir.mkdir(parents=True, exist_ok=True)
        name_set = set(self._names)

        # Collect raw lines per cluster without loading full protein data.
        cluster_lines: list[list[bytes]] = [[] for _ in range(self._n_clusters + 1)]
        with open(self._jsonl_path, "rb") as f:
            for raw_line in f:
                entry: Mapping[str, object] = json.loads(raw_line)
                if entry["name"] not in name_set:  # type: ignore[operator]
                    continue
                seq_len = len(entry["seq"])  # type: ignore[arg-type]
                k = self.assign_cluster(seq_len)
                cluster_lines[k].append(raw_line)

        # Write cluster files and record byte offsets.
        self.cluster_offsets = []
        for k in range(self._n_clusters + 1):
            offsets: list[int] = []
            with open(self.cluster_file(k), "wb") as f:
                pos = 0
                for raw_line in cluster_lines[k]:
                    offsets.append(pos)
                    f.write(raw_line)
                    pos += len(raw_line)
            self.cluster_offsets.append(offsets)

        self._build_flat_index()

    def _load_from_cache(self) -> None:
        """Read cluster files to rebuild byte-offset arrays. Does not parse protein data."""
        self.cluster_offsets = []
        for k in range(self._n_clusters + 1):
            offsets: list[int] = []
            byte_pos = 0
            with open(self.cluster_file(k), "rb") as f:
                for raw_line in f:
                    offsets.append(byte_pos)
                    byte_pos += len(raw_line)
            self.cluster_offsets.append(offsets)

        self._build_flat_index()

    def _build_flat_index(self) -> None:
        """Populate flat_to_cluster and flat_to_local from cluster_offsets."""
        self.flat_to_cluster = []
        self.flat_to_local = []
        for k, offsets in enumerate(self.cluster_offsets):
            for local_idx in range(len(offsets)):
                self.flat_to_cluster.append(k)
                self.flat_to_local.append(local_idx)
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_cluster_index.py -v
```

Expected: **PASSED** (all tests).

- [ ] **Step 5: Run full suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pallatom/helpers/cluster_index.py pallatom/tests/helpers/test_cluster_index.py
git commit -m "feat: implement ClusterIndex with 64-bucket length partitioning"
```

---

## Task 4: `token_budget` in `LoaderConfig`

**Files:**
- Modify: `pallatom/train/train_config.py`
- Test: `pallatom/tests/train/test_train_config.py`

- [ ] **Step 1: Write the failing test**

Open `pallatom/tests/train/test_train_config.py` and append:

```python
def test_loader_config_default_token_budget() -> None:
    """LoaderConfig.token_budget defaults to 512."""
    from train.train_config import LoaderConfig
    cfg = LoaderConfig()
    assert cfg.token_budget == 512


def test_loader_config_custom_token_budget() -> None:
    """LoaderConfig accepts a custom token_budget."""
    from train.train_config import LoaderConfig
    cfg = LoaderConfig(token_budget=256)
    assert cfg.token_budget == 256
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/test_train_config.py::test_loader_config_default_token_budget pallatom/tests/train/test_train_config.py::test_loader_config_custom_token_budget -v
```

Expected: **FAILED** — `ValidationError` or `AttributeError` (field does not exist).

- [ ] **Step 3: Add `token_budget` to `LoaderConfig`**

In `pallatom/train/train_config.py`, update `LoaderConfig`:

```python
class LoaderConfig(BaseModel):
    """DataLoader sequence length cap, batch size, and token budget."""

    model_config = ConfigDict(frozen=True)

    max_seq_length: int = Field(default=128, gt=0)
    batch_size: int = Field(default=2, gt=0)
    token_budget: int = Field(default=512, gt=0)
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/train/test_train_config.py -v
```

Expected: **PASSED** (all tests including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add pallatom/train/train_config.py pallatom/tests/train/test_train_config.py
git commit -m "feat: add token_budget field to LoaderConfig"
```

---

## Task 5: `ClusteredProteinDataset`

**Files:**
- Modify: `pallatom/helpers/data.py`
- Modify: `pallatom/tests/helpers/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `pallatom/tests/helpers/test_data.py`. First, add imports at the top of the file (add to the existing import block):

```python
import pathlib

from helpers.cluster_index import ClusterIndex
from helpers.data import ClusteredProteinDataset, _to_protein_batch_dynamic
```

Add a helper at module level (after `_make_coords`):

```python
def _make_entry(name: str, seq_len: int) -> dict[str, object]:
    """Build a minimal JSONL entry with the given name and sequence length."""
    return {
        "name": name,
        "seq": "A" * seq_len,
        "coords": _make_coords(seq_len),
    }


def _write_jsonl(path: pathlib.Path, entries: list[dict[str, object]]) -> None:
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
```

Then add tests:

```python
# ---------------------------------------------------------------------------
# ClusteredProteinDataset
# ---------------------------------------------------------------------------


def test_clustered_dataset_len(tmp_path: pathlib.Path) -> None:
    """ClusteredProteinDataset.__len__ returns the number of included proteins."""
    entries = [_make_entry(f"p{i}", 8) for i in range(5)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", [f"p{i}" for i in range(5)])
    assert len(ds) == 5


def test_clustered_dataset_item_keys(tmp_path: pathlib.Path) -> None:
    """__getitem__ returns a dict with atom_positions, atom_mask, residue_index, seq."""
    entries = [_make_entry("p1", 10)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1"])
    item = ds[0]
    assert set(item.keys()) == {"atom_positions", "atom_mask", "residue_index", "seq"}


def test_clustered_dataset_variable_lengths(tmp_path: pathlib.Path) -> None:
    """Items have their actual length, not a fixed padded length."""
    entries = [_make_entry("p1", 8), _make_entry("p2", 16), _make_entry("p3", 32)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1", "p2", "p3"])
    lengths = {ds[i]["atom_positions"].shape[0] for i in range(len(ds))}
    assert lengths == {8, 16, 32}


def test_clustered_dataset_truncates_to_budget(tmp_path: pathlib.Path) -> None:
    """Proteins longer than token_budget are truncated to token_budget."""
    entries = [_make_entry("p1", 600)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1"], token_budget=512)
    assert ds[0]["atom_positions"].shape[0] == 512


def test_clustered_dataset_pickles(tmp_path: pathlib.Path) -> None:
    """ClusteredProteinDataset survives pickle round-trip (for DataLoader workers)."""
    entries = [_make_entry("p1", 8)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1"])
    restored = pickle.loads(pickle.dumps(ds))
    item = restored[0]
    assert item["atom_positions"].shape[0] == 8


# ---------------------------------------------------------------------------
# _to_protein_batch_dynamic
# ---------------------------------------------------------------------------


def test_to_protein_batch_dynamic_pads_to_max(tmp_path: pathlib.Path) -> None:
    """_to_protein_batch_dynamic pads shorter items to the longest item in the batch."""
    entries = [_make_entry("p1", 8), _make_entry("p2", 16)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1", "p2"])
    batch = _to_protein_batch_dynamic([ds[0], ds[1]])
    assert batch.atom_positions.shape == (2, 16, 37, 3)
    assert batch.atom_mask.shape == (2, 16, 37)
    assert batch.residue_index.shape == (2, 16)


def test_to_protein_batch_dynamic_uniform_lengths(tmp_path: pathlib.Path) -> None:
    """_to_protein_batch_dynamic is a no-op when all items have the same length."""
    entries = [_make_entry("p1", 10), _make_entry("p2", 10)]
    _write_jsonl(tmp_path / "p.jsonl", entries)
    ds = ClusteredProteinDataset(tmp_path / "p.jsonl", ["p1", "p2"])
    batch = _to_protein_batch_dynamic([ds[0], ds[1]])
    assert batch.atom_positions.shape == (2, 10, 37, 3)
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_data.py::test_clustered_dataset_len pallatom/tests/helpers/test_data.py::test_clustered_dataset_item_keys pallatom/tests/helpers/test_data.py::test_clustered_dataset_variable_lengths -v
```

Expected: **FAILED** — `ImportError: cannot import name 'ClusteredProteinDataset'`.

- [ ] **Step 3: Implement `ClusteredProteinDataset` and `_to_protein_batch_dynamic`**

In `pallatom/helpers/data.py`, add these imports at the top (after existing imports):

```python
import torch.nn.functional as F  # noqa: N812
from helpers.atom_utils import truncate_to_length
from helpers.cluster_index import ClusterIndex
```

Then add the class and collate function after `ProteinDataset`:

```python
class ClusteredProteinDataset(torch.utils.data.Dataset[Mapping[str, torch.Tensor | str]]):
    """Lazy-loading Dataset backed by per-cluster JSONL files built by ClusterIndex.

    At construction, builds or loads a ClusterIndex (writing 65 cluster files the first
    time, reading cached files thereafter). __getitem__ seeks directly into the correct
    cluster file and returns the protein at its actual residue count — no padding. Items
    are variable-length; padding is deferred to the collate function.

    Compatible with num_workers > 0: all file handles are excluded from pickling and
    re-opened lazily inside each worker process.

    Args:
        jsonl_path:   Path to the source JSONL protein dataset.
        names:        List of entry names to include.
        token_budget: Hard truncation ceiling; proteins longer than this are truncated.
                      Default 512.
        n_clusters:   Number of regular length clusters. Default 64.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        names: list[str],
        token_budget: int = 512,
        n_clusters: int = 64,
    ) -> None:
        self.token_budget = token_budget
        self.cluster_index = ClusterIndex(jsonl_path, names, token_budget, n_clusters)
        # One file handle per cluster file (lazily opened).
        self._files: list[io.BufferedReader | None] = [None] * (n_clusters + 1)

    # ------------------------------------------------------------------
    # File-handle lifecycle — excluded from pickle so multiprocessing works

    def _open(self, k: int) -> None:
        if self._files[k] is None:
            self._files[k] = open(self.cluster_index.cluster_file(k), "rb")  # noqa: SIM115

    def __getstate__(self) -> dict[str, object]:
        """Return picklable state with all open file handles set to None."""
        state = self.__dict__.copy()
        state["_files"] = [None] * len(self._files)
        return state

    def __del__(self) -> None:
        """Close all open cluster file handles on deletion."""
        for f in self._files:
            if f is not None:
                f.close()

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of proteins across all clusters."""
        return len(self.cluster_index)

    def __getitem__(self, idx: int) -> Mapping[str, Float[torch.Tensor, "..."] | str]:
        """Return the protein at flat index idx at its actual (un-padded) length.

        Args:
            idx: Flat dataset index in [0, len(self)).

        Returns:
            Dict with atom_positions (N, 37, 3), atom_mask (N, 37),
            residue_index (N,), and seq (str), where N <= token_budget.
        """
        cluster_id = self.cluster_index.flat_to_cluster[idx]
        local_idx = self.cluster_index.flat_to_local[idx]
        offset = self.cluster_index.cluster_offsets[cluster_id][local_idx]

        self._open(cluster_id)
        self._files[cluster_id].seek(offset)  # type: ignore[union-attr]
        entry = json.loads(self._files[cluster_id].readline())  # type: ignore[union-attr]

        np_example = make_np_example(entry["coords"])
        center_positions(np_example)
        truncate_to_length(np_example, self.token_budget)

        sample = {k: torch.tensor(v, dtype=torch.float32) for k, v in np_example.items()}
        sample["seq"] = entry["seq"][: self.token_budget]
        return sample


def _to_protein_batch_dynamic(
    samples: list[Mapping[str, Float[torch.Tensor, "..."] | str]],
) -> ProteinBatch:
    """Collate variable-length protein samples into a ProteinBatch, padding to batch max.

    Pads each sample's tensors along axis 0 to the longest sample in the batch.
    Within a length-bucketed batch, all samples are within one bucket width of each
    other, so padding is near-zero.

    Args:
        samples: List of per-protein dicts from ClusteredProteinDataset.__getitem__.

    Returns:
        ProteinBatch with tensors of shape (B, max_len, ...).
    """
    max_len = max(cast("torch.Tensor", s["atom_positions"]).shape[0] for s in samples)
    padded_positions: list[torch.Tensor] = []
    padded_mask: list[torch.Tensor] = []
    padded_residue_index: list[torch.Tensor] = []
    seqs: list[str] = []

    for s in samples:
        pos = cast("torch.Tensor", s["atom_positions"])   # (n, 37, 3)
        mask = cast("torch.Tensor", s["atom_mask"])        # (n, 37)
        ridx = cast("torch.Tensor", s["residue_index"])   # (n,)
        n = pos.shape[0]
        pad = max_len - n
        if pad > 0:
            pos = F.pad(pos, (0, 0, 0, 0, 0, pad))
            mask = F.pad(mask, (0, 0, 0, pad))
            ridx = F.pad(ridx, (0, pad))
        padded_positions.append(pos)
        padded_mask.append(mask)
        padded_residue_index.append(ridx)
        seqs.append(cast("str", s["seq"]))

    return ProteinBatch(
        atom_positions=torch.stack(padded_positions),
        atom_mask=torch.stack(padded_mask),
        residue_index=torch.stack(padded_residue_index),
        seq=seqs,
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_data.py -v --tb=short
```

Expected: all pass (existing + new).

- [ ] **Step 5: Commit**

```bash
git add pallatom/helpers/data.py pallatom/tests/helpers/test_data.py
git commit -m "feat: implement ClusteredProteinDataset and dynamic collate"
```

---

## Task 6: `_compute_batch_plan`

**Files:**
- Create: `pallatom/helpers/bucketed_sampler.py`
- Create: `pallatom/tests/helpers/test_bucketed_sampler.py`

- [ ] **Step 1: Write the failing tests for `_compute_batch_plan`**

Create `pallatom/tests/helpers/test_bucketed_sampler.py`:

```python
"""Tests for _compute_batch_plan and BucketedBatchSampler."""

import json
import pathlib

import pytest
from helpers.bucketed_sampler import _compute_batch_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rep_lens(n_clusters: int = 64, token_budget: int = 512) -> list[int]:
    """Return the cluster_rep_len list for n_clusters regular clusters + overflow."""
    bin_width = token_budget // n_clusters
    return [bin_width * (k + 1) for k in range(n_clusters)] + [token_budget + 1]


# ---------------------------------------------------------------------------
# _compute_batch_plan
# ---------------------------------------------------------------------------


def test_batch_plan_respects_token_budget() -> None:
    """No batch in the plan exceeds the token budget (using representative lengths)."""
    n = 200
    flat_to_cluster = [0] * n  # all in cluster 0, rep_len=8
    rep_lens = _rep_lens()
    batches = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=0)
    for batch in batches:
        total = sum(rep_lens[flat_to_cluster[i]] for i in batch)
        assert total <= 512, f"Batch budget exceeded: {total}"


def test_batch_plan_covers_all_proteins() -> None:
    """Every protein index appears exactly once across all batches."""
    n = 100
    flat_to_cluster = list(range(64)) * (n // 64) + list(range(n % 64))
    rep_lens = _rep_lens()
    batches = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=0)
    all_indices = sorted(i for batch in batches for i in batch)
    assert all_indices == list(range(n))


def test_batch_plan_overflow_is_singleton() -> None:
    """A protein in the overflow cluster is always a singleton batch."""
    flat_to_cluster = [64]  # one overflow protein
    rep_lens = _rep_lens()
    batches = _compute_batch_plan(flat_to_cluster, rep_lens, 1, 512, 16, seed=0)
    assert len(batches) == 1
    assert batches[0] == [0]


def test_batch_plan_different_seeds_differ() -> None:
    """Different seeds produce different orderings for the same data."""
    n = 200
    flat_to_cluster = [0] * n
    rep_lens = _rep_lens()
    batches_a = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=0)
    batches_b = _compute_batch_plan(flat_to_cluster, rep_lens, n, 512, 16, seed=1)
    # The flat index lists should differ (probability of collision is negligible)
    indices_a = [i for batch in batches_a for i in batch]
    indices_b = [i for batch in batches_b for i in batch]
    assert indices_a != indices_b
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_respects_token_budget pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_covers_all_proteins pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_overflow_is_singleton pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_different_seeds_differ -v
```

Expected: **ERROR** — `ModuleNotFoundError: No module named 'helpers.bucketed_sampler'`.

- [ ] **Step 3: Implement `_compute_batch_plan`**

Create `pallatom/helpers/bucketed_sampler.py`:

```python
"""BucketedBatchSampler: sortish token-budget batching with a process-pool prefetch queue."""

import math
import multiprocessing as mp
import queue
import random
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor

import torch
import torch.utils.data

from helpers.cluster_index import ClusterIndex


def _compute_batch_plan(
    flat_to_cluster: list[int],
    cluster_rep_len: list[int],
    n_proteins: int,
    token_budget: int,
    chunk_multiplier: int,
    seed: int,
) -> list[list[int]]:
    """Compute one epoch's batch plan using sortish sampling and greedy token packing.

    This is a module-level function so it can be submitted to a ProcessPoolExecutor
    (must be picklable). All inputs and outputs are plain Python lists.

    Algorithm:
        1. Shuffle all indices with Random(seed).
        2. Split into chunks of chunk_size = chunk_multiplier * (token_budget // median_rep_len).
        3. Sort each chunk ascending by cluster_rep_len (shortest first).
        4. Greedy pack: accumulate proteins until adding the next would exceed token_budget.
           Overflow proteins (rep_len > token_budget) always become singleton batches.

    Args:
        flat_to_cluster:  cluster id for each global protein index.
        cluster_rep_len:  representative length for each cluster id.
        n_proteins:       total number of proteins.
        token_budget:     maximum cumulative rep_len per batch.
        chunk_multiplier: controls sortish-window width (default 16 ≈ 16 full batches).
        seed:             RNG seed; use seed + epoch to get per-epoch shuffles.

    Returns:
        List of batches; each batch is a list of flat protein indices.
    """
    rng = random.Random(seed)
    indices = list(range(n_proteins))
    rng.shuffle(indices)

    # Chunk size: ~chunk_multiplier full batches per sortish window.
    # Use the median regular cluster's rep_len as the denominator.
    n_regular = len(cluster_rep_len) - 1  # last entry is overflow
    median_rep_len = cluster_rep_len[n_regular // 2]
    avg_proteins_per_budget = max(1, token_budget // median_rep_len)
    chunk_size = chunk_multiplier * avg_proteins_per_budget

    batches: list[list[int]] = []

    for chunk_start in range(0, n_proteins, chunk_size):
        chunk = indices[chunk_start : chunk_start + chunk_size]
        chunk.sort(key=lambda i: cluster_rep_len[flat_to_cluster[i]])

        current_batch: list[int] = []
        current_budget = 0

        for i in chunk:
            rep_len = cluster_rep_len[flat_to_cluster[i]]
            if rep_len > token_budget:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_budget = 0
                batches.append([i])
            elif current_budget + rep_len > token_budget:
                batches.append(current_batch)
                current_batch = [i]
                current_budget = rep_len
            else:
                current_batch.append(i)
                current_budget += rep_len

        if current_batch:
            batches.append(current_batch)

    return batches
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_respects_token_budget pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_covers_all_proteins pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_overflow_is_singleton pallatom/tests/helpers/test_bucketed_sampler.py::test_batch_plan_different_seeds_differ -v
```

Expected: **PASSED**.

- [ ] **Step 5: Commit (partial — sampler file started)**

```bash
git add pallatom/helpers/bucketed_sampler.py pallatom/tests/helpers/test_bucketed_sampler.py
git commit -m "feat: implement _compute_batch_plan for bucketed epoch planning"
```

---

## Task 7: `BucketedBatchSampler`

**Files:**
- Modify: `pallatom/helpers/bucketed_sampler.py`
- Modify: `pallatom/tests/helpers/test_bucketed_sampler.py`

- [ ] **Step 1: Write the failing tests**

Append to `pallatom/tests/helpers/test_bucketed_sampler.py`. Add the following imports at the top of the file:

```python
import json
import pathlib

import pytest
from helpers.bucketed_sampler import BucketedBatchSampler, _compute_batch_plan
from helpers.cluster_index import ClusterIndex
```

Add a shared fixture and sampler tests:

```python
# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_coords(n: int) -> dict[str, list[list[float]]]:
    return {atom: [[0.0, 0.0, 0.0]] * n for atom in ("N", "CA", "C", "O")}


@pytest.fixture
def small_cluster_index(tmp_path: pathlib.Path) -> ClusterIndex:
    """ClusterIndex over 80 synthetic proteins spread across 8 clusters."""
    entries = [
        {"name": f"p{i}", "seq": "A" * ((i % 8 + 1) * 8), "coords": _make_coords((i % 8 + 1) * 8)}
        for i in range(80)
    ]
    path = tmp_path / "proteins.jsonl"
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return ClusterIndex(path, [f"p{i}" for i in range(80)], token_budget=512, n_clusters=64)


# ---------------------------------------------------------------------------
# BucketedBatchSampler
# ---------------------------------------------------------------------------


def test_sampler_covers_all_proteins(small_cluster_index: ClusterIndex) -> None:
    """Every protein index appears exactly once per epoch."""
    sampler = BucketedBatchSampler(
        small_cluster_index, token_budget=512, seed=0, prefetch_epochs=1
    )
    sampler.set_epoch(0)
    all_indices = sorted(i for batch in sampler for i in batch)
    assert all_indices == list(range(len(small_cluster_index)))


def test_sampler_respects_token_budget(small_cluster_index: ClusterIndex) -> None:
    """No batch exceeds the token budget (by representative lengths)."""
    sampler = BucketedBatchSampler(
        small_cluster_index, token_budget=512, seed=0, prefetch_epochs=1
    )
    sampler.set_epoch(0)
    for batch in sampler:
        total = sum(
            small_cluster_index.cluster_rep_len[small_cluster_index.flat_to_cluster[i]]
            for i in batch
        )
        assert total <= 512


def test_sampler_ddp_equal_length(small_cluster_index: ClusterIndex) -> None:
    """Both DDP ranks receive the same number of batches per epoch."""
    sampler_r0 = BucketedBatchSampler(
        small_cluster_index, token_budget=512, world_size=2, rank=0, seed=0, prefetch_epochs=1
    )
    sampler_r1 = BucketedBatchSampler(
        small_cluster_index, token_budget=512, world_size=2, rank=1, seed=0, prefetch_epochs=1
    )
    sampler_r0.set_epoch(0)
    sampler_r1.set_epoch(0)
    assert len(list(sampler_r0)) == len(list(sampler_r1))


def test_sampler_set_epoch_reshuffles(small_cluster_index: ClusterIndex) -> None:
    """Different epochs produce different batch orderings."""
    sampler = BucketedBatchSampler(
        small_cluster_index, token_budget=512, seed=0, prefetch_epochs=3
    )
    sampler.set_epoch(0)
    batches_e0 = [batch[:] for batch in sampler]
    sampler.set_epoch(1)
    batches_e1 = [batch[:] for batch in sampler]
    assert batches_e0 != batches_e1


def test_sampler_len_after_set_epoch(small_cluster_index: ClusterIndex) -> None:
    """__len__ returns the correct batch count after set_epoch."""
    sampler = BucketedBatchSampler(
        small_cluster_index, token_budget=512, seed=0, prefetch_epochs=1
    )
    sampler.set_epoch(0)
    assert len(sampler) == len(list(sampler))
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_bucketed_sampler.py::test_sampler_covers_all_proteins pallatom/tests/helpers/test_bucketed_sampler.py::test_sampler_respects_token_budget pallatom/tests/helpers/test_bucketed_sampler.py::test_sampler_ddp_equal_length -v
```

Expected: **FAILED** — `ImportError: cannot import name 'BucketedBatchSampler'`.

- [ ] **Step 3: Implement `BucketedBatchSampler`**

Append to `pallatom/helpers/bucketed_sampler.py` (after the existing `_compute_batch_plan` function):

```python
class BucketedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Token-budget batch sampler with a process-pool prefetch queue.

    Implements sortish sampling: shuffles all indices, splits into chunks, sorts
    within each chunk by representative length, then greedily packs proteins into
    batches that respect the token budget. Overflow proteins (rep_len > token_budget)
    are always singleton batches.

    Batch plans are precomputed in a subprocess via ProcessPoolExecutor so that the
    next epoch's plan is ready before the current epoch finishes. A bounded queue
    of `prefetch_epochs` futures ensures the pipeline stays full.

    DDP-aware: pads the batch list to a multiple of world_size and yields only the
    batches assigned to this rank (rank-strided).

    Usage::

        sampler = BucketedBatchSampler(cluster_index, token_budget=512)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=...)
        for epoch in range(n_epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...

    Args:
        cluster_index:    ClusterIndex exposing flat_to_cluster and cluster_rep_len.
        token_budget:     Maximum cumulative representative length per batch.
        chunk_multiplier: Sortish window width in multiples of avg proteins per batch.
        world_size:       Number of DDP processes. Default 1 (single GPU).
        rank:             This process's DDP rank. Default 0.
        seed:             Base RNG seed; epoch is added before each call to
                          _compute_batch_plan for per-epoch diversity.
        prefetch_epochs:  Queue depth — how many future epochs to precompute. Default 2.
    """

    def __init__(
        self,
        cluster_index: ClusterIndex,
        token_budget: int,
        chunk_multiplier: int = 16,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        prefetch_epochs: int = 2,
    ) -> None:
        self._cluster_index = cluster_index
        self._token_budget = token_budget
        self._chunk_multiplier = chunk_multiplier
        self._world_size = world_size
        self._rank = rank
        self._seed = seed
        self._prefetch_epochs = prefetch_epochs
        self._epoch = 0
        self._current_batches: list[list[int]] = []

        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
        )
        self._queue: queue.Queue[Future[list[list[int]]]] = queue.Queue(
            maxsize=prefetch_epochs
        )

        for offset in range(prefetch_epochs):
            self._queue.put_nowait(self._submit(seed + offset))

    # ------------------------------------------------------------------

    def _submit(self, seed: int) -> "Future[list[list[int]]]":
        return self._executor.submit(
            _compute_batch_plan,
            self._cluster_index.flat_to_cluster,
            self._cluster_index.cluster_rep_len,
            len(self._cluster_index),
            self._token_budget,
            self._chunk_multiplier,
            seed,
        )

    def set_epoch(self, epoch: int) -> None:
        """Swap in the precomputed batch plan and enqueue the next epoch's plan.

        Must be called once per epoch before iterating. Blocks if the prefetch
        thread has not yet finished (rare — it should finish well before the
        previous epoch ends).

        Args:
            epoch: Current epoch number (0-indexed or 1-indexed; only the delta
                   between epochs matters for RNG diversity).
        """
        self._epoch = epoch
        self._current_batches = self._queue.get().result()
        self._queue.put_nowait(self._submit(self._seed + epoch + self._prefetch_epochs))

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches for this rank.

        Pads the batch list to a multiple of world_size (repeating the first batch)
        so every rank gets the same number of batches, then yields rank-strided.
        """
        all_batches = list(self._current_batches)
        remainder = len(all_batches) % self._world_size
        if remainder != 0:
            all_batches = all_batches + all_batches[: self._world_size - remainder]
        yield from all_batches[self._rank :: self._world_size]

    def __len__(self) -> int:
        """Return the number of batches this rank will yield.

        Returns an exact count after set_epoch has been called, or a rough
        estimate before the first call (safe for tqdm progress bars).
        """
        if self._current_batches:
            n_total = len(self._current_batches)
            remainder = n_total % self._world_size
            if remainder != 0:
                n_total += self._world_size - remainder
            return n_total // self._world_size
        # Estimate before first set_epoch.
        n = len(self._cluster_index)
        n_regular = len(self._cluster_index.cluster_rep_len) - 1
        median_rep = self._cluster_index.cluster_rep_len[n_regular // 2]
        avg_per_batch = max(1, self._token_budget // median_rep)
        return math.ceil(math.ceil(n / avg_per_batch) / self._world_size)

    def __del__(self) -> None:
        """Shut down the executor without blocking process exit."""
        self._executor.shutdown(wait=False)
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_bucketed_sampler.py -v --tb=short
```

Expected: all pass.

- [ ] **Step 5: Run full suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pallatom/helpers/bucketed_sampler.py pallatom/tests/helpers/test_bucketed_sampler.py
git commit -m "feat: implement BucketedBatchSampler with process-pool prefetch queue"
```

---

## Task 8: Factory functions `make_bucketed_data_loaders` and `make_ddp_bucketed_data_loaders`

**Files:**
- Modify: `pallatom/helpers/data.py`
- Modify: `pallatom/tests/helpers/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `pallatom/tests/helpers/test_data.py`. Add to the existing import block:

```python
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.data import make_bucketed_data_loaders
```

Add fixtures and tests:

```python
# ---------------------------------------------------------------------------
# Fixtures for bucketed loader tests
# ---------------------------------------------------------------------------


@pytest.fixture
def bucketed_jsonl(tmp_path: pathlib.Path) -> str:
    """Write a JSONL with 5 proteins of varying lengths."""
    entries = [
        _make_entry("p1", 8),
        _make_entry("p2", 16),
        _make_entry("p3", 24),
        _make_entry("p4", 32),
        _make_entry("p5", 40),
    ]
    path = tmp_path / "proteins.jsonl"
    _write_jsonl(path, entries)
    return str(path)


@pytest.fixture
def bucketed_splits(tmp_path: pathlib.Path) -> str:
    """Write a splits JSON using the 5-protein JSONL names."""
    path = tmp_path / "splits.json"
    with open(path, "w") as f:
        json.dump(
            {
                "train": ["p1", "p2", "p3"],
                "validation": ["p4"],
                "test": ["p5"],
            },
            f,
        )
    return str(path)


@pytest.fixture
def bucketed_cfg() -> TrainConfig:
    """Minimal TrainConfig with token_budget=512 for bucketed loader tests."""
    return TrainConfig(
        train_loader=TrainLoaderConfig(token_budget=512, max_seq_length=64, batch_size=2),
        test_loader=EvalLoaderConfig(batch_size=1, max_seq_length=64),
    )


# ---------------------------------------------------------------------------
# make_bucketed_data_loaders
# ---------------------------------------------------------------------------


def test_bucketed_train_loader_uses_batch_sampler(
    bucketed_cfg: TrainConfig,
    bucketed_jsonl: str,
    bucketed_splits: str,
) -> None:
    """Training loader uses BucketedBatchSampler as its batch_sampler."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=bucketed_jsonl,
        splits_path=bucketed_splits,
        num_workers=0,
        debug_run=False,
    )
    assert isinstance(train_loader.batch_sampler, BucketedBatchSampler)


def test_bucketed_val_loader_uses_fixed_batch_size(
    bucketed_cfg: TrainConfig,
    bucketed_jsonl: str,
    bucketed_splits: str,
) -> None:
    """Validation loader yields ProteinBatch instances (not BucketedBatchSampler)."""
    _, val_loader, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=bucketed_jsonl,
        splits_path=bucketed_splits,
        num_workers=0,
        debug_run=False,
    )
    assert not isinstance(val_loader.batch_sampler, BucketedBatchSampler)


def test_bucketed_train_loader_yields_protein_batch(
    bucketed_cfg: TrainConfig,
    bucketed_jsonl: str,
    bucketed_splits: str,
) -> None:
    """Training loader yields ProteinBatch objects."""
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        jsonl_path=bucketed_jsonl,
        splits_path=bucketed_splits,
        num_workers=0,
        debug_run=False,
    )
    batch = next(iter(train_loader))
    assert isinstance(batch, ProteinBatch)
```

- [ ] **Step 2: Run failing tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_data.py::test_bucketed_train_loader_uses_batch_sampler pallatom/tests/helpers/test_data.py::test_bucketed_val_loader_uses_fixed_batch_size pallatom/tests/helpers/test_data.py::test_bucketed_train_loader_yields_protein_batch -v
```

Expected: **FAILED** — `ImportError: cannot import name 'make_bucketed_data_loaders'`.

- [ ] **Step 3: Implement factory functions**

In `pallatom/helpers/data.py`, add these imports at the top (alongside existing imports):

```python
from helpers.bucketed_sampler import BucketedBatchSampler
```

Then append the two factory functions after `make_ddp_data_loaders`:

```python
def make_bucketed_data_loaders(
    *,
    cfg: TrainConfig,
    jsonl_path: str | Path,
    splits_path: str | Path,
    num_workers: int,
    debug_run: bool,
) -> tuple[
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
]:
    """Build bucketed train loader and fixed-batch val/test loaders.

    The training loader uses ClusteredProteinDataset + BucketedBatchSampler for
    near-zero-padding token-budget batching. Val/test loaders use the original
    ProteinDataset with a fixed batch_size.

    Args:
        cfg:         TrainConfig. cfg.train_loader.token_budget controls packing budget.
        jsonl_path:  Path to the JSONL protein dataset.
        splits_path: Path to a JSON file with keys "train", "validation", "test".
        num_workers: DataLoader worker processes.
        debug_run:   If True, restrict training to the first 252 protein names.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_names = splits["train"][:252] if debug_run else splits["train"]

    train_set = ClusteredProteinDataset(
        jsonl_path,
        train_names,
        token_budget=cfg.train_loader.token_budget,
    )
    val_set = ProteinDataset(
        jsonl_path,
        splits["validation"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    test_set = ProteinDataset(
        jsonl_path,
        splits["test"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )

    train_sampler = BucketedBatchSampler(
        train_set.cluster_index,
        token_budget=cfg.train_loader.token_budget,
    )
    train_sampler.set_epoch(0)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch_dynamic,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.test_loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.test_loader.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )

    return cast(
        """tuple[
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch]
        ]""",
        (train_loader, val_loader, test_loader),
    )


def make_ddp_bucketed_data_loaders(
    cfg: TrainConfig,
    jsonl_path: str | Path,
    splits_path: str | Path,
    rank: int,
    world_size: int,
    num_workers: int = 0,
) -> tuple[
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
    torch.utils.data.DataLoader[ProteinBatch],
]:
    """Build DDP-aware bucketed train loader and fixed-batch val/test loaders.

    The training loader uses ClusteredProteinDataset + BucketedBatchSampler configured
    for this rank. Val/test loaders use DistributedSampler + fixed batch_size.

    Args:
        cfg:        TrainConfig.
        jsonl_path: Path to the JSONL protein dataset.
        splits_path: Path to the splits JSON.
        rank:       This process's DDP rank.
        world_size: Total number of DDP processes.
        num_workers: DataLoader worker processes. Default 0.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    with open(splits_path) as f:
        splits: dict[str, list[str]] = json.load(f)

    train_set = ClusteredProteinDataset(
        jsonl_path,
        splits["train"],
        token_budget=cfg.train_loader.token_budget,
    )
    val_set = ProteinDataset(
        jsonl_path,
        splits["validation"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )
    test_set = ProteinDataset(
        jsonl_path,
        splits["test"],
        max_seq_length=cfg.test_loader.max_seq_length,
    )

    train_sampler = BucketedBatchSampler(
        train_set.cluster_index,
        token_budget=cfg.train_loader.token_budget,
        world_size=world_size,
        rank=rank,
    )
    train_sampler.set_epoch(0)

    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch_dynamic,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.test_loader.batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.test_loader.batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_to_protein_batch,
    )
    return cast(
        """tuple[
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch],
            torch.utils.data.DataLoader[ProteinBatch]
        ]""",
        (train_loader, val_loader, test_loader),
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_data.py -v --tb=short
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pallatom/helpers/data.py pallatom/tests/helpers/test_data.py
git commit -m "feat: add make_bucketed_data_loaders factory functions"
```

---

## Task 9: Training loop integration

**Files:**
- Modify: `pallatom/train/train_loop.py`
- Modify: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Write the failing test**

In `pallatom/tests/train/test_train_loop.py`, add a new import and test. First add to the import block at the top:

```python
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.data import make_bucketed_data_loaders
```

Then append this test function:

```python
def test_train_uses_bucketed_loader(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
    jsonl_path: str,
    splits_path: str,
) -> None:
    """train() runs one epoch end-to-end using the bucketed DataLoader."""
    from train.train_loop import train
    train(
        model,
        tcfg,
        distogram_res,
        distogram_atom,
        jsonl_path=jsonl_path,
        splits_path=splits_path,
        num_workers=0,
        device="cpu",
    )
```

This test requires `train()` to accept `jsonl_path` and `splits_path` keyword arguments and use `make_bucketed_data_loaders` internally. If the existing `train()` signature does not accept those arguments, add them in the implementation step below. Check the current `train()` signature before writing the test — if it already has these params, use the existing fixture names; if it doesn't, extend the signature.

Look at the top of `pallatom/train/train_loop.py` to find the `train()` and `train_ddp()` signatures. The existing non-DDP `train()` may not accept `jsonl_path`/`splits_path` if those are currently passed via `args`. In that case, add those params and adjust accordingly.

- [ ] **Step 2: Implement loader swap in `train()` and `train_ddp()`**

**In `train_loop.py`**, add these imports near the top (alongside existing data imports):

```python
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.data import make_bucketed_data_loaders, make_ddp_bucketed_data_loaders
```

**In `train_ddp()`**, replace the `make_ddp_data_loaders` call with:

```python
train_loader, val_loader, _ = make_ddp_bucketed_data_loaders(
    tcfg,
    args.data,
    args.splits,
    rank=rank,
    world_size=world_size,
    num_workers=args.num_workers,
)
```

Replace the `set_epoch` cast (currently `train_loader.sampler`) with:

```python
cast(BucketedBatchSampler, train_loader.batch_sampler).set_epoch(epoch)
```

**In `train()`**, replace `make_data_loaders` with `make_bucketed_data_loaders`, supplying `jsonl_path`, `splits_path`, `num_workers`, and `debug_run` arguments. Update the `set_epoch` call analogously if `train()` iterates epochs with sampler reshuffling (check whether it does — the DDP version definitely does; the single-GPU version may not, in which case no cast is needed since `BucketedBatchSampler` handles its own epoch internally via `set_epoch(0)` at factory time for epoch 0, but you must add the per-epoch `set_epoch` call to `train()` for epochs > 0).

Specifically, in `train()`, add before each epoch's iteration:

```python
cast(BucketedBatchSampler, train_loader.batch_sampler).set_epoch(epoch)
```

- [ ] **Step 3: Run full test suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -v --tb=short
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat: integrate BucketedBatchSampler into training loop"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Section 1 (Metrics) → Task 1
- [x] Section 2 (ClusterIndex) → Task 3
- [x] Section 3 (ClusteredProteinDataset) → Task 5
- [x] Section 4 (BucketedBatchSampler + prefetch queue) → Tasks 6 + 7
- [x] Section 5 (dynamic collate + factory functions) → Tasks 5 + 8
- [x] Section 6 (training loop integration) → Task 9
- [x] Config change (token_budget) → Task 4

**Type consistency across tasks:**
- `ClusterIndex` attributes `flat_to_cluster`, `flat_to_local`, `cluster_rep_len`, `cluster_offsets`, `cluster_file(k)`, `assign_cluster(seq_len)` — used consistently in Tasks 3, 5, 6, 7, 8.
- `ClusteredProteinDataset.cluster_index` exposes the `ClusterIndex` — used in Tasks 5, 8.
- `BucketedBatchSampler.set_epoch(epoch)` — called in Tasks 8 and 9.
- `_to_protein_batch_dynamic` — referenced in Tasks 5 and 8.
- `make_bucketed_data_loaders` / `make_ddp_bucketed_data_loaders` — defined in Task 8, used in Task 9.

**No placeholders:** All code blocks are complete; no TBD or TODO present.
