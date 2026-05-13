# API Request Schemas for Conditional Sampling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `REST_APIs/api.py` to support all six conditioning use cases (unconditional, sequence-only, sequence+partial atoms, partial template, full template, atoms+template) through a single `/sample` endpoint with a unified `SampleRequest` schema.

**Architecture:** Three optional conditioning fields (`sequence`, `structure_pdb`, `template_pdb`) are added to `SampleRequest`; the server translates them into `build_sampling_context` arguments at runtime. A small fix to `build_AA_context` makes `'X'` residues valid. The lifespan is updated to construct distogram modules at startup (removing the stale `index_embedding` load).

**Tech Stack:** FastAPI, Pydantic v2, PyTorch, pytest, `fastapi.testclient.TestClient`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pallatom/sample/sampling.py` | Modify line 133 | Fix `restype_order[r]` → `.get(r, 20)` for `'X'` support |
| `pallatom/tests/sample/test_sampling.py` | Modify | Add tests for `'X'` handling in `build_AA_context` |
| `REST_APIs/api.py` | Modify | New schema, updated lifespan, rewritten `_run_sampling` |
| `REST_APIs/tests/__init__.py` | Create | Empty package marker |
| `REST_APIs/tests/test_api.py` | Create | Validation tests + `_run_sampling` integration tests |

---

## Task 1: Fix `build_AA_context` to handle `'X'` residues

**Files:**
- Modify: `pallatom/sample/sampling.py:133`
- Modify: `pallatom/tests/sample/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `pallatom/tests/sample/test_sampling.py`:

```python
# ---------------------------------------------------------------------------
# build_AA_context — 'X' residue handling
# ---------------------------------------------------------------------------


def test_build_aa_context_x_sequence_does_not_raise(atom_disto_fn, residue_idx_aa):
    with torch.no_grad():
        ctx = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="X" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
            c_res=C_RES_AA,
        )
    assert isinstance(ctx, AllAtomContext)


def test_build_aa_context_x_maps_to_index_20(atom_disto_fn, residue_idx_aa):
    with torch.no_grad():
        ctx = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="X" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
            c_res=C_RES_AA,
        )
    assert (ctx.aa_indices == 20).all()


def test_build_aa_context_x_is_distinct_from_alanine(atom_disto_fn, residue_idx_aa):
    with torch.no_grad():
        ctx_x = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="X" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
            c_res=C_RES_AA,
        )
        ctx_a = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="A" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
            c_res=C_RES_AA,
        )
    assert not torch.equal(ctx_x.aa_indices, ctx_a.aa_indices)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/sample/test_sampling.py::test_build_aa_context_x_sequence_does_not_raise pallatom/tests/sample/test_sampling.py::test_build_aa_context_x_maps_to_index_20 pallatom/tests/sample/test_sampling.py::test_build_aa_context_x_is_distinct_from_alanine -v
```

Expected: `FAILED` — `KeyError: 'X'`

- [ ] **Step 3: Apply the fix**

In `pallatom/sample/sampling.py`, find line 133 (inside `build_AA_context`):

```python
    _aa_vals = [restype_order[r] for r in aa_sequence]
```

Replace with:

```python
    _aa_vals = [restype_order.get(r, 20) for r in aa_sequence]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/sample/test_sampling.py::test_build_aa_context_x_sequence_does_not_raise pallatom/tests/sample/test_sampling.py::test_build_aa_context_x_maps_to_index_20 pallatom/tests/sample/test_sampling.py::test_build_aa_context_x_is_distinct_from_alanine -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Verify no regressions**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/sample/test_sampling.py -v
```

Expected: all existing tests still `PASSED`

- [ ] **Step 6: Commit**

```bash
git add pallatom/sample/sampling.py pallatom/tests/sample/test_sampling.py
git commit -m "fix(sampling): handle X residues in build_AA_context via restype_order.get"
```

---

## Task 2: Create API test infrastructure and `SampleRequest` validation tests

**Files:**
- Create: `REST_APIs/tests/__init__.py`
- Create: `REST_APIs/tests/test_api.py`

- [ ] **Step 1: Create the test package**

```bash
mkdir -p /workspaces/diffusion/REST_APIs/tests
touch /workspaces/diffusion/REST_APIs/tests/__init__.py
```

- [ ] **Step 2: Write the failing validation tests**

Create `REST_APIs/tests/test_api.py`:

```python
"""Tests for SampleRequest validation and _protein_from_pdb_string."""
import sys
import os

import numpy as np
import pytest

# Ensure pallatom is importable (api.py does this at module level too)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "pallatom"))

from pydantic import ValidationError


# ---------------------------------------------------------------------------
# SampleRequest validation
# ---------------------------------------------------------------------------


def _import_request():
    """Import SampleRequest lazily so errors surface per-test."""
    from REST_APIs.api import SampleRequest
    return SampleRequest


def test_sample_request_minimal_valid():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=10)
    assert req.n_res == 10
    assert req.sequence is None
    assert req.structure_pdb is None
    assert req.template_pdb is None


def test_sample_request_sequence_length_must_match_n_res():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError, match="sequence length"):
        SampleRequest(n_res=10, sequence="ACDE")  # len 4 ≠ 10


def test_sample_request_sequence_exact_length_accepted():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="ACDE")
    assert req.sequence == "ACDE"


def test_sample_request_sequence_x_characters_accepted():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="AXXA")
    assert req.sequence == "AXXA"


def test_sample_request_sequence_all_x_accepted():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="XXXX")
    assert req.sequence == "XXXX"


def test_sample_request_sequence_invalid_character_rejected():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError, match="Invalid characters"):
        SampleRequest(n_res=3, sequence="ABZ")  # B and Z are not valid


def test_sample_request_n_res_must_be_positive():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=0)


def test_sample_request_n_res_max_512():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=513)


def test_sample_request_n_samples_max_10():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=10, n_samples=11)


def test_sample_request_defaults():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=50)
    assert req.n_samples == 1
    assert req.ddim_steps == 40
    assert req.rho == 7.0
    assert req.S_churn == 0.0
    assert req.S_noise == 1.003
```

- [ ] **Step 3: Run tests to verify they fail (SampleRequest not yet updated)**

```bash
cd /workspaces/diffusion && python -m pytest REST_APIs/tests/test_api.py -v
```

Expected: many `FAILED` — `SampleRequest` doesn't have `sequence`, `structure_pdb`, or `template_pdb` fields yet, and doesn't validate sequence length/characters.

- [ ] **Step 4: Commit the test file (red state)**

```bash
git add REST_APIs/tests/__init__.py REST_APIs/tests/test_api.py
git commit -m "test(api): add SampleRequest validation tests (red)"
```

---

## Task 3: Replace `SampleRequest` with the validated conditional schema

**Files:**
- Modify: `REST_APIs/api.py`

- [ ] **Step 1: Update imports at the top of `api.py`**

Replace the existing pydantic import line:

```python
from pydantic import BaseModel, Field
```

With:

```python
from pydantic import BaseModel, Field, field_validator, model_validator
```

- [ ] **Step 2: Replace the `SampleRequest` class**

Replace the entire `SampleRequest` class (lines 94–101) with:

```python
_VALID_AA: frozenset[str] = frozenset("ARNDCQEGHILKMFPSTWYVX")


class SampleRequest(BaseModel):
    # --- target ---
    n_res: int = Field(..., gt=0, le=512, description="Number of residues to generate")

    # --- conditioning (all optional) ---
    sequence: str | None = Field(
        None,
        description=(
            "Amino-acid sequence of length n_res. "
            "Standard 20 AAs plus 'X' for unknown. "
            "Omit for no sequence conditioning ('X' * n_res is used internally)."
        ),
    )
    structure_pdb: str | None = Field(
        None,
        description=(
            "PDB string for atom-level conditioning. "
            "Residues present in the PDB fill r_gt/atom5_mask; uncovered positions are zeroed. "
            "Must cover ≤ n_res residues."
        ),
    )
    template_pdb: str | None = Field(
        None,
        description=(
            "PDB string for template-distogram conditioning. "
            "May cover fewer than n_res residues (padded with zeros)."
        ),
    )

    # --- sampler ---
    n_samples: int = Field(1, gt=0, le=10, description="Number of structures to generate in parallel")
    ddim_steps: int = Field(40, gt=1, description="ODE solver steps (more = higher quality)")
    rho: float = Field(7.0, gt=0, description="Karras noise-schedule exponent")
    S_churn: float = Field(0.0, ge=0, description="Stochasticity per step (0 = deterministic)")
    S_noise: float = Field(1.003, gt=0, description="Noise scaling for stochastic steps")

    @field_validator("sequence")
    @classmethod
    def sequence_valid_characters(cls, v: str | None) -> str | None:
        if v is not None:
            invalid = set(v) - _VALID_AA
            if invalid:
                raise ValueError(f"Invalid characters in sequence: {sorted(invalid)!r}")
        return v

    @model_validator(mode="after")
    def sequence_length_matches_n_res(self) -> "SampleRequest":
        if self.sequence is not None and len(self.sequence) != self.n_res:
            raise ValueError(
                f"sequence length {len(self.sequence)} must equal n_res {self.n_res}"
            )
        return self
```

- [ ] **Step 3: Run the validation tests**

```bash
cd /workspaces/diffusion && python -m pytest REST_APIs/tests/test_api.py -v
```

Expected: all validation tests `PASSED` (the `_run_sampling` tests don't exist yet)

- [ ] **Step 4: Commit**

```bash
git add REST_APIs/api.py
git commit -m "feat(api): add conditional SampleRequest schema with sequence validation"
```

---

## Task 4: Update `_load_model` and `lifespan` to drop `index_embedding` and add distogram functions

**Files:**
- Modify: `REST_APIs/api.py`

- [ ] **Step 1: Add missing imports to `api.py`**

After the existing imports block (after `from train.train_config import ModelParams, NoiseScheduleParams`), add:

```python
import tempfile

from helpers.atom_utils import protein_from_pdb
from helpers.featurize import Distogram
```

- [ ] **Step 2: Rewrite `_load_model`**

Replace the existing `_load_model` function with:

```python
def _load_model(
    checkpoint_path: str,
    mp: ModelParams,
    noise: NoiseScheduleParams,
    device: str,
) -> MainTrunk:
    model = MainTrunk(
        f_ref_dim=mp.f_ref_dim,
        n_bins=mp.n_bins,
        c_atom=mp.c_atom,
        c_pair=mp.c_pair,
        c_res=mp.c_res,
        c_atompair=mp.c_atompair,
        K_unit=mp.K_unit,
        sigma_data=noise.sigma_data,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
```

- [ ] **Step 3: Rewrite `lifespan`**

Replace the existing `lifespan` function with:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore
    _semaphore = asyncio.Semaphore(1)

    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    mp = ModelParams()
    noise = NoiseScheduleParams()
    model = _load_model(CHECKPOINT_PATH, mp, noise, DEVICE)
    atom_disto = Distogram(n_bins=22, min_dist=2.0, max_dist=22.0).to(DEVICE)
    templ_disto = Distogram(
        n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
    ).to(DEVICE)
    _state.update(
        model=model,
        mp=mp,
        noise=noise,
        atom_disto=atom_disto,
        templ_disto=templ_disto,
    )
    yield
    _state.clear()
```

- [ ] **Step 4: Verify `api.py` is importable (no syntax errors)**

```bash
cd /workspaces/diffusion && python -c "import sys; sys.path.insert(0, 'pallatom'); import importlib.util; spec = importlib.util.spec_from_file_location('api', 'REST_APIs/api.py'); m = importlib.util.module_from_spec(spec); print('import OK')"
```

Expected: `import OK`

- [ ] **Step 5: Commit**

```bash
git add REST_APIs/api.py
git commit -m "feat(api): drop index_embedding from _load_model, add distogram functions to lifespan"
```

---

## Task 5: Rewrite `_run_sampling` with full conditional logic

**Files:**
- Modify: `REST_APIs/api.py`

- [ ] **Step 1: Add helper `_protein_from_pdb_string` before `_run_sampling`**

Insert this function just before `_run_sampling`:

```python
def _protein_from_pdb_string(pdb_string: str):
    """Write pdb_string to a temp file, parse it, delete the temp file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
        f.write(pdb_string)
        path = f.name
    try:
        return protein_from_pdb(path)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Rewrite `_run_sampling`**

Replace the entire `_run_sampling` function with:

```python
def _run_sampling(req: SampleRequest) -> list[str]:
    model: MainTrunk = _state["model"]
    mp: ModelParams = _state["mp"]
    noise: NoiseScheduleParams = _state["noise"]
    atom_disto: Distogram = _state["atom_disto"]
    templ_disto: Distogram = _state["templ_disto"]

    N_res = req.n_res
    B = req.n_samples
    N_atom = N_res * NATOM

    # ── atom-level conditioning ───────────────────────────────────────────────
    if req.structure_pdb is not None:
        prot = _protein_from_pdb_string(req.structure_pdb)
        n_pdb: int = prot.atom_positions.shape[0]
        if n_pdb > N_res:
            raise ValueError(
                f"structure_pdb has {n_pdb} residues but n_res={N_res}; "
                "structure_pdb must cover ≤ n_res residues"
            )
        atom_positions = torch.zeros(N_res, 37, 3)
        atom_mask = torch.zeros(N_res, 37)
        atom_positions[:n_pdb] = torch.tensor(prot.atom_positions, dtype=torch.float32)
        atom_mask[:n_pdb] = torch.tensor(prot.atom_mask, dtype=torch.float32)
        pdb_idx = torch.tensor(prot.residue_index, dtype=torch.float32)
        if n_pdb < N_res:
            last = int(pdb_idx[-1].item()) if n_pdb > 0 else -1
            tail = torch.arange(last + 1, last + 1 + (N_res - n_pdb), dtype=torch.float32)
            residue_index = torch.cat([pdb_idx, tail])
        else:
            residue_index = pdb_idx
    else:
        atom_positions = torch.zeros(N_res, 37, 3)
        atom_mask = torch.zeros(N_res, 37)
        residue_index = torch.arange(N_res, dtype=torch.float32)

    # ── sequence ─────────────────────────────────────────────────────────────
    seq: str = req.sequence if req.sequence is not None else "X" * N_res

    # ── template-distogram conditioning ──────────────────────────────────────
    pdb_files: list[str] = []
    tmp_path: str | None = None
    try:
        if req.template_pdb is not None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdb", delete=False
            ) as f:
                f.write(req.template_pdb)
                tmp_path = f.name
            pdb_files = [tmp_path]

        context = build_sampling_context(
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            residue_index=residue_index,
            seq=seq,
            pdb_files=pdb_files,
            atom_distogram_fn=atom_disto,
            templ_distogram_fn=templ_disto,
            batch_size=B,
            device=DEVICE,
            c_res=mp.c_res,
        )
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    edm_precond = EDMPrecond(
        model,
        context,
        sigma_min=noise.sigma_min,
        sigma_max=noise.sigma_max,
    ).to(DEVICE)
    edm_precond.eval()

    edm_sampler = EDMSampler(
        edm_precond,
        sigma_min=noise.sigma_min,
        sigma_max=noise.sigma_max,
        rho=req.rho,
        S_churn=req.S_churn,
        S_tmin=0.0,
        S_tmax=float("inf"),
        S_noise=req.S_noise,
    )

    coords_batch = edm_sampler.sample(
        shape=(B, N_atom, 3),
        steps=req.ddim_steps,
        device=DEVICE,
    )

    pdb_strings: list[str] = []
    for b in range(B):
        coords_np = rearrange(
            coords_batch[b].cpu().numpy(), "(n a) d -> n a d", n=N_res, a=NATOM
        )
        x_37, mask_37 = atom5_to_atom37(coords_np)
        prot_out = Protein(
            atom_positions=x_37,
            atom_mask=mask_37,
            residue_index=np.arange(N_res, dtype=np.int32),
            aatype=np.zeros(N_res, dtype=np.int32),
            chain_index=np.zeros(N_res, dtype=np.int32),
            b_factors=np.ones((N_res, 37), dtype=np.float32),
        )
        pdb_strings.append(to_pdb(prot_out))

    return pdb_strings
```

- [ ] **Step 3: Verify `api.py` still imports cleanly**

```bash
cd /workspaces/diffusion && python -c "import sys; sys.path.insert(0, 'pallatom'); import importlib.util; spec = importlib.util.spec_from_file_location('api', 'REST_APIs/api.py'); m = importlib.util.module_from_spec(spec); print('import OK')"
```

Expected: `import OK`

- [ ] **Step 4: Commit**

```bash
git add REST_APIs/api.py
git commit -m "feat(api): rewrite _run_sampling to support all six conditioning use cases"
```

---

## Task 6: Integration tests for `_run_sampling` via mocked `_state`

**Files:**
- Modify: `REST_APIs/tests/test_api.py`

- [ ] **Step 1: Add `_run_sampling` integration tests**

Append to `REST_APIs/tests/test_api.py`:

```python
# ---------------------------------------------------------------------------
# _run_sampling integration tests (mocked model, no checkpoint required)
# ---------------------------------------------------------------------------

import torch
import numpy as np
from unittest.mock import MagicMock, patch

from helpers.atom_utils import Protein, to_pdb
from helpers.featurize import Distogram


N_RES_TEST = 4
N_ATOM_TEST = N_RES_TEST * 5  # NATOM = 5


def _make_trunk_mock():
    mock = MagicMock()

    def _forward(batch):
        B = batch.r_input.shape[0]
        N_atom = batch.r_input.shape[1]
        n_res = int(batch.tok_idx.max().item()) + 1
        return (
            torch.zeros(B, N_atom, 3),
            torch.zeros(B, n_res, 20),
            torch.zeros(B, n_res, n_res, 38),
            torch.zeros(B, N_atom, 1, 22),
            [],
            [],
        )

    mock.side_effect = _forward
    return mock


def _make_mock_state():
    from train.train_config import ModelParams, NoiseScheduleParams
    mp = ModelParams()
    noise = NoiseScheduleParams()
    return {
        "model": _make_trunk_mock(),
        "mp": mp,
        "noise": noise,
        "atom_disto": Distogram(n_bins=22, min_dist=2.0, max_dist=22.0),
        "templ_disto": Distogram(
            n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
        ),
    }


def _make_pdb_string(n_res: int) -> str:
    """Produce a minimal valid PDB string with n_res alanine residues."""
    pos = np.random.RandomState(42).randn(n_res, 37, 3).astype(np.float32)
    mask = np.ones((n_res, 37), dtype=np.float32)
    prot = Protein(
        atom_positions=pos,
        aatype=np.zeros(n_res, dtype=np.int32),
        atom_mask=mask,
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_index=np.zeros(n_res, dtype=np.int32),
        b_factors=np.ones((n_res, 37), dtype=np.float32),
    )
    return to_pdb(prot)


def _run(req_kwargs: dict) -> list[str]:
    from REST_APIs.api import SampleRequest, _run_sampling
    req = SampleRequest(**req_kwargs)
    mock_state = _make_mock_state()
    with patch("REST_APIs.api._state", mock_state):
        return _run_sampling(req)


# --- use case 1: unconditional ---

def test_run_sampling_unconditional_returns_pdb_strings():
    result = _run({"n_res": N_RES_TEST})
    assert isinstance(result, list)
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_unconditional_n_samples_controls_output_count():
    result = _run({"n_res": N_RES_TEST, "n_samples": 3})
    assert len(result) == 3


# --- use case 2: sequence only ---

def test_run_sampling_sequence_only_returns_pdb_strings():
    result = _run({"n_res": N_RES_TEST, "sequence": "ACDE"})
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_sequence_with_x_returns_pdb_strings():
    result = _run({"n_res": N_RES_TEST, "sequence": "AXXE"})
    assert len(result) == 1
    assert "ATOM" in result[0]


# --- use case 3: sequence + partial atoms ---

def test_run_sampling_seq_partial_atoms_returns_pdb_strings():
    partial_pdb = _make_pdb_string(N_RES_TEST // 2)
    result = _run({
        "n_res": N_RES_TEST,
        "sequence": "ACDE",
        "structure_pdb": partial_pdb,
    })
    assert len(result) == 1
    assert "ATOM" in result[0]


# --- use case 4: partial template only ---

def test_run_sampling_partial_template_returns_pdb_strings():
    partial_pdb = _make_pdb_string(N_RES_TEST // 2)
    result = _run({"n_res": N_RES_TEST, "template_pdb": partial_pdb})
    assert len(result) == 1
    assert "ATOM" in result[0]


# --- use case 5: full template ---

def test_run_sampling_full_template_returns_pdb_strings():
    full_pdb = _make_pdb_string(N_RES_TEST)
    result = _run({"n_res": N_RES_TEST, "template_pdb": full_pdb})
    assert len(result) == 1
    assert "ATOM" in result[0]


# --- use case 6: all atoms + partial template ---

def test_run_sampling_full_atoms_partial_template_returns_pdb_strings():
    full_struct = _make_pdb_string(N_RES_TEST)
    partial_templ = _make_pdb_string(N_RES_TEST // 2)
    result = _run({
        "n_res": N_RES_TEST,
        "structure_pdb": full_struct,
        "template_pdb": partial_templ,
    })
    assert len(result) == 1
    assert "ATOM" in result[0]


# --- structure_pdb validation ---

def test_run_sampling_structure_pdb_too_many_residues_raises():
    oversized_pdb = _make_pdb_string(N_RES_TEST + 2)
    from REST_APIs.api import SampleRequest, _run_sampling
    req = SampleRequest(n_res=N_RES_TEST, structure_pdb=oversized_pdb)
    mock_state = _make_mock_state()
    with patch("REST_APIs.api._state", mock_state):
        with pytest.raises(ValueError, match="structure_pdb has"):
            _run_sampling(req)


# --- residue_index from PDB is non-contiguous ---

def test_run_sampling_non_contiguous_residue_index_does_not_raise():
    """PDB with non-contiguous residue numbers (gap between residues)."""
    # Build a PDB where residues are numbered 1, 2, 10, 11
    n_res = 4
    pos = np.zeros((n_res, 37, 3), dtype=np.float32)
    mask = np.ones((n_res, 37), dtype=np.float32)
    prot = Protein(
        atom_positions=pos,
        aatype=np.zeros(n_res, dtype=np.int32),
        atom_mask=mask,
        residue_index=np.array([1, 2, 10, 11], dtype=np.int32),
        chain_index=np.zeros(n_res, dtype=np.int32),
        b_factors=np.ones((n_res, 37), dtype=np.float32),
    )
    pdb_str = to_pdb(prot)
    result = _run({"n_res": N_RES_TEST, "structure_pdb": pdb_str})
    assert len(result) == 1


# --- template tempfile is always cleaned up ---

def test_run_sampling_tempfile_deleted_after_sampling():
    import glob
    import tempfile as _tf
    tmpdir = _tf.gettempdir()
    pdb_before = set(glob.glob(os.path.join(tmpdir, "*.pdb")))
    full_pdb = _make_pdb_string(N_RES_TEST)
    _run({"n_res": N_RES_TEST, "template_pdb": full_pdb})
    pdb_after = set(glob.glob(os.path.join(tmpdir, "*.pdb")))
    assert pdb_after == pdb_before
```

- [ ] **Step 2: Run all API tests**

```bash
cd /workspaces/diffusion && python -m pytest REST_APIs/tests/test_api.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 3: Run all sampling tests to check for regressions**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/sample/test_sampling.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 4: Commit**

```bash
git add REST_APIs/tests/test_api.py
git commit -m "test(api): integration tests for all six conditional sampling use cases"
```

---

## Self-Review

**Spec coverage:**
- ✅ Single endpoint (`POST /sample`) — all use cases go through `_run_sampling`
- ✅ Three optional fields: `sequence`, `structure_pdb`, `template_pdb`
- ✅ `seq` defaults to `"X" * n_res` (not `"A" * n_res`)
- ✅ `residue_index` always `arange(n_res)` when no PDB; PDB's actual indices + sequential tail when PDB present
- ✅ `restype_order.get(r, 20)` fix in `build_AA_context`
- ✅ `index_embedding` removed from `_load_model` and `lifespan`
- ✅ Distogram functions constructed at startup and stored in `_state`
- ✅ `template_pdb` written to `tempfile` with `try/finally` cleanup
- ✅ `structure_pdb` validated: parsed residue count ≤ `n_res`
- ✅ `sequence` validated: length = `n_res`, characters in 20 AAs + `'X'`
- ✅ `SampleResponse` unchanged
- ✅ Known limitation (training gap) documented in spec — no code change required

**Placeholder scan:** None found. All steps contain full code.

**Type consistency:** `_make_mock_state()` returns the same keys (`model`, `mp`, `noise`, `atom_disto`, `templ_disto`) that `_run_sampling` reads from `_state`. `_protein_from_pdb_string` returns a `Protein` object whose `.atom_positions` and `.atom_mask` are `np.ndarray`, correctly converted to `torch.tensor(...)` in `_run_sampling`. Consistent throughout.
