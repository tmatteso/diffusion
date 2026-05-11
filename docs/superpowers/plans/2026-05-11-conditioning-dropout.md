# Conditioning Dropout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add residue-level classifier-free guidance conditioning dropout for three independent signals (template distogram, atom mask, sequence) so the model sees all six inference use cases during training.

**Architecture:** A standalone `apply_conditioning_dropout` function (in `featurize.py`) takes a `FeaturizedBatch` and applies independent Bernoulli residue-level masks to `gt_res_distogram`, `atom5_mask`, and `aa_indices`. Dropout probabilities live in a new `ConditioningDropoutConfig` Pydantic model added to `TrainConfig`. A new `nn.Embedding(21, c_res)` is added to `MainTrunk` to enable actual sequence conditioning (currently `aa_indices` is in the batch but not used by the model forward pass).

**Tech Stack:** PyTorch, einops, jaxtyping + beartype, Pydantic v2, pytest

**Key invariant:** Sequence dropout sets dropped positions to index 20 (`"X"`). The CE loss target must mask 20→-100 before computing cross-entropy (since `F.cross_entropy` only ignores -100 by default and the output head has 20 classes). A helper `_mask_seq_target` in `train_loop.py` handles this everywhere.

---

## File Map

| File | Change |
|------|--------|
| `pallatom/helpers/atom_utils.py` | Add `"X"` to `restypes` at position 20 |
| `pallatom/train/train_config.py` | Add `ConditioningDropoutConfig` + field on `TrainConfig` |
| `pallatom/helpers/featurize.py` | Add `apply_conditioning_dropout` function |
| `pallatom/architecture/main_trunk.py` | Add `self.aa_embedding = nn.Embedding(21, c_res)` and use in `forward` |
| `pallatom/train/train_loop.py` | Add `_mask_seq_target` helper; call dropout in two training loops; apply masking in CE losses |
| `pallatom/tests/helpers/test_featurize.py` | New tests for `apply_conditioning_dropout` |
| `pallatom/tests/architecture/test_main_trunk.py` | New test: forward pass with aa_indices=20 does not crash |
| `pallatom/tests/train/test_train_loop.py` | Smoke test: training step with dropout enabled produces finite loss |

---

### Task 1: Add "X" to `restypes`

**Files:**
- Modify: `pallatom/helpers/atom_utils.py:103-106`
- Test: `pallatom/tests/helpers/test_atom_utils.py` (create if absent; check `test_featurize.py` — tests there import `restype_order`)

- [ ] **Step 1: Write the failing test**

Add to `pallatom/tests/helpers/test_featurize.py` (at the top, with the other imports):

```python
from helpers.atom_utils import restype_order, restype_num, restypes
```

Then add these two test functions after the existing `restype_order`-related imports:

```python
def test_restype_order_x_is_20():
    assert restype_order["X"] == 20


def test_restype_num_is_21():
    assert restype_num == 21
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest pallatom/tests/helpers/test_featurize.py::test_restype_order_x_is_20 pallatom/tests/helpers/test_featurize.py::test_restype_num_is_21 -v
```

Expected: FAIL — `KeyError: 'X'` and `AssertionError: assert 20 == 21`

- [ ] **Step 3: Add "X" to `restypes`**

In `pallatom/helpers/atom_utils.py`, change lines 83–106 from:

```python
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
```

to:

```python
restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",  # mask token: unknown / conditioning-dropout placeholder
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 21.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest pallatom/tests/helpers/test_featurize.py::test_restype_order_x_is_20 pallatom/tests/helpers/test_featurize.py::test_restype_num_is_21 -v
```

Expected: PASS

- [ ] **Step 5: Run full suite to catch regressions**

```bash
pytest pallatom/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add pallatom/helpers/atom_utils.py pallatom/tests/helpers/test_featurize.py
git commit -m "feat(atom_utils): add X as mask token at restype index 20"
```

---

### Task 2: Add `ConditioningDropoutConfig` to `TrainConfig`

**Files:**
- Modify: `pallatom/train/train_config.py`
- Test: new file `pallatom/tests/train/test_train_config.py`

- [ ] **Step 1: Write the failing tests**

Create `pallatom/tests/train/test_train_config.py`:

```python
import pytest
from train.train_config import ConditioningDropoutConfig, TrainConfig


def test_conditioning_dropout_config_defaults():
    cfg = ConditioningDropoutConfig()
    assert cfg.p_distogram == 0.15
    assert cfg.p_atom == 0.15
    assert cfg.p_seq == 0.15


def test_conditioning_dropout_config_custom_values():
    cfg = ConditioningDropoutConfig(p_distogram=0.3, p_atom=0.1, p_seq=0.2)
    assert cfg.p_distogram == 0.3
    assert cfg.p_atom == 0.1
    assert cfg.p_seq == 0.2


def test_conditioning_dropout_config_rejects_negative():
    with pytest.raises(Exception):
        ConditioningDropoutConfig(p_distogram=-0.1)


def test_conditioning_dropout_config_rejects_above_one():
    with pytest.raises(Exception):
        ConditioningDropoutConfig(p_seq=1.1)


def test_train_config_has_conditioning_dropout():
    cfg = TrainConfig()
    assert hasattr(cfg, "conditioning_dropout")
    assert isinstance(cfg.conditioning_dropout, ConditioningDropoutConfig)


def test_train_config_conditioning_dropout_defaults():
    cfg = TrainConfig()
    assert cfg.conditioning_dropout.p_distogram == 0.15
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest pallatom/tests/train/test_train_config.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ConditioningDropoutConfig'`

- [ ] **Step 3: Add `ConditioningDropoutConfig` and field to `TrainConfig`**

In `pallatom/train/train_config.py`, add the new class before `TrainConfig` and add the field:

```python
class ConditioningDropoutConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    p_distogram: float = Field(0.15, ge=0.0, le=1.0)
    p_atom:      float = Field(0.15, ge=0.0, le=1.0)
    p_seq:       float = Field(0.15, ge=0.0, le=1.0)
```

Then in `TrainConfig`, add after the `test_loader` field:

```python
    conditioning_dropout: ConditioningDropoutConfig = Field(default_factory=ConditioningDropoutConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest pallatom/tests/train/test_train_config.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pallatom/train/train_config.py pallatom/tests/train/test_train_config.py
git commit -m "feat(train_config): add ConditioningDropoutConfig"
```

---

### Task 3: Implement `apply_conditioning_dropout`

**Files:**
- Modify: `pallatom/helpers/featurize.py`
- Test: `pallatom/tests/helpers/test_featurize.py`

- [ ] **Step 1: Write the failing tests**

Add a fixture and six test functions to `pallatom/tests/helpers/test_featurize.py`. The fixture reuses the existing `featurized_batch` fixture (which already exists in this file).

Add these imports at the top of the file if not already present:

```python
from helpers.featurize import apply_conditioning_dropout
```

Add the tests after the existing `featurize_batch` tests:

```python
# ---------------------------------------------------------------------------
# apply_conditioning_dropout
# ---------------------------------------------------------------------------


def test_conditioning_dropout_p1_distogram_zeroes_all(featurized_batch):
    out = apply_conditioning_dropout(featurized_batch, p_distogram=1.0, p_atom=0.0, p_seq=0.0, device="cpu")
    # All valid residues should have their rows/cols zeroed
    assert out.gt_res_distogram.sum() == 0
    assert out.f_pseudo_beta_mask.sum() == 0


def test_conditioning_dropout_p1_atom_zeroes_all(featurized_batch):
    out = apply_conditioning_dropout(featurized_batch, p_distogram=0.0, p_atom=1.0, p_seq=0.0, device="cpu")
    assert not out.atom5_mask.any()


def test_conditioning_dropout_p1_seq_sets_all_to_mask_token(featurized_batch):
    out = apply_conditioning_dropout(featurized_batch, p_distogram=0.0, p_atom=0.0, p_seq=1.0, device="cpu")
    valid = featurized_batch.residue_mask
    assert (out.aa_indices[valid] == 20).all()


def test_conditioning_dropout_p0_is_noop(featurized_batch):
    out = apply_conditioning_dropout(featurized_batch, p_distogram=0.0, p_atom=0.0, p_seq=0.0, device="cpu")
    assert torch.equal(out.gt_res_distogram, featurized_batch.gt_res_distogram)
    assert torch.equal(out.atom5_mask, featurized_batch.atom5_mask)
    assert torch.equal(out.aa_indices, featurized_batch.aa_indices)


def test_conditioning_dropout_distogram_symmetric(featurized_batch):
    torch.manual_seed(0)
    out = apply_conditioning_dropout(featurized_batch, p_distogram=0.5, p_atom=0.0, p_seq=0.0, device="cpu")
    # If row i is zeroed, column i must also be zeroed (and vice versa)
    row_sums = out.gt_res_distogram.sum(dim=(2, 3))   # (B, N_res)
    col_sums = out.gt_res_distogram.sum(dim=(1, 3))   # (B, N_res)
    assert torch.equal(row_sums == 0, col_sums == 0)


def test_conditioning_dropout_respects_residue_mask(featurized_batch):
    # Padding residues (residue_mask=False) must not be changed
    batch_with_padding = dataclasses.replace(
        featurized_batch,
        residue_mask=torch.zeros_like(featurized_batch.residue_mask, dtype=torch.bool),
    )
    out = apply_conditioning_dropout(batch_with_padding, p_distogram=1.0, p_atom=1.0, p_seq=1.0, device="cpu")
    assert torch.equal(out.aa_indices, batch_with_padding.aa_indices)
```

Also add `import dataclasses` at the top of `test_featurize.py` if not already imported.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest pallatom/tests/helpers/test_featurize.py::test_conditioning_dropout_p1_distogram_zeroes_all -v
```

Expected: FAIL — `ImportError: cannot import name 'apply_conditioning_dropout'`

- [ ] **Step 3: Implement `apply_conditioning_dropout` in `featurize.py`**

Add this function to `pallatom/helpers/featurize.py`, after the `featurize_batch` function. The imports `dataclasses`, `rearrange`, `repeat`, `Bool`, `Int`, `Float`, `jaxtyped`, `beartype` are already present in the file.

```python
@jaxtyped(typechecker=beartype)
def apply_conditioning_dropout(
    batch: FeaturizedBatch,
    p_distogram: float,
    p_atom: float,
    p_seq: float,
    device: str,
) -> FeaturizedBatch:
    residue_mask: Bool[torch.Tensor, "B N_res"] = batch.residue_mask
    B, N_res = residue_mask.shape

    drop_d: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_distogram, device=device)).bool() & residue_mask
    )
    drop_a: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_atom, device=device)).bool() & residue_mask
    )
    drop_s: Bool[torch.Tensor, "B N_res"] = (
        torch.bernoulli(torch.full((B, N_res), p_seq, device=device)).bool() & residue_mask
    )

    # Distogram: zero rows AND columns for dropped residues (matrix is symmetric)
    keep_d: Bool[torch.Tensor, "B N_res"] = ~drop_d
    disto_mask: Bool[torch.Tensor, "B N_res N_res"] = (
        rearrange(keep_d, "b i -> b i 1") & rearrange(keep_d, "b j -> b 1 j")
    )
    new_distogram = batch.gt_res_distogram * rearrange(disto_mask.long(), "b i j -> b i j 1")
    new_pseudo_beta_mask = batch.f_pseudo_beta_mask * keep_d.float()

    # Atom mask: expand residue-level drop to atom level (5 atoms per residue)
    drop_a_expanded: Bool[torch.Tensor, "B N_atom"] = repeat(drop_a, "b n -> b (n a)", a=5)
    new_atom5_mask: Bool[torch.Tensor, "B N_atom"] = batch.atom5_mask & ~drop_a_expanded

    # Sequence: replace dropped tokens with mask-token index 20 ("X")
    new_aa_indices: Int[torch.Tensor, "B N_res"] = batch.aa_indices.masked_fill(drop_s, 20)

    return dataclasses.replace(
        batch,
        gt_res_distogram=new_distogram,
        f_pseudo_beta_mask=new_pseudo_beta_mask,
        atom5_mask=new_atom5_mask,
        aa_indices=new_aa_indices,
    )
```

- [ ] **Step 4: Run all dropout tests**

```bash
pytest pallatom/tests/helpers/test_featurize.py -k "conditioning_dropout" -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Run full featurize test suite**

```bash
pytest pallatom/tests/helpers/test_featurize.py -v --tb=short 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add pallatom/helpers/featurize.py pallatom/tests/helpers/test_featurize.py
git commit -m "feat(featurize): add apply_conditioning_dropout"
```

---

### Task 4: Add sequence embedding to `MainTrunk`

**Context:** `aa_indices` is computed in `featurize_batch` and stored in `FeaturizedBatch`, but `MainTrunk.forward` currently never reads it — there is no amino acid embedding table. This task creates one and wires it into `s_init`.

**Files:**
- Modify: `pallatom/architecture/main_trunk.py:307-308, 416-428`
- Test: `pallatom/tests/architecture/test_main_trunk.py`

- [ ] **Step 1: Write the failing test**

Add to `pallatom/tests/architecture/test_main_trunk.py`:

```python
def test_main_trunk_forward_with_mask_token_aa_indices(model, featurized_batch):
    # aa_indices containing 20 (mask token "X") must not raise IndexError
    masked_batch = dataclasses.replace(
        featurized_batch,
        aa_indices=torch.full((B, N_RES), 20, dtype=torch.long),
    )
    with torch.no_grad():
        r_denoised, *_ = model(masked_batch)
    assert r_denoised.shape == (B, N_ATOM, 3)
    assert torch.isfinite(r_denoised).all()
```

Also add at the top of `test_main_trunk.py`:

```python
import dataclasses
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest pallatom/tests/architecture/test_main_trunk.py::test_main_trunk_forward_with_mask_token_aa_indices -v
```

Expected: FAIL — either `IndexError` (embedding out of range) or the test catches that the current model doesn't use `aa_indices` at all. Either way it should fail because the model fixture is instantiated with default `n_amino=20` embedding that doesn't exist yet.

- [ ] **Step 3: Add `aa_embedding` to `MainTrunk.__init__`**

In `pallatom/architecture/main_trunk.py`, locate the `__init__` method. After line 308 (`n_amino: int = 20,`), no change to the signature. After line 316 (`self.time_fourier = TimeFourierEmbedding(c_res)`), add:

```python
        # Amino-acid sequence conditioning: 21 entries (0-19 = amino acids, 20 = mask token "X")
        self.aa_embedding = nn.Embedding(21, c_res)
```

The full block around it should look like:

```python
        # Step 3: time Fourier embedding
        self.time_fourier = TimeFourierEmbedding(c_res)

        # Amino-acid sequence conditioning: 21 entries (0-19 = amino acids, 20 = mask token "X")
        self.aa_embedding = nn.Embedding(21, c_res)

        # Step 5: relative position encoding → z_init
        self.rel_pos_enc = RelativePositionEncoding(c_pair)
```

- [ ] **Step 4: Use `aa_embedding` in `forward`**

In `pallatom/architecture/main_trunk.py`, find the forward method. After line 400 (`center_uid: Int[torch.Tensor, "B N_res"] = batch.center_uid`), add:

```python
        aa_indices: Int[torch.Tensor, "B N_res"] = batch.aa_indices
```

Then find step 2 / step 4 block (lines 414-428):

```python
        # Step 2: s_init = LinearNoBias(f_residue_idx)         [B, N_res, c_res]
        s_init: Float[torch.Tensor, "B N_res c_res"] = self.proj_residue_idx(f_residue_idx)

        # Step 3: t_i = TimeFourierEmbedding(¼·log(t̂/σ_data))  [B, N_res, c_res]
        log_val = 0.25 * math.log(t_hat / sd + 1e-8)
        log_arg = torch.full((B, N_res), log_val, device=device)
        t_i: Float[torch.Tensor, "B N_res c_res"] = self.time_fourier(log_arg)

        # Step 4: s_init += t_i
        s_init = s_init + t_i
```

Replace it with:

```python
        # Step 2: s_init = LinearNoBias(f_residue_idx)         [B, N_res, c_res]
        s_init: Float[torch.Tensor, "B N_res c_res"] = self.proj_residue_idx(f_residue_idx)

        # Step 2b: s_init += aa_embedding(aa_indices)
        # Clamp to [0, 20]: padding residues have aa_indices=-100 which clamps to 0
        # (padding is excluded from all downstream losses; embedding value doesn't matter).
        # Dropout-masked residues have aa_indices=20 and get the learned "X" embedding.
        aa_idx_clamped: Int[torch.Tensor, "B N_res"] = aa_indices.clamp(min=0, max=20)
        s_init = s_init + self.aa_embedding(aa_idx_clamped)

        # Step 3: t_i = TimeFourierEmbedding(¼·log(t̂/σ_data))  [B, N_res, c_res]
        log_val = 0.25 * math.log(t_hat / sd + 1e-8)
        log_arg = torch.full((B, N_res), log_val, device=device)
        t_i: Float[torch.Tensor, "B N_res c_res"] = self.time_fourier(log_arg)

        # Step 4: s_init += t_i
        s_init = s_init + t_i
```

- [ ] **Step 5: Run the new test**

```bash
pytest pallatom/tests/architecture/test_main_trunk.py::test_main_trunk_forward_with_mask_token_aa_indices -v
```

Expected: PASS

- [ ] **Step 6: Run full main_trunk test suite**

```bash
pytest pallatom/tests/architecture/test_main_trunk.py -v --tb=short 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pallatom/architecture/main_trunk.py pallatom/tests/architecture/test_main_trunk.py
git commit -m "feat(main_trunk): add aa_embedding for sequence conditioning (21 tokens incl. X)"
```

---

### Task 5: Mask dropped sequence tokens in CE losses

**Context:** `F.cross_entropy` only ignores target value `-100` by default. After `apply_conditioning_dropout`, dropped positions have `aa_indices=20`. These must be masked to `-100` before computing any CE loss, or the loss will index a non-existent class (output head has 20 classes, indices 0–19).

**Files:**
- Modify: `pallatom/train/train_loop.py`

- [ ] **Step 1: Add `_mask_seq_target` helper**

Near the top of `pallatom/train/train_loop.py` (after the imports, before the first function), add:

```python
def _mask_seq_target(aa_indices: torch.Tensor) -> torch.Tensor:
    """Replace mask-token index 20 with -100 so CE loss ignores dropped positions."""
    return aa_indices.masked_fill(aa_indices == 20, -100)
```

- [ ] **Step 2: Apply `_mask_seq_target` at every CE loss call**

Find every occurrence of `rearrange(featurized_batch.aa_indices, "b n -> (b n)")` used as a CE loss target. There are multiple call sites across all four functions (`evaluate_loop`, `ddp_evaluate_loop`, `train`, `ddp_train`). Replace each with `rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)")`.

Run this search to find all lines before editing:

```bash
grep -n "featurized_batch.aa_indices" pallatom/train/train_loop.py
```

For each line shown, wrap the `featurized_batch.aa_indices` with `_mask_seq_target(...)`. Example — change:

```python
CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
    rearrange(f_seq_logits, "b n c -> (b n) c"),
    rearrange(featurized_batch.aa_indices, "b n -> (b n)"),
)
```

to:

```python
CE_loss: Float[torch.Tensor, ""] = F.cross_entropy(
    rearrange(f_seq_logits, "b n c -> (b n) c"),
    rearrange(_mask_seq_target(featurized_batch.aa_indices), "b n -> (b n)"),
)
```

Apply the same pattern to every other occurrence (intermediate loss CE computations, ddp variants).

- [ ] **Step 3: Run existing tests to verify no regressions**

```bash
pytest pallatom/tests/train/ -v --tb=short 2>&1 | tail -15
```

Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add pallatom/train/train_loop.py
git commit -m "fix(train_loop): mask sequence dropout token (20) to -100 before CE loss"
```

---

### Task 6: Wire `apply_conditioning_dropout` into training loops

**Context:** There are four `featurize_batch` call sites in `train_loop.py`. Only the two training ones (inside `train` and `ddp_train`) should apply dropout; the two evaluation ones (`evaluate_loop`, `ddp_evaluate_loop`) must not.

The four call sites (by approximate line number — run `grep -n "featurize_batch" pallatom/train/train_loop.py` to confirm):
- `evaluate_loop` ≈ line 68 — **skip**
- `ddp_evaluate_loop` ≈ line 190 — **skip**
- `train` ≈ line 318 — **add dropout**
- `ddp_train` ≈ line 533 — **add dropout**

**Files:**
- Modify: `pallatom/train/train_loop.py:20` (import), `≈318`, `≈533`
- Test: `pallatom/tests/train/test_train_loop.py`

- [ ] **Step 1: Write the failing integration test**

Add to `pallatom/tests/train/test_train_loop.py`. The existing fixtures `model`, `loader`, `distogram_res`, `distogram_atom`, `index_embedding`, and `tcfg` are already defined in the file. Add these imports at the top if not already present:

```python
from train.train_config import ConditioningDropoutConfig
from helpers.featurize import featurize_batch, apply_conditioning_dropout
from train.train_loop import _to_protein_batch
```

Then add these tests:

```python
def test_training_step_with_full_dropout_produces_finite_outputs(
    model, loader, tcfg, distogram_res, distogram_atom, index_embedding
):
    batch = next(iter(loader))
    featurized = featurize_batch(
        _to_protein_batch(batch), tcfg, distogram_res, distogram_atom, index_embedding, device="cpu"
    )
    featurized = apply_conditioning_dropout(
        featurized,
        p_distogram=0.5,
        p_atom=0.5,
        p_seq=0.5,
        device="cpu",
    )
    with torch.no_grad():
        r_denoised, f_seq_logits, *_ = model(featurized)
    assert torch.isfinite(r_denoised).all()
    assert torch.isfinite(f_seq_logits).all()


def test_evaluate_loop_unchanged_by_dropout_config(
    model, loader, tcfg, distogram_res, distogram_atom, index_embedding
):
    from train.train_loop import evaluate_loop
    result = evaluate_loop(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert all(isinstance(v, float) for v in result.values())
```

- [ ] **Step 2: Run test to verify it fails (or passes partially)**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_training_step_with_full_dropout_produces_finite_loss -v
```

Expected: FAIL or ERROR (because `_to_protein_batch` may not be importable; or because dropout hasn't been wired into the train loop yet).

- [ ] **Step 3: Add import to `train_loop.py`**

In `pallatom/train/train_loop.py`, change line 20:

```python
from helpers.featurize import Distogram, ProteinBatch, featurize_batch
```

to:

```python
from helpers.featurize import Distogram, ProteinBatch, featurize_batch, apply_conditioning_dropout
```

- [ ] **Step 4: Wire dropout into `train` loop**

In the `train` function (around line 318), after the `featurize_batch` call, add:

```python
            featurized_batch = featurize_batch(
                _to_protein_batch(batch),
                tcfg,
                distogram_res,
                distogram_atom,
                index_embedding,
                device,
            )
            featurized_batch = apply_conditioning_dropout(
                featurized_batch,
                p_distogram=tcfg.conditioning_dropout.p_distogram,
                p_atom=tcfg.conditioning_dropout.p_atom,
                p_seq=tcfg.conditioning_dropout.p_seq,
                device=device,
            )
```

- [ ] **Step 5: Wire dropout into `ddp_train` loop**

In the `ddp_train` function (around line 533), apply the same addition after its `featurize_batch` call:

```python
            featurized_batch = featurize_batch(
                _to_protein_batch(batch),
                tcfg,
                distogram_res,
                distogram_atom,
                index_embedding,
                device,
            )
            featurized_batch = apply_conditioning_dropout(
                featurized_batch,
                p_distogram=tcfg.conditioning_dropout.p_distogram,
                p_atom=tcfg.conditioning_dropout.p_atom,
                p_seq=tcfg.conditioning_dropout.p_seq,
                device=device,
            )
```

- [ ] **Step 6: Run integration test**

```bash
pytest pallatom/tests/train/test_train_loop.py::test_training_step_with_full_dropout_produces_finite_loss -v
```

Expected: PASS

- [ ] **Step 7: Run full train test suite**

```bash
pytest pallatom/tests/train/ -v --tb=short 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 8: Run full test suite**

```bash
pytest pallatom/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add pallatom/train/train_loop.py pallatom/tests/train/test_train_loop.py
git commit -m "feat(train_loop): wire conditioning dropout into training loops"
```
