# API Request Schemas for Conditional Sampling

**Date:** 2026-05-09
**Scope:** `REST_APIs/api.py` — extend the existing FastAPI service to support all six
conditioning use cases identified in the test fixtures.

---

## Context

The existing `/sample` endpoint only supports unconditional generation and calls an
outdated `build_sampling_context` signature. `build_sampling_context` (in
`pallatom/sample/sampling.py`) already accepts the full set of conditioning inputs;
the API just needs schemas and server-side logic to expose them.

---

## Use Cases

Six conditioning scenarios, each a combination of three optional signals:

| # | Name | sequence | atom coords | template distogram |
|---|------|----------|-------------|-------------------|
| 1 | Unconditional | — | — | — |
| 2 | Sequence only | ✓ | — | — |
| 3 | Sequence + partial atoms | ✓ | partial | — |
| 4 | Partial template only | — | — | partial |
| 5 | Full template only | — | — | full |
| 6 | All atoms + partial template | — | full | partial |

All six use the same model and the same `/sample` endpoint.

---

## Request Schema

```python
class SampleRequest(BaseModel):
    # --- target ---
    n_res: int = Field(..., gt=0, le=512,
        description="Number of residues to generate")

    # --- conditioning (all optional) ---
    sequence: str | None = Field(None,
        description="AA sequence of length n_res. Characters from the standard 20 AAs "
                    "plus 'X' for unknown. Omit for no sequence conditioning.")
    structure_pdb: str | None = Field(None,
        description="PDB string providing atom-level conditioning. Residues present in "
                    "the PDB are filled into r_gt/atom5_mask; uncovered positions are "
                    "zeroed. May cover fewer than n_res residues.")
    template_pdb: str | None = Field(None,
        description="PDB string providing template-distogram conditioning. May cover "
                    "fewer than n_res residues (padded with zeros).")

    # --- sampler ---
    n_samples: int  = Field(1,     gt=0, le=10)
    ddim_steps: int = Field(40,    gt=1)
    rho: float      = Field(7.0,   gt=0)
    S_churn: float  = Field(0.0,   ge=0)
    S_noise: float  = Field(1.003, gt=0)
```

### Response (unchanged)

```python
class SampleResponse(BaseModel):
    pdb_strings: list[str]
    n_res: int
    n_samples: int
    device: str
```

---

## Mapping: Request Fields → `build_sampling_context` Arguments

| `build_sampling_context` param | Derived from |
|---|---|
| `atom_positions` | Parse `structure_pdb` with `protein_from_pdb`; zero-pad to `(n_res, 37, 3)`. If None → `zeros(n_res, 37, 3)`. |
| `atom_mask` | Same parse; zero-pad to `(n_res, 37)`. If None → `zeros(n_res, 37)`. |
| `seq` | `sequence` if provided; else `"X" * n_res`. Never `"A" * n_res` for the no-sequence case. |
| `residue_index` | If `structure_pdb` provided → PDB's actual `residue_index` (non-contiguous integers preserved; tail positions beyond PDB length continue sequentially from last PDB index + 1). If None → `arange(n_res)`. |
| `pdb_files` | Write `template_pdb` to a `tempfile.NamedTemporaryFile` and pass its path; if None → `[]`. |

### Residue index detail

PDB files commonly have non-contiguous residue numbering (e.g. 1…12, 25…50 due to
disordered loops). `protein_from_pdb` already preserves the actual `_resseq` column.
When `structure_pdb` covers `n_pdb < n_res` residues, the remaining `n_res - n_pdb`
positions are padded sequentially: `[pdb_last_idx + 1, pdb_last_idx + 2, ...]`.
`residue_index` is **never** a zeros tensor — at minimum it is `arange(n_res)`.

### 'X' residues

`restype_order` contains only the 20 standard amino acids; `'X'` is not in it.
`build_AA_context` currently uses `restype_order[r]`, which raises `KeyError` on `'X'`.
This must be changed to `restype_order.get(r, 20)` to match the convention already
used in `protein_from_pdb` (index 20 = unknown). `'X'` and `'A'` must map to
**different indices** (0 vs. 20) — using `"A" * n_res` as the no-sequence default
would incorrectly signal alanine conditioning.

---

## Server-Side Implementation Notes

### Startup (`lifespan`)

Two distogram functions are constructed once and stored in `_state`:

```python
_state["atom_disto"] = Distogram(n_bins=22, min_dist=2.0, max_dist=22.0).to(DEVICE)
_state["templ_disto"] = Distogram(
    n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
).to(DEVICE)
```

The checkpoint save format needs updating: the current checkpoint only saves
`model` weights; `index_embedding` (used in the old API) is no longer needed since
`build_sampling_context` handles residue encoding internally.

### `_run_sampling` logic

```
1. Parse structure_pdb  →  (atom_positions, atom_mask, residue_index_from_pdb)
   - if None            →  zeros, zeros, None
2. Build residue_index  →  pdb residue indices (padded) or arange(n_res)
3. Build seq            →  sequence or "X" * n_res
4. Write template_pdb to tempfile (within a try/finally)  →  pdb_files=[path] or []
5. Call build_sampling_context(...)
6. Run EDMPrecond + EDMSampler
7. Convert atom5 → atom37 → Protein → PDB string per sample
8. Delete tempfile in finally block (guaranteed even on exception)
```

### Runtime validation (raises HTTP 422)

- If `sequence` is provided: `len(sequence) == n_res`
- If `sequence` is provided: all characters in `restype_order` or `== 'X'`
- If `structure_pdb` is provided: parsed protein must have `≤ n_res` residues

---

## Known Limitations

### Training gap — conditioning dropout not implemented

The model was trained exclusively via `featurize_batch`, which always provides:
- the protein's own full Cβ template distogram as `gt_res_distogram`
- all real atom positions (non-zero `atom5_mask`)
- the actual amino-acid sequence (no masking)

**None of the six inference use cases except approximately use case 5 (full template)
were present in the training distribution.** Use cases 1–4 and 6 pass zero or partial
conditioning signals that the model has never seen during training.

To fix this properly, `featurize_batch` needs classifier-free-guidance-style dropout:
randomly zeroing `gt_res_distogram`, `atom5_mask`, and/or `aa_indices` with some
probability during training. This is tracked as a future training requirement and is
out of scope for this API work.

---

## Files Changed

| File | Change |
|---|---|
| `REST_APIs/api.py` | Replace `SampleRequest`, rewrite `_run_sampling`, update `lifespan` |
| `pallatom/sample/sampling.py` | `build_AA_context`: `restype_order[r]` → `restype_order.get(r, 20)` |
