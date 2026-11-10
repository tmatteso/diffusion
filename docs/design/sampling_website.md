# Design: minimum viable sampling website

**Status:** Draft, for review
**Scope:** A local web UI that samples proteins unconditionally from one
trained `MainTrunk` checkpoint and shows the results in 3D.
**Stack:** FastAPI (Python 3.12) backend, React + TypeScript (Vite) frontend.

---

## 1. Problem

Today the only way to look at samples is the notebook path described in
[`pallatom/README.md`](../../pallatom/README.md): write a `SampleConfig` to
JSON, run `sample/sampling.py` as a subprocess, read `samples.json` back, and
render one PDB string at a time with `py3Dmol`. Each step is manual, and you
need a Jupyter kernel on the GPU box.

We want a page where you choose a length and a sample count, click
**Sample**, and get rotatable structures, their sequences and PDB downloads
back.

## 2. Goals and non-goals

**Goals (MVP)**

1. Load **one** checkpoint (EMA weights) when the server starts, and keep it
   resident on the GPU, or on the CPU on arm64.
2. Run unconditional sampling with a user-chosen `n_res` and `n_samples`, an
   optional `seed`, and an optional `ddim_steps` override.
3. Show each sample in an interactive 3D viewer, coloured as a cartoon
   spectrum like the notebook, next to its decoded one-letter sequence.
4. Let the user download each sample as a `.pdb` file.
5. Serve the whole thing as one process on one port, so that
   `ssh -L 8000:localhost:8000 <gpu-box>` is the whole deployment story.
6. Hold the new Python code to the same quality bar as the rest of
   `pallatom/` (basedpyright `all`, ruff `ALL`, jaxtyping, einops, and so on),
   and give the TypeScript an equally strict bar.

**Non-goals (explicitly deferred)**

- Conditional sampling, such as templates, motifs or a fixed sequence.
  `build_sampling_context(pdb_file_path=None, ...)` stays `None`.
- Switching checkpoints from the UI, or uploading checkpoints.
- Authentication, multiple users, persistent job history and databases.
- Streaming the denoising trajectory, or a progress bar (see §9, Phase 2).
- Public internet hosting.

## 3. Constraints from the existing code

| Fact | Consequence for the design |
|------|----------------------------|
| A checkpoint `.pt` holds only `model`, `optimizer` and `ema` state dicts (`train/train_loop.py`). It holds **no** architecture config. | The server cannot rebuild `MainTrunk` from the `.pt` file alone. It must start from a `SampleConfig` JSON, which already carries `model`, `noise`, the distogram params and `checkpoint.checkpoint_path`. |
| `sample/sampling.py::main` mixes four jobs: model loading, context building, `EDMSampler.sample`, and conversion to PDB plus writing the file. | Refactor these into reusable functions (§5.1) so that the CLI and the API share one code path. |
| `EDMSampler.sample` draws from torch's global RNG. | Seeding is `torch.manual_seed(seed)`. That is safe only if a single sampling job runs at a time, which the MVP enforces anyway (§5.4). |
| `fastapi`, `uvicorn[standard]` and `pydantic` are already in `pyproject.toml`. | The backend needs no new Python dependencies. |
| basedpyright type-checks only `pallatom/`, and `extraPaths` resolves imports without the `pallatom.` prefix. | Put the backend at `pallatom/serve/` so that it gets full type checking and imports like `from sample.sampling import ...`. |
| The devcontainer has no Node toolchain. | Add the `node` devcontainer feature (§7). |
| `CLAUDE.md` bans `Any` and `TypedDict`. | Every API payload is a Pydantic model. The FastAPI handlers avoid `app.state`, which is `Any`, and bind a typed service object instead (§5.3). |

## 4. Architecture

```
┌──────────────────────────── browser ────────────────────────────┐
│  React + TS (Vite build)                                         │
│  SampleForm ──► api.ts (typed fetch) ──► SampleGallery           │
│                                         └─ StructureViewer (3Dmol)│
└───────────────▲──────────────────────────────┬───────────────────┘
                │ static assets (web/dist)     │ JSON /api/*
┌───────────────┴──────────────────────────────▼───────────────────┐
│  uvicorn ── FastAPI app  (pallatom/serve/app.py)                  │
│    StaticFiles("/")        APIRouter("/api")                      │
│                              └─ SamplingService (one instance)    │
│                                   ├─ MainTrunk (EMA, eval, device)│
│                                   ├─ distogram modules            │
│                                   └─ threading.Lock               │
│                                        │                          │
│                     sample/sampling.py: build_sampling_context,   │
│                     EDMSampler, structures_to_pdb  (refactored)   │
└───────────────────────────────────────────────────────────────────┘
```

Repository layout:

```
pallatom/
  sample/sampling.py        # refactored: load_model / run_sampling / to_samples
  serve/
    __init__.py
    README.md
    api_models.py           # Pydantic request/response models
    service.py              # SamplingService: owns model + lock
    app.py                  # create_app(), CLI entry point (uvicorn)
  tests/serve/              # TestClient tests against a tiny CPU model
web/
  package.json  tsconfig.json  vite.config.ts  eslint.config.js
  index.html
  src/
    main.tsx  App.tsx
    api/client.ts           # typed fetch wrapper
    api/schema.d.ts         # generated from FastAPI's OpenAPI (committed)
    components/SampleForm.tsx
    components/SampleGallery.tsx
    components/SampleCard.tsx
    components/StructureViewer.tsx
```

## 5. Backend

### 5.1 Refactor `sample/sampling.py` (no behaviour change)

Split `main` into three public functions. The CLI then becomes a thin
composition of them, and its output stays byte-for-byte identical for a
fixed seed.

```python
def load_model(scfg: SampleConfig, device: str) -> LoadedModel: ...
    # builds MainTrunk + distogram modules, loads ckpt["ema"], .eval()

def run_sampling(
    loaded: LoadedModel,
    sampler_params: SamplerParams,
    n_res: int,
    n_samples: int,
) -> tuple[Float[torch.Tensor, "B N_atom 3"], Float[torch.Tensor, "B N_res"]]: ...
    # build_sampling_context + EDMSampler.sample, under torch.inference_mode()

def to_samples(
    coords: Float[torch.Tensor, "B N_atom 3"],
    seqs: Float[torch.Tensor, "B N_res"],
) -> list[SampledProtein]: ...
    # atom5_to_atom37 → Protein → to_pdb, plus one-letter sequence
```

`LoadedModel` and `SampledProtein` (`pdb: str`, `sequence: str`) are frozen
`@dataclass`es. The refactor lands as its own PR, together with a regression
test showing that the CLI output is unchanged.

### 5.2 API surface

All routes live under `/api`. Every request and response is a Pydantic model
in `serve/api_models.py`. FastAPI publishes them as OpenAPI, and the frontend
types are generated from that (§6.3).

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `GET` | `/api/health` | — | `{status: "ok", device: str}` |
| `GET` | `/api/model` | — | `ModelInfo`: checkpoint file name (basename only, never the full path), the default `SamplerParams`, and the server's limits (`max_n_res`, `max_n_samples`, `max_ddim_steps`) |
| `POST` | `/api/sample` | `SampleRequest` | `SampleResponse` |

```python
class SampleRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    n_res: int = Field(default=100, ge=MIN_N_RES, le=MAX_N_RES)
    n_samples: int = Field(default=4, ge=1, le=MAX_N_SAMPLES)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    ddim_steps: int | None = Field(default=None, gt=1, le=MAX_DDIM_STEPS)

class SampleOut(BaseModel):
    index: int
    sequence: str          # one-letter, length n_res
    pdb: str

class SampleResponse(BaseModel):
    seed: int              # echoed; server draws one if the request had None
    n_res: int
    ddim_steps: int
    elapsed_s: float
    samples: list[SampleOut]
```

The proposed limits are `MIN_N_RES = 32`, `MAX_N_RES = 256`,
`MAX_N_SAMPLES = 8` and `MAX_DDIM_STEPS = 500`. They bound GPU memory, since
pair features are O(B·N_res²). Tune them once we have the measurements in
§8.

Errors:

| Status | When |
|--------|------|
| `422` | Validation failed (FastAPI default) |
| `409` | A sampling job is already running (§5.4) |
| `503` | The model failed to load at startup |
| `500` | CUDA OOM or another sampler failure, with a short message; the full traceback is logged through structlog only |

### 5.3 Typed wiring without `app.state`

`app.state` is typed as `Any`, so going through it would trip
`reportAny`. Instead, `create_app` takes a fully constructed service and
registers the service's **bound methods** as routes. This needs no closures
and no globals:

```python
def create_app(service: SamplingService, static_dir: Path | None) -> FastAPI:
    app = FastAPI(title="pallatom sampler")
    app.add_api_route("/api/health", service.health, methods=["GET"])
    app.add_api_route("/api/model", service.model_info, methods=["GET"])
    app.add_api_route("/api/sample", service.sample, methods=["POST"])
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True))
    return app
```

The entry point is `python pallatom/serve/app.py --config
sample_config.json [--host 127.0.0.1] [--port 8000] [--static web/dist]`. It
validates the `SampleConfig`, builds the service (which loads the model
eagerly), and calls `uvicorn.run(app, ...)`. Loading eagerly means a bad
config or checkpoint fails at startup, not on the first request.

### 5.4 Concurrency model

- `service.sample` is a **sync** `def`, so FastAPI runs it in its threadpool,
  and a long GPU call never blocks the event loop or `/api/health`.
- A `threading.Lock` is held for the whole sampling job. If
  `lock.acquire(blocking=False)` fails, the server returns `409 Busy`
  immediately instead of queueing. With one GPU and one model, a queue adds
  little for a single-user MVP. The frontend disables the button while a
  request is in flight, so the 409 only shows up when two tabs are open.
- `torch.manual_seed(seed)` runs inside the lock, so a given seed is
  reproducible on a given device.
- The request is synchronous. If sampling takes longer than about 60 s, the
  long-request problem moves to the reverse proxy or tunnel, and Phase 2
  replaces this with jobs (§9).

### 5.5 Logging and security

- Reuse `StructlogConfig` (wrapped around `uvicorn.run`). Log one line per
  request with `n_res`, `n_samples`, `seed`, `elapsed_s` and the device.
- Bind to `127.0.0.1` by default. There is no auth, so remote access goes
  through an SSH tunnel.
- The checkpoint path comes **only** from the server-side config, never from a
  request. `torch.load` runs with `weights_only=True`, which is already the
  default on torch 2.11; write it explicitly anyway.
- Do not add CORS middleware. In dev, Vite proxies `/api`, and in prod the
  page and the API share an origin.

## 6. Frontend

### 6.1 Stack

- **Vite + React 19 + TypeScript**, strict mode. No router and no state
  library: one page, `useState` and `useReducer`.
- **3Dmol.js** (npm `3dmol`) for the viewer. It is the JavaScript library
  under `py3Dmol`, so the notebook view carries over directly:
  `addModel(pdb, "pdb")`, cartoon with `color: "spectrum"`, then `zoomTo()`.
- No CSS framework. Use a small hand-written stylesheet with a responsive
  CSS grid for the gallery.

### 6.2 UI

```
┌──────────────────────────────────────────────────────────────┐
│ pallatom sampler        checkpoint: pallatom_best.pt · cuda  │
├──────────────────────────────────────────────────────────────┤
│ Length [ 100 ]  Samples [ 4 ]  Seed [      ] (blank = random)│
│ ▸ Advanced: DDIM steps [ 200 ]                    [ Sample ] │
├──────────────────────────────────────────────────────────────┤
│ seed 12345 · 4 × 100 res · 200 steps · 38.2 s                │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────┐ │
│ │  3D viewer   │ │  3D viewer   │ │  3D viewer   │ │  ...  │ │
│ │              │ │              │ │              │ │       │ │
│ │ MKLV…(copy)  │ │ ...          │ │ ...          │ │       │ │
│ │ [Download]   │ │ [Download]   │ │ [Download]   │ │       │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └───────┘ │
└──────────────────────────────────────────────────────────────┘
```

- The form's limits and defaults come from `GET /api/model`, so the UI
  never hard-codes a range the server would reject.
- While sampling, the button shows a spinner with elapsed time and is
  disabled. Errors (409, 422, 500) appear in an inline banner.
- **Download** builds a `Blob` client-side and saves it as
  `pallatom_seed{seed}_{index}.pdb`. There is no extra endpoint.
- Each `StructureViewer` owns one 3Dmol viewer. It creates the viewer in a
  `useEffect` and calls `viewer.clear()` on cleanup, so WebGL contexts don't
  leak. Browsers cap these at about 16, which is another reason
  `MAX_N_SAMPLES` is 8.

### 6.3 Keeping types in sync

`npm run gen:api` runs `openapi-typescript` against
`http://localhost:8000/openapi.json` and writes `src/api/schema.d.ts`, which
is committed. `api/client.ts` takes its request and response types only from
that file. CI regenerates the file from a CPU-only app instance and fails on
a diff, so a backend schema change cannot silently break the frontend.

### 6.4 Strictness (the TypeScript counterpart of the Python bar)

- `tsconfig`: `strict`, `noUncheckedIndexedAccess`,
  `exactOptionalPropertyTypes`, `noImplicitOverride`.
- ESLint (`typescript-eslint` `strictTypeChecked`):
  `@typescript-eslint/no-explicit-any: error` and the `no-unsafe-*` rules.
  These mirror the repo-wide `Any` ban.
- Prettier for formatting, with an 80-column print width to match Black.

## 7. Tooling, devcontainer and CI

- **Devcontainer:** add `"features": {"ghcr.io/devcontainers/features/node:1":
  {"version": "22"}}` to both `cuda/` and `arm64/` `devcontainer.json`. The
  CUDA container already uses `--network=host`. For arm64, add
  `forwardPorts: [8000, 5173]`.
- **`.gitignore`:** `web/node_modules/`, `web/dist/`.
- **Pre-commit** (`language: system`, scoped with `files: ^web/`):
  `tsc --noEmit`, `eslint`, `prettier --check`. The `language: system` choice
  follows the existing rule that tool versions come from the lockfile
  (`web/package-lock.json`), not from `rev:` pins.
- **CI:** the static-checks job already runs `pre-commit run --all-files`
  inside the devcontainer, so the new hooks run in CI automatically once Node
  is in the image. Add `npm ci && npm run build && npm test` to the
  `test` job, next to `pytest pallatom`.
- **Python lint scope:** `pallatom/serve/` falls under every existing hook
  unchanged. It needs no new carve-outs.

## 8. Testing

**Backend** (`pallatom/tests/serve/`, following the tests style guide)

- Build a **tiny** `MainTrunk` on CPU from small `ModelParams` with random
  weights. It is saved to `tmp_path` as `{"ema": state_dict}`, which
  exercises the real `load_model` path.
- Use `TestClient(create_app(service, static_dir=None))` to test:
  - that `/api/health` and `/api/model` have the right shapes and limits
  - that `/api/sample` with `ddim_steps=2`, `n_res=MIN_N_RES` and
    `n_samples=2` returns 2 parseable PDBs, sequences of length `n_res`, and
    the echoed seed
  - that the same seed gives identical PDBs
  - `422` for out-of-range or extra fields
  - `409` when the lock is already held, by acquiring
    `service.lock` in the test
- A regression test for the refactor: the old and new CLI output match for a
  fixed seed.

**Frontend** (Vitest + React Testing Library)

- `SampleForm` clamps values to the limits from `/api/model`, and disables
  itself while busy.
- `client.ts` maps 409, 422 and 500 to typed error results (`fetch` is
  mocked).
- `StructureViewer` is smoke-tested with `3dmol` mocked, because jsdom has no
  WebGL.

**Manual acceptance:** on the GPU box, take `pallatom_toy_best.pt` and its
config, sample 4 × 100 residues, and check that the structures render and
the downloaded PDBs open in PyMOL. Record the wall time and peak GPU memory
at `n_res ∈ {64, 128, 256}` and `n_samples ∈ {1, 4, 8}`, and use those numbers
to finalise the limits in §5.2.

## 9. Delivery plan

| PR | Content | Ships value on its own? |
|----|---------|-------------------------|
| 1 | Refactor `sampling.py` into `load_model` / `run_sampling` / `to_samples`, plus the regression test | Yes: the CLI is cleaner and reusable |
| 2 | `pallatom/serve/` (API models, service, app) and its tests | Yes: the API is usable with `curl` |
| 3 | `web/` scaffold, Node devcontainer feature, pre-commit hooks, CI steps | Partly: tooling only |
| 4 | Frontend UI, OpenAPI type generation, `StaticFiles` mount, README section | Yes: the MVP is complete |

**Phase 2 (after the MVP, not in scope here):** `POST /api/jobs` returning a
job id, Server-Sent Events for per-step progress (`EDMSampler.sample` would
take an optional step callback), cancellation, a checkpoint picker that lists
a configured directory, and an animated view of the trajectory.

## 10. Open questions

1. **Embed the config in checkpoints?** If `train_loop.py` also saved
   `TrainConfig.model_dump()` into the `.pt`, the server could start from a
   checkpoint alone, and a config and checkpoint could never be mismatched.
   That would be a small change to training, but it would leave older
   checkpoints in a separate format.
2. **Where will it run?** Locally on the CUDA box behind an SSH tunnel (this
   doc's assumption), or on a rented instance that people reach from
   outside? The second option would bring auth and job queueing into scope.
3. **Sync request vs jobs.** We need §8's timing numbers. If a typical
   request takes much longer than 60 s, the job API should move from Phase 2
   into the MVP.
4. **Node in the arm64 image.** Is the extra image size acceptable, or
   should the frontend be built in a separate CI job outside the
   devcontainer?
