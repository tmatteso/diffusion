# Design: unconditional sampling website

**Status:** Draft, for review
**Scope of this doc:** The MVP (Phase 1) in full detail. Phases 2–4 are
covered in enough detail that the MVP does not paint them into a corner.
**Stack:** FastAPI (Python 3.12) backend; React + TypeScript (Vite) frontend
deployed to GitHub Pages.

---

## 1. Problem

Today the only way to look at samples is the notebook path described in
[`pallatom/README.md`](../../pallatom/README.md). You write a `SampleConfig`
to JSON, run `sample/sampling.py` as a subprocess, read `samples.json` back,
and render one PDB string at a time with `py3Dmol`. Every step is manual,
and it only works for someone with a Jupyter kernel on the GPU box.

We want a public web page where anyone can pick a length and a sample count,
click **Sample**, and get back rotatable 3D structures, their sequences, and
PDB downloads.

## 2. Roadmap at a glance

| Phase | What | Auth | Job transport | Hosts |
|-------|------|------|---------------|-------|
| **1 (MVP)** | Sampling website | **none** | in-process queue + worker thread | Frontend on GitHub Pages; API + worker on the current GPU box |
| 2 | Two-machine job queue | yes | **RabbitMQ** | ARM64 cloud host (gateway, broker, auth) + GPU box (workers) |
| 3 | Orchestration | yes | RabbitMQ | **k3s** + NVIDIA GPU Operator, Gateway API `HTTPRoute`, **KEDA** worker autoscaling |
| 4 | Observability | yes | RabbitMQ | **Prometheus, Grafana, Alertmanager, Loki** on the cluster |

The rest of this doc is about Phase 1. §9 describes the later phases and
lists the MVP decisions that exist to make them cheap.

## 3. MVP goals and non-goals

**Goals**

1. Load **one** checkpoint (EMA weights) when the server starts, and keep it
   resident on one GPU of the current machine.
2. Run unconditional sampling with a user-chosen `n_res` and `n_samples`, an
   optional `seed`, and an optional `ddim_steps` override.
3. Shape the API around jobs: submit a job, then poll it. Put the transport
   behind an interface, so that Phase 2 swaps in RabbitMQ without changing
   the HTTP contract or the frontend.
4. Show each sample in an interactive 3D viewer (cartoon, spectrum colouring,
   as in the notebook) next to its one-letter sequence, with a `.pdb`
   download.
5. Host the frontend on **GitHub Pages**
   (`https://tmatteso.github.io/diffusion/`). The repo is public and Pages
   is already set to deploy from Actions.
6. Hold the new Python code to the same quality bar as the rest of
   `pallatom/`, and give the TypeScript an equally strict bar.

**Non-goals for the MVP**

- Auth, RabbitMQ, or a second machine (Phase 2).
- Containers or Kubernetes (Phase 3).
- Metrics endpoints or dashboards (Phase 4). We still emit structured logs
  now (§5.6).
- Conditional sampling (templates, motifs, fixed sequence), switching
  checkpoints, cancelling jobs, or streaming the denoising trajectory.
- Keeping jobs across server restarts. Job results live in memory.

## 4. Constraints from the existing code and environment

| Fact | Consequence |
|------|-------------|
| A checkpoint `.pt` holds only `model`, `optimizer` and `ema` state dicts (`train/train_loop.py`). It has **no** architecture config. | The server starts from a `SampleConfig` JSON, which carries `model`, `noise`, the distogram params and `checkpoint.checkpoint_path`. |
| `sample/sampling.py::main` mixes model loading, context building, `EDMSampler.sample`, and PDB conversion and writing. | Refactor these into reusable functions (§5.1). The CLI, the MVP worker thread and the Phase 2 RabbitMQ consumer then share one code path. |
| `EDMSampler.sample` draws from torch's global RNG. | `torch.manual_seed(seed)` is only safe when one job runs at a time per process. The single worker thread guarantees that. |
| `fastapi`, `uvicorn[standard]` and `pydantic` are already dependencies. | The MVP backend needs no new Python dependencies. |
| basedpyright type-checks `pallatom/`, and `extraPaths` resolves imports without the `pallatom.` prefix. | The backend lives at `pallatom/serve/`. |
| `CLAUDE.md` bans `Any` and `TypedDict`. | Every payload is a Pydantic model. The handlers do not use `app.state`, which is `Any`; they are bound methods on typed objects (§5.4). |
| GitHub Pages serves over **HTTPS** from a different origin than the API. | The API must be reachable over HTTPS, or browsers block it as mixed content, and it must send **CORS** headers for the Pages origin (§5.5). |
| The GPU box has two RTX 4070 SUPERs. | The MVP runs one worker on one GPU (`--device cuda:0`). The second GPU becomes a second consumer in Phase 2. |
| The devcontainer has no Node toolchain. | Install Node and pnpm in the shared `.devcontainer/Dockerfile` (§7). |

## 5. Backend (Phase 1)

### 5.1 Refactor `sample/sampling.py` (no behaviour change)

Split `main` into three public functions. The CLI then becomes a thin
composition of them, and its output for a fixed seed stays byte-for-byte
the same.

```python
def load_model(scfg: SampleConfig, device: str) -> LoadedModel: ...
    # builds MainTrunk + distogram modules, loads ckpt["ema"], .eval()

def run_sampling(
    loaded: LoadedModel,
    sampler_params: SamplerParams,
    n_res: int,
    n_samples: int,
) -> tuple[Float[torch.Tensor, "B N_atom 3"], Float[torch.Tensor, "B N_res"]]: ...
    # build_sampling_context + EDMSampler.sample under torch.inference_mode()

def to_samples(
    coords: Float[torch.Tensor, "B N_atom 3"],
    seqs: Float[torch.Tensor, "B N_res"],
) -> list[SampledProtein]: ...
    # atom5_to_atom37 → Protein → to_pdb, plus one-letter sequence
```

`LoadedModel` and `SampledProtein` (`pdb: str`, `sequence: str`) are frozen
`@dataclass`es.

### 5.2 Layout: the transport is separate from the work

```
pallatom/serve/
  messages.py     # Pydantic: SampleJobSpec, SampleJob, SampleResult, JobStatus…
  worker.py       # execute_job(job, loaded) -> SampleResult   (no I/O, no transport)
  queue.py        # JobQueue / JobStore Protocols + InProcess implementations
  api.py          # SamplerApi: typed HTTP handlers (bound methods)
  app.py          # create_app(), CLI entry point (uvicorn)
  README.md
pallatom/tests/serve/
```

Phase 2 will use these parts as follows:

- `messages.py` already defines the **wire format**. The JSON that
  `SampleJob.model_dump_json()` produces in the MVP is exactly the AMQP
  message body that Phase 2 publishes.
- `execute_job` is a pure function of a job and a loaded model. In the MVP,
  a worker thread calls it. In Phase 2, a RabbitMQ consumer process calls it
  unchanged.
- The MVP has `InProcessJobQueue`, built on `queue.Queue` with a single
  worker thread, and `InMemoryJobStore`, a bounded dict with a TTL. Phase 2
  adds `RabbitMQJobQueue` and a persistent store behind the same
  `typing.Protocol`s.

```python
class JobQueue(Protocol):
    def submit(self, job: SampleJob) -> None: ...      # raises QueueFull
    def depth(self) -> int: ...

class JobStore(Protocol):
    def put(self, status: JobStatus) -> None: ...
    def get(self, job_id: UUID) -> JobStatus | None: ...
```

### 5.3 API surface

All routes live under `/api`. Request and response types are the Pydantic
models in `messages.py`, which FastAPI publishes as OpenAPI. The frontend
types are generated from that (§6.3).

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `GET` | `/api/health` | — | `{status: "ok", device, queue_depth}` |
| `GET` | `/api/model` | — | `ModelInfo`: checkpoint file name (basename only), default `SamplerParams`, and the limits `max_n_res`, `max_n_samples` and `max_ddim_steps` |
| `POST` | `/api/jobs` | `SampleJobSpec` | `202` + `JobStatus` (`job_id`, `state: "queued"`, `position`) |
| `GET` | `/api/jobs/{job_id}` | — | `JobStatus` |

```python
class SampleJobSpec(BaseModel):          # what the user asks for
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    n_res: int = Field(default=100, ge=MIN_N_RES, le=MAX_N_RES)
    n_samples: int = Field(default=4, ge=1, le=MAX_N_SAMPLES)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    ddim_steps: int | None = Field(default=None, gt=1, le=MAX_DDIM_STEPS)

class SampleJob(BaseModel):              # what goes on the queue (AMQP body in Phase 2)
    job_id: UUID
    spec: SampleJobSpec
    seed: int                            # resolved: drawn by the API if spec.seed is None
    submitted_at: datetime
    schema_version: Literal[1] = 1

class SampleOut(BaseModel):
    index: int
    sequence: str
    pdb: str

class SampleResult(BaseModel):
    job_id: UUID
    seed: int
    n_res: int
    ddim_steps: int
    elapsed_s: float
    samples: list[SampleOut]

class JobStatus(BaseModel):
    job_id: UUID
    state: Literal["queued", "running", "succeeded", "failed"]
    position: int | None = None          # queued only
    result: SampleResult | None = None   # succeeded only
    error: str | None = None             # failed only: short, user-safe message
```

`schema_version` lets Phase 2 consumers reject or upgrade messages that a
newer or older gateway puts on the queue.

The proposed limits are `MIN_N_RES = 32`, `MAX_N_RES = 256`,
`MAX_N_SAMPLES = 8`, `MAX_DDIM_STEPS = 500` and `MAX_QUEUE_DEPTH = 4`. They
bound GPU memory (pair features are O(B·N_res²)) and bound the queue, since
the endpoint is public with no auth. Tune them once we have the measurements
in §8.

| Status | When |
|--------|------|
| `202` | Job accepted |
| `404` | Unknown or expired `job_id` |
| `422` | Validation failed |
| `429` | The queue is full, or this client exceeded its rate limit (§5.5); includes `Retry-After` |
| `503` | The model failed to load |

Sampler failures, including CUDA OOM, never surface as HTTP 500. They become
`state: "failed"` with a short message, and the traceback goes to the logs
only.

**Why jobs instead of one long request, even in the MVP:** a 200-step sample
can take tens of seconds or more. Polling keeps every HTTP call short, which
avoids proxy and tunnel timeouts (Cloudflare cuts requests off at 100 s).
Polling is also the contract the RabbitMQ design needs anyway. The frontend
polls `GET /api/jobs/{id}` every 1.5 s.

### 5.4 Typed wiring and the worker

```python
def create_app(api: SamplerApi, allowed_origins: list[str]) -> FastAPI:
    app = FastAPI(title="pallatom sampler")
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.add_api_route("/api/health", api.health, methods=["GET"])
    app.add_api_route("/api/model", api.model_info, methods=["GET"])
    app.add_api_route("/api/jobs", api.submit, methods=["POST"], status_code=202)
    app.add_api_route("/api/jobs/{job_id}", api.status, methods=["GET"])
    return app
```

- Routes are bound methods on `SamplerApi`, which holds a `JobQueue`, a
  `JobStore` and the `ModelInfo`. This avoids closures, globals and
  `app.state`.
- `InProcessJobQueue` starts **one** daemon worker thread. The thread loops
  over `queue.get()`, sets the store to `running`, seeds torch, calls
  `execute_job`, and stores `succeeded` or `failed`. Because there is a
  single consumer per process, no lock is needed, and seeds are
  reproducible on a given device.
- The entry point is `python pallatom/serve/app.py --config
  sample_config.json --device cuda:0 --allowed-origin
  https://tmatteso.github.io [--host 127.0.0.1] [--port 8000]`. It loads the
  model eagerly, so a bad config or checkpoint fails at startup.

### 5.5 Public exposure without auth

The GitHub Pages page needs an HTTPS URL for the API. The server keeps
binding to `127.0.0.1` and is published through **Cloudflare Tunnel**
(`cloudflared`). The tunnel connects outbound, so we open no inbound ports
and need no TLS certificates on the box. That is also the same
outbound-only pattern the GPU box keeps in Phase 2.

- **Named tunnel (recommended):** a stable hostname such as
  `sampler.<your-domain>`. This needs a domain on Cloudflare (the free plan
  is enough).
- **Quick tunnel (fallback if there is no domain):** a random
  `*.trycloudflare.com` URL that changes every time the tunnel restarts. The
  frontend therefore accepts a runtime override, `?api=<url>`, remembered in
  `localStorage`, on top of the URL baked in at build time (§6.4).

The endpoint is public and has no auth, so we rely on hard limits instead:

- The request limits and the bounded queue (§5.3) cap how much GPU time any
  burst can consume.
- A per-client token bucket allows about 10 job submissions an hour per IP.
  It keys on the `CF-Connecting-IP` header, which is trusted only because
  the server listens solely on loopback behind the tunnel. It is in memory
  and implemented in `api.py`, with no new dependency.
- CORS allows only `https://tmatteso.github.io`. This stops other websites
  from embedding the API, but it does **not** stop scripts like `curl`;
  only the limits above stop those.
- The checkpoint path comes only from the server config. No endpoint reads
  or writes files. `torch.load(..., weights_only=True)`.
- Kill switch: stop `cloudflared`. The model and the local API keep running.

### 5.6 Logging (ready for Loki in Phase 4)

Reuse `StructlogConfig`. Log one JSON line per job transition with
`job_id`, `state`, `n_res`, `n_samples`, `seed`, `elapsed_s`, `device` and
`queue_depth`. Keep the label-worthy fields low-cardinality (`state`,
`device`) and keep `job_id` in the message body, so Phase 4 can send these
lines to Loki as they are.

## 6. Frontend (Phase 1)

### 6.1 Stack

- **Vite + React 19 + TypeScript**, strict. No router and no client-state
  library; server data goes through **TanStack Query** (decision 3, §10).
  Scaffold with the `react-ts` template, which uses `@vitejs/plugin-react`
  rather than the SWC variant. At this size SWC's dev-speed edge is
  negligible, `plugin-react` is the officially supported path for the React
  Compiler, and type checking comes from `tsc --noEmit` either way (§6.5).
- **Function components only.** Every component is a plain function
  declaration with a typed props interface and an explicit return type, for
  example
  `function SampleCard({ sample }: SampleCardProps): React.JSX.Element`.
  Don't use `React.FC`. State and effects use hooks (`useState`,
  `useReducer`, `useEffect`), and shared logic lives in custom hooks rather
  than higher-order components or render props. The three data hooks are
  thin wrappers over TanStack Query's `useQuery`, so retries, caching and
  polling are not hand-written:
  - `useHealth()` polls `/api/health` (`refetchInterval`).
  - `useModelInfo()` fetches `/api/model` once (`staleTime: Infinity`).
  - `useJob(jobId)` polls `/api/jobs/{id}` every 1.5 s through queued →
    running → succeeded or failed. `refetchInterval` returns `false` once
    the job reaches a terminal state, which stops the polling.

  The one thing React still has no hook for is an **error boundary**. Use
  the `react-error-boundary` package's `<ErrorBoundary>` component rather
  than hand-writing a class, so the codebase itself contains no class
  components.
- **3Dmol.js** (npm `3dmol`), the JavaScript library under `py3Dmol`:
  `addModel(pdb, "pdb")`, cartoon with `color: "spectrum"`, then `zoomTo()`.
- **StyleX** for all styling (see *Styling* below). There is no CSS
  framework and no hand-written stylesheets beyond a tiny reset.

**Layout and naming.** Components use **PascalCase**, both for the function
and for the file, with one component per file and the file named after its
component:

```
web/src/
  main.tsx  App.tsx
  components/  SampleForm.tsx  SampleGallery.tsx  SampleCard.tsx
               StructureViewer.tsx  HealthIndicator.tsx
  hooks/       useHealth.ts  useModelInfo.ts  useJob.ts
  api/         client.ts  generated/    # generated/ is types + Zod schemas (§6.3)
  styles/      tokens.stylex.ts  reset.css
```

| Kind | Case | Example |
|------|------|---------|
| Components, and their `.tsx` files | PascalCase | `SampleCard`, `SampleCard.tsx` |
| Props types | PascalCase, `<Component>Props` | `SampleCardProps` |
| Types, interfaces, enums | PascalCase | `JobState`, `ApiError` |
| Hooks, and their files | camelCase with a `use` prefix | `useJob`, `useJob.ts` |
| Other functions, variables, non-component files | camelCase | `submitJob`, `client.ts` |
| StyleX token files | camelCase with the `.stylex.ts` suffix that StyleX requires | `tokens.stylex.ts` |
| StyleX style keys | camelCase | `styles.grid`, `styles.failed` |
| Module-level constants | UPPER_SNAKE_CASE | `POLL_INTERVAL_MS` |

Hooks are the deliberate exception to PascalCase. React's hook lint rules
only recognise functions whose names start with `use`, so a hook named
`UseJob` would silently escape `rules-of-hooks`. The API types generated
from the Pydantic models (`SampleJobSpec`, `JobStatus`, …) are already
PascalCase, so names match end to end, from Python to TypeScript.

**Styling: StyleX.** Styles are typed TypeScript objects, compiled at build
time to static atomic CSS. This is Meta's system (facebook.com, Instagram,
WhatsApp web, Threads), and it is the same build-time approach as
Microsoft's Griffel. We chose it over Bootstrap, Tailwind, CSS Modules and
runtime CSS-in-JS because:

- styles are plain TypeScript, with camelCase properties and no second
  syntax to learn
- conditional styles are ordinary expressions
- misspelled properties and missing tokens are compile errors
- it has no runtime cost

Plain CSS files read poorly next to the rest of a strictly typed codebase;
StyleX keeps styling inside the type system.

*Design tokens* live in one file, as typed variables. Dark mode is part of
the token definitions, so components never mention it:

```tsx
// styles/tokens.stylex.ts
import * as stylex from "@stylexjs/stylex";

const DARK = "@media (prefers-color-scheme: dark)";

export const colors = stylex.defineVars({
  bg: { default: "#ffffff", [DARK]: "#121212" },
  surface: { default: "#f5f6f8", [DARK]: "#1e1f22" },
  fg: { default: "#1a1a1a", [DARK]: "#e8e8e8" },
  muted: { default: "#5f6368", [DARK]: "#9aa0a6" },
  accent: "#2f6fdf",
  danger: { default: "#c5221f", [DARK]: "#f28b82" },
});

export const space = stylex.defineVars({ xs: "0.25rem", sm: "0.5rem",
                                         md: "1rem", lg: "1.5rem" });
export const radius = stylex.defineVars({ sm: "4px", md: "8px" });
```

*Component styles* sit at module scope in the component's own file, as one
`styles` constant per file:

```tsx
// components/SampleGallery.tsx
import * as stylex from "@stylexjs/stylex";
import { space } from "../styles/tokens.stylex";

const styles = stylex.create({
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: space.md,
  },
});

export function SampleGallery({ samples }: SampleGalleryProps): React.JSX.Element {
  return <div {...stylex.props(styles.grid)}>{/* SampleCard × n */}</div>;
}
```

```tsx
// components/SampleCard.tsx: a variant is just a boolean expression
const styles = stylex.create({
  card: { backgroundColor: colors.surface, borderRadius: radius.md,
          padding: space.md },
  failed: { outlineColor: colors.danger, outlineStyle: "solid",
            outlineWidth: 2 },
});
// <article {...stylex.props(styles.card, failed && styles.failed)}>
```

*Rules:*

- **Every colour, spacing and radius value comes from `tokens.stylex.ts`.**
  Components contain no raw hex values or pixel spacing. Layout values that
  aren't tokens, such as the `280px` grid minimum, are fine.
- **Styles are always applied with `stylex.props(...)`.** No `className`
  strings and no inline `style={{…}}`. The one exception is dynamic values
  that 3Dmol needs on its own canvas.
- **Overrides are typed.** A component that accepts style overrides from its
  parent types the prop as `StyleXStyles<{…}>`, which limits which
  properties may be overridden. The MVP needs none.
- **The only `.css` file is `styles/reset.css`**, a few lines (box-sizing,
  body margin) imported once in `main.tsx`. StyleX does not do global
  element styles. The page background and font are set by `App`'s root
  StyleX style.
- **The 3Dmol viewer needs a sized container.** 3Dmol sizes its canvas from
  its container, so `StructureViewer`'s container style must include
  `position: "relative"`, `width: "100%"` and `aspectRatio: "1"`. Without
  them the canvas collapses to 0 px.

*Build integration:*

- StyleX ships a Babel plugin (`@stylexjs/babel-plugin`), which runs through
  `@vitejs/plugin-react`'s `babel.plugins` option. This is one more reason
  for `plugin-react` over SWC.
- In dev, styles are injected at runtime for fast reloads. Production builds
  extract them into a single static `.css` file.
- Vitest reuses the Vite config, so tests see the same transform.
- StyleX's Vite integration has been evolving. In PR B2, use whatever package
  the StyleX docs currently recommend for Vite, and record the choice in
  `web/README.md`.

### 6.2 UI

```
┌──────────────────────────────────────────────────────────────┐
│ pallatom sampler   checkpoint: pallatom_best.pt · cuda:0 · ● │
├──────────────────────────────────────────────────────────────┤
│ Length [ 100 ]  Samples [ 4 ]  Seed [      ] (blank = random)│
│ ▸ Advanced: DDIM steps [ 200 ]                    [ Sample ] │
├──────────────────────────────────────────────────────────────┤
│ Queued (position 2)…  →  Running… 00:23  →  Done in 38.2 s   │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────┐ │
│ │  3D viewer   │ │  3D viewer   │ │  3D viewer   │ │  ...  │ │
│ │ MKLV…(copy)  │ │ ...          │ │ ...          │ │       │ │
│ │ [Download]   │ │ [Download]   │ │ [Download]   │ │       │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ └───────┘ │
└──────────────────────────────────────────────────────────────┘
```

- The header dot shows the result of `GET /api/health`. When the API is
  unreachable, the page shows "sampler offline", not a broken form.
- The form's limits and defaults come from `GET /api/model`.
- The page polls `GET /api/jobs/{id}` until the job reaches a terminal
  state. A `429` appears as "busy, try again in N s".
- **Download** saves a client-side `Blob` as
  `pallatom_seed{seed}_{index}.pdb`.
- Each `StructureViewer` creates one 3Dmol viewer in `useEffect` and calls
  `clear()` on cleanup. Browsers cap WebGL contexts at about 16, which is
  another reason `MAX_N_SAMPLES` is 8.

### 6.3 Keeping types in sync

TypeScript types are erased at runtime, so the frontend parses every API
response instead of trusting it (decision 4, §10). `pnpm gen:api` reads the
app's OpenAPI spec and writes both the TypeScript types and matching Zod
schemas to `src/api/generated/`, which is committed. `client.ts` parses
every response with the generated schema, so the types and the runtime
checks come from one source and cannot drift apart. Pick the generator in
PR B5 (for example `@hey-api/openapi-ts` with its Zod plugin, or `orval`),
and record the choice in `web/README.md`.

The spec comes from `python pallatom/serve/app.py --dump-openapi`, which
builds the app without loading a model, so it runs on CPU-only CI. CI
regenerates `src/api/generated/` and fails on a diff.

### 6.4 GitHub Pages deployment

- A new workflow, `.github/workflows/pages.yml`, runs on `push` to `main`
  when files under `web/**` change, and on `workflow_dispatch`. It runs
  `actions/configure-pages`, `pnpm/action-setup`, `setup-node`,
  `pnpm install --frozen-lockfile`, `pnpm build`,
  `actions/upload-pages-artifact` (`web/dist`) and `actions/deploy-pages`.
- Vite sets `base: "/diffusion/"` so asset paths resolve under the project
  site.
- The API URL comes from `VITE_API_BASE_URL`, read from a repository
  variable (`vars.SAMPLER_API_URL`). It is not a secret: the frontend ships
  it to every browser anyway. The runtime override `?api=` (§5.5) takes
  precedence when present.
- The workflow is validated by the existing `check-github-workflows` hook.

### 6.5 Strictness

The base is the full toolchain in
[`typescript_react_tooling.md`](typescript_react_tooling.md) (decision 2,
§10): `@tsconfig/strictest` plus the extra compiler flags listed there;
ESLint starting from the `all` presets, with type-aware typescript-eslint
rules, `eslint-plugin-jsdoc`, unicorn, sonarjs and `react-hooks`; Prettier;
`type-coverage --strict --at-least 100`; and knip. `no-explicit-any` and the
`no-unsafe-*` rules mirror the repo's `Any` ban, and
`explicit-module-boundary-types` mirrors its rule that every function is
annotated. This website adds the rules below on top of that base.

- Class components are banned by a `no-restricted-syntax` rule that matches
  any class extending `Component` or `PureComponent`, whether imported bare
  or as `React.Component`. Other classes, such as an `ApiError extends Error`
  in `api/client.ts`, are still allowed.
- Naming (§6.1) is enforced by lint errors:
  - `@typescript-eslint/naming-convention`: type-like names must be
    PascalCase; functions must be camelCase or PascalCase; variables must
    be camelCase or UPPER_CASE.
  - `react/jsx-pascal-case` covers component usage in JSX. React itself
    treats any lowercase JSX tag as a DOM element, so a component that isn't
    PascalCase would not render anyway.
  - `eslint-plugin-check-file` requires PascalCase file names under
    `components/` and `use*` camelCase file names under `hooks/`.
- `@stylexjs/eslint-plugin`: `valid-styles` (rejects invalid CSS
  properties and values inside `stylex.create`) and `no-unused` (flags style
  keys that are never used). Both are errors.
- Prettier with an 80-column print width, to match Black.

## 7. Tooling, devcontainer and CI (Phase 1)

- **Devcontainer:** install Node 22 and pnpm (through `corepack`) in the
  shared `.devcontainer/Dockerfile`, which both the `cuda/` and `arm64/`
  devcontainers build from, so the two cannot diverge (decision 5, §10).
  ARM64 CI runs every pre-commit hook, including the new frontend ones.
- **`.gitignore`:** `web/node_modules/`, `web/dist/`.
- **Pre-commit** (`language: system`, `files: ^web/`, with versions taken
  from `web/pnpm-lock.yaml`, decision 1, §10): `pnpm exec` for
  `tsc --noEmit`, `eslint --max-warnings 0`, `prettier --check`,
  `type-coverage` and `knip`.
- **CI:** add `pnpm install --frozen-lockfile && pnpm build && pnpm test`
  and the OpenAPI drift check (§6.3) to the `test` job, next to
  `pytest pallatom`.
- **Running it on the box:** a short `pallatom/serve/README.md` section
  covers how to start uvicorn and `cloudflared` in `tmux`. The MVP has no
  systemd units or containers; Phase 3 replaces all of this.

## 8. Testing (Phase 1)

**Backend** (`pallatom/tests/serve/`, following the tests style guide)

- Build a tiny `MainTrunk` from small `ModelParams` with random weights,
  save it to `tmp_path` as `{"ema": state_dict}`, and load it through the
  real `load_model` on CPU.
- Test `execute_job` directly: `ddim_steps=2`, `n_res=MIN_N_RES` and
  `n_samples=2` give 2 parseable PDBs and sequences of length `n_res`, and
  the same seed gives identical output.
- Test through `TestClient`:
  - submit, then poll until `succeeded`
  - `422` on out-of-range or extra fields
  - `404` for an unknown id
  - `429` when the queue is full (fill it with a queue whose worker is not
    started) and when the rate limit is hit
  - CORS headers present for the allowed origin and absent for others
- Round-trip `SampleJob` through JSON. This guards the future AMQP wire
  format.
- A regression test for the refactor: the old and new CLI output match for a
  fixed seed.

**Frontend** (Vitest + React Testing Library, with MSW mocking the API at
the request level)

- The form respects the limits from `/api/model`. The polling state machine
  goes through queued, running and succeeded or failed (with timers mocked).
  `429` and offline states render. The `?api=` override is used. A response
  that fails its Zod schema surfaces as an error, not a crash.
  `StructureViewer` is smoke-tested with `3dmol` mocked.
- Playwright end-to-end tests with `@axe-core/playwright` follow once the MVP
  is live (PR B11, decision 6 in §10).

**Manual acceptance:** on the GPU box, take `pallatom_toy_best.pt` and its
config, open the Pages URL through the tunnel, sample 4 × 100 residues, and
check that the PDBs open in PyMOL. Record the wall time and peak GPU memory
for `n_res ∈ {64, 128, 256}` × `n_samples ∈ {1, 4, 8}` to finalise the
limits.

## 9. Later phases

### 9.1 Phase 2: two machines, RabbitMQ, auth

```
browser ──HTTPS──► ARM64 cloud host (always on)
                   ├─ auth proxy (oauth2-proxy w/ GitHub login, or Cloudflare Access)
                   ├─ FastAPI gateway  (same api.py; RabbitMQJobQueue + persistent JobStore)
                   └─ RabbitMQ
                         ▲ AMQP over TLS, opened outbound from the GPU box
GPU box ─ worker per GPU (CUDA_VISIBLE_DEVICES=0 / =1), each: consume → execute_job → publish result
```

- **Queues:** a durable quorum queue `sample.jobs`. Workers consume with
  `prefetch_count=1` and **manual ack after the result is published**, so
  a worker that crashes mid-job causes redelivery. A dead-letter exchange
  catches jobs that fail repeatedly (track redeliveries with a header
  counter). A message TTL drops jobs nobody will poll for.
- **Results:** workers publish `SampleResult` or a failure to
  `sample.results`. The gateway consumes that queue and writes it to the
  `JobStore`, so the GPU box only ever makes outbound connections. The store
  becomes persistent (Redis or Postgres; to be decided in Phase 2). The
  queue position comes from the queue's message count.
- **Two GPUs means two consumers.** No code changes; this is the reason for
  the single-consumer-per-process design in the MVP.
- **Auth with Pages:** cookies sent across origins (Pages to the API) are
  increasingly blocked as third-party cookies. When auth arrives, either
  serve the frontend from the gateway's origin, or put Pages and the API
  under one domain that we control. A custom domain on Pages, with the API
  on a subdomain, keeps cookies first-party. Decide this at the start of
  Phase 2. The frontend code doesn't change either way, because the API base
  URL is already configurable.
- **The HTTP contract doesn't change.** The frontend is untouched apart from
  login and logout.

### 9.2 Phase 3: k3s + NVIDIA GPU Operator + HTTPRoute + KEDA (about a week or more)

```
browser ─► Gateway (Traefik) ─HTTPRoute /api─► FastAPI pods ─► RabbitMQ sample.jobs
                                                                      │ queue length
                                                    KEDA ◄────────────┘
                                                      │ scales 1..2
                                                      ▼
                                         GPU worker Deployment (1 pod = 1 GPU)
```

- **k3s** runs a mixed-architecture cluster: the ARM64 host is the server
  node and the GPU box joins as an amd64 agent. Images are built for both
  architectures with `docker buildx`. GPU workloads are pinned to the GPU
  node with `nodeSelector`.
- **NVIDIA GPU Operator** manages the driver and container toolkit, the
  device plugin (`nvidia.com/gpu: 1` per worker pod) and the **DCGM
  exporter**, which Phase 4 scrapes for free.
- **RabbitMQ** moves in-cluster, through the RabbitMQ Cluster Operator.
- **Health probes:** split `/api/health` into `/healthz` (liveness) and
  `/readyz` (model loaded; broker connected).

**Ingress: Gateway API `HTTPRoute`.** A `Gateway` (the `:443` listener, the
hostname and TLS) plus one `HTTPRoute` that sends `/api` to the FastAPI
`Service`. We use Traefik, which k3s already ships; its Gateway API provider
and the Gateway API CRDs have to be enabled once. TLS comes from
cert-manager, or `cloudflared` keeps running in-cluster and points at the
Gateway, so the public URL is unchanged from the MVP. The route only ever
carries cheap CPU calls (submit and poll), so it needs no GPU-aware routing.

**Why not the Gateway API Inference Extension (GIE)?** GIE's endpoint picker
routes *synchronous* inference requests straight to model-server pods, using
live load metrics from those pods. Our GPU workers receive no HTTP traffic:
they *pull* from RabbitMQ with `prefetch_count=1`, and that already gives
least-loaded dispatch. GIE would have nothing to route. Revisit it only if
we add a synchronous low-latency path to GPU pods, or serve several
checkpoints from separate pools.

**Worker autoscaling: KEDA, with no scale to zero.** A KEDA `ScaledObject`
scales the worker Deployment on the length of `sample.jobs`.
`minReplicaCount: 1` keeps one worker warm at all times, so no job ever
waits for a pod to be scheduled and the checkpoint to load.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: { name: sample-worker }
spec:
  scaleTargetRef: { name: sample-worker }
  minReplicaCount: 1               # always one warm worker: no cold starts
  maxReplicaCount: 2               # one per GPU
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown: { stabilizationWindowSeconds: 600 }   # 10 min idle before 2 → 1
  triggers:
    - type: rabbitmq
      metadata: { queueName: sample.jobs, mode: QueueLength, value: "1" }
      authenticationRef: { name: rabbitmq-conn }        # Secret with the AMQP URL
```

Trade-off: one GPU stays reserved for sampling at all times. Training runs
get the second GPU whenever KEDA has scaled down to one worker. Using both
GPUs for training means scaling the worker Deployment to 0 by hand, or
pausing the `ScaledObject`.

**Worker Deployment and graceful shutdown.** The autoscaler doesn't know
which worker is busy. Rollouts, node drains and scaling from 2 to 1 can
therefore all send `SIGTERM` to a worker in the middle of a job. The worker
must exit cleanly instead of losing that job:

1. On `SIGTERM`, the worker cancels its RabbitMQ consumer, so the broker
   sends it no new jobs.
2. It finishes the current job, publishes the result, **then** acks the
   message.
3. It closes the connection and exits 0.

- `terminationGracePeriodSeconds` is set to about 1.5× the slowest allowed
  job, taken from the §8 measurements at `MAX_N_RES` × `MAX_N_SAMPLES` ×
  `MAX_DDIM_STEPS`.
- If the grace period still runs out, Kubernetes sends `SIGKILL`. The
  message was never acked, so RabbitMQ redelivers it to the other worker.
  Delivery is at-least-once. Because the seed is fixed when the job is
  submitted, re-running a job gives the same output, and the result write is
  keyed by `job_id`, so a duplicate run just overwrites an identical result.
- The rollout strategy is `RollingUpdate` with `maxSurge: 0` and
  `maxUnavailable: 1`. With both GPUs in use, a surge pod could never be
  scheduled, so the rollout would stall. With these settings it replaces one
  worker at a time, and the other keeps consuming.
- `readinessProbe` is `/readyz` (model loaded, broker connected). The worker
  starts consuming only once it is ready.

The shutdown logic belongs in the Phase 2 worker process. `systemctl stop`
sends the same `SIGTERM`, so Phase 3 only adds the manifests around it.

### 9.3 Phase 4: observability

- **Prometheus** scrapes everything, so targets only need to expose a port.
  Use a single instance with `--storage.tsdb.retention.time` set to 15–30
  days on local disk, and **no** Thanos or Mimir. Add **Alertmanager** and
  **Grafana**. On k3s, `kube-prometheus-stack` installs all of these, plus
  node-exporter and kube-state-metrics.
- **Scrape targets:**
  - the gateway's `/metrics` via `prometheus_client`: request rate and
    status, jobs submitted, accepted and rejected (`429`), and queue depth
  - the workers' `/metrics`: a histogram of job duration by `n_res` bucket,
    failures by reason, and peak GPU memory per job
  - RabbitMQ's built-in `rabbitmq_prometheus` plugin on `:15692`
  - DCGM exporter (GPU utilisation, memory, temperature)
  - node-exporter
- **Alerts to start with:** worker down or zero consumers on `sample.jobs`;
  queue depth above N for 10 minutes; job failure rate above X% over
  15 minutes; GPU memory near its limit or temperature too high; disk filling
  up, including Prometheus' own TSDB.
- **Loki** with Grafana Alloy (or Promtail) collects the structlog JSON from
  §5.6.
- Phase 3 must land before Phase 4. With the Phase 2 topology alone, the GPU
  box only makes outbound connections, so a Prometheus server on the ARM
  host could not scrape it. Inside the k3s cluster network that problem goes
  away.

## 10. Delivery plan

### 10.1 Decisions reconciling this doc with the tooling doc

[`typescript_react_tooling.md`](typescript_react_tooling.md) sets the
TypeScript bar for the repo. Where it and this doc disagreed, these
decisions apply, and the sections above have been updated to match.

| # | Topic | Decision | Why |
|---|-------|----------|-----|
| 1 | Package manager | **pnpm**, `web/pnpm-lock.yaml` committed | Stricter dependency resolution: packages can't import dependencies they don't declare. |
| 2 | Lint bar | **The tooling doc's full toolchain** (`all` presets, jsdoc, unicorn, sonarjs, type-coverage, knip); §6.5 lists only this site's additions | Matches the Python side's `select = ["ALL"]`. |
| 3 | Fetching server data | **TanStack Query** behind `useHealth`, `useModelInfo` and `useJob` | Retries, caching and polling are not hand-written; the hook names stay the same. |
| 4 | Checking API payloads at runtime | **Generate types and Zod schemas** from the OpenAPI spec | TS types are erased at runtime; one source means the types and checks can't drift. |
| 5 | Node in the devcontainer | **The shared `.devcontainer/Dockerfile`**, not a devcontainer feature | The ARM64 and CUDA images can't diverge. |
| 6 | Browser (end-to-end) tests | **Playwright with axe after the MVP ships**, as PR B11 | Keeps the MVP path short. |
| 7 | Frontend style guide | **Its own PR** (B4) | Repo convention keeps guide changes out of tooling PRs. |

### 10.2 PR sequence

Each PR passes ARM64 CI and CUDA CI on its own. Tracks A and B don't depend
on each other until B5, so they can be worked on alternately.

**A. Backend (`pallatom/serve/`)**

| PR | Type | Content | Depends on |
|----|------|---------|------------|
| A1 | `refactor(sample)` | Split `sampling.py::main` into `load_model`, `run_sampling` and `to_samples`, with frozen `LoadedModel` and `SampledProtein` (§5.1). Regression test: old and new CLI output match for a fixed seed. | — |
| A2 | `feat(serve)` | `messages.py` (the Pydantic models and limits) and `worker.execute_job`. Tests: `SampleJob` JSON round trip; `execute_job` is deterministic on a tiny CPU model. | A1 |
| A3 | `feat(serve)` | `queue.py`: the `JobQueue` and `JobStore` protocols, `InProcessJobQueue` (one worker thread, raises `QueueFull`) and `InMemoryJobStore` (TTL). | A2 |
| A4 | `feat(serve)` | `api.py` and `app.py`: `SamplerApi` with bound-method handlers, CORS, `429` with `Retry-After` when the queue is full, `503` when the model fails to load, structlog lines per job transition (§5.6), the CLI, and `--dump-openapi` (§6.3). Tests through `TestClient`. | A3 |
| A5 | `feat(serve)` | Per-IP token-bucket rate limit keyed on `CF-Connecting-IP` (§5.5), with tests. | A4 |

**B. Frontend (`web/`)**

| PR | Type | Content | Depends on |
|----|------|---------|------------|
| B1 | `build(devcontainer)` | Node 22 and pnpm in the shared Dockerfile; `web/node_modules/` and `web/dist/` in `.gitignore`. | — |
| B2 | `feat(web)` | Scaffold: Vite `react-ts`, `tsconfig` from `@tsconfig/strictest`, ESLint flat config with the full rule set (tooling doc plus §6.5), Prettier, the StyleX Babel plugin, Vitest with Testing Library, type-coverage and knip. One trivial component and one test. | B1 |
| B3 | `ci` | Pre-commit hooks for `web/` (§7), plus `pnpm build` and `pnpm test` in the `test` job of `devcontainer-checks.yml`. | B2 |
| B4 | `docs` | `web/CLAUDE.md` style guide; update the root `CLAUDE.md` hook inventory and `docs/best_practices.md`. | B3 |
| B5 | `feat(web)` | API layer: generated types and Zod schemas (§6.3), `client.ts` with `ApiError`, the `?api=` override in `localStorage`, and the CI drift check. | A4, B3 |
| B6 | `feat(web)` | `useHealth`, `useModelInfo` and `useJob` on TanStack Query, tested with MSW and fake timers. | B5 |
| B7 | `feat(web)` | Design tokens, `reset.css`, the `App` shell, `HealthIndicator` and `SampleForm` (limits from `/api/model`; `429` and offline states). | B6 |
| B8 | `feat(web)` | `StructureViewer` (3Dmol, sized container, cleanup on unmount), `SampleCard`, `SampleGallery`, PDB download, and `ErrorBoundary`. | B7 |
| B9 | `ci(pages)` | `pages.yml`, `base: "/diffusion/"`, the `SAMPLER_API_URL` repository variable, and `pallatom/serve/README.md` on running uvicorn and `cloudflared`. **The MVP goes live.** | A5, B8 |
| B10 | `chore(serve)` | Manual acceptance and the timing and GPU-memory measurements (§8); tune the limits. | B9 |
| B11 | `test(web)` | Playwright with `@axe-core/playwright`. Optional. | B9 |

**C. Phases 2–4.** Each phase gets a short design addendum before it is
split into PRs.

- **Phase 2:** store the training config in checkpoints (open question 1;
  can land any time); a standalone worker process with graceful `SIGTERM`
  shutdown; `RabbitMQJobQueue` and the result consumer (a new dependency,
  plus a RabbitMQ service in CI); a persistent `JobStore` (Redis or
  Postgres); deploy the gateway on the ARM64 host and one worker per GPU;
  auth, once the domain question is settled.
- **Phase 3:** multi-arch images; split `/healthz` and `/readyz`; k3s with
  the GPU Operator; the RabbitMQ Cluster Operator; `Gateway` and
  `HTTPRoute`; the KEDA `ScaledObject`; worker Deployment manifests.
- **Phase 4:** `kube-prometheus-stack`; `/metrics` on the gateway and the
  workers; alert rules; Loki with Alloy; dashboards.

## 11. Open questions

1. **Embed the config in checkpoints?** If `train_loop.py` also saved
   `TrainConfig.model_dump()` into the `.pt`, the server could start from a
   checkpoint alone. That is a small change to training; older checkpoints
   would still need a separate config.
2. **Is a domain available on Cloudflare?** With one, we get a named tunnel
   and a stable API URL. Without one, the MVP uses a quick tunnel plus the
   `?api=` override.
3. **Where does the frontend live once auth arrives?** Gateway origin versus
   a custom domain on Pages (§9.1).
