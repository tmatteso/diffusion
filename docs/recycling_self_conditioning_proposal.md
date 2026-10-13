# Proposal: recycling-style self-conditioning for training

**Status:** proposal — no implementation in this branch.
**Base:** `origin/main` @ `eaf88ba` (`fix: run the sampler from maximum to minimum noise (#6)`).
**Scope:** `pallatom/helpers/data.py`, `pallatom/train/train_loop.py`,
`pallatom/sample/sampling.py`, `pallatom/train/train_config.py`,
`pallatom/sample/sample_config.py`.

This document proposes a four-commit change set. It is written against the
atom5 representation on `main` (`ATOMS_PER_RES = 5`, `NATOM = 5` in
`sampling.py`, `atom5_mask` on the batch). The in-flight atom14 refactor
conflicts with Commit 2 in exactly one hunk — the Cβ extraction — and
whichever lands second rewrites roughly ten lines.

---

## The defect chain

Three things are wrong, and they are causally linked, which is why they belong
in one change set.

1. **Training's self-conditioning template is built from noised ground
   truth.** `featurize_protein` bins the Cβ distogram of `r_gt_noised`. That
   template's errors are isotropic Gaussian perturbations of the true
   structure. The sampler's template errors are the model's own —
   systematically structured, correlated across residues, and largest exactly
   where the model is weakest. Training on the former and deploying on the
   latter is exposure bias: the model calibrates its trust to a noise process
   it never sees again.

2. **When the sampler has no template, it fabricates one that is confidently
   wrong rather than absent.** See Commit 1.

3. **The one channel that could let the trunk discount an unreliable template,
   `t_normalized`, is drawn independently of anything that determines template
   reliability.** It is noise, so the model learns to ignore it. See Commit 3.

Fixing (1) alone makes things worse. It hands the model a template whose
quality varies by orders of magnitude across the noise range, while leaving it
no means to tell which regime it is in, and while the no-template case stays
actively misleading.

---

## Commit 1 — `feat: add a null template distogram shared by training and sampling`

### What is wrong

`build_sampling_context` is called with `pdb_file_path=None` by default
(`sampling.py`, comment: *"single template only. for now it is None"*). On that
path it fabricates a protein:

```python
Protein(
    atom_positions=np.zeros((residue_number, 37, 3)),
    atom_mask=np.ones((residue_number, 37)),
    ...
)
```

which flows through `build_template_context` → `atom37_to_cb` →
`distogram_fn`. Inside `Distogram.forward`:

```python
bin_idx = torch.bucketize(pairwise_dist_mat, self.edges[1:])
bin_idx = bin_idx.clamp(min=0)          # overflow_bin=True
f_distogram = F.one_hot(bin_idx, num_classes=n_classes).float()
```

Every coordinate is zero, so every pairwise distance is zero, so
`bucketize(0.0, edges[1:])` returns 0 for every pair. `min_dist` for the
template distogram is **3.25 Å**. The result is a crisp one-hot in bin 0 for
all N² pairs: *every residue is within 3.25 Å of every other residue.* For a
128-residue protein that asserts a physically impossible structure — all 128
residues mutually coincident — at full confidence.

Meanwhile `f_pseudo_beta_mask = residue_mask.long()`, where
`residue_mask = atom37_mask[:, :, ATOM37_CA].bool()` and `atom_mask` is all
ones. So the mask says *all of this is valid*. The mask is **correct** — those
residues genuinely are real, we are generating them — and should not change.
Only the distogram is wrong.

### Why it matters more under recycling

Today this is a soft problem. The trunk has learned to ignore the
template-time channel, and in training the template is always GT-derived and
uniformly reliable, so step 0's garbage is partially absorbed by a trunk that
was never taught to lean on templates hard. After Commit 2 the trunk *does*
lean on them. Then the first step of every unconditioned sample starts from a
confident assertion that the protein is a single point — a bad initial
condition for a trajectory that cannot revisit it.

### The change

One definition of "no template information", shared by both regimes:

```python
@jaxtyped(typechecker=beartype)
def null_template_distogram(
    shape: tuple[int, int, int, int],
    device: torch.device | str,
) -> Float[torch.Tensor, "B N_res N_res n_templ_bins"]:
    """Build the all-zero template distogram meaning "no template".

    Every real distogram is a one-hot over the bin axis, so it sums to one
    there. An all-zero tensor sums to zero and is therefore unambiguously
    distinguishable from any binned structure, which lets the trunk learn
    "no information" as its own case rather than inferring it from an
    implausible geometry.

    This replaces the previous no-template path, which fabricated zero
    coordinates with a full atom mask and binned them: zero distance falls
    below ``min_dist`` for every pair, producing a confident one-hot in bin 0
    that asserts every residue is mutually coincident.

    Args:
        shape: Target shape ``(B, N_res, N_res, n_templ_bins)``.
        device: Device for the returned tensor.

    Returns:
        An all-zero template distogram.
    """
    return torch.zeros(shape, device=device)
```

In `build_sampling_context`, the `pdb_file_path=None` branch stops routing
through `build_template_context` for the distogram and uses this instead,
keeping the all-ones `f_pseudo_beta_mask` it already derives.
`build_template_context` itself is untouched: with a real PDB it is correct,
and this commit should not change templated sampling at all.

### Why zeros rather than a uniform distribution over bins

A uniform `1 / n_bins` across the bin axis is the other natural encoding of
ignorance, and it has one advantage: it preserves the row-sum invariant of
1.0, so `proj_a`'s input statistics do not shift between the null and real
cases.

Zeros are still preferable, because uniform is a *claim* — "all distances
equally likely" — which an untrained `proj_a` will happily project into a
non-zero pair bias. Zeros multiply through `LinearNoBias` to exactly zero
contribution from the distogram channels, so the template embedder's output
reduces to the mask and time channels alone. That is the honest "I was told
nothing" state. Worth an ablation eventually; not worth blocking on.

### Risk

This changes unconditioned sampling output for every existing checkpoint.
There is no way around that — the point is that the old behavior was wrong —
but it means A/B comparisons against pre-change samples are invalid.

### Tests

- `NullTemplateScenario` enum: one member for the null case (distogram
  all-zero, mask all-ones), one for the templated case (distogram one-hot
  summing to 1 on the bin axis, mask from the PDB). Factory fixture builds the
  context; a single parametrized test.
- A regression assertion that the null distogram's bin-0 slice is *not*
  all-ones — the specific thing that was broken. Reintroduce the
  zeros-coordinate path to confirm the test fails, then restore.

---

## Commit 2 — `feat: derive the training self-conditioning template from the model's own prediction`

### What is wrong

`featurize_protein` builds the template from the Cβ of the noised ground
truth:

```python
c_beta_pos_noised, _ = ...        # Cβ of the *noised ground truth*
noised_res_distogram, _ = template_distogram_fn(c_beta_pos_noised, residue_mask)
```

See the defect chain above for why that distribution is the wrong one.

### The change

100% self-conditioning, per Pallatom. No Bernoulli draw and no rate parameter,
which means no per-rank decision and no DDP interaction to reason about — a
material simplification over a mixed-rate design.

```python
@jaxtyped(typechecker=beartype)
def self_conditioned_template(
    batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> Float[torch.Tensor, "B N_res N_res n_templ_bins"]:
    """Build the self-conditioning template from a detached forward pass.

    Runs the trunk once under ``no_grad`` against the null template, then bins
    the Cβ distogram of its prediction. This is the template the sampler feeds
    the trunk at every step. Building it from noised ground-truth coordinates
    instead leaks true pair geometry that is unavailable at inference, and
    trains the model to trust a noise process it never sees again.

    Calls the unwrapped ``MainTrunk`` rather than the DDP wrapper so the pass
    cannot touch the gradient reducer.

    Args:
        batch: Micro-batch already moved to the target device.
        model_params: Bundled model and the template distogram module.

    Returns:
        One-hot template distogram over ``n_templ_bins`` distance bins.
    """
    trunk: MainTrunk = (
        cast(MainTrunk, model_params.model.module)
        if isinstance(model_params.model, DDP)
        else model_params.model
    )
    with torch.no_grad():
        pred: PredictedOutputs = trunk(
            dataclasses.replace(
                batch,
                noised_res_distogram=null_template_distogram(
                    batch.noised_res_distogram.shape,
                    batch.noised_res_distogram.device,
                ),
            ),
        )
    atom37_positions, atom37_mask = atom5_to_atom37(
        rearrange(pred.r_denoised, "b (n a) d -> b n a d", a=ATOMS_PER_RES),
        rearrange(
            batch.atom5_mask,
            "b (n a) -> b n a",
            a=ATOMS_PER_RES,
        ).float(),
    )
    cb_positions, _ = atom37_to_cb(atom37_positions, atom37_mask)
    template, _ = model_params.distogram_template(
        cb_positions,
        batch.f_pseudo_beta_mask.bool(),
    )
    return template
```

Wired into `take_step` immediately after the device transfer, before
`StepContext`:

```python
    if model_params.tcfg.training.self_condition:
        featurized_batch = dataclasses.replace(
            featurized_batch,
            noised_res_distogram=self_conditioned_template(
                featurized_batch,
                model_params,
            ),
        )
```

It is a separate function rather than inline because `take_step` is already
long and xenon enforces `--max-absolute B`. `ModelSetup` already carries
`distogram_template`, so there is no plumbing. New imports in `train_loop.py`:
`rearrange` from einops (only `reduce` is imported today), plus
`atom5_to_atom37`, `atom37_to_cb` and `ATOMS_PER_RES` from
`helpers.atom_utils`.

A single config flag, `self_condition: bool = Field(default=True)` on
`TrainingParams`, keeps the change revertible by config and makes the ablation
runnable without a code change.

`evaluate` routes through `take_step` with `train_mode=False`, so validation
picks this up automatically and measures the sampling-time condition.

### Why `.module` rather than the DDP wrapper

Under `no_grad`, DDP skips `prepare_for_backward` because it checks
`torch.is_grad_enabled()`, so going through the wrapper would probably be
fine. "Probably fine" is the wrong standard for something that fails as a
multi-hour hang at an all-reduce. Unwrapping makes it unconditional and
documents the intent.

There is **no DDP hazard here at all**, because every rank takes the same
branch — the flag is config, not a random draw. This is the key simplification
from a 100% rate. A per-rank Bernoulli would have needed a rank-0 broadcast,
and specifically *not* an OR-reduction, which would inflate an intended rate
`p` to `1 − (1 − p)^world_size`.

### Why the detached pass keeps dropout on

The trunk stays in train mode for the detached pass, matching AF2 recycling:
the template is then a sample from the model rather than its mean, which is
closer to a trajectory where every step is itself stochastic. The
counterargument — eval mode gives lower-variance templates and therefore
lower-variance gradients — is real but weaker. Easy to flip; worth an ablation
if training proves noisy.

### What this costs

About **+50% wall-clock per step**. A forward is roughly half of
forward-plus-backward, and at a 100% rate there is no averaging-down. The pass
stores no activations, so peak memory rises only by the transient forward
working set, not by a second activation graph.

If that is unacceptable, the lever is batch size, not the rate. Dropping to
50% would reintroduce the per-rank decision, the broadcast, and the
null-template training case for a 25% saving.

### Early training, and why there is no warmup ramp

At initialization the trunk's predictions are noise, so the template is noise,
so for an initial period the extra 50% compute conditions on garbage.

A ramp from 0 was considered and rejected. With Commit 3 the model can see the
noise level, and a ramp would make the *relationship* between `t` and template
quality non-stationary during exactly the period when the model is learning
that relationship. One fixed mapping from the first step is preferable. This
is a deliberate choice, not an oversight, and it is a one-line change if the
loss curve argues otherwise.

### Tests

- **Gradient isolation:** `self_conditioned_template`'s output has
  `requires_grad=False`, and `take_step` still produces finite, non-zero
  gradients on `template_embedder`. Prove the test works by dropping the
  `no_grad` and showing it fails, then restore.
- **Template provenance:** with a stub trunk returning known coordinates, the
  resulting template equals `distogram_template` of those coordinates' Cβ —
  i.e. the template came from the prediction, not from `r_gt_noised`.
- `self_condition=False` reproduces the pre-commit batch exactly.
- Derive configs from the existing `model_params` fixture via
  `tcfg.model_copy(update=...)` plus `dataclasses.replace`. Pin
  `max_seq_length` to the longest test protein: `ProteinDataset` pads
  everything to 128 otherwise, at roughly 9 s per protein per rank.

---

## Commit 3 — `fix: define the template weight as the normalized noise level`

### What is wrong, precisely

`featurize_protein` draws:

```python
t_template = repeat(torch.rand(()), "-> n m", n=N_res, m=N_res)
```

independent of `t_hat` and of everything else. The sampler instead passes
`normalized_timestep=t_p`, the loop's own time variable.

Why that matters hinges on a fact about the architecture that is not obvious.
In `MainTrunk.embed_inputs`:

- **Step 3–4:** `t_i = TimeFourierEmbedding(0.25 · log(t̂))`, a 256-dim
  (`c_res`) embedding, added into **`s_init`** — the single track.
- **Step 5:** `z_ij = rel_pos_enc(residue_index, asym_id, entity_id,
  token_index, sym_id)` — **purely positional, no noise-level content
  whatsoever.**
- **Step 6:** `z_ij += template_embedder(f_distogram, f_pseudo_beta_mask,
  z_ij, t)`.

`s_init` is **never passed to the `TemplateEmbedder`**. The module runs before
the atom encoder and receives only the distogram, the mask, a positional
`z_ij`, and `t`. Inside it:

```python
b_time = rearrange(t * b_mask[..., 0], ...)
a_ij = torch.cat([f_distogram, b_mask, b_time], dim=-1)   # a_dim = n_bins + 1 + 1
v_ij = self.proj_z(...) + self.proj_a(a_ij)
```

So `t_normalized` is the **only** noise-level signal available to the one
module whose job is deciding how much pair geometry to extract from the
template. And by the time Step 6 returns, the template is already additively
fused into `z_ij`: downstream blocks can re-weight `z_ij`, but they cannot
un-mix a template that was over-trusted.

With `t ~ U(0, 1)` independent of template quality, the cheapest thing for
`proj_a` to learn is to zero its `b_time` column. The channel is dead. That is
survivable today, when the template is always GT-derived and uniformly
reliable. After Commit 2, template quality varies by orders of magnitude
across the noise range and the model has no way to know which regime it is in.

### Why "churn breaks this" is not an objection

One can argue against this fix on the grounds that churn sets
`t_hat = c_T · (gamma + 1)` while passing `normalized_timestep=t_p`
unchanged, so `t` and `t_hat` are decoupled at sampling, and tying them in
training would push inference off-distribution.

That objection kills a *half* measure — changing only the featurizer — but not
the fix. The resolution is to change both sides. Define:

> **`t` ≡ the normalized noise level, `noise_schedule⁻¹(t̂)`**

and compute it that way in training *and* in the sampler, replacing `t_p`.
Then the two agree by construction, churn or not.

Under churn it is strictly *more* informative than `t_p`. Because
`t_hat = c_T · (1 + gamma) > c_T`, and larger σ maps to larger `t` on main's
convention (`t = 1` is maximum noise after `eaf88ba`), we get
`noise_schedule⁻¹(t_hat) > t_p`. The template weight then correctly reports
"effectively noisier than this nominal step", which is exactly what the
inflated `t_hat` means. Passing `t_p` understates the noise on every churned
step — and with `S_churn = 0.2`, `S_tmin = 0.01`, `S_tmax = 1.0`, that is
essentially every step of the trajectory.

### The change

```python
@jaxtyped(typechecker=beartype)
def normalized_time_from_sigma(
    t_hat: Float[torch.Tensor, ""],
    noise_params: NoiseScheduleParams,
) -> Float[torch.Tensor, ""]:
    """Invert the rho-spaced noise schedule to recover normalized time.

    The exact inverse of ``StochasticSampler.noise_schedule``, which maps
    t = 1 (maximum noise) to t = 0 (minimum). Defining the template weight
    this way makes it mean the same thing in training and sampling: the noise
    level the template was produced at. The sampler's loop variable ``t_p``
    is not that quantity whenever churn is active, since churn inflates
    ``t_hat`` above ``noise_schedule(t_p)``.

    Clamped to [0, 1] because the lognormal training schedule can draw sigmas
    outside the sampler's [sigma_min, sigma_max] range.

    Args:
        t_hat: Noise level sigma.
        noise_params: Schedule parameters supplying sigma_data, sigma_min,
            sigma_max, and rho.

    Returns:
        Normalized time in [0, 1]; 1 is maximum noise, 0 is minimum.
    """
    inv_rho = 1.0 / noise_params.rho
    sigma_min_r = noise_params.sigma_min**inv_rho
    sigma_max_r = noise_params.sigma_max**inv_rho
    ratio = (t_hat / noise_params.sigma_data) ** inv_rho
    return ((ratio - sigma_min_r) / (sigma_max_r - sigma_min_r)).clamp(0.0, 1.0)
```

Three call-site changes:

1. `featurize_protein`: replace the `torch.rand(())` draw with
   `normalized_time_from_sigma(t_hat, noise_params)`, still broadcast over
   residue pairs.
2. `sampling.py`, **both** `denoise` call sites:
   `normalized_timestep=normalized_time_from_sigma(t_hat, ...)` instead of
   `t_p`.
3. `rho` moves from `SamplerParams` to `NoiseScheduleParams` (default 7.0),
   and `StochasticSampler.__init__` reads `self.rho = noise_params.rho`
   instead of `sampler_params.rho`. It already receives `noise_params`, so
   this is a two-line change. Without it the featurizer has no access to `rho`
   at all: it currently exists only on the sampler config, which the training
   path never sees.

### The consequence to scrutinize hardest

This changes the **marginal** distribution of `t` at training time, and not in
a neutral direction.

Today `t ~ U(0, 1)`. After the fix, `t = noise_schedule⁻¹(t̂)` where
`t̂ = σ_data · exp(𝒩(−1.2, 1.5²))` — the EDM lognormal. Working the median
through: `t̂ ≈ 16 · e^{−1.2} ≈ 4.8`, and with σ_min = 4e-4, σ_max = 160,
ρ = 7, that maps to `t ≈ 0.30`. So training mass concentrates around 0.3
instead of spreading uniformly, with the lognormal tails giving little
coverage near `t = 0` and `t = 1`. Meanwhile the sampler sweeps `t_p` from 1
down to 0 roughly uniformly. The marginals now *disagree*, where before they
matched.

This is still correct, for a specific reason: **it is the same mismatch EDM
already accepts for `t̂` itself.** Training draws σ lognormally while sampling
sweeps it along the ρ schedule, and `lambda_sigma_loss_weight` is precisely
the machinery for reconciling that choice. Making `t` a deterministic function
of `t̂` means `t` simply *inherits* `t̂`'s coverage — the already-made design
decision — rather than having a second, independent and unjustified coverage
story. The matching marginals today are coincidental, not designed, and they
come at the price of a destroyed joint distribution.

It does mean early sampling steps (`t_p` near 1, maximum noise) sit in `t̂`'s
lognormal tail and get little training mass. That was **already true** before
this commit: it is a property of `P_mean` / `P_std` versus `σ_max`, not
something this change introduces. It merely becomes visible in `t` as well. If
it bites, the lever is `P_mean` / `P_std` or `σ_max`, not re-randomizing `t`.

The tempting mitigation — jitter, `t = clamp(inverse(t̂) + ε)` — should be
skipped initially, because it partially reintroduces the decorrelation this
commit exists to remove. Reach for it only if the model demonstrably fails at
trajectory start.

### Tests

- **Round trip:** `normalized_time_from_sigma(noise_schedule(t)) ≈ t` across a
  grid of `t`. This single assertion pins the whole commit and fails on any
  `rho` or `sigma_data` disagreement between configs — the specific breakage
  the `rho` move prevents.
- **Clamping:** σ above `σ_data · σ_max` → 1.0; below `σ_data · σ_min` → 0.0.
- **Churn direction:** for `gamma > 0`,
  `normalized_time_from_sigma(c_T · (1 + gamma)) > t_p`. This encodes *why*
  the sampler no longer passes `t_p`, which is what a future reader needs.
- Scenario values live in the factory fixture, not in module constants.

---

## Commit 4 — `refactor: stop featurizing the noised-GT template`

### What is wrong

After Commit 2 at a 100% rate, nothing reads
`FeaturizedBatch.noised_res_distogram` as produced by the featurizer: the
detached pass uses the null, the gradient pass uses the prediction. The field
is dead, but the work to produce it is not. `featurize_protein` still calls
`template_distogram_fn` per item on a dataloader worker, and still ships a
`128 × 128 × 39` float32 tensor — about **2.5 MB per item** — through the
worker IPC queue, into pinned memory, and across the PCIe bus. At batch size 8
that is roughly 20 MB per batch of pure waste on the critical path of every
step.

### The change

Remove the field from `FeaturizedItem` and `FeaturizedBatch`, drop the
`template_distogram_fn` call from `featurize_protein`, and have `take_step`
and the sampler construct the null directly from shape metadata.

Touchpoints on `origin/main`: 9 in `helpers/data.py` (field declaration ×2,
docstrings ×4, `to()`, `pin_memory()`, collate), 4 in `sample/sampling.py`, 1
each in `architecture/main_trunk.py` and `train/train_config.py`, plus 18
across four test files.

### Why it is last, and severable

It is the only commit with no behavioral content — pure deletion — and it has
the widest blast radius in tests. Landing it last means Commits 1–3 review on
their merits. If the diff grows unwieldy this one drops out and becomes a
follow-up, at the cost of some dataloader bandwidth. It is also the commit
most likely to conflict with the atom14 refactor, since both edit the
`FeaturizedItem` / `FeaturizedBatch` field list.

One caveat: this forecloses cheaply A/B-ing the old GT-derived template, since
`self_condition=False` would then have no template to fall back to except the
null. To keep that ablation runnable, either hold Commit 4 back or accept that
`self_condition=False` means "null template always" rather than "old
behavior".

---

## Ordering

`1 → 2 → 3 → 4` is the only order in which each commit is individually
coherent.

- Commit 2 needs Commit 1's null to exist.
- Commit 3 is only *motivated* by Commit 2 — before it, the dead `b_time`
  channel is harmless — though it would be correct standalone.
- Commit 4 requires Commit 2.
- Reviewing 2 before 1 would mean reviewing a recycling pass whose null
  template is the confidently-wrong one.

## Size estimate

Roughly **460–620 changed lines, about 180 of it production code.**

| Commit | Prod | Tests |
|---|---|---|
| 1. Null template | ~45 | ~90 |
| 2. Recycling template | ~75 | ~120 |
| 3. Normalized-noise template weight | ~55 | ~70 |
| 4. Drop featurized template | ~40 (deletions) | ~60 |

Two things roughly double raw counts versus the same change elsewhere: Google
docstrings with full `Args` / `Returns` on every function including tests, and
the 80-column limit. The enum-plus-factory-fixture test shape is front-loaded
— roughly 40 lines before the first assertion — but near-free per additional
scenario. Without Commit 4 the change set is about 330 lines.

## Compatibility

- **Validation loss is discontinuous** and not comparable to pre-change runs,
  for two independent reasons: template source and `t` semantics.
- **Sampling output changes** for every checkpoint, including templated
  sampling, because `t` changes meaning.
- **Existing checkpoints load** — no parameter shapes change, `a_dim` is
  untouched — but they trained with `b_time` as noise and will behave oddly
  until they re-learn to use it. Prefer a fresh run for the headline
  comparison over fine-tuning.
- `SamplerParams.rho` is removed, so any serialized sampling config setting it
  will fail validation.

## Open questions

1. **Commit 4 in or out?** In costs 18 test touchpoints and most of the atom14
   conflict surface. Out costs roughly 20 MB per batch of dataloader waste and
   leaves `self_condition=False` meaning "null always" rather than "old
   behavior".
2. **Dropout in the detached pass** — train mode (proposed, matches AF2) or
   eval mode (lower-variance gradients)?
3. **Null encoding** — zeros (proposed) or uniform `1 / n_bins`? Zeros give
   exactly zero contribution through `LinearNoBias`; uniform preserves the
   row-sum invariant.
