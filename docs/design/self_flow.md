# Design: Self-Flow and noise-schedule shifting for pallatom

**Status:** Draft, for discussion. This PR contains only this document.

This doc proposes two related changes to pallatom training:

- **Self-Flow** (dual-timestep self-distillation from an EMA teacher).
- **Noise-schedule shifting** (where the training and sampling noise levels
  sit).

It explains the design, then gives the order in which to land PRs and run
experiments, with a go/no-go gate after each stage.

**References:**

- Chefer, Esser et al., *Self-Supervised Flow Matching for Scalable
  Multi-Modal Synthesis*, Black Forest Labs, arXiv
  [2603.06507](https://arxiv.org/abs/2603.06507), March 2026.
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution
  Image Synthesis*, ICML 2024, arXiv
  [2403.03206](https://arxiv.org/abs/2403.03206), §5.3.2.
- Black Forest Labs, *FLUX.2: Analyzing and Enhancing the Latent Space of
  FLUX*, [techblog](https://bfl.ai/techblog/representation-comparison/),
  November 2025.

## 1. Summary

Self-Flow trains a shallow layer of the model to match a deeper layer of its
EMA copy. The student sees an input where each residue is noised at one of
two sampled levels. The teacher sees every residue at the lower level. To
match the teacher, the student has to infer clean structure from noisy
context.

In pallatom, the matching is one extra term in the intermediate-loss loop:
the Kabsch-aligned MSE between student decoder unit 2 and EMA-teacher decoder
unit 6 (`0.3·K` and `0.7·K` with `K_unit = 8`). Most of the work is
elsewhere: making the noise level per residue rather than per protein.

Noise-schedule shifting matters for two reasons:

- The BFL FLUX.2 study shows that comparing two methods at one fixed noise
  distribution can rank them the wrong way round.
- Representation objectives like Self-Flow shift where the best noise level
  sits.

So every comparison below is made at each run's own best `P_mean`. A
length-dependent shift is evaluated as a separate, later question.

### Roadmap at a glance

| Stage | Lands | Runs | Question | Gate to continue |
|---|---|---|---|---|
| 0 | PR 1: sampler `α` multiplier | sampling-`α` sweep; baseline `P_mean` sweep | Where are today's best training and sampling noise levels? | none (reference numbers) |
| 1 | PR 2: per-residue σ refactor | none | Is the refactor behaviour-preserving? | bitwise regression tests pass |
| 2 | PR 3: dual-timestep featurizer | `dual-t` × 3 `P_mean` | Does mixed per-residue noise alone hurt? | `dual-t` ≥ `baseline` |
| 3 | PR 4: EMA teacher + Arm A loss | `sf-coords` × 3 `P_mean` | Does coordinate-space Self-Flow help? | `sf-coords` > `dual-t` |
| 4 | none | Self-Flow ablations | Which `R_M`, layer pair and λ weighting? | none (pick defaults) |
| 5 | PR 5: Arm B (conditional) | `sf-latents` | Does the paper's representation-space version do better? | only if Stage 3 is flat |
| 6 | PR 6: length-dependent shift | `β` sweep, then a 2×2 | Should the noise level grow with protein length? | none |

Stages 0 and 1 can proceed in parallel: Stage 0 needs only a small sampler
change and config edits.

## 2. Background

### 2.1 Self-Flow

The facts below were checked against the paper text (Sec. 3.3, 3.4, 4.4;
App. A.3, B, H).

- **Two timesteps.** Draw `t, s ~ p(t)` independently from the usual
  training distribution.
- **Per-token mask.** Each token is masked independently with probability
  `R_M ≤ 0.5`: a Bernoulli draw per token, not a fixed count. The paper uses
  `R_M = 0.25` for images, `0.5` for audio and `0.1` for video ("due to
  significant temporal redundancies"). It applies the schedule to every
  sample.
- **Per-token noise level.** `τ_i = s` for masked tokens, otherwise `t`. The
  model is conditioned on the per-token vector `τ ∈ R^N`, not a scalar.
- **Teacher.** The EMA copy of the model (decay 0.9999), which is also the
  evaluation model. Its input has all tokens at `τ_min = min(t, s)`. The
  notation implies the teacher and student share the same noise sample; the
  paper does not say so explicitly.
- **Loss.** `L = L_gen + γ·L_rep`, with `γ = 0.8` fixed and
  `L_rep = −cos(h(student layer 0.3D), teacher layer 0.7D)`. `h` is a small
  MLP head on the student only. ℓ1 was unstable as feature norms grew.
- **Ablations.**
  - Removing `L_rep` costs more than 4 FID.
  - Removing the masking but keeping `L_rep` costs more than 1 FID.
  - The dual-timestep schedule alone gives a small gain.
  - Results are stable for layer choices near 0.3D/0.7D.
- **Cost.** One extra no-grad teacher forward pass. The paper reports no
  overhead number.

### 2.2 Noise-schedule shifting

Both shifting papers use the same shift,
`t' = α·t / (1 + (α−1)·t)`, with the rule of thumb `α = √(m/n)`. They apply
it along different axes:

| | Esser et al. (SD3) | BFL FLUX.2 study |
|---|---|---|
| Axis | number of **tokens** (resolution) | **channels per token** (dimension of the representation), with tokens fixed at 256 |
| m, n | pixel counts | channels per token (SD-VAE's 16 as baseline) |
| Chosen by | derived, then tuned (α = 3 at 1024²) | swept `α ∈ {1, 1.78, 2.95, 4.63, 6.93}`, separately for training and sampling |

The common principle: **the right noise level depends on how redundant the
representation is.** With more redundant tokens or channels, a given noise
level destroys less information, so the schedule should move toward more
noise.

**The shift in EDM terms.** A rectified-flow input `(1−t)·x0 + t·ε`, divided
by `(1−t)`, is `x0 + σ·ε` with `σ = t/(1−t)`. Substituting `t'` gives

```
σ' = α · σ
```

So the shift is a multiplicative rescale of σ. Since `ln σ = logit(t)`, our
log-normal sampler over σ is exactly a logit-normal over t. A shift by α is
then `P_mean += ln α`. The BFL study found that a shifted logit-normal always
beat a shifted uniform, so we are already in the preferred family, and any
shift is a change to `P_mean` alone.

**BFL findings that shape the plan:**

1. **The fixed-shift ranking can be wrong.** With each representation at its
   own best shift, RAE beat FLUX.2 (gFID 3.10 vs 3.70). Without shifting, the
   order reversed. The best vs worst training shift changed FID by 61–86%.
   → §4.1: compare runs at their own best `P_mean`.
2. **REPA lowers the best shift** ("generally results in a small reduction
   of the optimal shifting value"). Self-Flow is a REPA-style objective.
   → Self-Flow runs get their own `P_mean` sweep.
3. **The best sampling shift is above the best training shift** for every
   representation, and is "very dependent on the number of sampling steps
   and choice of sampler." → Stage 0 adds a sampling-only sweep.
4. **`√(m/n)` is only a starting point.** It matched RAE (`√(768/16)` =
   6.93) but not FLUX.2, which preferred 4.63 against a predicted 2.83. The
   study attributes the gap to FLUX.2's higher intrinsic dimension, i.e. less
   redundancy across channels. → Shifts are swept, never computed (§3.7, §6).

### 2.3 What pallatom already has

| Self-Flow needs | pallatom has |
|---|---|
| EMA teacher weights | `EMA` in `architecture/main_trunk.py`, updated every optimizer step |
| Per-layer outputs | `intermediate_denoised_coord_stack`, one entry per decoder unit |
| A loss on per-layer outputs | `med_loss`: Kabsch-aligned MSE plus CE per unit |
| Per-token time conditioning | `TimeFourierEmbedding` already takes `log t̂` as `[B, N_res]` (a broadcast scalar today) |
| A logit-normal noise distribution | the EDM log-normal over σ (§2.2) |

## 3. Design

### 3.1 Notation

pallatom uses EDM, so "timestep" means the noise level σ (`t_hat` in code):

```
x_σ = x0 + σ · ε,        σ = σ_data · exp(N(P_mean, P_std²))
```

The tokens are **residues**. All 5 atoms of a residue share that residue's
`σ_i`. Masking happens at residue level, never atom level.

### 3.2 Per-residue σ (PR 2)

`t_hat` changes from `Float[Tensor, "B"]` to `Float[Tensor, "B N_res"]`
everywhere. Wherever it is used, it is broadcast to atoms with
`repeat(t_hat, "b n -> b (n a)", a=atoms_per_res)`.

| Site | Today | With per-residue σ |
|---|---|---|
| `main_trunk.py` step 1, `c_in` (`denom`) | `[B]` | `[B, N_atom]`: per-atom scaling of `r_input` |
| `main_trunk.py` step 3, `TimeFourierEmbedding` | `repeat(log t̂, "b -> b n")` | pass `0.25·log(t_hat)` directly, already `[B, N_res]` |
| `main_trunk.py` step 14, `c_skip`/`c_out` (`w_gt`, `w_upd`) | `"b -> b 1 1"` | `"b n_atom -> b n_atom 1"` |
| `lambda_sigma_weight` (train loop) | `[B]` | `[B, N_atom]` |
| `atom_loss`, `med_loss` | λ scales the per-sample loss after reduction | λ multiplies each atom's squared error **before** the weighted mean |
| `sample/sampling.py` | `repeat(t_hat, " -> b", b=B)` | `repeat(t_hat, " -> b n", b=B, n=N_res)`; inference always uses uniform σ |

**Why per-atom λ.** EDM's denoiser is `D = c_skip·x + c_out·F`, so
`D − x0 = c_out·(F − F_target)`. Since `λ = 1/c_out²`,

```
λ(σ_i) · ‖D_i − x0_i‖²  =  ‖F_i − F_target,i‖²
```

Per-atom λ therefore weights every atom equally in terms of the raw network
output F. Low-σ residues have large λ, but their coordinate errors are
smaller by the same factor, so a clean subset does not automatically dominate
a sample's loss. This relies on `c_skip` and `c_out` also being per-atom.
Smooth-lDDT, the distogram losses and CE have never been noise-weighted, and
stay unchanged.

With uniform σ every change in this table is an identity, so PR 2 must
reproduce `main` exactly.

### 3.3 Dual-timestep featurization (PR 3)

For each item in `featurize_single_item`:

1. Draw `σ_s, σ_t` independently from the log-normal.
2. Draw `m ~ Bernoulli(R_M)` of length `N_res`, and set
   `σ_i = where(m_i, σ_s, σ_t)`.
3. Draw one `ε ~ N(0, I)` of shape `[N_atom, 3]`, shared by both views.
4. Student coordinates: `r_student = r_gt + center(σ_atom ⊙ ε)`, where
   `σ_atom = repeat(σ_res, "n -> (n a)", a=5)`.
5. Teacher coordinates: `r_teacher = r_gt + center(σ_min · ε)`, where
   `σ_min = min(σ_s, σ_t)`.
6. Build the self-conditioning template distogram **separately for each
   view**, from that view's noised Cβ.

**Centering.** Today we compute `t_hat · (ε − CoM(ε))`. With per-residue σ,
`σ_i · (ε_i − CoM(ε))` does not have zero centre of mass. Center *after*
scaling: `n = σ_atom ⊙ ε`, then `n − CoM(n)`. For uniform σ the two forms are
identical.

**The template distogram.** Self-Flow fits pallatom unusually well here. The
template distogram for pair `(i, j)` carries noise from both `σ_i` and `σ_j`.
Some regions of the pair map are sharp and others blurred, which acts as
span-masking of the distogram for free.

**Train/inference mismatch and `apply_prob`.** With `R_M = 0.25`, the chance
that no residue is masked is `0.75^N_res`, about 6·10⁻⁶ at 50 residues. So
under this schedule the model essentially never trains on uniform σ, yet
sampling always uses uniform σ. The paper reports no problem on images, but
our template distogram makes pallatom more exposed. `apply_prob` is the
fraction of samples that get the dual-timestep schedule. The rest use
`R_M = 0`, which is the paper's "no masking, keep `L_rep`" setting and still
a valid training signal. The default is 1.0, matching the paper. It is a
Stage 2 fallback (§4).

**Validation is always uniform σ** (`R_M = 0`, no teacher pass), so
validation numbers stay comparable across all runs.

### 3.4 EMA teacher forward pass (PR 4)

```python
with torch.no_grad(), teacher_mode(inner_model):
    teacher_out = cast(
        PredictedOutputs,
        torch.func.functional_call(
            inner_model,               # MainTrunk, unwrapped from DDP
            model_params.ema.shadow,   # params + buffers, already on device
            (teacher_batch,),
        ),
    )
```

- **Do not use `EMA.swap`.** `swap` reassigns `.data` on the live
  parameters, which does not bump autograd's version counter. If the teacher
  ran between the student forward and `backward()`, backward would silently
  compute gradients with the EMA weights. `functional_call` never mutates the
  module, so call order cannot matter.
- **Bypass DDP.** Call the unwrapped module. There are no gradients, so no
  all-reduce hooks are needed.
- **Use eval mode.** `NodeUpdate` uses `DropoutRowwise(0.25)`, and a teacher
  target should be deterministic. `teacher_mode` is a small context manager
  in `helpers/context_managers.py`, next to `StepContext`, that calls
  `eval()` and restores the previous mode.
- **Reuse the batch.** `teacher_batch` is
  `dataclasses.replace(batch, r_gt_noised=..., noised_res_distogram=..., t_hat=...)`.
  Every clean field is shared, not copied.
- **EMA decay.** The paper uses 0.9999; we use 0.999. One EMA serves as both
  teacher and evaluation model, as in the paper. The bias-correction ramp
  `min(decay, (step+1)/(step+10))` makes the teacher roughly equal to the
  student for the first few hundred steps. That is harmless: the target is
  then just "the deeper unit on a cleaner input". No warm-up gate.

### 3.5 The Self-Flow loss

**Arm A, coordinate space (PR 4).**

```
L_sf = mean_b  KabschMSE( student.coord_stack[k_s],
                          stopgrad(teacher.coord_stack[k_t]) ;  λ(σ_student) )
k_s = round(0.3 · K_unit) − 1 = 1     (decoder unit 2 when K_unit = 8)
k_t = round(0.7 · K_unit) − 1 = 5     (decoder unit 6)
total_loss += alpha_5 · L_sf,   alpha_5 = 0.8
```

- **Reuse `atom_loss`** as is, with the teacher coordinates in the `r_gt`
  slot. It already aligns the target onto the prediction under `no_grad`.
- **The sign is `+`.** The paper's leading minus comes from *maximizing*
  cosine similarity. We *minimize* an MSE, so a minus sign would push the
  student *away* from the teacher.
- **The student is supposed to differ from the teacher.** Unit 2's
  coordinates include `c_skip · r_input` from the noisier student input, so
  the student must denoise the masked residues up to the teacher's level.
  That is the intended signal, not a confound.
- **λ defaults to the student's per-atom `λ(σ_i)`**, so `L_sf` is on the
  same scale as `med_loss`. The risk: the target has its own error. At small
  σ the two views are nearly identical and λ is large (about 100 at
  σ = 0.1). So leftover teacher error could get amplified on exactly the
  samples where distillation teaches least. The paper's `L_rep` has no time
  weighting. The σ-bucketed diagnostic (§3.6) decides this in Stage 4.

**Arm B, representation space (PR 5, conditional).** This is the paper's
version:

- Take the student's `c_l` (or `s_i`) after unit 2 and pass it through a new
  `LinearNoBias` → SiLU → `LinearNoBias` head (about `2·c²` parameters).
- Compare it with the teacher's same tensor after unit 6, with no head.
- Use `1 − cos` with a masked mean.

Both tensors are already SE(3)-invariant. This needs the forward pass to also
return a stack of hidden states.

**Why Arm A first.** It adds no modules and is a very small change to the
intermediate-loss loop. Its risk is redundancy: unit 2 is already supervised
against ground truth by `med_loss`, and the teacher's target is itself
trained toward ground truth. If that redundancy makes Arm A flat, Arm B is
the next thing to try.

### 3.6 Configuration, data structures and diagnostics

```python
class SelfFlowParams(BaseModel):
    enabled: bool = False
    mask_ratio: float = Field(default=0.25, ge=0, le=0.5)      # R_M
    apply_prob: float = Field(default=1.0, ge=0, le=1)
    student_unit_frac: float = Field(default=0.3, gt=0, lt=1)
    teacher_unit_frac: float = Field(default=0.7, gt=0, lt=1)
    target: Literal["coords", "latents"] = "coords"            # Arm A / B
```

The loss weight is `LossParams.alpha_5 = 0.8`. A `model_validator` enforces
`k_s < k_t` after rounding.

**Data structures:**

- `FeaturizedItem.t_hat` becomes `"N_res"`, and `FeaturizedBatch.t_hat`
  becomes `"B N_res"` (PR 2).
- PR 3 adds `teacher_view: NoisedView | None` to `FeaturizedBatch`.
  `NoisedView` is a new `@dataclass` with fields `r_noised`,
  `noised_res_distogram` and `t_hat`. `to()` and `pin_memory()` pass it
  through. It is `None` when Self-Flow is off, and for validation and
  sampling batches.

**Metrics and diagnostics** (in `LossMetrics`, logged to wandb):

| Metric | PR | Used by |
|---|---|---|
| **Group share:** per sample, each σ group's share of the λ-weighted `atom_loss` against its share of valid atoms | 3 | Stage 2 gate |
| `self_flow_loss` | 4 | Stage 3 |
| `self_flow_teacher_gt_mse`: the teacher's unit-6 error against ground truth (is the target better than what `med_loss` already gives?) | 4 | Stage 3 |
| `L_sf` bucketed by student σ (log-spaced buckets) | 4 | Stage 4 λ decision |
| `L_sf` and `self_flow_teacher_gt_mse` by length bin | 6 | Stage 6 |

If the per-atom λ argument in §3.2 holds, the two group shares roughly match.
A large, persistent gap means one group is taking most of the gradient.
Likely causes are poor calibration at one end of the σ range, or
whole-protein Kabsch alignment letting noisy residues inflate errors on clean
ones.

### 3.7 Noise-schedule shifts (PR 1, PR 6)

**Sampling shift (PR 1).** Add `α` to the sampler params (default 1.0) and
apply `t_hat ← α · t_hat` in `noise_schedule` in `sample/sampling.py`.
Evaluation-only; no retraining.

**Global training shift (no PR).** `P_mean` is already a config field.

**Length-dependent shift (PR 6).** Protein length is our resolution axis.
The proposal is a length-dependent mean,

```
P_mean(N) = P_mean + β · ln(N_res / N_ref),
```

with the same `α(N) = (N_res / N_ref)^β` applied to the sampler schedule, so
training and inference stay matched. The constant-image argument gives
`β = ½`. It does not carry over directly: local protein geometry (3.8 Å
Cα–Cα, bond angles) does not depend on length, and only the global fold gets
more redundant as N grows. So `β ∈ [0, ½]` is swept.

There is a related point: `σ_data = 16` is fixed, while coordinate spread
grows with length (`R_g ∝ N^~0.4`). Part of what a length shift corrects is
that preconditioning mismatch.

How the length shift interacts with Self-Flow:

- **They compose cleanly.** Both σ draws come from the shifted distribution,
  and `min(ασ_s, ασ_t) = α·min(σ_s, σ_t)`. The shift goes where σ is drawn,
  before the dual-timestep split. No Self-Flow code changes.
- **Both depend on length.** With a fixed `R_M`, long proteins have more
  clean anchor residues, so the student's task gets easier as N grows. That
  is the effect the shift counteracts. Without a shift, Self-Flow's benefit
  may concentrate on short proteins. With one, long proteins move to higher
  σ, where the gap between student and teacher is largest. But the teacher
  is also less accurate there. Logging by length bin settles which effect
  wins.
- **The shift confounds σ-bucketed diagnostics.** Short proteins
  over-populate the low-σ buckets. Under a shift, bucket by the unshifted
  `σ/α(N)`, or by (σ, length bin).

## 4. Rollout plan

### 4.1 Ground rules for every experiment

- **Matched settings.** Same seed, data split, step count and batch size.
  Report validation Kabsch MSE, lDDT and RMSD on EMA weights, uniform σ, plus
  sample metrics. Report wall-clock per step separately.
- **Compare at each run's own best noise level.** Runs that decide a gate
  are trained at `P_mean ∈ {−1.7, −1.2, −0.7}` and compared at their best
  value (§2.2, finding 1). Ablations within one stage use the best `P_mean`
  of the arm they ablate.
- **Sample at each run's best sampling `α`** from the PR 1 sweep, and also
  report `α = 1`.

### 4.2 Stage 0 — reference numbers (PR 1)

**PR 1 `feat:` sampler `α` multiplier** (§3.7).
Tests: `α = 1` reproduces the current schedule exactly; the schedule scales
linearly in `α`.

**Experiments:**

1. **Exp 0a.** Sweep sampling `α ∈ {1, 1.25, 1.5, 2}` on the current best
   checkpoint. It needs no training.
2. **Exp 0b.** Train `baseline` at the three `P_mean` values. This is config
   only.

**Output:** the baseline's best `P_mean` and best sampling `α`. This is the
reference every later stage is measured against. If either differs from
today's defaults, that is a win on its own.

### 4.3 Stage 1 — per-residue σ (PR 2)

**PR 2 `refactor:` per-residue `t_hat`** (§3.2). Covers `data.py`,
`main_trunk.py`, `losses.py`, `train_loop.py` and `sampling.py`.

Tests:

- With uniform σ, `PredictedOutputs` and every loss are bitwise identical to
  `main`.
- Per-atom λ with uniform σ equals the per-sample form.
- The sampler builds `[B, N_res]` σ.

**Gate:** the tests pass. No experiment, since behaviour is unchanged by
construction.

### 4.4 Stage 2 — does mixed noise alone hurt? (PR 3)

**PR 3 `feat:` dual-timestep featurizer** (§3.3). Adds `SelfFlowParams`,
`NoisedView`, the second template distogram and the group-share diagnostic.

Tests:

- `R_M = 0` reproduces PR 2's featurization.
- The student's σ per residue is in `{σ_s, σ_t}` and constant within a
  residue.
- The teacher's σ equals the minimum.
- Both views share ε.
- The noise centre of mass is zero over valid atoms.
- `apply_prob = 0` always yields `R_M = 0`.

**Exp 1: `dual-t`.** `R_M = 0.25`, no teacher loss, at the three `P_mean`
values.

**Gate:** `dual-t` ≥ `baseline`, each at its best `P_mean`. The paper found
the schedule alone helps slightly, so this is a fair bar.

**If the gate fails,** in this order:

1. Check the group share. If one σ group is badly over-represented, switch
   the generative loss to per-group normalization: take the mean within each
   σ group, then weight the group means by atom fraction.
2. Rerun with `apply_prob = 0.5` to address the train/inference mismatch.
3. If both fail, stop. Mixed per-residue noise does not suit this model, and
   Self-Flow's premise fails with it.

### 4.5 Stage 3 — does Self-Flow help? (PR 4)

**PR 4 `feat:` EMA teacher pass and Arm A loss** (§3.4, §3.5). Adds
`teacher_mode`, `functional_call`, `alpha_5`, and the `self_flow_*` metrics.

Tests:

- `L_sf = 0` when the student and teacher weights and inputs are identical.
- No gradient reaches the EMA shadow.
- Student gradients are identical with and without the teacher pass. This
  guards against the `swap`/autograd problem in §3.4.
- `K_unit = 8` gives unit indices `(1, 5)`.

**Exp 2: `sf-coords`.** Uses Stage 2's settings (`R_M = 0.25`, plus
`apply_prob` or the group-normalized loss if Stage 2 needed them), at the
three `P_mean` values. Expect its best `P_mean` to be at or below
`dual-t`'s (§2.2, finding 2).

**Gate:** `sf-coords` beats `dual-t` on validation Kabsch MSE and lDDT at
matched steps, each at its best `P_mean`. Step overhead must stay under 35%.
The estimate is 25–30%: a no-grad forward is about a third of a forward plus
checkpointed backward, plus CPU time for the second view.

- **Pass:** go to Stage 4.
- **Flat or worse:** check `self_flow_teacher_gt_mse`. If the teacher's
  unit-6 coordinates are not clearly better than the student's unit-2
  coordinates against ground truth, the coordinate target is redundant with
  `med_loss`. Skip to Stage 5.

### 4.6 Stage 4 — tune Self-Flow (no PR)

All runs here use `sf-coords`'s best `P_mean`, one axis at a time:

| Run | Varies | Why |
|---|---|---|
| `sf-coords-R` | `R_M ∈ {0.1, 0.5}` | residues are locally redundant along the chain, so the best `R_M` may be nearer video's 0.1 than images' 0.25 |
| `sf-coords-layers` | `(k_s, k_t) ∈ {(2, 8), (3, 6)}` | sensitivity to the layer pair; the paper reports a broad optimum |
| `sf-coords-unweighted` | λ off for `L_sf` | run only if the σ-bucketed `L_sf` is dominated by the low-σ buckets (§3.5); clamping λ is the alternative |

**Output:** new defaults for `SelfFlowParams`.

### 4.7 Stage 5 — representation-space Self-Flow (PR 5, conditional)

Do this only if Stage 3 was flat, or as a follow-up if Stage 3 passed by a
small margin.

**PR 5 `feat:` Arm B** (§3.5). Adds the hidden-state stack and the projection
head.
Tests: head shapes; zero loss for identical inputs; no gradient to the
teacher.

**Exp 4: `sf-latents`.** Run at its own three `P_mean` values. Compare with
`dual-t` and with `sf-coords`.

### 4.8 Stage 6 — length-dependent shift (PR 6)

This stage is kept apart from the Self-Flow stages. It changes the noise
distribution for each length, and mixing it with Self-Flow would make the
earlier results impossible to interpret.

**PR 6 `feat:` length-dependent `P_mean(N)` and sampler `α(N)`** (§3.7). Adds
length-binned metrics and unshifted-σ bucketing.
Tests: `β = 0` is the identity; the mean shift is `β·ln(N/N_ref)`; the
training and sampling multipliers match.

**Experiments:**

1. **Exp 5.** `baseline` with `β ∈ {0, 0.25, 0.5}` at its best global
   `P_mean`.
2. **Exp 6** (only if Exp 5 and Stage 3 both helped). A 2×2 of length shift ×
   the best Self-Flow arm, each cell at its own best global `P_mean`.

### 4.9 Optional performance PR

If Stage 3's overhead is near the limit, add a `MainTrunk.forward` keyword
`n_units: int | None` so the teacher stops after unit `k_t`. This skips the
remaining units, the distogram heads and the sequence head, saving about 30%
of the teacher pass. It does not change behaviour. It can land any time
after PR 4.

## 5. Open question

**Shared conditioning dropout.** If conditioning dropout (sequence or
template) is added later, should the two views share the same draw? Sharing
seems right, so that the only difference between the views is the noise.
Otherwise the teacher could hold information the student cannot recover, and
`L_sf` would never shrink. `main` has no conditioning-dropout code path yet.
If it is applied in the featurizer before the views split, sharing comes for
free.

## 6. Out of scope: changing the per-residue representation

The BFL study's axis, channels per token, is constant for us: we diffuse
raw atom5 coordinates, 15 numbers per residue. It would matter if the
representation changed (atom14, atom37, or a learned protein latent). The
channel count also overstates the information: given the backbone frame,
N, C and Cβ are almost fully determined by ideal geometry, so a residue
carries about 6 numbers for the rigid frame plus a few torsions. That is the
same kind of gap that moved FLUX.2 off `√(m/n)`. Any such change should come
with its own `P_mean` and sampling-`α` sweeps (Stage 0's protocol), not a
computed shift.
