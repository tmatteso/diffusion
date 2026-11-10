# Design: Self-Flow dual-timestep self-distillation for pallatom

**Status:** Draft, for discussion. No code changes in this PR.
**Reference:** Chefer, Esser et al., *Self-Supervised Flow Matching for
Scalable Multi-Modal Synthesis* (Black Forest Labs, arXiv
[2603.06507](https://arxiv.org/abs/2603.06507), March 2026).

## 1. Summary

Self-Flow adds a self-supervised term to diffusion training. Each step builds
two views of one example:

- **Student view.** Every token is noised at one of two sampled noise levels,
  chosen per token by a random mask.
- **Teacher view.** Every token is noised at the *lower* of the two levels.

The teacher is the EMA copy of the model. A shallow student layer is trained to
match a deeper teacher layer. Because the teacher sees the cleaner input, the
student has to infer clean structure from noisy context. The paper reports
faster convergence than REPA and does not need an external encoder.

pallatom already has the main pieces:

| Self-Flow needs | pallatom already has |
|---|---|
| EMA teacher weights | `EMA` in `architecture/main_trunk.py`, updated on every optimizer step |
| Per-layer outputs | `intermediate_denoised_coord_stack` (one entry per decoder unit) |
| A loss on per-layer outputs | `med_loss` (Kabsch-aligned MSE plus CE for each block) |
| Per-token time conditioning | `TimeFourierEmbedding` already takes `log t̂` with shape `[B, N_res]` (it is a scalar broadcast today) |

The proposal adds one more intermediate-loss term. It compares student
decoder unit 2 (`0.3·K`) with EMA-teacher decoder unit 6 (`0.7·K`), using the
Kabsch-aligned MSE we already have. The training input becomes a mixed-noise
student view plus a cleaner teacher view.

## 2. What the paper does

The facts below were checked against the paper text (Sec. 3.3, 3.4, 4.4,
App. A.3, B, H).

1. **Two timesteps.** Draw `t, s ~ p(t)` independently from the usual training
   distribution.
2. **Per-token mask.** `M = {i : u_i < R_M}` with `u_i ~ U(0,1)`, which is an
   independent Bernoulli(`R_M`) draw per token. `R_M ≤ 0.5`. The paper uses
   `R_M = 0.25` for images, `0.5` for audio and `0.1` for video ("due to
   significant temporal redundancies"). It is applied to every sample. There is
   no per-sample gate.
3. **Per-token noise level.** `τ_i = s` if `i ∈ M`, otherwise `t`. Then
   `x_τ = diag(1−τ)·x0 + diag(τ)·x1`. The model gets a per-token timestep
   vector `t ∈ R^N` in place of a scalar.
4. **Teacher input.** All tokens are noised at `τ_min = min(t, s)`. The
   notation implies the same noise sample `x1` as the student view. The paper
   does not say this explicitly.
5. **Teacher weights.** The EMA copy (decay 0.9999), which is also the
   evaluation model.
6. **Representation loss.**
   `L_rep = −E[cos(h_θ(student layer 0.3D), teacher layer 0.7D)]`.
   `h_θ` is a small MLP projection head on the student only. ℓ1 was tried and
   was unstable as feature norms grew.
7. **Total loss.** `L = L_gen + γ·L_rep` with a fixed `γ = 0.8`. γ is not
   ablated.
8. **Ablations.** Removing `L_rep` costs more than 4 FID. Removing the masking
   but keeping `L_rep` costs more than 1 FID. The dual-timestep schedule alone,
   with no `L_rep`, gives a small gain. Results are stable across layer
   choices near 0.3D/0.7D. A shallow teacher layer hurts, and a deep student
   layer hurts generation.
9. **Cost.** One extra teacher forward pass with no gradients. The paper gives
   no overhead number.

## 3. Mapping to pallatom

### 3.1 Notation

pallatom uses EDM, not rectified flow, so "timestep" here means the noise
level `σ` (`t_hat` in code):

```
x_σ = x0 + σ · ε,        σ = σ_data · exp(N(P_mean, P_std²))
```

The tokens are **residues**. All 5 atoms of a residue share that residue's
`σ_i`. Masking happens at residue level, never atom level.

### 3.2 Training-time algorithm

For each item in `featurize_single_item`:

1. Draw `σ_s, σ_t` independently from the existing log-normal.
2. Draw `m ~ Bernoulli(R_M)` of length `N_res`, and set
   `σ_i = where(m_i, σ_s, σ_t)`. This is the **ρ vector** from the original
   notes.
3. Draw a single `ε ~ N(0, I)` of shape `[N_atom, 3]`, shared by both views.
4. Student coordinates: `r_student = r_gt + center(σ_atom ⊙ ε)`, where
   `σ_atom = repeat(σ_res, "n -> (n a)", a=5)`. This is the
   "diagonalize ρ and multiply by the noise" step.
5. Teacher coordinates: `r_teacher = r_gt + center(σ_min · ε)` with
   `σ_min = min(σ_s, σ_t)`.
6. Build the self-conditioning template distogram **separately for each
   view**, from that view's noised Cβ.

Step 6 is where Self-Flow fits pallatom unusually well. The template distogram
for pair `(i, j)` now carries noise from both `σ_i` and `σ_j`. Some regions of
the pair map are sharp and others are blurred, which acts as span-masking of
the distogram for free.

**Centering detail.** Today we compute `t_hat · (ε − CoM(ε))`. With
per-residue σ, `σ_i · (ε_i − CoM(ε))` does not have zero centre of mass.
Center *after* scaling: `n = σ_atom ⊙ ε`, then `n − CoM(n)`. For uniform σ the
two forms are identical, so the baseline is unchanged.

### 3.3 Model changes (per-residue σ)

`t_hat` changes from `Float[Tensor, "B"]` to `Float[Tensor, "B N_res"]`
everywhere. Wherever it is used, it is broadcast to atoms with
`repeat(t_hat, "b n -> b (n a)", a=atoms_per_res)`.

| Site (`main_trunk.py`) | Today | With per-residue σ |
|---|---|---|
| Step 1, `c_in` (`denom`) | `[B]` | `[B, N_atom]`, per-atom scaling of `r_input` |
| Step 3, `TimeFourierEmbedding` | `repeat(log t̂, "b -> b n")` | pass `0.25·log(t_hat)` directly, already `[B, N_res]` |
| Step 14, `c_skip` / `c_out` (`w_gt`, `w_upd`) | `"b -> b 1 1"` | `"b n_atom -> b n_atom 1"` |
| `lambda_sigma_weight` (train loop) | `[B]` | `[B, N_atom]` |
| `atom_loss` / `med_loss` | `λ` scales the per-sample loss after reduction | `λ` multiplies each atom's squared error **before** the weighted mean |

The sampler (`sample/sampling.py`) builds `t_hat_batch` with
`repeat(t_hat, " -> b n", b=B, n=N_res)`. Inference always uses uniform σ.

This refactor is **behaviour-preserving when `R_M = 0`**. It should land
first, on its own, with a regression test showing bitwise-equal outputs for
uniform σ.

### 3.4 Teacher forward pass

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

- **Do not use `EMA.swap` here.** `swap` reassigns `.data` on the live
  parameters, and that does not bump autograd's version counter. If the
  teacher ran between the student forward and `backward()`, the backward pass
  would silently use the EMA weights in the saved tensors. Running the teacher
  first would avoid that, but the code would then depend on call order to stay
  correct. `functional_call` never mutates the module, so call order does not
  matter.
- **Run the teacher on the unwrapped module.** It must not go through DDP:
  there are no gradients, so no all-reduce hooks are needed.
- **Use eval mode for the teacher.** `NodeUpdate` uses `DropoutRowwise(0.25)`,
  and a teacher target should be deterministic. `teacher_mode` is a small
  context manager in `helpers/context_managers.py` (alongside `StepContext`)
  that calls `eval()` and then restores the previous mode.
- `teacher_batch` is
  `dataclasses.replace(batch, r_gt_noised=..., noised_res_distogram=..., t_hat=...)`.
  It reuses every clean field, so there is no second copy of the batch on the
  GPU.
- **Follow-up optimisation.** The teacher only needs decoder units `0..5`. A
  later `MainTrunk.forward` keyword such as `n_units: int | None` could skip
  units 7–8, the distogram heads and the sequence head. That saves roughly 30%
  of the teacher pass. Not needed for v1.

### 3.5 The Self-Flow loss

```
L_sf = mean_b  KabschMSE( student.coord_stack[k_s],
                          stopgrad(teacher.coord_stack[k_t]) ;  λ(σ_student) )
k_s = round(0.3 · K_unit) − 1 = 1     (2nd decoder unit, K_unit = 8)
k_t = round(0.7 · K_unit) − 1 = 5     (6th decoder unit)
```

`atom_loss` can be reused as is, with the teacher coordinates in the `r_gt`
slot. It already aligns the target onto the prediction under `no_grad`.

`total_loss += γ_sf · L_sf` with `γ_sf = 0.8`, which becomes `alpha_5` in
`LossParams`.

**The sign is `+`.** The paper's `−1·` comes from *maximising* cosine
similarity. We *minimise* an MSE, so adding a `−1·` would push the student
*away* from the teacher.

### 3.6 Two loss variants (A/B)

The paper aligns **hidden representations** with cosine similarity through a
projection head. The original notes propose aligning the **per-unit coordinate
readout** with Kabsch MSE. The two are different bets, so both should run:

| | **Arm A: coordinate space** (original notes) | **Arm B: representation space** (paper) |
|---|---|---|
| Student signal | `intermediate_denoised_coord_stack[1]` | `c_l` or `s_i` after unit 2, through a new `LinearNoBias` + SiLU + `LinearNoBias` head |
| Teacher signal | `intermediate_denoised_coord_stack[5]` | same tensor after unit 6, no head |
| Distance | Kabsch-aligned MSE with `λ(σ)` weighting | `1 − cos`, per atom or per residue, masked mean |
| New params | none | about `2·c²` |
| Invariance | SE(3) handled by Kabsch | `c_l`/`s_i` are already invariant (no coordinates enter them except through `r_scaled` and the template) |
| Risk | Unit 2 is already supervised against **GT** by `med_loss`. A teacher target that is itself trained toward GT may be redundant | Needs the coordinate stacks extended with a stack of hidden states |

**Recommendation:** implement Arm A first. It has no new modules and the
change to the intermediate-loss loop is very small. Add Arm B only if A is
flat against the dual-timestep-only control (§6). In Arm A, the student
coordinates at unit 2 include `c_skip · r_input` from a *noisier* input, so
the student must denoise the masked residues up to the teacher's level. That
is the intended signal, not a confound.

## 4. Configuration

A new `SelfFlowParams` block in `train/train_config.py` (pydantic, validated):

```python
class SelfFlowParams(BaseModel):
    enabled: bool = False
    mask_ratio: float = Field(default=0.25, ge=0, le=0.5)      # R_M
    student_unit_frac: float = Field(default=0.3, gt=0, lt=1)
    teacher_unit_frac: float = Field(default=0.7, gt=0, lt=1)
    target: Literal["coords", "latents"] = "coords"            # Arm A / B
```

The loss weight goes in `LossParams.alpha_5 = 0.8`. A `model_validator`
enforces `k_s < k_t` after rounding.

- **EMA decay.** The paper uses 0.9999, and we use 0.999. One EMA serves both
  the teacher and evaluation, which matches the paper. The bias-corrected ramp
  `min(decay, (step+1)/(step+10))` means the teacher is about the same as the
  student for the first few hundred steps. That is harmless, because the
  target is then just "the deeper unit on a cleaner input". No warm-up gate in
  v1.
- **Validation always runs with `R_M = 0` and no teacher pass.** This keeps
  validation numbers comparable with existing runs. Only the training collate
  path uses the dual-timestep featurizer.

## 5. Data-structure changes

- `FeaturizedItem` / `FeaturizedBatch`: `t_hat` becomes `"N_res"` /
  `"B N_res"`.
- Add `teacher_view: NoisedView | None`, a new `@dataclass` with fields
  `r_noised`, `noised_res_distogram` and `t_hat`. `to()` and `pin_memory()`
  must pass it through. It is `None` when Self-Flow is disabled, and on
  validation and sampling batches.
- `LossMetrics`: add `self_flow_loss`, logged to wandb next to
  `intermediate_loss`. Add these diagnostics:
  - `self_flow_teacher_gt_mse`: the teacher's unit-6 error against GT. It
    shows whether the target is better than what the student already gets
    from `med_loss`.
  - `L_sf` bucketed by student σ (§9.1).
  - Each σ group's share of the λ-weighted `atom_loss` against its share of
    atoms (§9.2).

## 6. Experiment plan

Same seed, data, steps and batch for each run. Compare EMA-weight validation
metrics and sample quality. Wall-clock time is reported separately.

| Run | `R_M` | Teacher loss | Purpose |
|---|---|---|---|
| `baseline` | 0 | none | control |
| `dual-t` | 0.25 | none (`alpha_5 = 0`) | the paper finds the dual-timestep schedule alone helps a little; isolates it |
| `sf-coords` | 0.25 | Arm A | main hypothesis |
| `sf-latents` | 0.25 | Arm B | the paper's version |
| `sf-coords-R` | {0.1, 0.5} | Arm A | protein residues have strong local redundancy along the chain, so the right `R_M` may be closer to video (0.1) than to images (0.25) |
| `sf-coords-layers` | 0.25 | Arm A, (k_s, k_t) ∈ {(2,8), (3,6)} | check sensitivity to the layer pair |

"Sucks or not" criterion: `sf-coords` should beat `dual-t` on validation
Kabsch MSE and lDDT at matched steps, and `dual-t` should be at least as good
as `baseline`. If `dual-t` falls short of `baseline`, revisit the
mixed-noise loss weighting first (§9.2), before judging the teacher loss. The overhead should be under 35% per step. With a no-grad
forward at about one third of forward plus checkpointed backward, the expected
overhead is about 25–30%, plus CPU time to featurize the second view.

## 7. Implementation phases (separate PRs)

1. **`refactor:` per-residue `t_hat`.** Covers the shape change across
   `data.py`, `main_trunk.py`, `losses.py`, `train_loop.py` and `sampling.py`.
   Regression test: uniform σ gives bitwise-identical `PredictedOutputs` and
   losses compared with `main`.
2. **`feat:` dual-timestep featurizer.** Adds `SelfFlowParams`, `NoisedView`
   and the second template distogram. Tests: with `R_M = 0` the student view
   equals the old path; student σ ∈ {σ_s, σ_t} per residue and is constant
   within a residue; teacher σ equals `min`; both views share ε; the noise
   CoM is zero over valid atoms.
3. **`feat:` EMA teacher pass and `L_sf` (Arm A).** Adds `teacher_mode`,
   `functional_call` and the new `LossMetrics` field. Tests: `L_sf = 0` when
   student and teacher are identical and inputs are identical; no gradient
   reaches the EMA shadow; student gradients are identical with and without
   the teacher pass computed (guards against the swap/autograd trap in §3.4);
   for `K_unit = 8` the unit indices are `(1, 5)`.
4. **`feat:` Arm B (optional).** Adds the hidden-state stack and projection
   head.

## 8. Open questions

- **Teacher dropout.** If conditioning dropout (sequence or template) is added
  later, should it be *shared* between the views? Sharing seems right, so the
  only difference between views is the noise. On `main` there is no
  conditioning-dropout code path yet: only a mask-token placeholder in
  `atom_utils.py` and docstring references. Applying it in the featurizer
  before both views are built would share it automatically.

## 9. Decisions

### 9.1 λ weighting for `L_sf`

**Default:** use the student's per-atom `λ(σ_i)`, so `L_sf` is on the same
scale as `med_loss` and `γ_sf = 0.8` keeps a similar meaning.

**Why it might be wrong:** `L_sf` has a target with its own error, the
teacher's prediction. At small σ the two views are nearly identical, and λ is
large (about 100 at σ = 0.1, and about 15% of draws have σ < 1). So the
leftover student–teacher disagreement, including the teacher's own mistakes,
gets amplified on exactly the samples where distillation teaches the least.
The paper's `L_rep` has no time weighting.

**Decision rule:** log `L_sf` bucketed by the student's σ (for example,
log-spaced buckets over the sampler's range). If the low-σ buckets dominate
the total, switch to an unweighted MSE or clamp λ.

### 9.2 Per-residue weighting of the mixed-noise `L_gen`

**Default:** in `atom_loss` and `med_loss`, multiply each atom's squared error
by its own `λ(σ_i)` *before* the weighted mean. Smooth-lDDT, the distogram
losses and CE have never been noise-weighted and stay unchanged.

**Why it is the principled choice:** EDM's denoiser is
`D = c_skip·x + c_out·F`, so `D − x0 = c_out·(F − F_target)`, and
`λ = 1/c_out²`. That gives

```
λ(σ_i) · ‖D_i − x0_i‖²  =  ‖F_i − F_target,i‖²
```

Per-atom λ therefore weights every atom equally in terms of the raw network
output F. Low-σ residues have large λ, but their coordinate-space errors are
smaller by the same factor, so a clean subset does not automatically dominate
the sample's loss. For uniform σ this reduces exactly to today's per-sample
form. It relies on the per-atom `c_skip`/`c_out` from §3.3.

**Decision rule:**

1. **Gate on `dual-t` vs `baseline` (§6).** `dual-t` uses mixed noise with no
   teacher loss. If it underperforms `baseline` on uniform-σ validation, the
   mixed-noise generative loss is the prime suspect, and its weighting is the
   first thing to change. The paper found that the dual-timestep schedule
   alone helps slightly, so "matches or beats baseline" is a fair bar.
2. **Log each group's share of the loss.** Per sample, split residues into the
   lower-σ and higher-σ groups. Log each group's share of the λ-weighted
   `atom_loss` next to its share of valid atoms. If the cancellation above
   holds, the two shares should roughly match. A large, persistent gap means
   one group is taking most of the gradient. Likely causes are poor
   calibration at one end of the σ range, or whole-protein Kabsch alignment
   letting noisy residues inflate errors on clean ones.

**Fallback (if either check fails):** take the mean within each σ group
separately, then combine the group means weighted by each group's fraction of
atoms. Each group then contributes in proportion to its size, whatever its λ.

**Test (either way):** with `R_M = 0`, the per-atom form must reproduce
`main`'s `atom_loss` and `med_loss` exactly. This belongs in phase 1 (§7).
