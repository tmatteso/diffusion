"""Self-Flow's EMA-teacher pass and loss terms for one training step.

Self-Flow (Chefer et al. 2026) trains a shallow student decoder unit to
match a deeper decoder unit of the EMA teacher, which sees a cleaner input
(the batch's ``teacher_view``). In pallatom both units' outputs are their
denoised coordinates, so the loss is the Kabsch-aligned, EDM-weighted MSE
``atom_loss`` already computes, with the teacher's coordinates as the
target.

The teacher runs through ``torch.func.functional_call`` with the EMA shadow
weights. That never touches the live parameters, so it cannot disturb the
student's autograd graph. ``EMA.swap`` reassigns ``.data`` in place, which
autograd does not track, and would silently give the student's backward pass
the teacher's weights if it ran between the forward and backward passes.
"""

import dataclasses
from types import TracebackType
from typing import Self, cast

import torch
from architecture.losses import atom_loss
from architecture.main_trunk import MainTrunk, PredictedOutputs
from beartype import beartype
from einops import rearrange
from helpers.atom_utils import ATOMS_PER_RES, atom5_to_atom37, atom37_to_cb
from helpers.data import Distogram, FeaturizedBatch, null_template_distogram
from helpers.useful_objects import ModelSetup
from jaxtyping import Float, jaxtyped
from torch.nn.parallel import DistributedDataParallel as DDP
from train.train_config import self_flow_unit_indices

LOW_SIGMA_ANGSTROM: float = 1.0
"""Teacher noise level (Å) below which a protein counts as low-sigma.

Below about 1 Å the EDM weight exceeds 1 and both views are nearly clean,
so this is where Self-Flow's loss weighting is most likely to amplify the
teacher's own errors (design doc §3.5).
"""


class TeacherMode:
    """Put a trunk in eval mode for the teacher pass, then restore its mode.

    The teacher's target should be deterministic, so dropout is off for its
    passes even when the student trains with it on.

    Args:
        trunk: The trunk whose mode to switch.
    """

    def __init__(self, trunk: MainTrunk) -> None:
        self.trunk: MainTrunk = trunk
        self.was_training: bool = trunk.training

    def __enter__(self) -> Self:
        """Switch the trunk to eval mode.

        Returns:
            This context manager.
        """
        self.was_training = self.trunk.training
        _ = self.trunk.eval()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the trunk's previous mode.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            traceback: Traceback, if the block raised.
        """
        _ = self.trunk.train(self.was_training)


@jaxtyped(typechecker=beartype)
def template_from_prediction(
    prediction: PredictedOutputs,
    batch: FeaturizedBatch,
    distogram_template: Distogram,
) -> Float[torch.Tensor, "B N_res N_res n_templ_bins"]:
    """Bin the Cβ distogram of a trunk prediction into a template.

    This is the self-conditioning template: the sampler feeds the trunk its
    own previous prediction, binned this way.

    Args:
        prediction: Trunk outputs whose denoised coordinates to bin.
        batch: Batch supplying the atom and residue masks.
        distogram_template: Template distogram module.

    Returns:
        One-hot template distogram of the predicted Cβ positions.
    """
    atom37_positions, atom37_mask = atom5_to_atom37(
        rearrange(
            prediction.r_denoised,
            "b (n a) d -> b n a d",
            a=ATOMS_PER_RES,
        ),
        rearrange(
            batch.atom5_mask,
            "b (n a) -> b n a",
            a=ATOMS_PER_RES,
        ).float(),
    )
    cb_positions, _ = atom37_to_cb(atom37_positions, atom37_mask)
    template, _ = distogram_template(
        cb_positions,
        batch.f_pseudo_beta_mask.bool(),
    )
    return template


@jaxtyped(typechecker=beartype)
def teacher_input(batch: FeaturizedBatch) -> FeaturizedBatch:
    """The batch as the teacher sees it: its ``teacher_view`` noising.

    Every clean field is shared with the student batch, not copied.

    Args:
        batch: Student batch carrying a teacher view.

    Returns:
        The batch with the teacher view's noised fields and no teacher view.

    Raises:
        ValueError: If the batch has no teacher view.
    """
    view = batch.teacher_view
    if view is None:
        msg = "batch has no teacher_view; enable Self-Flow to build one"
        raise ValueError(msg)
    return dataclasses.replace(
        batch,
        r_gt_noised=view.r_noised,
        noised_res_distogram=view.noised_res_distogram,
        t_hat=view.t_hat,
        t_normalized=view.t_normalized,
        teacher_view=None,
    )


@jaxtyped(typechecker=beartype)
def teacher_predictions(
    model_params: ModelSetup,
    batch: FeaturizedBatch,
) -> PredictedOutputs:
    """Run the EMA teacher on the batch's teacher view, without gradients.

    The teacher uses the EMA shadow weights, in eval mode, on the unwrapped
    trunk (never through DDP). With self-conditioning on, it first builds
    its own template from a pass against the null template, as the student
    does, so both see the template the sampler would give them.

    Args:
        model_params: Bundle holding the trunk, its EMA and the config.
        batch: Student batch carrying a teacher view.

    Returns:
        The teacher's outputs, including its per-unit coordinates.
    """
    trunk: MainTrunk = (
        cast(MainTrunk, model_params.model.module)
        if isinstance(model_params.model, DDP)
        else model_params.model
    )
    weights = model_params.ema.state_dict()
    teacher_batch = teacher_input(batch)
    with torch.no_grad(), TeacherMode(trunk):
        if model_params.tcfg.training.self_condition:
            draft = cast(
                PredictedOutputs,
                torch.func.functional_call(
                    trunk,
                    weights,
                    (
                        dataclasses.replace(
                            teacher_batch,
                            noised_res_distogram=null_template_distogram(
                                teacher_batch.noised_res_distogram.shape,
                                teacher_batch.noised_res_distogram.device,
                            ),
                        ),
                    ),
                ),
            )
            teacher_batch = dataclasses.replace(
                teacher_batch,
                noised_res_distogram=template_from_prediction(
                    draft,
                    teacher_batch,
                    model_params.distogram_template,
                ),
            )
        return cast(
            PredictedOutputs,
            torch.func.functional_call(trunk, weights, (teacher_batch,)),
        )


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class SelfFlowTerms:
    """The Self-Flow loss of one step and its diagnostics.

    All zero when the step has no teacher pass (Self-Flow off, evaluation,
    or ``alpha_5`` of 0).

    Attributes:
        loss: Mean Kabsch-aligned, EDM-weighted MSE between the student unit
            and the stop-gradient teacher unit. Carries gradients.
        teacher_gt_mse: Mean unweighted Kabsch MSE of the teacher unit
            against ground truth.
        student_gt_mse: The same for the student unit. If the teacher is not
            clearly below this, its target adds little over ``med_loss``.
        low_sigma_loss_share: Share of the loss from proteins whose teacher
            noise level is below ``LOW_SIGMA_ANGSTROM``.
        low_sigma_sample_share: Share of proteins below that level.
    """

    loss: Float[torch.Tensor, ""]
    teacher_gt_mse: Float[torch.Tensor, ""]
    student_gt_mse: Float[torch.Tensor, ""]
    low_sigma_loss_share: Float[torch.Tensor, ""]
    low_sigma_sample_share: Float[torch.Tensor, ""]

    @classmethod
    def zero(cls, device: torch.device) -> "SelfFlowTerms":
        """Terms for a step without a teacher pass.

        Args:
            device: Device of the zero tensors.

        Returns:
            Every term zero.
        """
        zero = torch.zeros((), device=device)
        return cls(
            loss=zero,
            teacher_gt_mse=zero,
            student_gt_mse=zero,
            low_sigma_loss_share=zero,
            low_sigma_sample_share=zero,
        )


@jaxtyped(typechecker=beartype)
def self_flow_terms(
    model_params: ModelSetup,
    batch: FeaturizedBatch,
    student: PredictedOutputs,
    lambda_sigma_weight: Float[torch.Tensor, "B N_atom"],
) -> SelfFlowTerms:
    """Run the teacher and compute the Self-Flow loss and diagnostics.

    The caller decides whether the step needs a teacher pass; this always
    runs one.

    Args:
        model_params: Bundle holding the trunk, its EMA and the config.
        batch: Student batch carrying a teacher view.
        student: The student's outputs for ``batch``.
        lambda_sigma_weight: The student's per-atom EDM weight.

    Returns:
        The loss, with gradients through the student only, and diagnostics.
    """
    cfg = model_params.tcfg
    student_unit, teacher_unit = self_flow_unit_indices(
        cfg.model.K_unit,
        cfg.self_flow.student_unit_frac,
        cfg.self_flow.teacher_unit_frac,
    )
    teacher = teacher_predictions(model_params, batch)
    student_coords = student.intermediate_denoised_coord_stack[student_unit]
    teacher_coords = teacher.intermediate_denoised_coord_stack[
        teacher_unit
    ].detach()
    mask = batch.atom5_mask.float()
    per_sample = atom_loss(
        student_coords,
        teacher_coords,
        mask,
        aa_indices=batch.aa_indices,
        lambda_sigma_weight=lambda_sigma_weight,
    )
    with torch.no_grad():
        unit_weight = torch.ones_like(lambda_sigma_weight)
        teacher_gt = atom_loss(
            teacher_coords,
            batch.r_gt,
            mask,
            aa_indices=batch.aa_indices,
            lambda_sigma_weight=unit_weight,
        )
        student_gt = atom_loss(
            student_coords.detach(),
            batch.r_gt,
            mask,
            aa_indices=batch.aa_indices,
            lambda_sigma_weight=unit_weight,
        )
        low = (teacher_input(batch).t_hat[:, 0] < LOW_SIGMA_ANGSTROM).float()
        detached = per_sample.detach()
        low_loss_share = (detached * low).sum() / detached.sum().clamp(
            min=1e-12,
        )
    return SelfFlowTerms(
        loss=per_sample.mean(),
        teacher_gt_mse=teacher_gt.mean(),
        student_gt_mse=student_gt.mean(),
        low_sigma_loss_share=low_loss_share,
        low_sigma_sample_share=low.mean(),
    )
