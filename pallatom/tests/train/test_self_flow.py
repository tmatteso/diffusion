"""Tests for Self-Flow's EMA-teacher pass and loss terms.

Covers the teacher input, the teacher pass (against a copy of the trunk with
the EMA weights loaded), its side effects on the live trunk, the Self-Flow
loss and when ``take_step`` runs it, and the decoder-unit selection.
"""

import copy
import dataclasses
import enum

import numpy as np
import pytest
import torch
from architecture.main_trunk import EMA, MainTrunk, PredictedOutputs
from einops import rearrange
from helpers.atom_utils import Protein
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    build_distogram_module,
    featurize_batch,
    null_template_distogram,
)
from helpers.useful_objects import LossMetrics, ModelSetup, manual_seed
from jaxtyping import Float
from pydantic import ValidationError
from torch.optim import Adam
from train.self_flow import (
    SelfFlowTerms,
    TeacherMode,
    self_flow_terms,
    teacher_input,
    teacher_predictions,
    template_from_prediction,
)
from train.train_config import (
    ModelParams,
    SelfFlowParams,
    TrainConfig,
    self_flow_unit_indices,
)
from train.train_loop import build_scheduler, take_step

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


@pytest.fixture
def self_flow_config() -> TrainConfig:
    """Three-unit trunk config with Self-Flow on and half the residues masked.

    Three units are the fewest that give distinct student (0) and teacher
    (1) units with the default depth fractions.

    Returns:
        The training configuration.
    """
    return TrainConfig(
        model=ModelParams(K_unit=3),
        self_flow=SelfFlowParams(enabled=True, mask_ratio=0.5),
    )


@pytest.fixture
def template_distogram(self_flow_config: TrainConfig) -> Distogram:
    """Template distogram module for the config.

    Args:
        self_flow_config: Training configuration.

    Returns:
        The template distogram module.
    """
    return build_distogram_module(self_flow_config.distogram_template).eval()


@pytest.fixture
def residue_distogram(self_flow_config: TrainConfig) -> Distogram:
    """Residue distogram module for the config.

    Args:
        self_flow_config: Training configuration.

    Returns:
        The residue distogram module.
    """
    return build_distogram_module(self_flow_config.distogram_residue).eval()


@pytest.fixture
def atom_distogram(self_flow_config: TrainConfig) -> Distogram:
    """Atom distogram module for the config.

    Args:
        self_flow_config: Training configuration.

    Returns:
        The atom distogram module.
    """
    return build_distogram_module(self_flow_config.distogram_atom).eval()


@pytest.fixture
def model_setup(
    self_flow_config: TrainConfig,
    template_distogram: Distogram,
    residue_distogram: Distogram,
    atom_distogram: Distogram,
) -> ModelSetup:
    """Seeded trunk in train mode whose EMA differs from its live weights.

    The EMA shadow is perturbed by small seeded noise, so a pass that used
    the live weights instead of the EMA weights gives different outputs.

    Args:
        self_flow_config: Training configuration.
        template_distogram: Template distogram module.
        residue_distogram: Residue distogram module.
        atom_distogram: Atom distogram module.

    Returns:
        The bundled training objects.
    """
    _ = manual_seed(0)
    trunk = MainTrunk(
        model_params=self_flow_config.model,
        template_distogram_params=self_flow_config.distogram_template,
        residue_distogram_params=self_flow_config.distogram_residue,
        atom_distogram_params=self_flow_config.distogram_atom,
        noise_params=self_flow_config.noise,
    ).train()
    ema = EMA(trunk, decay=self_flow_config.training.ema_decay)
    for tensor in ema.state_dict().values():
        if torch.is_floating_point(tensor):
            _ = tensor.add_(0.01 * torch.randn_like(tensor))
    optimizer = Adam(trunk.parameters(), lr=self_flow_config.training.lr)
    return ModelSetup(
        model=trunk,
        tcfg=self_flow_config,
        distogram_template=template_distogram,
        distogram_residue=residue_distogram,
        distogram_atom=atom_distogram,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, self_flow_config.training),
        ema=ema,
    )


@pytest.fixture
def self_flow_batch(
    self_flow_config: TrainConfig,
    template_distogram: Distogram,
    residue_distogram: Distogram,
    atom_distogram: Distogram,
) -> FeaturizedBatch:
    """Seeded Self-Flow batch of two alanine proteins, six and four residues.

    Args:
        self_flow_config: Training configuration.
        template_distogram: Template distogram module.
        residue_distogram: Residue distogram module.
        atom_distogram: Atom distogram module.

    Returns:
        The featurized batch with its teacher view.
    """
    proteins = [
        Protein(
            atom_positions=rearrange(
                0.5 * np.arange(n_res * 37 * 3, dtype=np.float64),
                "(n a d) -> n a d",
                a=37,
                d=3,
            ),
            aatype=np.zeros(n_res, dtype=np.intp),
            atom_mask=np.ones((n_res, 37), dtype=np.float64),
            residue_index=np.arange(n_res, dtype=np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )
        for n_res in (6, 4)
    ]
    _ = manual_seed(1)
    return featurize_batch(
        proteins,
        self_flow_config,
        template_distogram,
        residue_distogram,
        atom_distogram,
    )


# ---------------------------------------------------------------------------
# teacher_input
# ---------------------------------------------------------------------------


def test_teacher_input_swaps_in_the_teacher_view(
    self_flow_batch: FeaturizedBatch,
) -> None:
    """The teacher batch carries the teacher view's noising and no view.

    No other cases: the swap is fixed field for field, and clean fields are
    shared rather than copied.

    Args:
        self_flow_batch: Student batch with a teacher view.
    """
    view = self_flow_batch.teacher_view
    assert view is not None
    teacher = teacher_input(self_flow_batch)
    assert teacher.teacher_view is None
    assert teacher.r_gt_noised is view.r_noised
    assert teacher.t_hat is view.t_hat
    assert teacher.t_normalized is view.t_normalized
    assert teacher.noised_res_distogram is view.noised_res_distogram
    assert teacher.r_gt is self_flow_batch.r_gt


def test_teacher_input_requires_a_teacher_view(
    self_flow_batch: FeaturizedBatch,
) -> None:
    """A batch without a teacher view cannot be given to the teacher.

    No other cases: there is one way to lack a view.

    Args:
        self_flow_batch: Batch whose view is removed for the call.
    """
    with pytest.raises(ValueError, match="teacher_view"):
        _ = teacher_input(
            dataclasses.replace(self_flow_batch, teacher_view=None),
        )


# ---------------------------------------------------------------------------
# teacher_predictions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TeacherCase:
    """Whether the teacher builds its own self-conditioning template.

    Attributes:
        self_condition: Self-conditioning setting of the run.
    """

    self_condition: bool


class TeacherScenario(enum.Enum):
    """Teacher passes with and without self-conditioning.

    Members:
        self_conditioned: The default; the teacher first predicts against
            the null template and bins that prediction into its template.
        featurized_template: The teacher uses the featurized template of
            its own noised view.
    """

    self_conditioned = TeacherCase(self_condition=True)
    featurized_template = TeacherCase(self_condition=False)


@pytest.fixture
def teacher_setup(
    teacher_scenario: TeacherScenario,
    model_setup: ModelSetup,
) -> ModelSetup:
    """``model_setup`` with the scenario's self-conditioning setting.

    Args:
        teacher_scenario: Self-conditioning on or off; the requesting test
            parametrizes it.
        model_setup: Base bundle.

    Returns:
        The bundle with ``training.self_condition`` set.
    """
    cfg = model_setup.tcfg
    return dataclasses.replace(
        model_setup,
        tcfg=cfg.model_copy(
            update={
                "training": cfg.training.model_copy(
                    update={
                        "self_condition": (
                            teacher_scenario.value.self_condition
                        ),
                    },
                ),
            },
        ),
    )


@pytest.fixture
def ema_copy_prediction(
    teacher_scenario: TeacherScenario,
    teacher_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
) -> PredictedOutputs:
    """The teacher's outputs computed the slow way, as a reference.

    Deep-copies the trunk, loads the EMA weights into the copy, switches it
    to eval mode, and runs the teacher view through it, with a
    self-conditioning draft first when the scenario asks for one.

    Args:
        teacher_scenario: Self-conditioning on or off; the requesting test
            parametrizes it.
        teacher_setup: Bundle holding the trunk and its EMA.
        self_flow_batch: Student batch with a teacher view.

    Returns:
        The reference teacher outputs.
    """
    reference = copy.deepcopy(teacher_setup.model)
    _ = reference.load_state_dict(teacher_setup.ema.state_dict())
    _ = reference.eval()
    teacher_batch = teacher_input(self_flow_batch)
    with torch.no_grad():
        if teacher_scenario.value.self_condition:
            draft: PredictedOutputs = reference(
                dataclasses.replace(
                    teacher_batch,
                    noised_res_distogram=null_template_distogram(
                        teacher_batch.noised_res_distogram.shape,
                        teacher_batch.noised_res_distogram.device,
                    ),
                ),
            )
            teacher_batch = dataclasses.replace(
                teacher_batch,
                noised_res_distogram=template_from_prediction(
                    draft,
                    teacher_batch,
                    teacher_setup.distogram_template,
                ),
            )
        return reference(teacher_batch)


@pytest.mark.parametrize(
    "teacher_scenario",
    list(TeacherScenario),
    ids=[case.name for case in TeacherScenario],
)
def test_teacher_predictions_match_an_ema_loaded_copy(
    teacher_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
    ema_copy_prediction: PredictedOutputs,
) -> None:
    """The teacher is the trunk with EMA weights, in eval mode, on its view.

    Arrange: a trunk whose EMA differs from its live weights.
    Act: ``teacher_predictions``, and the same pass on a deep copy with the
    EMA weights loaded.
    Assert: final and per-unit coordinates match, so ``functional_call``
    used the EMA weights, eval mode and the teacher view, and built the
    teacher's own template when self-conditioning is on.

    Args:
        teacher_setup: Bundle holding the trunk and its EMA.
        self_flow_batch: Student batch with a teacher view.
        ema_copy_prediction: Reference outputs from the EMA-loaded copy.
    """
    teacher = teacher_predictions(teacher_setup, self_flow_batch)
    torch.testing.assert_close(
        teacher.r_denoised,
        ema_copy_prediction.r_denoised,
    )
    for unit, reference in zip(
        teacher.intermediate_denoised_coord_stack,
        ema_copy_prediction.intermediate_denoised_coord_stack,
        strict=True,
    ):
        torch.testing.assert_close(unit, reference)


@pytest.fixture
def live_weights_before(
    model_setup: ModelSetup,
) -> dict[str, Float[torch.Tensor, "..."]]:
    """Copies of the live trunk's weights and buffers before any pass.

    Args:
        model_setup: Bundle holding the trunk.

    Returns:
        Each state-dict entry, cloned.
    """
    trunk = model_setup.model
    return {
        name: tensor.detach().clone()
        for name, tensor in [
            *trunk.named_parameters(),
            *trunk.named_buffers(),
        ]
    }


def test_teacher_pass_leaves_the_live_trunk_untouched(
    model_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
    live_weights_before: dict[str, Float[torch.Tensor, "..."]],
) -> None:
    """Running the teacher changes neither the live weights nor the mode.

    No other cases: the guard is that ``functional_call`` never writes the
    live parameters. A ``.data`` swap that was not undone, or a mode left
    at eval, would show up here.

    Args:
        model_setup: Bundle holding a trunk in train mode.
        self_flow_batch: Student batch with a teacher view.
        live_weights_before: The live weights before the pass.
    """
    _ = teacher_predictions(model_setup, self_flow_batch)
    assert model_setup.model.training
    trunk = model_setup.model
    for name, tensor in [*trunk.named_parameters(), *trunk.named_buffers()]:
        assert torch.equal(tensor, live_weights_before[name]), name


def test_teacher_mode_restores_the_previous_mode(
    model_setup: ModelSetup,
) -> None:
    """``TeacherMode`` switches to eval inside the block and back after.

    No other cases: the trunk starts in train mode, the mode the teacher
    pass is used from.

    Args:
        model_setup: Bundle holding a trunk in train mode.
    """
    trunk = model_setup.model
    assert isinstance(trunk, MainTrunk)
    with TeacherMode(trunk):
        assert not trunk.training
    assert trunk.training


# ---------------------------------------------------------------------------
# self_flow_terms
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_lambda(
    self_flow_batch: FeaturizedBatch,
) -> Float[torch.Tensor, "B N_atom"]:
    """EDM weight of 1 on every atom of ``self_flow_batch``.

    Args:
        self_flow_batch: Batch to weight.

    Returns:
        A tensor of ones.
    """
    return torch.ones_like(self_flow_batch.atom5_mask, dtype=torch.float32)


@pytest.fixture
def teacher_copy_student(
    model_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
) -> PredictedOutputs:
    """A student whose every unit equals the teacher's target unit.

    Built from the teacher's own outputs, so the Self-Flow loss must be 0.

    Args:
        model_setup: Bundle holding the trunk and its EMA.
        self_flow_batch: Student batch with a teacher view.

    Returns:
        The student outputs.
    """
    teacher = teacher_predictions(model_setup, self_flow_batch)
    target = teacher.intermediate_denoised_coord_stack[1]
    return dataclasses.replace(
        teacher,
        intermediate_denoised_coord_stack=[
            target for _ in teacher.intermediate_denoised_coord_stack
        ],
    )


def test_self_flow_loss_is_zero_when_the_student_matches_the_teacher(
    model_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
    teacher_copy_student: PredictedOutputs,
    unit_lambda: Float[torch.Tensor, "B N_atom"],
) -> None:
    """Identical student and teacher units give a zero Self-Flow loss.

    No other cases: zero is the only exact value; that the loss is positive
    otherwise is checked through ``take_step`` below.

    Args:
        model_setup: Bundle holding the trunk and its EMA.
        self_flow_batch: Student batch with a teacher view.
        teacher_copy_student: Student equal to the teacher's target unit.
        unit_lambda: EDM weight of 1.
    """
    terms: SelfFlowTerms = self_flow_terms(
        model_setup,
        self_flow_batch,
        teacher_copy_student,
        unit_lambda,
    )
    torch.testing.assert_close(terms.loss, torch.tensor(0.0))


@pytest.fixture
def student_with_grad(
    model_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
) -> PredictedOutputs:
    """The live trunk's outputs on the student batch, with gradients.

    Args:
        model_setup: Bundle holding the trunk.
        self_flow_batch: Student batch.

    Returns:
        The student outputs.
    """
    return model_setup.model(self_flow_batch)


def test_self_flow_loss_trains_the_student_only(
    model_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
    student_with_grad: PredictedOutputs,
    unit_lambda: Float[torch.Tensor, "B N_atom"],
) -> None:
    """Gradients reach the live trunk and never the EMA teacher.

    No other cases: one backward pass shows both directions.

    Args:
        model_setup: Bundle holding the trunk and its EMA.
        self_flow_batch: Student batch with a teacher view.
        student_with_grad: Student outputs with gradients.
        unit_lambda: EDM weight of 1.
    """
    terms = self_flow_terms(
        model_setup,
        self_flow_batch,
        student_with_grad,
        unit_lambda,
    )
    torch.autograd.backward([terms.loss])
    grads = [
        p.grad for p in model_setup.model.parameters() if p.grad is not None
    ]
    assert grads
    assert any(bool(g.abs().sum() > 0) for g in grads)
    assert not any(
        t.requires_grad for t in model_setup.ema.state_dict().values()
    )


# ---------------------------------------------------------------------------
# take_step: when the teacher runs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GateCase:
    """A training-step setting and whether it must run the teacher.

    Attributes:
        alpha_5: Weight of the Self-Flow loss.
        train_mode: Whether the step trains (backward pass included).
        teacher_runs: Whether the step must run the teacher pass.
    """

    alpha_5: float
    train_mode: bool
    teacher_runs: bool


class TeacherGate(enum.Enum):
    """Steps on a Self-Flow batch that do and do not run the teacher.

    Members:
        self_flow: Training with the default weight 0.8 runs it.
        dual_t_only: Weight 0, the Stage 2 ``dual-t`` control, skips it.
        evaluation: Evaluation steps skip it.
    """

    self_flow = GateCase(alpha_5=0.8, train_mode=True, teacher_runs=True)
    dual_t_only = GateCase(alpha_5=0.0, train_mode=True, teacher_runs=False)
    evaluation = GateCase(alpha_5=0.8, train_mode=False, teacher_runs=False)


@pytest.fixture
def gated_metrics(
    teacher_gate_scenario: TeacherGate,
    model_setup: ModelSetup,
    self_flow_batch: FeaturizedBatch,
) -> LossMetrics:
    """Metrics of one ``take_step`` with the scenario's weight and mode.

    Args:
        teacher_gate_scenario: Weight and mode; the requesting test
            parametrizes it.
        model_setup: Bundle holding the trunk and its EMA.
        self_flow_batch: Self-Flow batch to step on.

    Returns:
        The step's loss metrics.
    """
    case = teacher_gate_scenario.value
    cfg = model_setup.tcfg
    setup = dataclasses.replace(
        model_setup,
        tcfg=cfg.model_copy(
            update={
                "loss": cfg.loss.model_copy(update={"alpha_5": case.alpha_5}),
            },
        ),
    )
    metrics, _ = take_step(
        batch=self_flow_batch,
        model_params=setup,
        train_mode=case.train_mode,
    )
    return metrics


@pytest.mark.parametrize(
    "teacher_gate_scenario",
    list(TeacherGate),
    ids=[case.name for case in TeacherGate],
)
def test_take_step_runs_the_teacher_only_when_its_loss_counts(
    gated_metrics: LossMetrics,
    teacher_gate_scenario: TeacherGate,
) -> None:
    """Only training steps with a non-zero weight pay for the teacher.

    Arrange: a Self-Flow batch, and a step setting from the scenario.
    Act: one ``take_step``.
    Assert: the teacher's ground-truth error, which only the teacher pass
    produces, is positive exactly when the teacher must run, and so is the
    Self-Flow loss. The total loss stays finite either way.

    Args:
        gated_metrics: Metrics of the step.
        teacher_gate_scenario: Whether the teacher must run; parametrized.
    """
    runs = teacher_gate_scenario.value.teacher_runs
    assert bool(gated_metrics.self_flow_teacher_gt_mse > 0) is runs
    assert bool(gated_metrics.self_flow_loss > 0) is runs
    assert bool(torch.isfinite(gated_metrics.total_loss))


# ---------------------------------------------------------------------------
# Decoder-unit selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UnitCase:
    """Decoder count and the zero-based units Self-Flow must align.

    Attributes:
        k_unit: Number of decoder units.
        student: Expected student unit index.
        teacher: Expected teacher unit index.
    """

    k_unit: int
    student: int
    teacher: int


class UnitSelection(enum.Enum):
    """Units chosen by the default fractions 0.3 and 0.7.

    Members:
        default_depth: 8 units give units 2 and 6, indices 1 and 5.
        smallest: 3 units give indices 0 and 1.
        deeper: 10 units give units 3 and 7, indices 2 and 6.
    """

    default_depth = UnitCase(k_unit=8, student=1, teacher=5)
    smallest = UnitCase(k_unit=3, student=0, teacher=1)
    deeper = UnitCase(k_unit=10, student=2, teacher=6)


@pytest.mark.parametrize(
    "unit_selection",
    list(UnitSelection),
    ids=[case.name for case in UnitSelection],
)
def test_self_flow_unit_indices_follow_the_depth_fractions(
    unit_selection: UnitSelection,
) -> None:
    """The default fractions pick 0.3D and 0.7D, rounded, as indices.

    Arrange: a decoder count.
    Act: ``self_flow_unit_indices`` with the default fractions.
    Assert: the student and teacher indices match the hand-worked values.

    Args:
        unit_selection: Decoder count and expected indices; parametrized.
    """
    case = unit_selection.value
    defaults = SelfFlowParams()
    assert self_flow_unit_indices(
        case.k_unit,
        defaults.student_unit_frac,
        defaults.teacher_unit_frac,
    ) == (case.student, case.teacher)


def test_config_rejects_self_flow_without_distinct_units(
    self_flow_config: TrainConfig,
) -> None:
    """Two decoder units cannot host distinct student and teacher units.

    No other cases: two units is the largest count where both default
    fractions round to the same unit. Pydantic reports the validator's
    ``SelfFlowUnitsError`` as a ``ValidationError``.

    Args:
        self_flow_config: A valid Self-Flow config to shrink.
    """
    with pytest.raises(ValidationError, match="student unit"):
        _ = TrainConfig(
            model=ModelParams(K_unit=2),
            self_flow=self_flow_config.self_flow,
        )
