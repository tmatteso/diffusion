"""Tests for the training loop.

Covers take_step in eval and train mode (including gradient scaling and
gradient reach), process_accum_window's protein-weighted scaling,
component_grad_norms, optimizer_step with its EMA update, evaluate,
swapped_in_ema_weights with and without DDP and on error, checkpoint
save/load, log_epoch's W&B gating, the AlphaFold learning-rate schedule
through a real LambdaLR, token-budget validation, and train() end to end.

The model, distograms, configuration and loaders are all built by
module-scoped factory fixtures from one ``TrainModelConfig``, so the trunk
and the data agree by construction. Each ``train()`` scenario runs once per
module and the tests read a frozen record of it.
"""

import copy
import dataclasses
import enum
import hashlib
import json
import multiprocessing
import pathlib
from datetime import timedelta
from multiprocessing.queues import SimpleQueue
from typing import cast
from unittest import mock

import numpy as np
import pytest
import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
from architecture.main_trunk import EMA, MainTrunk, PredictedOutputs
from einops import rearrange, reduce
from helpers.atom_utils import (
    RESTYPE_NUM,
    RESTYPE_NUM_NO_X,
    Protein,
    ResidueRepresentation,
)
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    build_distogram_module,
    featurize_batch,
    make_bucketed_data_loaders,
)
from helpers.useful_objects import (
    ComponentNorms,
    EpochMetrics,
    LossMetrics,
    ModelSetup,
    TensorAccumulatorMixin,
    ThroughputStatistics,
    manual_seed,
)
from jaxtyping import Float
from pydantic import ValidationError
from structlog.typing import FilteringBoundLogger
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from train.train_config import (
    AtomDistogramParams,
    CheckpointParams,
    LoaderConfig,
    LoggingParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
    TrainArgs,
    TrainConfig,
    TrainingParams,
    TrainLoaderConfig,
)
from train.train_loop import (
    alphafold_lr_lambda,
    build_lr_scheduler,
    component_grad_norms,
    evaluate,
    load_checkpoint,
    log_epoch,
    optimizer_step,
    process_accum_window,
    save_checkpoint,
    self_conditioned_template,
    swapped_in_ema_weights,
    take_step,
    train,
)

type StateDict = dict[str, Float[torch.Tensor, "..."]]
type DataLoader = torch.utils.data.DataLoader[FeaturizedBatch]


@dataclasses.dataclass(frozen=True)
class TrainModelConfig:
    """The model and distogram settings every fixture builds from.

    Attributes:
        model: Small trunk sizes, fast on CPU.
        template: Self-conditioning template distogram.
        residue: Residue-level distogram head.
        atom: Atom-level distogram head.
        noise: Noise schedule.
    """

    model: ModelParams
    template: TemplateDistogramParams
    residue: ResidueDistogramParams
    atom: AtomDistogramParams
    noise: NoiseScheduleParams


@dataclasses.dataclass(frozen=True)
class Distograms:
    """The three distogram modules a ModelSetup carries.

    Attributes:
        template: Self-conditioning template distogram.
        residue: Residue-level distogram.
        atom: Atom-level distogram.
    """

    template: Distogram
    residue: Distogram
    atom: Distogram


@dataclasses.dataclass(frozen=True)
class TrainCase:
    """Training settings for one configuration.

    Attributes:
        num_epochs: Number of training epochs.
        token_budget: Tokens accumulated before an optimizer step.
        grad_clip: Gradient clipping threshold, or None to disable it.
        use_wandb: Whether W&B logging is enabled.
    """

    num_epochs: int
    token_budget: int
    grad_clip: float | None
    use_wandb: bool


class TrainScenario(enum.Enum):
    """Configurations ``train()`` runs under, each run once per module.

    A real epoch takes seconds, so ``recorded_train_runs`` runs each
    member once and every test reads the record. Tests share this enum
    rather than subclassing it, because members of separate subclasses are
    separate parameters and would each trigger their own run.

    Every budget, including ``large_token_budget``'s, is below the 168
    tokens of the two synthetic train batches (84 each). So each epoch
    flushes the first batch when the second arrives, and flushes the
    second at the end of the epoch: two optimizer steps per epoch.

    Members:
        default_budget: One epoch with clipping on and W&B off.
        no_grad_clip: As ``default_budget`` with clipping disabled.
        large_token_budget: A budget above one batch's tokens, W&B on.
        multi_epoch: Two epochs with W&B on, the fewest that can show the
            logged epoch advancing.
    """

    default_budget = TrainCase(1, 16, 1.0, use_wandb=False)
    no_grad_clip = TrainCase(1, 16, None, use_wandb=False)
    large_token_budget = TrainCase(1, 160, 1.0, use_wandb=True)
    multi_epoch = TrainCase(2, 16, 1.0, use_wandb=True)


@dataclasses.dataclass(frozen=True)
class TrainRun:
    """What one ``train()`` run did, recorded once per scenario.

    Attributes:
        parameters_changed: True if any model parameter changed.
        reduced_metrics: Every instance ``all_reduce_mean()`` was called on.
        wandb_payloads: One dict per ``wandb.log`` call.
        checkpoint: The saved checkpoint, loaded back from disk.
        num_epochs: Number of epochs the run was configured for.
    """

    parameters_changed: bool
    reduced_metrics: list[TensorAccumulatorMixin]
    wandb_payloads: list[dict[str, object]]
    checkpoint: dict[str, dict[str, object]]
    num_epochs: int


@dataclasses.dataclass(frozen=True)
class Loaders:
    """Train and test loaders over the synthetic proteins.

    Attributes:
        train: Bucketed training loader.
        test: Evaluation loader.
    """

    train: DataLoader
    test: DataLoader


@dataclasses.dataclass(frozen=True)
class SwapSetup:
    """A setup ready for ``swapped_in_ema_weights``, with reference weights.

    Attributes:
        setup: Setup whose model may be DDP-wrapped.
        inner: The underlying MainTrunk, read for its live weights.
        raw: The trunk's weights before the swap.
        shadow: The EMA shadow before the swap.
    """

    setup: ModelSetup
    inner: MainTrunk
    raw: StateDict
    shadow: StateDict


@dataclasses.dataclass(frozen=True)
class GateCase:
    """``log_epoch`` flags and how many ``wandb.log`` calls they allow.

    Attributes:
        do_log: Value of ``log_epoch``'s ``do_log``.
        use_wandb: Value of ``logging.use_wandb``.
        calls: Expected number of ``wandb.log`` calls.
    """

    do_log: bool
    use_wandb: bool
    calls: int


class WandbGate(enum.Enum):
    """Flag combinations for ``log_epoch``.

    Members:
        do_log_false: Logging suppressed for this epoch, W&B on.
        wandb_disabled: Logging requested, W&B off.
        wandb_enabled: Logging requested, W&B on: one call.
    """

    do_log_false = GateCase(do_log=False, use_wandb=True, calls=0)
    wandb_disabled = GateCase(do_log=True, use_wandb=False, calls=0)
    wandb_enabled = GateCase(do_log=True, use_wandb=True, calls=1)


@dataclasses.dataclass(frozen=True)
class TrainHarness:
    """Everything a full ``train()`` run is assembled from.

    The methods build fresh objects from the frozen settings each time they
    are called, so every run starts from the same seeded trunk.

    Attributes:
        model_config: Model and distogram settings.
        distograms: Distogram modules every setup carries.
        dataset_files: Dataset and splits paths; shard and log paths are
            replaced per run.
    """

    model_config: TrainModelConfig
    distograms: Distograms
    dataset_files: TrainArgs

    def trunk(self) -> MainTrunk:
        """Build the small trunk under a fixed seed.

        Returns:
            A freshly initialised MainTrunk.
        """
        _ = manual_seed(0)
        config = self.model_config
        return MainTrunk(
            model_params=config.model,
            template_distogram_params=config.template,
            residue_distogram_params=config.residue,
            atom_distogram_params=config.atom,
            noise_params=config.noise,
        )

    def tcfg(
        self,
        work_dir: pathlib.Path,
        case: TrainCase,
        *,
        loader_budget: int = 63,
    ) -> TrainConfig:
        """Build a TrainConfig for ``case`` matching the shared model.

        Args:
            work_dir: Directory for the checkpoint.
            case: Training settings.
            loader_budget: Token budget of each loader batch.

        Returns:
            The configuration.
        """
        config = self.model_config
        return TrainConfig(
            training=TrainingParams(
                num_epochs=case.num_epochs,
                lr=1e-4,
                grad_clip=case.grad_clip,
                accumulated_token_budget=case.token_budget,
            ),
            model=config.model,
            distogram_template=config.template,
            distogram_residue=config.residue,
            distogram_atom=config.atom,
            noise=config.noise,
            checkpoint=CheckpointParams(
                checkpoint_path=work_dir / "best.pt",
                save_every=100,
            ),
            logging=LoggingParams(use_wandb=case.use_wandb),
            test_loader=LoaderConfig(batch_size=2),
            train_loader=TrainLoaderConfig(
                token_budget=loader_budget,
                noise_magnitude=0,
                n_shards=1,
                num_workers=1,
                epoch_prefetch_depth=1,
                batch_prefetch_depth=1,
                n_threads=1,
            ),
        )

    def setup(self, trunk: MainTrunk | DDP, tcfg: TrainConfig) -> ModelSetup:
        """Bundle ``trunk`` with Adam, the training LR schedule and an EMA.

        Args:
            trunk: Model, plain or DDP-wrapped.
            tcfg: Training configuration.

        Returns:
            The setup on CPU.
        """
        optimizer = Adam(trunk.parameters(), lr=tcfg.training.lr)
        return ModelSetup(
            model=trunk,
            tcfg=tcfg,
            distogram_template=self.distograms.template,
            distogram_residue=self.distograms.residue,
            distogram_atom=self.distograms.atom,
            device=torch.device("cpu"),
            optimizer=optimizer,
            scheduler=build_lr_scheduler(optimizer, tcfg.training),
            ema=EMA(trunk, decay=tcfg.training.ema_decay),
        )

    def loaders(self, work_dir: pathlib.Path, tcfg: TrainConfig) -> Loaders:
        """Build loaders over the synthetic proteins, sharding in ``work_dir``.

        Args:
            work_dir: Directory for shards, config and logs.
            tcfg: Configuration whose loader settings apply.

        Returns:
            The loaders.
        """
        args = dataclasses.replace(
            self.dataset_files,
            shard_dir=work_dir / "shards",
            config=work_dir / "config.json",
            structlog_jsonl=work_dir / "structlog.jsonl",
        )
        train_loader, test_loader, _ = make_bucketed_data_loaders(
            cfg=tcfg,
            extra_train_args=args,
        )
        return Loaders(train=train_loader, test=test_loader)


@dataclasses.dataclass
class TrainRunCache:
    """Runs each ``TrainScenario`` at most once per module and keeps the record.

    The cache is explicit because a parametrized module-scoped fixture does
    not guarantee one run per scenario: pytest groups parameters by their
    position in each ``parametrize`` list. Each run spies on
    ``all_reduce_mean`` and captures ``wandb.log`` inside the run itself,
    since function-scoped ``monkeypatch`` cannot serve a module-scoped
    cache.

    Attributes:
        tmp_path_factory: Factory for each run's directory.
        harness: Builds the trunk, configuration, setup and loaders.
        runs: Records made so far, by scenario.
    """

    tmp_path_factory: pytest.TempPathFactory
    harness: TrainHarness
    runs: dict[TrainScenario, TrainRun] = dataclasses.field(
        default_factory=dict,
    )

    def get(self, scenario: TrainScenario) -> TrainRun:
        """Return the scenario's record, running it on first use.

        Args:
            scenario: Configuration to look up.

        Returns:
            The record.
        """
        if scenario not in self.runs:
            self.runs[scenario] = self.record(scenario)
        return self.runs[scenario]

    def record(self, scenario: TrainScenario) -> TrainRun:
        """Run ``train()`` for the scenario and record what it did.

        ``all_reduce_mean`` is wrapped by an autospec mock whose side effect
        is the real method, so every reduction still runs and the mock
        records which instance each call reduced.

        Args:
            scenario: Configuration to run.

        Returns:
            The record.
        """
        work = self.tmp_path_factory.mktemp(scenario.name)
        tcfg = self.harness.tcfg(work, scenario.value)
        trunk = self.harness.trunk()
        built = self.harness.loaders(work, tcfg)
        before = [p.clone().detach() for p in trunk.parameters()]
        payloads: list[dict[str, object]] = []
        with (
            pytest.MonkeyPatch.context() as patch,
            mock.patch.object(
                TensorAccumulatorMixin,
                "all_reduce_mean",
                autospec=True,
                side_effect=TensorAccumulatorMixin.all_reduce_mean,
            ) as reduce_spy,
        ):
            patch.setattr("train.train_loop.wandb.log", payloads.append)
            train(
                best_val_loss=torch.tensor(float("inf")),
                train_loader=built.train,
                test_loader=built.test,
                model_params=self.harness.setup(trunk, tcfg),
                log=cast(FilteringBoundLogger, structlog.get_logger()),
            )
        return TrainRun(
            parameters_changed=any(
                not torch.equal(b, a)
                for b, a in zip(before, trunk.parameters(), strict=True)
            ),
            reduced_metrics=[
                cast(TensorAccumulatorMixin, call.args[0])
                for call in reduce_spy.call_args_list
            ],
            wandb_payloads=payloads,
            checkpoint=cast(
                dict[str, dict[str, object]],
                torch.load(work / "best.pt", weights_only=True),
            ),
            num_epochs=scenario.value.num_epochs,
        )


@dataclasses.dataclass(frozen=True)
class CheckpointWriteCase:
    """A rank passed to ``save_checkpoint`` and whether it writes a file.

    Attributes:
        rank: Rank passed to ``save_checkpoint``.
        writes: Whether a checkpoint file must appear.
    """

    rank: int
    writes: bool


class CheckpointWriteScenario(enum.Enum):
    """Ranks that do and do not write the checkpoint.

    Members:
        rank0: Rank 0 writes every state.
        rank1: Any other rank writes nothing.
    """

    rank0 = CheckpointWriteCase(rank=0, writes=True)
    rank1 = CheckpointWriteCase(rank=1, writes=False)


@dataclasses.dataclass(frozen=True)
class SwapWrapCase:
    """How the model is wrapped before its EMA weights are swapped in.

    Attributes:
        ddp: Wrap the model in one-rank DDP, so the swap must reach
            ``.module``.
    """

    ddp: bool


class SwapWrap(enum.Enum):
    """Model wrappings the EMA swap must see through.

    Members:
        plain: The bare ``MainTrunk``.
        ddp: The trunk inside one-rank DDP.
    """

    plain = SwapWrapCase(ddp=False)
    ddp = SwapWrapCase(ddp=True)


@pytest.fixture(scope="module")
def model_config() -> TrainModelConfig:
    """Return the small model and distogram settings shared by every fixture.

    Module-scoped because the module-scoped train-run cache builds models
    from it; the settings are frozen pydantic models.

    Returns:
        The settings.
    """
    return TrainModelConfig(
        model=ModelParams(
            f_ref_dim=35,
            c_atom=32,
            c_pair=32,
            c_res=32,
            c_atompair=16,
            K_unit=1,
            n_blocks_atom_transformer_encoder=3,
            n_heads_atom_transformer_encoder=4,
            n_blocks_atom_transformer_decoder=3,
            n_heads_atom_transformer_decoder=4,
            n_pairformer_blocks_template_embedder=2,
            n_paiformer_heads_template_embedder=4,
            n_amino=RESTYPE_NUM_NO_X,
            max_residues=50,
        ),
        template=TemplateDistogramParams(
            n_bins=8,
            min_dist=3.25,
            max_dist=50.75,
        ),
        residue=ResidueDistogramParams(n_bins=8, min_dist=2.0, max_dist=22.0),
        atom=AtomDistogramParams(n_bins=5, min_dist=0.0, max_dist=10.0),
        noise=NoiseScheduleParams(sigma_data=16.0),
    )


@pytest.fixture(scope="module")
def distograms(model_config: TrainModelConfig) -> Distograms:
    """Build the three distogram modules from the shared settings.

    Module-scoped: the modules hold only fixed bin edges.

    Args:
        model_config: Model and distogram settings.

    Returns:
        The modules in eval mode.
    """
    return Distograms(
        template=build_distogram_module(model_config.template).eval(),
        residue=build_distogram_module(model_config.residue).eval(),
        atom=build_distogram_module(model_config.atom).eval(),
    )


@pytest.fixture(scope="module")
def dataset_files(tmp_path_factory: pytest.TempPathFactory) -> TrainArgs:
    """Write six 42-residue all-alanine proteins and their splits.

    Four go to train, one each to validation and test. Coordinates are all
    zero: the loop only needs well-formed inputs.

    Args:
        tmp_path_factory: Factory for the module's data directory.

    Returns:
        TrainArgs pointing at the files; shard and log paths are replaced
        per run.
    """
    root = tmp_path_factory.mktemp("dataset")
    names = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A", "6ff.A"]
    coords = {atom: [[0.0] * 3] * 42 for atom in ("N", "CA", "C", "O")}
    jsonl = root / "proteins.jsonl"
    _ = jsonl.write_text(
        "".join(
            json.dumps({"name": n, "seq": "A" * 42, "coords": coords}) + "\n"
            for n in names
        ),
    )
    splits = root / "splits.json"
    _ = splits.write_text(
        json.dumps(
            {
                "train": ["1aa.A", "2bb.A", "3cc.A", "6ff.A"],
                "validation": ["4dd.A"],
                "test": ["5ee.A"],
            },
        ),
    )
    return TrainArgs(
        dataset_jsonl=jsonl,
        shard_dir=root / "shards",
        keys_for_splits_json=splits,
        config=root / "config.json",
        structlog_jsonl=root / "structlog.jsonl",
        ddp=False,
        debug_run=False,
    )


@pytest.fixture(scope="module")
def harness(
    model_config: TrainModelConfig,
    distograms: Distograms,
    dataset_files: TrainArgs,
) -> TrainHarness:
    """Bundle the settings and data a full ``train()`` run is built from.

    Module-scoped: every field is frozen settings, fixed modules or paths.

    Args:
        model_config: Model and distogram settings.
        distograms: Distogram modules.
        dataset_files: Dataset and splits paths.

    Returns:
        The harness.
    """
    return TrainHarness(
        model_config=model_config,
        distograms=distograms,
        dataset_files=dataset_files,
    )


@pytest.fixture
def model_params(tmp_path: pathlib.Path, harness: TrainHarness) -> ModelSetup:
    """Build a fresh setup with the default one-epoch configuration.

    Args:
        tmp_path: Per-test directory for the checkpoint.
        harness: Builds the trunk, configuration and setup.

    Returns:
        The setup, with W&B off.
    """
    tcfg = harness.tcfg(tmp_path, TrainScenario.default_budget.value)
    return harness.setup(harness.trunk(), tcfg)


@pytest.fixture
def featurized_batch(model_params: ModelSetup) -> FeaturizedBatch:
    """Featurize ten copies of a deterministic 16-residue protein.

    Residue ``r`` is residue type ``r % RESTYPE_NUM`` and atom ``a`` of it
    sits at ``(3.8 r + 0.1 a, sin(r + a), 0)``.

    Args:
        model_params: Supplies the configuration and distograms.

    Returns:
        The batch.
    """
    n = 16
    r = np.arange(n, dtype=np.float64)[:, None]
    a = np.arange(37, dtype=np.float64)[None, :]
    protein = Protein(
        atom_positions=np.stack(
            [3.8 * r + 0.1 * a, np.sin(r + a), np.zeros((n, 37))],
            axis=-1,
        ),
        atom_mask=np.ones((n, 37)),
        residue_index=np.arange(n, dtype=np.intp),
        aatype=np.arange(n, dtype=np.intp) % RESTYPE_NUM,
        chain_index=np.zeros(n, dtype=np.intp),
        b_factors=np.zeros((n, 37)),
    )
    _ = manual_seed(1)
    return featurize_batch(
        [protein] * 10,
        model_params.tcfg,
        model_params.distogram_template,
        model_params.distogram_residue,
        model_params.distogram_atom,
    )


@pytest.fixture
def loaders(
    tmp_path: pathlib.Path,
    model_params: ModelSetup,
    harness: TrainHarness,
) -> Loaders:
    """Build loaders over the synthetic proteins in a per-test directory.

    Args:
        tmp_path: Per-test directory.
        model_params: Supplies the configuration.
        harness: Builds the loaders.

    Returns:
        The loaders.
    """
    return harness.loaders(tmp_path, model_params.tcfg)


@pytest.fixture
def log() -> FilteringBoundLogger:
    """Return a structlog logger for the functions that take one.

    Returns:
        The logger.
    """
    return cast(FilteringBoundLogger, structlog.get_logger())


@pytest.fixture
def wandb_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch ``wandb.log`` and return the list of captured payloads.

    Args:
        monkeypatch: Patches the attribute for this test only.

    Returns:
        List that receives one dict per ``wandb.log`` call.
    """
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", captured.append)
    return captured


@pytest.fixture
def epoch_metrics() -> EpochMetrics:
    """Return EpochMetrics for epoch 7 with every metric at zero.

    Returns:
        The metrics.
    """
    device = torch.device("cpu")
    loss = LossMetrics.zero_init(device)
    return EpochMetrics(
        epoch=7,
        global_step=1,
        train_loss_metrics=loss,
        train_throughput_stats=ThroughputStatistics.zero_init(device),
        train_gradient_norms=ComponentNorms.zero_init(device),
        val_loss_metrics=loss,
    )


@pytest.fixture
def diverged_ema(model_params: ModelSetup) -> ModelSetup:
    """Shift every weight by 1 and update the EMA, so model and shadow differ.

    Args:
        model_params: Setup to modify.

    Returns:
        The same setup, with raw weights and EMA shadow now distinct.
    """
    with torch.no_grad():
        for param in model_params.model.parameters():
            _ = param.add_(1.0)
    model_params.ema.update(model_params.model, step=0)
    return model_params


@pytest.fixture
def swap_setup(
    request: pytest.FixtureRequest,
    swap_wrap: SwapWrap,
    diverged_ema: ModelSetup,
) -> SwapSetup:
    """Prepare a swap, wrapping the model in one-rank DDP when asked.

    With DDP the swap must reach ``.module``, whose keys carry no
    ``module.`` prefix. Reference weights are cloned before any swap.

    Args:
        request: Pytest request used to start the one-rank process group.
        swap_wrap: Whether to wrap in DDP; the requesting test
            parametrizes it.
        diverged_ema: Setup whose raw weights and shadow differ.

    Returns:
        The setup and its reference weights.
    """
    inner = cast(MainTrunk, diverged_ema.model)
    setup = diverged_ema
    if swap_wrap.value.ddp:
        request.getfixturevalue("single_process_gloo_group")
        setup = dataclasses.replace(diverged_ema, model=DDP(inner))
    raw = {k: v.clone() for k, v in cast(StateDict, inner.state_dict()).items()}
    shadow = {k: v.clone() for k, v in diverged_ema.ema.state_dict().items()}
    return SwapSetup(setup=setup, inner=inner, raw=raw, shadow=shadow)


@pytest.fixture
def lr_schedule_optimizer() -> Adam:
    """Return Adam over one dummy parameter at a base learning rate of 2.

    Returns:
        The optimizer.
    """
    return Adam([nn.Parameter(torch.zeros(1))], lr=2.0)


@pytest.fixture
def lr_schedule_training() -> TrainingParams:
    """Return 4 warmup steps, then a halving every 3 steps.

    Returns:
        Training parameters supplying the schedule's shape.
    """
    return TrainingParams(
        lr_warmup_steps=4,
        lr_decay_steps=3,
        lr_decay_factor=0.5,
    )


@pytest.fixture
def lr_scheduler(
    lr_schedule_optimizer: Adam,
    lr_schedule_training: TrainingParams,
) -> LambdaLR:
    """Build the training LR schedule over ``lr_schedule_optimizer``.

    Args:
        lr_schedule_optimizer: Adam at base rate 2.
        lr_schedule_training: Warmup and decay settings.

    Returns:
        The scheduler at step 0.
    """
    return build_lr_scheduler(lr_schedule_optimizer, lr_schedule_training)


@pytest.fixture(scope="module")
def recorded_train_runs(
    tmp_path_factory: pytest.TempPathFactory,
    harness: TrainHarness,
) -> TrainRunCache:
    """Return the module's cache of ``train()`` runs, one per scenario.

    Args:
        tmp_path_factory: Factory for each run's directory.
        harness: Builds each run.

    Returns:
        An empty cache that runs each scenario on first use.
    """
    return TrainRunCache(tmp_path_factory=tmp_path_factory, harness=harness)


@pytest.fixture
def train_run(
    train_scenario: TrainScenario,
    recorded_train_runs: TrainRunCache,
) -> TrainRun:
    """Return the shared record for ``train_scenario``.

    The record is a frozen snapshot rather than live training state, so the
    tests sharing it stay order-independent.

    Args:
        train_scenario: Run to look up; the requesting test parametrizes
            it.
        recorded_train_runs: Module-wide cache of runs.

    Returns:
        The scenario's record.
    """
    return recorded_train_runs.get(train_scenario)


def test_take_step_eval_reports_finite_metrics_without_gradients(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """An eval step returns finite losses and sane throughput, no gradients.

    The token pack rate lies in ``(0, 1]`` and both rates are positive. Not
    parametrized: train mode is tested below.

    Args:
        featurized_batch: Ten-protein batch.
        model_params: Setup holding the model.
    """
    loss, tput = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=False,
    )
    losses = torch.tensor(list(loss.to_float_dict().values()))
    assert bool(torch.isfinite(losses).all())
    assert 0 < tput.token_pack_rate.item() <= 1
    assert tput.residues_per_sec.item() > 0
    assert tput.atoms_per_sec.item() > 0
    assert all(p.grad is None for p in model_params.model.parameters())


def test_take_step_train_reaches_every_submodule(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """A train step gives every top-level submodule a finite nonzero gradient.

    Submodules whose parameters get no gradient at all count as missing,
    so a head cut out of the graph fails. Not parametrized: one step on a
    full batch reaches every head.

    Args:
        featurized_batch: Ten-protein batch.
        model_params: Setup holding the model.
    """
    _ = model_params.model.train()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
    )
    usable: dict[str, bool] = {}
    for name, param in model_params.model.named_parameters():
        grad = param.grad
        good = (
            grad is not None
            and bool(torch.isfinite(grad).all())
            and bool(grad.abs().max() > 0)
        )
        prefix = name.split(".")[0]
        usable[prefix] = usable.get(prefix, False) or good
    assert usable
    assert usable == dict.fromkeys(usable, True)


@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_take_step_grad_scale_divides_the_gradient(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
    scale: float,
) -> None:
    """``grad_scale = s`` gives ``1/s`` of the unscaled gradient norm.

    The same seed is used for both steps, so only the scale differs.

    Args:
        featurized_batch: Ten-protein batch.
        model_params: Setup holding the model.
        scale: Gradient scale to compare with 1.
    """
    _ = model_params.model.train()
    norms: list[float] = []
    for grad_scale in (1.0, scale):
        _ = manual_seed(0)
        model_params.model.zero_grad()
        _ = take_step(
            batch=featurized_batch,
            model_params=model_params,
            train_mode=True,
            grad_scale=grad_scale,
        )
        squares = [
            reduce(p.grad**2, "... -> ", "sum")
            for p in model_params.model.parameters()
            if p.grad is not None
        ]
        norms.append(float(torch.stack(squares).sum().sqrt()))
    torch.testing.assert_close(norms[1], norms[0] / scale, rtol=1e-3, atol=0)


def test_process_accum_window_scales_each_batch_by_protein_share(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """Batch ``i`` of the window gets ``grad_scale = total / proteins_i``.

    With 1 and 3 proteins the total is 4, so the scales are 4 and 4/3, and
    the window's losses are finite. Not parametrized: two unequal batches
    show the weighting.

    Args:
        featurized_batch: Batch used for both window slots.
        model_params: Setup holding the model.
    """
    _ = model_params.model.train()
    with mock.patch(
        "train.train_loop.take_step",
        autospec=True,
        side_effect=take_step,
    ) as take_step_spy:
        loss, _ = process_accum_window(
            [featurized_batch, featurized_batch],
            [1, 3],
            model_params,
        )
    losses = torch.tensor(list(loss.to_float_dict().values()))
    scales = [
        cast(float, call.kwargs["grad_scale"])
        for call in take_step_spy.call_args_list
    ]
    torch.testing.assert_close(
        torch.tensor(scales),
        torch.tensor([4.0, 4.0 / 3]),
    )
    assert bool(torch.isfinite(losses).all())


def test_component_grad_norms_are_positive_after_a_step(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """After a train step every component's gradient norm is positive.

    Not parametrized: one backward pass fills every component.

    Args:
        featurized_batch: Ten-protein batch.
        model_params: Setup holding the model.
    """
    _ = model_params.model.train()
    _ = take_step(
        batch=featurized_batch,
        model_params=model_params,
        train_mode=True,
    )
    norms = component_grad_norms(model_params.model).to_float_dict()
    assert {k: v > 0 for k, v in norms.items()} == dict.fromkeys(norms, True)


def test_optimizer_step_moves_weights_and_blends_the_ema(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> None:
    """One optimizer step changes the weights, bumps the step, updates EMA.

    The shadow afterwards is ``d * before + (1 - d) * after`` with the
    bias-corrected ``d = min(decay, (step + 1) / (step + 10))`` at the
    pre-step count. Not parametrized: the EMA formula has its own
    parametrized test in ``test_main_trunk``.

    Args:
        featurized_batch: Ten-protein batch.
        model_params: Setup holding the model and EMA.
    """
    name, _ = next(model_params.model.named_parameters())
    shadow_before = model_params.ema.shadow[name].clone()
    before = [p.clone().detach() for p in model_params.model.parameters()]
    _ = model_params.model.train()
    step = 5
    _, _, norms, new_step = optimizer_step(
        [featurized_batch],
        [1],
        model_params,
        global_step=step,
    )
    after = cast(StateDict, model_params.model.state_dict())[name]
    ramp = (step + 1) / (step + 10)
    decay = min(model_params.tcfg.training.ema_decay, ramp)
    expected = decay * shadow_before + (1 - decay) * after
    assert new_step == step + 1
    assert isinstance(norms, ComponentNorms)
    assert any(
        not torch.equal(b, a)
        for b, a in zip(before, model_params.model.parameters(), strict=True)
    )
    torch.testing.assert_close(model_params.ema.shadow[name], expected)


def test_evaluate_reports_finite_metrics_without_gradients(
    loaders: Loaders,
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Evaluation over the test loader gives finite, non-negative losses.

    It must leave no gradients. Not parametrized: one loader covers it.

    Args:
        loaders: Train and test loaders.
        model_params: Setup holding the model.
        log: Logger.
    """
    loss, _ = evaluate(loaders.test, model_params, log)
    losses = torch.tensor(list(loss.to_float_dict().values()))
    assert bool(torch.isfinite(losses).all())
    assert loss.total_loss.item() >= 0
    assert loss.RMSD.item() >= 0
    assert all(p.grad is None for p in model_params.model.parameters())


@pytest.mark.parametrize(
    "swap_wrap",
    list(SwapWrap),
    ids=[case.name for case in SwapWrap],
)
def test_swapped_in_ema_weights_swaps_in_and_back(
    swap_setup: SwapSetup,
) -> None:
    """Inside the block the model holds the shadow; after it, the raw weights.

    Args:
        swap_setup: Setup, plain or DDP-wrapped, with reference weights.
    """
    x = swap_setup
    with swapped_in_ema_weights(x.setup):
        inside = cast(StateDict, x.inner.state_dict())
        matched = all(torch.equal(inside[k], v) for k, v in x.shadow.items())
    after = cast(StateDict, x.inner.state_dict())
    assert matched
    assert all(torch.equal(after[k], v) for k, v in x.raw.items())


@pytest.mark.parametrize(
    "swap_wrap",
    list(SwapWrap),
    ids=[case.name for case in SwapWrap],
)
def test_swapped_in_ema_weights_restores_raw_weights_on_error(
    swap_setup: SwapSetup,
) -> None:
    """An exception in the block propagates and the raw weights come back.

    Otherwise a failed evaluation would leave training on EMA weights.

    Args:
        swap_setup: Setup, plain or DDP-wrapped, with reference weights.

    Raises:
        RuntimeError: Raised inside the block to exercise the error path;
            caught by ``pytest.raises``.
    """
    x = swap_setup
    message = "simulated eval failure"
    with (
        pytest.raises(RuntimeError, match=message),
        swapped_in_ema_weights(x.setup),
    ):
        raise RuntimeError(message)
    after = cast(StateDict, x.inner.state_dict())
    assert all(torch.equal(after[k], v) for k, v in x.raw.items())


@pytest.mark.parametrize(
    "checkpoint_write_scenario",
    list(CheckpointWriteScenario),
    ids=[case.name for case in CheckpointWriteScenario],
)
def test_save_checkpoint_writes_only_on_rank_zero(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    checkpoint_write_scenario: CheckpointWriteScenario,
) -> None:
    """Rank 0 writes a checkpoint with every state; other ranks write none.

    Args:
        model_params: Setup to save.
        log: Logger.
        checkpoint_write_scenario: Rank and whether it must write a file.
    """
    case = checkpoint_write_scenario.value
    writes = case.writes
    path = model_params.tcfg.checkpoint.checkpoint_path
    save_checkpoint(
        model_params=model_params,
        rank=case.rank,
        log=log,
        best_val_loss=torch.tensor(0.5),
    )
    saved = (
        cast(dict[str, object], torch.load(path, weights_only=True))
        if path.exists()
        else {}
    )
    expected = {"model", "optimizer", "scheduler", "ema", "best_val_loss"}
    assert path.exists() == writes
    assert set(saved) == (expected if writes else set())


def test_load_checkpoint_restores_every_saved_state(
    diverged_ema: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """Loading undoes corruption of the weights, EMA, Adam and scheduler.

    After one optimizer and scheduler step, the setup is saved, then its
    weights and shadow are zeroed, Adam's first moments zeroed and the
    scheduler rewound. Loading must restore each saved value and the best
    validation loss. Not parametrized: every state goes in one file.

    Args:
        diverged_ema: Setup whose EMA differs from its weights.
        log: Logger.
    """
    setup = diverged_ema
    for param in setup.model.parameters():
        param.grad = torch.ones_like(param)
    _ = setup.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    setup.scheduler.step()
    save_checkpoint(
        model_params=setup,
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.271),
    )
    saved = cast(
        dict[str, dict[str, object]],
        torch.load(setup.tcfg.checkpoint.checkpoint_path, weights_only=True),
    )
    with torch.no_grad():
        for param in setup.model.parameters():
            _ = param.zero_()
        for tensor in setup.ema.shadow.values():
            _ = tensor.zero_()
        live_state = cast(dict[nn.Parameter, StateDict], setup.optimizer.state)
        for state in live_state.values():
            _ = state["exp_avg"].zero_()
    setup.scheduler.last_epoch = -1
    restored, best = load_checkpoint(model_params=setup, rank=0, log=log)
    model_sd = cast(StateDict, saved["model"])
    ema_sd = cast(StateDict, saved["ema"])
    opt_sd = cast(dict[int, StateDict], saved["optimizer"]["state"])
    restored_opt = cast(
        dict[int, StateDict],
        cast(dict[str, object], restored.optimizer.state_dict())["state"],
    )
    live = cast(StateDict, restored.model.state_dict())
    assert all(torch.equal(live[k], v) for k, v in model_sd.items())
    assert all(
        torch.equal(restored.ema.shadow[k], v) for k, v in ema_sd.items()
    )
    assert all(
        torch.equal(restored_opt[i]["exp_avg"], s["exp_avg"])
        for i, s in opt_sd.items()
    )
    assert restored.scheduler.last_epoch == saved["scheduler"]["last_epoch"]
    torch.testing.assert_close(best, torch.tensor(0.271))


@pytest.mark.usefixtures("single_process_gloo_group")
def test_save_checkpoint_strips_ddp_wrapper_prefix(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
) -> None:
    """A checkpoint saved from a DDP model has the plain model's keys.

    ``save_checkpoint`` must unwrap ``.module`` before ``state_dict()``, so
    no key carries a ``module.`` prefix. Not parametrized: the plain case
    is the rank-0 test above.

    Args:
        model_params: Setup whose model is wrapped here.
        log: Logger.
    """
    plain_keys = set(model_params.model.state_dict())
    save_checkpoint(
        model_params=dataclasses.replace(
            model_params,
            model=DDP(model_params.model),
        ),
        rank=0,
        log=log,
        best_val_loss=torch.tensor(0.3),
    )
    saved = cast(
        dict[str, StateDict],
        torch.load(
            model_params.tcfg.checkpoint.checkpoint_path,
            weights_only=True,
        ),
    )
    assert set(saved["model"]) == plain_keys


@pytest.mark.parametrize(
    "gate",
    list(WandbGate),
    ids=[case.name for case in WandbGate],
)
def test_log_epoch_calls_wandb_only_when_asked_and_enabled(
    model_params: ModelSetup,
    epoch_metrics: EpochMetrics,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
    gate: WandbGate,
) -> None:
    """``wandb.log`` runs once when ``do_log`` and ``use_wandb`` are both set.

    Args:
        model_params: Setup whose logging flag is replaced here.
        epoch_metrics: Metrics to log.
        log: Logger.
        wandb_payloads: Captured ``wandb.log`` payloads.
        gate: Flags and the expected call count.
    """
    case = gate.value
    tcfg = model_params.tcfg.model_copy(
        update={"logging": LoggingParams(use_wandb=case.use_wandb)},
    )
    log_epoch(
        epoch_metrics=epoch_metrics,
        model_params=dataclasses.replace(model_params, tcfg=tcfg),
        log=log,
        do_log=case.do_log,
    )
    assert len(wandb_payloads) == case.calls


def test_log_epoch_payload_names_the_epoch_and_both_splits(
    model_params: ModelSetup,
    epoch_metrics: EpochMetrics,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """The payload carries the epoch and ``train/`` and ``val/`` metrics.

    Not parametrized: the gating is the test above.

    Args:
        model_params: Setup whose W&B flag is switched on here.
        epoch_metrics: Metrics for epoch 7.
        log: Logger.
        wandb_payloads: Captured ``wandb.log`` payloads.
    """
    tcfg = model_params.tcfg.model_copy(
        update={"logging": LoggingParams(use_wandb=True)},
    )
    log_epoch(
        epoch_metrics=epoch_metrics,
        model_params=dataclasses.replace(model_params, tcfg=tcfg),
        log=log,
        do_log=True,
    )
    payload = wandb_payloads[0]
    assert payload["epoch"] == epoch_metrics.epoch
    assert any(k.startswith("train/") for k in payload)
    assert any(k.startswith("val/") for k in payload)


@pytest.mark.parametrize("budget", [0, -5], ids=["zero", "negative"])
def test_accumulated_token_budget_rejects_non_positive_values(
    budget: int,
) -> None:
    """A token budget of zero or less fails validation.

    Args:
        budget: Invalid budget.
    """
    with pytest.raises(ValidationError):
        _ = TrainingParams(accumulated_token_budget=budget)


@pytest.mark.parametrize("budget", [1, 1_000_000], ids=["one", "large"])
def test_accumulated_token_budget_accepts_any_positive_value(
    budget: int,
) -> None:
    """Any positive token budget is accepted as given.

    Args:
        budget: Valid budget.
    """
    params = TrainingParams(accumulated_token_budget=budget)
    assert params.accumulated_token_budget == budget


@pytest.mark.parametrize(
    "n_steps",
    [0, 1, 3, 4, 5, 7, 10],
    ids=[
        "construction_only_first_warmup_value",
        "one_step_into_warmup",
        "last_warmup_step",
        "warmup_just_ended_no_decay",
        "one_step_past_warmup_before_decay_boundary",
        "first_decay_boundary_reached",
        "second_decay_boundary_reached",
    ],
)
def test_build_lr_scheduler_follows_alphafold_lr_lambda(
    lr_schedule_optimizer: Adam,
    lr_schedule_training: TrainingParams,
    lr_scheduler: LambdaLR,
    n_steps: int,
) -> None:
    """The training LR schedule reproduces ``alphafold_lr_lambda`` exactly.

    With 4 warmup steps, decay every 3 steps by 0.5 and base rate 2, the
    optimizer's live rate after ``n_steps`` must equal
    ``2 * alphafold_lr_lambda(n_steps)``, through PyTorch's scheduler.

    Args:
        lr_schedule_optimizer: Adam at base rate 2.
        lr_schedule_training: Warmup and decay settings.
        lr_scheduler: ``build_lr_scheduler`` over that optimizer.
        n_steps: Scheduler steps to take.
    """
    for _ in range(n_steps):
        lr_scheduler.step()
    actual = cast(float, lr_schedule_optimizer.param_groups[0]["lr"])
    expected = alphafold_lr_lambda(
        n_steps,
        warmup_steps=lr_schedule_training.lr_warmup_steps,
        decay_steps=lr_schedule_training.lr_decay_steps,
        decay_factor=lr_schedule_training.lr_decay_factor,
    )
    torch.testing.assert_close(actual, 2.0 * expected)


def test_train_preflushes_before_an_oversized_batch(
    tmp_path: pathlib.Path,
    harness: TrainHarness,
    log: FilteringBoundLogger,
) -> None:
    """Each window is flushed before a batch that would exceed the budget.

    The loader yields 4 one-protein batches; with a budget of 7 tokens, two
    batches always exceed it, so batches 1 to 3 are each flushed alone
    when the next arrives, and the end-of-epoch flush takes batch 4 so no
    batch is dropped: four windows of one. Not parametrized: the
    budget is chosen so every arrival triggers a flush.

    Args:
        tmp_path: Per-test directory.
        harness: Builds the trunk, configuration, setup and loaders.
        log: Logger.
    """
    tcfg = harness.tcfg(
        tmp_path,
        TrainCase(1, 7, 1.0, use_wandb=False),
        loader_budget=42,
    )
    tcfg = tcfg.model_copy(update={"test_loader": LoaderConfig(batch_size=1)})
    built = harness.loaders(tmp_path, tcfg)
    with mock.patch(
        "train.train_loop.process_accum_window",
        autospec=True,
        side_effect=process_accum_window,
    ) as process_spy:
        train(
            best_val_loss=torch.tensor(float("inf")),
            train_loader=built.train,
            test_loader=built.test,
            model_params=harness.setup(harness.trunk(), tcfg),
            log=log,
        )
    windows = [
        len(cast(list[FeaturizedBatch], call.kwargs["micro_buffer"]))
        for call in process_spy.call_args_list
    ]
    assert windows == [1, 1, 1, 1]


@pytest.mark.parametrize(
    "train_scenario",
    list(TrainScenario),
    ids=[case.name for case in TrainScenario],
)
def test_train_updates_model_parameters(train_run: TrainRun) -> None:
    """Training changes the model's parameters under every scenario.

    Args:
        train_run: Recorded run.
    """
    assert train_run.parameters_changed


@pytest.mark.parametrize(
    "train_scenario",
    [TrainScenario.default_budget],
    ids=["default_budget"],
)
def test_train_reduces_every_epoch_metric_across_ranks(
    train_run: TrainRun,
) -> None:
    """``all_reduce_mean`` runs on exactly the four EpochMetrics fields.

    Train and validation LossMetrics, the train ThroughputStatistics and
    the train ComponentNorms. The reduction itself no-ops without a process
    group; ``test_useful_objects`` proves its arithmetic on real ranks.
    One scenario suffices: the calls do not depend on configuration.

    Args:
        train_run: Recorded one-epoch run.
    """
    kinds = [type(m).__name__ for m in train_run.reduced_metrics]
    assert sorted(kinds) == sorted(
        [
            "LossMetrics",
            "LossMetrics",
            "ThroughputStatistics",
            "ComponentNorms",
        ],
    )


@pytest.mark.parametrize(
    "train_scenario",
    [TrainScenario.default_budget, TrainScenario.no_grad_clip],
    ids=["default_budget", "no_grad_clip"],
)
def test_train_does_not_call_wandb_when_disabled(train_run: TrainRun) -> None:
    """With ``use_wandb`` off, ``wandb.log`` is never called.

    Args:
        train_run: Recorded run with W&B off.
    """
    assert train_run.wandb_payloads == []


@pytest.mark.parametrize(
    "train_scenario",
    [TrainScenario.large_token_budget],
    ids=["large_token_budget"],
)
def test_train_logs_nonzero_loss_under_large_token_budget(
    train_run: TrainRun,
) -> None:
    """A budget above one batch but below two still runs an optimizer step.

    The first batch is flushed when the second would overflow the budget,
    so a real optimizer step runs mid-epoch, before the epoch-end flush.

    Args:
        train_run: Recorded run with W&B on.
    """
    assert len(train_run.wandb_payloads) == 1
    train_loss = train_run.wandb_payloads[0]["train/total_loss"]
    assert isinstance(train_loss, float)
    assert train_loss > 0


@pytest.mark.parametrize(
    "train_scenario",
    [TrainScenario.default_budget],
    ids=["default_budget"],
)
def test_train_checkpoint_holds_stepped_optimizer_and_scheduler(
    train_run: TrainRun,
) -> None:
    """The saved Adam moments are nonzero and the scheduler counts steps.

    The LR schedule is per optimizer step (AlphaFold 3-style warmup), so
    the scheduler advances once per optimizer step, not once per epoch.
    Adam's per-parameter ``step`` counter records the same number of
    optimizer steps, so the two must agree in the saved checkpoint, and
    both must be positive after a training epoch.

    Args:
        train_run: Recorded one-epoch run.
    """
    opt_state = cast(
        dict[int, StateDict],
        train_run.checkpoint["optimizer"]["state"],
    )
    first = next(iter(opt_state.values()))
    adam_steps = int(first["step"].item())
    assert bool(first["exp_avg"].abs().max() > 0)
    assert adam_steps > 0
    assert train_run.checkpoint["scheduler"]["last_epoch"] == adam_steps


@pytest.mark.parametrize(
    "train_scenario",
    [TrainScenario.multi_epoch],
    ids=["multi_epoch"],
)
def test_train_logs_once_per_epoch_with_the_epoch_advancing(
    train_run: TrainRun,
) -> None:
    """Two epochs give two W&B payloads, for epochs 1 and 2.

    Args:
        train_run: Recorded two-epoch run with W&B on.
    """
    assert [p["epoch"] for p in train_run.wandb_payloads] == [1, 2]


@dataclasses.dataclass(frozen=True)
class DDPLockstepRun:
    """Inputs for a CPU gloo DDP run in which each rank trains on its own data.

    Built by the ``lockstep_run`` factory fixture and pickled by
    ``torch.multiprocessing.spawn`` into one process per rank, where
    ``train_one_rank`` consumes it. The world size is the number of entries
    in ``batches_by_rank``.

    Attributes:
        setup: Model, optimizer, scheduler, EMA and config shared by every
            rank before training; each rank deep-copies it.
        batches_by_rank: The training batches each rank iterates over,
            indexed by rank. Every rank holds the same number of batches, as
            ``DistributedSampler`` guarantees in real training.
        store_path: File backing the gloo rendezvous store, so the test does
            not have to reserve a free TCP port.
    """

    setup: ModelSetup
    batches_by_rank: tuple[tuple[FeaturizedBatch, ...], ...]
    store_path: pathlib.Path


def straight_chain_protein(n_res: int) -> Protein:
    """Build a deterministic, human-readable protein of ``n_res`` residues.

    Residue ``i`` sits at ``(3.8 * i, 0, 0)`` angstroms, the typical spacing
    of consecutive alpha carbons, and atom slot ``a`` of every residue is
    offset from it by ``(0.1 * a, 0.3 * sin(a), 0.3 * cos(a))`` so the atoms
    are not coplanar. Residue types cycle through the standard amino acids
    in order, every atom is present, and all residues are on one chain. The
    token count the training loop sees for this protein is ``n_res``, which
    is the only property the lockstep test depends on.

    Args:
        n_res: Number of residues.

    Returns:
        The protein, with every array fully determined by ``n_res``.
    """
    residue_index = np.arange(n_res, dtype=np.float64)
    atom_index = np.arange(ResidueRepresentation.atom37, dtype=np.float64)
    residue_origins = np.stack(
        [3.8 * residue_index, np.zeros(n_res), np.zeros(n_res)],
        axis=-1,
    )
    atom_offsets = np.stack(
        [0.1 * atom_index, 0.3 * np.sin(atom_index), 0.3 * np.cos(atom_index)],
        axis=-1,
    )
    return Protein(
        atom_positions=rearrange(residue_origins, "n c -> n 1 c")
        + rearrange(atom_offsets, "a c -> 1 a c"),
        atom_mask=np.ones((n_res, ResidueRepresentation.atom37)),
        residue_index=np.arange(n_res, dtype=np.intp),
        aatype=np.arange(n_res, dtype=np.intp) % RESTYPE_NUM_NO_X,
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.zeros((n_res, ResidueRepresentation.atom37)),
    )


def train_one_rank(
    rank: int,
    run: DDPLockstepRun,
    reports: SimpleQueue[tuple[int, str]],
) -> None:
    """Train for one epoch as one rank of a CPU gloo group and report back.

    This is the per-process entry point handed to
    ``torch.multiprocessing.spawn``. It lives at module level because
    ``spawn`` can only run a picklable callable in the child process, and
    fixtures cannot run there.

    The setup is deep-copied first. ``spawn`` pickles tensors through shared
    memory, so without the copy every rank would train the very same
    parameter storage and the cross-rank weight comparison in the test would
    pass trivially. ``copy.deepcopy`` keeps the optimizer, scheduler and EMA
    bound to the copied model.

    After training, the rank reports its optimizer step count (the
    scheduler steps once per optimizer step) and a SHA-256 digest of its
    parameters, so the test can compare ranks exactly.

    Args:
        rank: This process's rank, supplied by ``spawn``.
        run: The shared run inputs; this rank trains on
            ``run.batches_by_rank[rank]``.
        reports: Queue receiving ``(optimizer_steps, weight_digest)``.
    """
    torch.set_num_threads(2)
    dist.init_process_group(
        backend="gloo",
        init_method=run.store_path.as_uri(),
        rank=rank,
        world_size=len(run.batches_by_rank),
        timeout=timedelta(seconds=60),
    )
    setup = copy.deepcopy(run.setup)
    loader: torch.utils.data.DataLoader[FeaturizedBatch] = (
        torch.utils.data.DataLoader(
            cast(
                torch.utils.data.Dataset[FeaturizedBatch],
                cast(object, run.batches_by_rank[rank]),
            ),
            batch_size=None,
        )
    )
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        test_loader=loader,
        model_params=setup,
        log=cast(FilteringBoundLogger, structlog.get_logger()),
    )
    weights = hashlib.sha256()
    for param in setup.model.parameters():
        weights.update(param.detach().numpy().tobytes())
    reports.put((setup.scheduler.last_epoch, weights.hexdigest()))
    dist.destroy_process_group()


@dataclasses.dataclass(frozen=True)
class RankResidues:
    """Residue count of each one-protein batch on each rank of a 2-rank run.

    Each rank gets its own named field, so a case such as
    ``RankResidues((8, 8), (16, 16))`` reads as "rank 0 holds two 8-residue
    batches and rank 1 holds two 16-residue batches". Each field lists that
    rank's batches in the order its DataLoader yields them, which is the
    order the flush rule accumulates tokens in. Every rank holds the same
    number of batches, as ``DistributedSampler`` guarantees in real
    training.

    Attributes:
        rank_0: Residue count of each batch rank 0 trains on, in order.
        rank_1: Residue count of each batch rank 1 trains on, in order.
    """

    rank_0: tuple[int, ...]
    rank_1: tuple[int, ...]

    @property
    def by_rank(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Each rank's batch lengths, indexed by rank number.

        Returns:
            ``(rank_0, rank_1)``, so ``by_rank[r]`` is rank ``r``'s batch
            lengths and ``len(by_rank)`` is the world size.
        """
        return (self.rank_0, self.rank_1)


@dataclasses.dataclass(frozen=True)
class LockstepCase:
    """Inputs and expected outcome of one DDP lockstep scenario.

    Shared by every ``LockstepScenario`` member, so each case has the same
    named, typed fields. A 16-residue protein has 16 tokens, so with a
    per-rank budget of 24 a rank holding two 16-residue batches wants to
    flush before its second batch (16 + 16 > 24), and a rank holding two
    8-residue batches does not (8 + 8 <= 24).

    Attributes:
        residues: Residue count of each one-protein batch on each rank.
        budget_per_rank: Tokens one rank may accumulate before it wants to
            flush.
        expected_steps: Optimizer steps every rank must take in the epoch,
            including the epoch-end flush of the buffered micro-batches.
    """

    residues: RankResidues
    budget_per_rank: int
    expected_steps: int


class LockstepScenario(enum.Enum):
    """Named DDP lockstep cases, each a ``LockstepCase``.

    Members:
        rank_1_over_budget: Only rank 1 wants to flush, so the vote must
            flush rank 0 too: one pre-flush plus the epoch-end flush.
        rank_0_over_budget: Only rank 0 wants to flush; catches a vote that
            listens to only one rank.
        neither_over_budget: No rank wants to flush, so the only step is the
            epoch-end flush of both batches.
        both_over_budget: Both ranks want to flush at the same batch.
    """

    rank_1_over_budget = LockstepCase(RankResidues((8, 8), (16, 16)), 24, 2)
    rank_0_over_budget = LockstepCase(RankResidues((16, 16), (8, 8)), 24, 2)
    neither_over_budget = LockstepCase(RankResidues((8, 8), (8, 8)), 24, 1)
    both_over_budget = LockstepCase(RankResidues((16, 16), (16, 16)), 24, 2)


@pytest.fixture
def lockstep_setup(
    lockstep_scenario: LockstepScenario,
    model_params: ModelSetup,
) -> ModelSetup:
    """Provide ``model_params`` with the scenario's token budget.

    Factory fixture: ``train()`` divides ``accumulated_token_budget`` by the
    world size to get each rank's budget, so the global budget is the
    scenario's ``budget_per_rank`` times its number of ranks.

    Args:
        lockstep_scenario: Scenario, parametrized by the requesting test,
            naming the per-rank budget and, through
            the length of ``residues.by_rank``, the number of ranks.
        model_params: Fixture providing the small model, optimizer,
            scheduler, EMA and base TrainConfig.

    Returns:
        A ModelSetup sharing every component of ``model_params`` except the
        TrainConfig, whose token budget is replaced.
    """
    base = model_params.tcfg
    world_size = len(lockstep_scenario.value.residues.by_rank)
    return dataclasses.replace(
        model_params,
        tcfg=base.model_copy(
            update={
                "training": base.training.model_copy(
                    update={
                        "accumulated_token_budget": world_size
                        * lockstep_scenario.value.budget_per_rank,
                    },
                ),
            },
        ),
    )


@pytest.fixture
def lockstep_run(
    lockstep_scenario: LockstepScenario,
    lockstep_setup: ModelSetup,
    tmp_path: pathlib.Path,
) -> DDPLockstepRun:
    """Build a DDP run in which each rank holds proteins of its own length.

    Factory fixture: rank ``r`` trains on one one-protein batch of
    ``straight_chain_protein(n_res)`` for each ``n_res`` in
    ``lockstep_scenario.value.residues.by_rank[r]``.

    Args:
        lockstep_scenario: Scenario naming each rank's batch lengths; the
            requesting test parametrizes it.
        lockstep_setup: Fixture providing the model setup with the
            scenario's token budget.
        tmp_path: Pytest-provided directory for the rendezvous store and the
            checkpoint that rank 0 writes.

    Returns:
        DDPLockstepRun ready to hand to ``torch.multiprocessing.spawn``.
    """
    return DDPLockstepRun(
        setup=lockstep_setup,
        batches_by_rank=tuple(
            tuple(
                featurize_batch(
                    [straight_chain_protein(n_res)],
                    lockstep_setup.tcfg,
                    lockstep_setup.distogram_template,
                    lockstep_setup.distogram_residue,
                    lockstep_setup.distogram_atom,
                )
                for n_res in rank_residues
            )
            for rank_residues in lockstep_scenario.value.residues.by_rank
        ),
        store_path=tmp_path / "gloo_store",
    )


@pytest.mark.parametrize(
    "lockstep_scenario",
    list(LockstepScenario),
    ids=[case.name for case in LockstepScenario],
)
def test_train_ddp_keeps_ranks_in_lockstep(
    lockstep_scenario: LockstepScenario,
    lockstep_run: DDPLockstepRun,
) -> None:
    """Ranks with different token counts take the same steps and match.

    Regression test for an end-of-epoch DDP deadlock. Each rank used to
    decide on its own when its next batch would overflow the token budget,
    so a rank holding long proteins took more optimizer steps than a rank
    holding short ones. Each extra step's gradient all-reduce had no
    partner, and the job hung until the collective timeout killed it. The
    flush decision is now a vote: if any rank's next batch would overflow,
    every rank flushes. Leftover micro-batches are flushed at epoch end
    instead of being dropped. ``LockstepScenario`` covers the vote being
    won by rank 1 alone, by rank 0 alone, by nobody, and by both ranks.

    Arrange: ``lockstep_run`` gives each rank the scenario's one-protein
    batches, and ``lockstep_setup`` applies the scenario's token budget.
    Act: ``torch.multiprocessing.spawn`` runs ``train_one_rank`` in two real
    OS processes forming a CPU gloo group; each trains one epoch and reports
    its optimizer step count and a digest of its weights.
    Assert: every rank reports the scenario's step count, and all ranks'
    weight digests are identical. A deadlock makes gloo abort the stuck
    collective after 60 seconds, and ``spawn`` then raises, so the test
    fails instead of hanging.

    Args:
        lockstep_scenario: Scenario naming the expected step count.
        lockstep_run: Factory-fixture-built run for this scenario.
    """
    world_size = len(lockstep_run.batches_by_rank)
    reports: SimpleQueue[tuple[int, str]] = multiprocessing.get_context(
        "spawn",
    ).SimpleQueue()

    _ = spawn(train_one_rank, args=(lockstep_run, reports), nprocs=world_size)

    reported = {reports.get() for _ in range(world_size)}
    assert {steps for steps, _ in reported} == {
        lockstep_scenario.value.expected_steps,
    }
    assert len({digest for _, digest in reported}) == 1


# ---------------------------------------------------------------------------
# self_conditioned_template — the training-side self-conditioning pass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrunkCall:
    """What the trunk saw on one forward call.

    Attributes:
        grad_enabled: Whether autograd was on during the call.
        template: The template distogram the call received.
    """

    grad_enabled: bool
    template: Float[torch.Tensor, "B N_res N_res n_templ_bins"]


@dataclasses.dataclass(frozen=True)
class RecordingForward:
    """Stand-in ``forward`` that records each call, then runs the real one.

    The real forward is reached through the class, ``MainTrunk.forward``,
    which the instance attribute set by ``monkeypatch`` does not shadow.

    Attributes:
        trunk: The trunk whose calls are recorded.
        calls: Receives one ``TrunkCall`` per call.
    """

    trunk: MainTrunk
    calls: list[TrunkCall]

    def __call__(self, batch: FeaturizedBatch) -> PredictedOutputs:
        """Record the autograd state and template, then run the trunk.

        Args:
            batch: The trunk's input.

        Returns:
            The real trunk's prediction.
        """
        self.calls.append(
            TrunkCall(
                grad_enabled=torch.is_grad_enabled(),
                template=batch.noised_res_distogram.detach().clone(),
            ),
        )
        return MainTrunk.forward(self.trunk, batch)


@pytest.fixture
def recorded_trunk_calls(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> list[TrunkCall]:
    """Run the self-conditioning pass in train mode and record trunk calls.

    Wraps the real trunk's ``forward`` so each call records its input
    template and autograd state, then runs unchanged. ``MainTrunk``
    overrides ``__call__`` to call ``forward`` directly, so module hooks
    never fire; the instance attribute set here is what ``__call__`` finds.
    Recording the input checks what the pass feeds the model rather than
    its binned output, which a small untrained model can leave unchanged at
    bin resolution. Returns a list because a regression could add forwards.

    Args:
        featurized_batch: Batch whose noised ground-truth template is a
            one-hot that must not reach the trunk.
        model_params: Model and template distogram module.
        monkeypatch: Installs and later removes the recording ``forward``.

    Returns:
        One ``TrunkCall`` per trunk forward made by the pass.
    """
    trunk = cast(MainTrunk, model_params.model)
    recorder = RecordingForward(trunk=trunk, calls=[])
    monkeypatch.setattr(trunk, "forward", recorder)
    _ = trunk.train()
    with torch.enable_grad():
        _ = self_conditioned_template(featurized_batch, model_params)
    return recorder.calls


def test_self_conditioning_pass_runs_detached_on_null(
    recorded_trunk_calls: list[TrunkCall],
) -> None:
    """The pass makes one trunk call, with autograd off, on the null.

    Autograd must be off even when the caller has it on, or the pass would
    store a second activation graph. The template must be the all-zero null,
    not the featurizer's noised ground truth, which leaks true pair geometry
    the sampler never has. No parametrization: the caller's grad-enabled
    train mode is the one state in which either property can fail.

    Args:
        recorded_trunk_calls: What the trunk saw on each call.
    """
    assert len(recorded_trunk_calls) == 1
    assert not recorded_trunk_calls[0].grad_enabled
    assert not recorded_trunk_calls[0].template.any()


@dataclasses.dataclass(frozen=True)
class TemplatePair:
    """Templates built from a batch before and after its noisy input moved.

    Attributes:
        baseline: Template from the unmodified batch.
        perturbed: Template after ``r_gt_noised`` was scaled and shifted.
    """

    baseline: Float[torch.Tensor, "B N_res N_res n_templ_bins"]
    perturbed: Float[torch.Tensor, "B N_res N_res n_templ_bins"]


@pytest.fixture
def template_pair(
    featurized_batch: FeaturizedBatch,
    model_params: ModelSetup,
) -> TemplatePair:
    """Build the template from a batch, then again with moved coordinates.

    The model runs in eval mode so dropout cannot make the two passes
    differ on its own. Scaling ``r_gt_noised`` by 2 and shifting it by 1
    moves every coordinate, so a template built from the prediction moves.

    Args:
        featurized_batch: Batch the templates are built from.
        model_params: Model and template distogram module.

    Returns:
        The baseline and perturbed templates.
    """
    _ = model_params.model.eval()
    perturbed_batch = dataclasses.replace(
        featurized_batch,
        r_gt_noised=featurized_batch.r_gt_noised * 2.0 + 1.0,
    )
    return TemplatePair(
        baseline=self_conditioned_template(featurized_batch, model_params),
        perturbed=self_conditioned_template(perturbed_batch, model_params),
    )


def test_self_conditioned_template_is_binned_prediction(
    featurized_batch: FeaturizedBatch,
    template_pair: TemplatePair,
) -> None:
    """The template is a one-hot per valid pair and follows the noisy input.

    A one-hot over the bins for every valid residue pair shows the output is
    a binned structure; changing with ``r_gt_noised`` shows that structure
    is the model's prediction. No parametrization: the batch's noisy
    coordinates are the only input the prediction is built from.

    Args:
        featurized_batch: Supplies the residue mask.
        template_pair: Templates before and after the input moved.
    """
    residue_mask = featurized_batch.f_pseudo_beta_mask.bool()
    pair_mask = rearrange(residue_mask, "b i -> b i 1") & rearrange(
        residue_mask,
        "b j -> b 1 j",
    )
    bin_sums = reduce(template_pair.baseline, "b i j k -> b i j", "sum")
    assert torch.equal(
        bin_sums[pair_mask],
        torch.ones_like(bin_sums[pair_mask]),
    )
    assert not torch.equal(template_pair.baseline, template_pair.perturbed)


@dataclasses.dataclass(frozen=True)
class TemplateSourceCase:
    """Training config for one template source.

    Attributes:
        self_condition: Value of ``training.self_condition``. True builds the
            template from the model's own detached prediction; False uses
            the noised ground-truth template from the featurizer.
    """

    self_condition: bool


class TemplateSource(enum.Enum):
    """Where ``take_step`` gets the template it feeds the trunk.

    Members:
        self_conditioned_template: The binned prediction of a detached pass
            run against the null template.
        noised_ground_truth_template: The featurized template binned from
            the noised ground truth.
    """

    self_conditioned_template = TemplateSourceCase(self_condition=True)
    noised_ground_truth_template = TemplateSourceCase(self_condition=False)


@pytest.fixture
def self_condition_setup(
    template_source: TemplateSource,
    model_params: ModelSetup,
) -> ModelSetup:
    """Return ``model_params`` with ``training.self_condition`` overridden.

    Args:
        template_source: Template source to configure, parametrized
            directly by the requesting test.
        model_params: Base model setup to copy.

    Returns:
        A ModelSetup whose config has the scenario's flag.
    """
    base_cfg = model_params.tcfg
    return dataclasses.replace(
        model_params,
        tcfg=base_cfg.model_copy(
            update={
                "training": base_cfg.training.model_copy(
                    update={
                        "self_condition": (
                            template_source.value.self_condition
                        ),
                    },
                ),
            },
        ),
    )


@pytest.mark.parametrize(
    "template_source",
    list(TemplateSource),
    ids=[case.name for case in TemplateSource],
)
def test_take_step_trains_template_embedder(
    featurized_batch: FeaturizedBatch,
    self_condition_setup: ModelSetup,
) -> None:
    """Both template sources leave finite, non-zero template embedder grads.

    The self-conditioning pass is detached, so it must not cut the gradient
    path of the grad-tracking pass that consumes its template.

    Args:
        featurized_batch: Batch to train on.
        self_condition_setup: Model setup configured for the template source.
    """
    trunk = cast(MainTrunk, self_condition_setup.model)
    _ = trunk.train()
    trunk.zero_grad()
    _ = take_step(
        batch=featurized_batch,
        model_params=self_condition_setup,
        train_mode=True,
    )
    grads = [
        p.grad
        for p in trunk.template_embedder.parameters()
        if p.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.any() for g in grads)
