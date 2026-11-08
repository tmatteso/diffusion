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

import dataclasses
import enum
import functools
import json
import pathlib
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import pytest
import structlog
import torch
import torch.nn as nn
from architecture.main_trunk import EMA, MainTrunk
from einops import reduce
from helpers.atom_utils import RESTYPE_NUM, RESTYPE_NUM_NO_X, Protein
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
    component_grad_norms,
    evaluate,
    load_checkpoint,
    log_epoch,
    optimizer_step,
    process_accum_window,
    save_checkpoint,
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
    flushes the first batch when the second arrives, and the second stays
    in the buffer because ``train()`` has no end-of-epoch flush.

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


class TcfgFactory(Protocol):
    """Builds a TrainConfig for a case, checkpointing under a directory."""

    def __call__(
        self,
        work_dir: pathlib.Path,
        case: TrainCase,
        *,
        loader_budget: int = 63,
    ) -> TrainConfig:
        """Build the configuration.

        Args:
            work_dir: Directory for the checkpoint.
            case: Training settings.
            loader_budget: Token budget of each loader batch.

        Returns:
            The configuration.
        """
        ...  # pylint: disable=unnecessary-ellipsis


@dataclasses.dataclass(frozen=True)
class TrainHarness:
    """The factories a full ``train()`` run is assembled from.

    Attributes:
        make_trunk: Builds a freshly seeded trunk.
        make_tcfg: Builds a configuration for a case.
        make_setup: Bundles a trunk with optimizer, scheduler and EMA.
        make_loaders: Builds loaders over the synthetic proteins.
    """

    make_trunk: Callable[[], MainTrunk]
    make_tcfg: TcfgFactory
    make_setup: Callable[[MainTrunk | DDP, TrainConfig], ModelSetup]
    make_loaders: Callable[[pathlib.Path, TrainConfig], Loaders]


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
def make_trunk(model_config: TrainModelConfig) -> Callable[[], MainTrunk]:
    """Return a factory for the small trunk, built under a fixed seed.

    Args:
        model_config: Model and distogram settings.

    Returns:
        Zero-argument factory.
    """

    def make() -> MainTrunk:
        """Build the trunk.

        Returns:
            A freshly initialised MainTrunk.
        """
        _ = manual_seed(0)
        return MainTrunk(
            model_params=model_config.model,
            template_distogram_params=model_config.template,
            residue_distogram_params=model_config.residue,
            atom_distogram_params=model_config.atom,
            noise_params=model_config.noise,
        )

    return make


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
def make_tcfg(model_config: TrainModelConfig) -> TcfgFactory:
    """Return a factory for TrainConfigs matching the shared model.

    Args:
        model_config: Model and distogram settings.

    Returns:
        ``make(work_dir, case, loader_budget=63)``.
    """

    def make(
        work_dir: pathlib.Path,
        case: TrainCase,
        *,
        loader_budget: int = 63,
    ) -> TrainConfig:
        """Build the configuration.

        Args:
            work_dir: Directory for the checkpoint.
            case: Training settings.
            loader_budget: Token budget of each loader batch.

        Returns:
            The configuration.
        """
        return TrainConfig(
            training=TrainingParams(
                num_epochs=case.num_epochs,
                lr=1e-4,
                grad_clip=case.grad_clip,
                accumulated_token_budget=case.token_budget,
            ),
            model=model_config.model,
            distogram_template=model_config.template,
            distogram_residue=model_config.residue,
            distogram_atom=model_config.atom,
            noise=model_config.noise,
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

    return make


@pytest.fixture(scope="module")
def make_setup(
    distograms: Distograms,
) -> Callable[[MainTrunk | DDP, TrainConfig], ModelSetup]:
    """Return a factory bundling a trunk with Adam, LambdaLR and an EMA.

    Args:
        distograms: Distogram modules to carry.

    Returns:
        ``make(trunk, tcfg)``.
    """

    def make(trunk: MainTrunk | DDP, tcfg: TrainConfig) -> ModelSetup:
        """Build the setup.

        Args:
            trunk: Model, plain or DDP-wrapped.
            tcfg: Training configuration.

        Returns:
            The setup on CPU.
        """
        optimizer = Adam(trunk.parameters(), lr=tcfg.training.lr)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=functools.partial(
                alphafold_lr_lambda,
                warmup_steps=tcfg.training.lr_warmup_steps,
                decay_steps=tcfg.training.lr_decay_steps,
                decay_factor=tcfg.training.lr_decay_factor,
            ),
        )
        return ModelSetup(
            model=trunk,
            tcfg=tcfg,
            distogram_template=distograms.template,
            distogram_residue=distograms.residue,
            distogram_atom=distograms.atom,
            device=torch.device("cpu"),
            optimizer=optimizer,
            scheduler=scheduler,
            ema=EMA(trunk, decay=tcfg.training.ema_decay),
        )

    return make


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
def make_loaders(
    dataset_files: TrainArgs,
) -> Callable[[pathlib.Path, TrainConfig], Loaders]:
    """Return a factory for loaders over the synthetic proteins.

    Args:
        dataset_files: Dataset and splits paths.

    Returns:
        ``make(work_dir, tcfg)``, sharding under ``work_dir``.
    """

    def make(work_dir: pathlib.Path, tcfg: TrainConfig) -> Loaders:
        """Build the loaders.

        Args:
            work_dir: Directory for shards, config and logs.
            tcfg: Configuration whose loader settings apply.

        Returns:
            The loaders.
        """
        args = dataclasses.replace(
            dataset_files,
            shard_dir=work_dir / "shards",
            config=work_dir / "config.json",
            structlog_jsonl=work_dir / "structlog.jsonl",
        )
        train_loader, test_loader, _ = make_bucketed_data_loaders(
            cfg=tcfg,
            extra_train_args=args,
        )
        return Loaders(train=train_loader, test=test_loader)

    return make


@pytest.fixture(scope="module")
def harness(
    make_trunk: Callable[[], MainTrunk],
    make_tcfg: TcfgFactory,
    make_setup: Callable[[MainTrunk | DDP, TrainConfig], ModelSetup],
    make_loaders: Callable[[pathlib.Path, TrainConfig], Loaders],
) -> TrainHarness:
    """Bundle the four factories a full ``train()`` run needs.

    Args:
        make_trunk: Trunk factory.
        make_tcfg: Configuration factory.
        make_setup: Setup factory.
        make_loaders: Loader factory.

    Returns:
        The bundle.
    """
    return TrainHarness(
        make_trunk=make_trunk,
        make_tcfg=make_tcfg,
        make_setup=make_setup,
        make_loaders=make_loaders,
    )


@pytest.fixture
def model_params(
    tmp_path: pathlib.Path,
    make_trunk: Callable[[], MainTrunk],
    make_tcfg: TcfgFactory,
    make_setup: Callable[[MainTrunk | DDP, TrainConfig], ModelSetup],
) -> ModelSetup:
    """Build a fresh setup with the default one-epoch configuration.

    Args:
        tmp_path: Per-test directory for the checkpoint.
        make_trunk: Trunk factory.
        make_tcfg: Configuration factory.
        make_setup: Setup factory.

    Returns:
        The setup, with W&B off.
    """
    tcfg = make_tcfg(tmp_path, TrainScenario.default_budget.value)
    return make_setup(make_trunk(), tcfg)


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
    make_loaders: Callable[[pathlib.Path, TrainConfig], Loaders],
) -> Loaders:
    """Build loaders over the synthetic proteins in a per-test directory.

    Args:
        tmp_path: Per-test directory.
        model_params: Supplies the configuration.
        make_loaders: Loader factory.

    Returns:
        The loaders.
    """
    return make_loaders(tmp_path, model_params.tcfg)


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
    diverged_ema: ModelSetup,
) -> SwapSetup:
    """Prepare a swap, wrapping the model in one-rank DDP when the param asks.

    With DDP the swap must reach ``.module``, whose keys carry no
    ``module.`` prefix. Reference weights are cloned before any swap.

    Args:
        request: Pytest request carrying whether to wrap in DDP.
        diverged_ema: Setup whose raw weights and shadow differ.

    Returns:
        The setup and its reference weights.
    """
    inner = cast(MainTrunk, diverged_ema.model)
    setup = diverged_ema
    if cast(bool, request.param):
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


@pytest.fixture(scope="module")
def recorded_train_runs(
    tmp_path_factory: pytest.TempPathFactory,
    harness: TrainHarness,
) -> Callable[[TrainScenario], TrainRun]:
    """Return a lookup that runs each TrainScenario at most once per module.

    The cache is explicit because a parametrized module-scoped fixture does
    not guarantee one run per scenario: pytest groups parameters by their
    position in each ``parametrize`` list. Each run spies on
    ``all_reduce_mean`` and captures ``wandb.log`` through
    ``MonkeyPatch.context``, since function-scoped ``monkeypatch`` cannot
    serve a module-scoped cache.

    Args:
        tmp_path_factory: Factory for each run's directory.
        harness: Model, configuration, setup and loader factories.

    Returns:
        Callable returning the scenario's TrainRun, running it on first use.
    """
    runs: dict[TrainScenario, TrainRun] = {}
    original_reduce = TensorAccumulatorMixin.all_reduce_mean

    def record(scenario: TrainScenario) -> TrainRun:
        """Run ``train()`` for the scenario and record what it did.

        Args:
            scenario: Configuration to run.

        Returns:
            The record.
        """
        work = tmp_path_factory.mktemp(scenario.name)
        tcfg = harness.make_tcfg(work, scenario.value)
        trunk = harness.make_trunk()
        built = harness.make_loaders(work, tcfg)
        before = [p.clone().detach() for p in trunk.parameters()]
        reduced: list[TensorAccumulatorMixin] = []
        payloads: list[dict[str, object]] = []

        def spy(self: TensorAccumulatorMixin) -> None:
            """Record the instance, then run the real reduction.

            Args:
                self: Metrics being reduced.
            """
            reduced.append(self)
            original_reduce(self)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(TensorAccumulatorMixin, "all_reduce_mean", spy)
            patch.setattr("train.train_loop.wandb.log", payloads.append)
            train(
                best_val_loss=torch.tensor(float("inf")),
                train_loader=built.train,
                test_loader=built.test,
                model_params=harness.make_setup(trunk, tcfg),
                log=cast(FilteringBoundLogger, structlog.get_logger()),
            )
        return TrainRun(
            parameters_changed=any(
                not torch.equal(b, a)
                for b, a in zip(before, trunk.parameters(), strict=True)
            ),
            reduced_metrics=reduced,
            wandb_payloads=payloads,
            checkpoint=cast(
                dict[str, dict[str, object]],
                torch.load(work / "best.pt", weights_only=True),
            ),
            num_epochs=scenario.value.num_epochs,
        )

    def lookup(scenario: TrainScenario) -> TrainRun:
        """Return the scenario's record, running it on first use.

        Args:
            scenario: Configuration to look up.

        Returns:
            The record.
        """
        if scenario not in runs:
            runs[scenario] = record(scenario)
        return runs[scenario]

    return lookup


@pytest.fixture
def train_run(
    request: pytest.FixtureRequest,
    recorded_train_runs: Callable[[TrainScenario], TrainRun],
) -> TrainRun:
    """Return the shared record for the scenario routed in via param.

    The record is a frozen snapshot rather than live training state, so the
    tests sharing it stay order-independent.

    Args:
        request: Pytest request carrying a ``TrainScenario``.
        recorded_train_runs: Module-wide lookup.

    Returns:
        The scenario's record.
    """
    return recorded_train_runs(cast(TrainScenario, request.param))


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch ``i`` of the window gets ``grad_scale = total / proteins_i``.

    With 1 and 3 proteins the total is 4, so the scales are 4 and 4/3, and
    the window's losses are finite. Not parametrized: two unequal batches
    show the weighting.

    Args:
        featurized_batch: Batch used for both window slots.
        model_params: Setup holding the model.
        monkeypatch: Wraps ``take_step`` to record its scale.
    """
    scales: list[float] = []
    real_take_step = take_step

    def recording_take_step(
        *,
        batch: FeaturizedBatch,
        model_params: ModelSetup,  # pylint: disable=redefined-outer-name
        train_mode: bool,
        grad_scale: float = 1.0,
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        """Record ``grad_scale``, then take the real step.

        Args:
            batch: Micro-batch.
            model_params: Setup.
            train_mode: Whether to backpropagate.
            grad_scale: Divisor applied to the loss.

        Returns:
            The real step's metrics.
        """
        scales.append(grad_scale)
        return real_take_step(
            batch=batch,
            model_params=model_params,
            train_mode=train_mode,
            grad_scale=grad_scale,
        )

    monkeypatch.setattr("train.train_loop.take_step", recording_take_step)
    _ = model_params.model.train()
    loss, _ = process_accum_window(
        [featurized_batch, featurized_batch],
        [1, 3],
        model_params,
    )
    losses = torch.tensor(list(loss.to_float_dict().values()))
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
    "swap_setup",
    [False, True],
    ids=["plain", "ddp"],
    indirect=True,
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
    "swap_setup",
    [False, True],
    ids=["plain", "ddp"],
    indirect=True,
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
    ("rank", "writes"),
    [(0, True), (1, False)],
    ids=["rank0", "rank1"],
)
def test_save_checkpoint_writes_only_on_rank_zero(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    rank: int,
    writes: bool,  # noqa: FBT001
) -> None:
    """Rank 0 writes a checkpoint with every state; other ranks write none.

    Args:
        model_params: Setup to save.
        log: Logger.
        rank: Rank passed to ``save_checkpoint``.
        writes: Whether a file must appear.
    """
    path = model_params.tcfg.checkpoint.checkpoint_path
    save_checkpoint(
        model_params=model_params,
        rank=rank,
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
def test_lambda_lr_matches_alphafold_lr_lambda_through_real_scheduler(
    lr_schedule_optimizer: Adam,
    n_steps: int,
) -> None:
    """A real LambdaLR driven by ``alphafold_lr_lambda`` reproduces it.

    With 4 warmup steps, decay every 3 steps by 0.5 and base rate 2, the
    optimizer's live rate after ``n_steps`` must equal
    ``2 * alphafold_lr_lambda(n_steps)``, through PyTorch's scheduler.

    Args:
        lr_schedule_optimizer: Adam at base rate 2.
        n_steps: Scheduler steps to take.
    """
    schedule = functools.partial(
        alphafold_lr_lambda,
        warmup_steps=4,
        decay_steps=3,
        decay_factor=0.5,
    )
    scheduler = LambdaLR(lr_schedule_optimizer, lr_lambda=schedule)
    for _ in range(n_steps):
        scheduler.step()
    actual = cast(float, lr_schedule_optimizer.param_groups[0]["lr"])
    torch.testing.assert_close(actual, 2.0 * schedule(n_steps))


def test_train_preflushes_before_an_oversized_batch(
    tmp_path: pathlib.Path,
    harness: TrainHarness,
    log: FilteringBoundLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each window is flushed before a batch that would exceed the budget.

    The loader yields 4 one-protein batches; with a budget of 7 tokens, two
    batches always exceed it, so batches 1 to 3 are each flushed alone
    when the next arrives and batch 4 stays buffered (there is no
    end-of-epoch flush): three windows of one. Not parametrized: the
    budget is chosen so every arrival triggers a flush.

    Args:
        tmp_path: Per-test directory.
        harness: Model, configuration, setup and loader factories.
        log: Logger.
        monkeypatch: Wraps ``process_accum_window`` to record windows.
    """
    windows: list[int] = []
    real_process = process_accum_window

    def tracking_process(
        micro_buffer: list[FeaturizedBatch],
        n_proteins_per_batch: list[int],
        model_params: ModelSetup,  # pylint: disable=redefined-outer-name
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        """Record the window size, then process it for real.

        Args:
            micro_buffer: Batches in the window.
            n_proteins_per_batch: Proteins per batch.
            model_params: Setup.

        Returns:
            The real window's metrics.
        """
        windows.append(len(micro_buffer))
        return real_process(micro_buffer, n_proteins_per_batch, model_params)

    monkeypatch.setattr(
        "train.train_loop.process_accum_window",
        tracking_process,
    )
    tcfg = harness.make_tcfg(
        tmp_path,
        TrainCase(1, 7, 1.0, use_wandb=False),
        loader_budget=42,
    )
    tcfg = tcfg.model_copy(update={"test_loader": LoaderConfig(batch_size=1)})
    built = harness.make_loaders(tmp_path, tcfg)
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=built.train,
        test_loader=built.test,
        model_params=harness.make_setup(harness.make_trunk(), tcfg),
        log=log,
    )
    assert windows == [1, 1, 1]


@pytest.mark.parametrize(
    "train_run",
    list(TrainScenario),
    ids=[case.name for case in TrainScenario],
    indirect=True,
)
def test_train_updates_model_parameters(train_run: TrainRun) -> None:
    """Training changes the model's parameters under every scenario.

    Args:
        train_run: Recorded run.
    """
    assert train_run.parameters_changed


@pytest.mark.parametrize(
    "train_run",
    [TrainScenario.default_budget],
    ids=["default_budget"],
    indirect=True,
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
    "train_run",
    [TrainScenario.default_budget, TrainScenario.no_grad_clip],
    ids=["default_budget", "no_grad_clip"],
    indirect=True,
)
def test_train_does_not_call_wandb_when_disabled(train_run: TrainRun) -> None:
    """With ``use_wandb`` off, ``wandb.log`` is never called.

    Args:
        train_run: Recorded run with W&B off.
    """
    assert train_run.wandb_payloads == []


@pytest.mark.parametrize(
    "train_run",
    [TrainScenario.large_token_budget],
    ids=["large_token_budget"],
    indirect=True,
)
def test_train_logs_nonzero_loss_under_large_token_budget(
    train_run: TrainRun,
) -> None:
    """A budget above one batch but below two still runs an optimizer step.

    A budget above both batches' tokens would train nothing, because
    ``train()`` has no end-of-epoch flush.

    Args:
        train_run: Recorded run with W&B on.
    """
    assert len(train_run.wandb_payloads) == 1
    train_loss = train_run.wandb_payloads[0]["train/total_loss"]
    assert isinstance(train_loss, float)
    assert train_loss > 0


@pytest.mark.parametrize(
    "train_run",
    [TrainScenario.default_budget],
    ids=["default_budget"],
    indirect=True,
)
def test_train_checkpoint_holds_stepped_optimizer_and_scheduler(
    train_run: TrainRun,
) -> None:
    """The saved Adam moments are nonzero and the scheduler counts epochs.

    Args:
        train_run: Recorded one-epoch run.
    """
    opt_state = cast(
        dict[int, StateDict],
        train_run.checkpoint["optimizer"]["state"],
    )
    first = next(iter(opt_state.values()))
    assert bool(first["exp_avg"].abs().max() > 0)
    assert (
        train_run.checkpoint["scheduler"]["last_epoch"] == train_run.num_epochs
    )


@pytest.mark.parametrize(
    "train_run",
    [TrainScenario.multi_epoch],
    ids=["multi_epoch"],
    indirect=True,
)
def test_train_logs_once_per_epoch_with_the_epoch_advancing(
    train_run: TrainRun,
) -> None:
    """Two epochs give two W&B payloads, for epochs 1 and 2.

    Args:
        train_run: Recorded two-epoch run with W&B on.
    """
    assert [p["epoch"] for p in train_run.wandb_payloads] == [1, 2]
