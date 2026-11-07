"""Hybrid sharding on a four-rank CPU gloo mesh; runs on every PR.

Each distributed test spawns four real processes and runs real collectives
on a tiny ``MainTrunk``, so ``pytest pallatom`` covers it on both CI runners
with no GPU. Tests marked ``xfail(strict=True)`` encode a required change from
``docs/fsdp_device_mesh.md``: they fail today, and the strict marker turns the
fix into a test failure until the marker is removed.
"""

import contextlib
import dataclasses
import enum
import multiprocessing
import pathlib
from collections.abc import Generator
from datetime import timedelta
from multiprocessing.queues import SimpleQueue
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from architecture.atom_transformers import (
    AtomAttentionDecoder,
    AtomFeatureEncoder,
)
from architecture.main_trunk import MainTrunk
from architecture.node_update import NodeUpdate
from architecture.pair_update import PairUpdate
from architecture.template_embedder import TemplateEmbedder
from helpers.useful_objects import manual_seed
from tests.fsdp.harness import (
    ParityGaps,
    Workload,
    build_hybrid_setup,
    build_mesh,
    build_model,
    build_setup,
    gather_state,
    model_state,
    parity_gaps,
    rank_batches,
    reshard_like,
    route_calls_through_module_hooks,
    run_windows,
    shard_model,
    state_gap,
    structlog_logger,
    write_proteins,
)
from torch.distributed.tensor import DTensor
from torch.multiprocessing.spawn import ProcessRaisedException, spawn
from train.train_config import (
    AtomDistogramParams,
    CheckpointParams,
    LoaderConfig,
    LoggingParams,
    ModelParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
    TrainConfig,
    TrainingParams,
)
from train.train_loop import save_checkpoint, take_step

_SHARDED_CLASSES: tuple[type[nn.Module], ...] = (
    MainTrunk,
    NodeUpdate,
    AtomAttentionDecoder,
    PairUpdate,
    TemplateEmbedder,
    AtomFeatureEncoder,
)
_WORLD_SIZE = 4
_MESH_SHAPE = (2, 2)
_N_RES = 8
_BATCH_SIZE = 1
_C_RES = 32
_C_ATOM = 32
_C_PAIR = 32
_C_ATOMPAIR = 16
_K_UNIT = 2
_N_BLOCKS = 1
_N_HEADS = 4
_TEMPLATE_BINS = 8
_RESIDUE_BINS = 8
_ATOM_BINS = 5
_ACTIVE_CLIP = 1e-3
_INACTIVE_CLIP = 1e6
_PARITY_TOLERANCE = 1e-4
_COLLECTIVE_TIMEOUT = timedelta(seconds=120)
_INFINITE_LOSS = float("inf")


class ParityScenario(enum.Enum):
    """Accumulation and clipping cases where hybrid must match DDP.

    Members:
        one_micro_batch: Plain step with clipping forced on.
        accumulated: Two micro-batches, syncing on every backward, which is
            what happens with no code changes because ``no_sync`` is absent.
        accumulated_window_sync: Two micro-batches, syncing once per window.
        no_clipping: Two micro-batches with clipping out of reach.
        dropout_active: Train mode, so dropout must agree under checkpointing.
    """

    one_micro_batch = (1, _ACTIVE_CLIP, False, False)
    accumulated = (2, _ACTIVE_CLIP, False, False)
    accumulated_window_sync = (2, _ACTIVE_CLIP, True, False)
    no_clipping = (2, _INACTIVE_CLIP, False, False)
    dropout_active = (2, _ACTIVE_CLIP, False, True)

    @property
    def micro_batches(self) -> int:
        """Micro-batches per accumulation window."""
        return self.value[0]

    @property
    def grad_clip(self) -> float:
        """Gradient clipping threshold."""
        return self.value[1]

    @property
    def window_sync(self) -> bool:
        """Whether hybrid gradients sync once per window."""
        return self.value[2]

    @property
    def train_mode(self) -> bool:
        """Whether dropout is active."""
        return self.value[3]


@dataclasses.dataclass(frozen=True)
class Gloo:
    """Rendezvous details for one spawned four-rank group.

    Attributes:
        store_path: File backing the gloo store.
        device: Device for every rank.
    """

    store_path: pathlib.Path
    device: torch.device


@contextlib.contextmanager
def gloo_rank(group: Gloo, rank: int) -> Generator[None, None, None]:
    """Join the gloo group as ``rank`` with hooks routed for FSDP.

    Args:
        group: Rendezvous details.
        rank: This process's rank, supplied by ``spawn``.

    Yields:
        None: The group is live for the duration of the block.
    """
    torch.set_num_threads(1)
    route_calls_through_module_hooks()
    dist.init_process_group(
        backend="gloo",
        init_method=group.store_path.as_uri(),
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=_COLLECTIVE_TIMEOUT,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def parity_rank(
    rank: int,
    group: Gloo,
    run: Workload,
    window_sync: bool,  # noqa: FBT001
    reports: SimpleQueue[ParityGaps],
) -> None:
    """Report this rank's DDP-versus-hybrid gaps.

    Args:
        rank: This process's rank.
        group: Rendezvous details.
        run: The run description.
        window_sync: Sync hybrid gradients once per window.
        reports: Queue receiving the gaps.
    """
    with gloo_rank(group, rank):
        reports.put(
            parity_gaps(run, group.device, window_sync=window_sync),
        )


def replica_rank(
    rank: int,
    group: Gloo,
    run: Workload,
    seeded: bool,  # noqa: FBT001
    reports: SimpleQueue[tuple[int, float, float]],
) -> None:
    """Report a digest of this rank's local shards and buffers after init.

    Args:
        rank: This process's rank.
        group: Rendezvous details.
        run: The run description.
        seeded: Seed identically on all ranks before building the model.
        reports: Queue receiving ``(rank, shard_digest, buffer_digest)``.
    """
    with gloo_rank(group, rank):
        _ = manual_seed(0 if seeded else rank + 1)
        model = shard_model(
            build_model(run.tcfg, group.device),
            build_mesh(run, group.device),
        )
        shard_digest = sum(
            float(tensor.to_local().double().pow(2).sum())
            for tensor in cast(
                dict[str, torch.Tensor],
                model.state_dict(),
            ).values()
            if isinstance(tensor, DTensor)
        )
        buffer_digest = sum(
            float(buffer.double().pow(2).sum()) for buffer in model.buffers()
        )
        reports.put((rank, shard_digest, buffer_digest))


def ema_swap_rank(
    rank: int,
    group: Gloo,
    run: Workload,
    reports: SimpleQueue[tuple[float, float]],
) -> None:
    """Evaluate through ``EMA.swap`` the way ``train()`` does.

    Args:
        rank: This process's rank.
        group: Rendezvous details.
        run: The run description.
        reports: Queue receiving ``(swap_round_trip_gap, ema_weight_gap)``.
    """
    with gloo_rank(group, rank):
        batches = rank_batches(run)
        setup = build_hybrid_setup(run, group.device)
        _ = run_windows(run, setup, batches)
        raw_before = gather_state(model_state(setup))
        shadow = gather_state(setup.ema.state_dict())
        _ = take_step(batch=batches[0], model_params=setup, train_mode=False)
        setup.ema.swap(setup.model)
        _ = take_step(batch=batches[0], model_params=setup, train_mode=False)
        swapped = gather_state(model_state(setup))
        setup.ema.swap(setup.model)
        restored = gather_state(model_state(setup))
        reports.put(
            (state_gap(raw_before, restored), state_gap(shadow, swapped)),
        )


def checkpoint_rank(
    rank: int,
    group: Gloo,
    run: Workload,
    truth_path: pathlib.Path,
) -> None:
    """Save through the real ``save_checkpoint`` and record the true EMA.

    Args:
        rank: This process's rank.
        group: Rendezvous details.
        run: The run description.
        truth_path: Where rank 0 writes the gathered EMA weights. A file,
            because queued tensors do not outlive the worker that sent them.
    """
    with gloo_rank(group, rank):
        setup = build_hybrid_setup(run, group.device)
        _ = run_windows(run, setup, rank_batches(run))
        save_checkpoint(
            model_params=setup,
            rank=rank,
            log=structlog_logger(),
            best_val_loss=torch.tensor(_INFINITE_LOSS),
        )
        truth = gather_state(setup.ema.state_dict())
        if rank == 0:
            torch.save(truth, truth_path)
        _ = dist.barrier()  # pyright: ignore[reportUnknownMemberType]


def resume_rank(
    rank: int,
    group: Gloo,
    run: Workload,
    reports: SimpleQueue[float],
) -> None:
    """Gather state, reshard it, and compare one more step to no restart.

    This pins the recipe the production resume path should use: full tensors
    on disk, ``load_state_dict`` before sharding, ``distribute_tensor`` for
    Adam's per-parameter slots.

    Args:
        rank: This process's rank.
        group: Rendezvous details.
        run: The run description.
        reports: Queue receiving the weight gap after the extra step.
    """
    with gloo_rank(group, rank):
        device = group.device
        batches = rank_batches(run)
        setup = build_hybrid_setup(run, device)
        _ = run_windows(run, setup, batches)
        saved_weights = gather_state(model_state(setup))
        saved_scheduler = setup.scheduler.state_dict()
        optimizer_state = setup.optimizer.state_dict()
        saved_slots = {
            index: gather_state(slots)
            for index, slots in cast(
                dict[int, dict[str, torch.Tensor]],
                optimizer_state["state"],
            ).items()
        }
        _ = run_windows(run, setup, batches)
        uninterrupted = gather_state(model_state(setup))

        model = build_model(run.tcfg, device)
        _ = model.load_state_dict(saved_weights)
        model = shard_model(model, build_mesh(run, device))
        _ = model.train(run.train_mode)
        resumed = build_setup(model, run.tcfg, device)
        parameters = list(model.parameters())
        slots_by_index = {
            index: {
                name: reshard_like(value.to(device), parameters[index])
                for name, value in slots.items()
            }
            for index, slots in saved_slots.items()
        }
        resumed.optimizer.load_state_dict(
            {
                "param_groups": optimizer_state["param_groups"],
                "state": slots_by_index,
            },
        )
        resumed.scheduler.load_state_dict(saved_scheduler)
        _ = run_windows(run, resumed, batches)
        reports.put(
            state_gap(uninterrupted, gather_state(model_state(resumed))),
        )


def tiny_config(checkpoint_path: pathlib.Path, grad_clip: float) -> TrainConfig:
    """Build a small TrainConfig that runs quickly on one CPU thread.

    Args:
        checkpoint_path: Where ``save_checkpoint`` writes.
        grad_clip: Gradient clipping threshold.

    Returns:
        The configuration.
    """
    return TrainConfig(
        training=TrainingParams(grad_clip=grad_clip, lr_warmup_steps=0),
        model=ModelParams(
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
            max_residues=_N_RES,
            n_blocks_atom_transformer_encoder=_N_BLOCKS,
            n_heads_atom_transformer_encoder=_N_HEADS,
            n_blocks_atom_transformer_decoder=_N_BLOCKS,
            n_heads_atom_transformer_decoder=_N_HEADS,
            n_pairformer_blocks_template_embedder=_N_BLOCKS,
            n_paiformer_heads_template_embedder=_N_HEADS,
        ),
        distogram_template=TemplateDistogramParams(n_bins=_TEMPLATE_BINS),
        distogram_residue=ResidueDistogramParams(n_bins=_RESIDUE_BINS),
        distogram_atom=AtomDistogramParams(n_bins=_ATOM_BINS),
        checkpoint=CheckpointParams(checkpoint_path=checkpoint_path),
        logging=LoggingParams(use_wandb=False),
        test_loader=LoaderConfig(
            max_seq_length=_N_RES,
            batch_size=_BATCH_SIZE,
        ),
    )


@pytest.fixture
def gloo(tmp_path: pathlib.Path) -> Gloo:
    """Provide rendezvous details for a CPU group.

    Args:
        tmp_path: Directory holding the rendezvous store.

    Returns:
        Gloo details targeting the CPU.
    """
    return Gloo(store_path=tmp_path / "gloo", device=torch.device("cpu"))


@pytest.fixture
def scenario(request: pytest.FixtureRequest) -> ParityScenario:
    """Provide the parametrized scenario, or the accumulated default.

    Args:
        request: Carries the optional scenario.

    Returns:
        The scenario.
    """
    return cast(
        ParityScenario,
        getattr(request, "param", ParityScenario.accumulated),
    )


@pytest.fixture
def workload(scenario: ParityScenario, tmp_path: pathlib.Path) -> Workload:
    """Build a run from a scenario.

    Args:
        scenario: Accumulation, clipping, and dropout settings.
        tmp_path: Directory for the proteins and checkpoint.

    Returns:
        The workload, two windows of the scenario's micro-batches.
    """
    n_windows = 2
    run = Workload(
        tcfg=tiny_config(tmp_path / "best.pt", scenario.grad_clip),
        jsonl_path=tmp_path / "proteins.jsonl",
        n_windows=n_windows,
        micro_batches=scenario.micro_batches,
        batch_size=_BATCH_SIZE,
        train_mode=scenario.train_mode,
        mesh_shape=_MESH_SHAPE,
    )
    write_proteins(run.jsonl_path, _WORLD_SIZE * run.proteins_per_rank, _N_RES)
    return run


@pytest.fixture
def hook_calls(workload: Workload) -> set[type[nn.Module]]:
    """Run one real forward, recording module classes whose hooks fired.

    Args:
        workload: Supplies the configuration and proteins.

    Returns:
        Every recorded class among the six that the design shards.
    """
    fired: set[type[nn.Module]] = set()

    def record(module: nn.Module, _: tuple[object, ...]) -> None:
        """Record which class this hook fired on.

        Args:
            module: The module about to run.
        """
        fired.add(type(module))

    model = build_model(workload.tcfg, torch.device("cpu"))
    for module in model.modules():
        if isinstance(module, _SHARDED_CLASSES):
            _ = module.register_forward_pre_hook(record)
    with single_rank_group(workload.jsonl_path.parent / "single"):
        batch = rank_batches(workload)[0]
    with torch.no_grad():
        _ = model(batch)
    return fired


@contextlib.contextmanager
def single_rank_group(store_path: pathlib.Path) -> Generator[None, None, None]:
    """Join a one-rank gloo group for code that reads ``dist.get_rank``.

    Args:
        store_path: File backing the gloo store.

    Yields:
        None: The group is live for the duration of the block.
    """
    dist.init_process_group(
        backend="gloo",
        init_method=store_path.as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _queue() -> SimpleQueue[object]:
    """Return a queue usable across spawned processes."""
    return multiprocessing.get_context("spawn").SimpleQueue()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="required change 1: typed __call__ overrides bypass module hooks",
)
@pytest.mark.parametrize(
    "module_class",
    _SHARDED_CLASSES,
    ids=[module_class.__name__ for module_class in _SHARDED_CLASSES],
)
def test_forward_reaches_module_hooks(
    hook_calls: set[type[nn.Module]],
    module_class: type[nn.Module],
) -> None:
    """Every class the design shards must run FSDP's forward hooks.

    Arrange: a real ``MainTrunk`` with a pre-forward hook on the six classes.
    Act: one real forward pass.
    Assert: the hook fired for this class. FSDP unshards parameters from such
    a hook, so a class that bypasses it runs on sharded parameters and raises
    on mixed ``Tensor`` and ``DTensor`` inputs.

    Args:
        hook_calls: Classes whose hooks fired during the forward.
        module_class: The class under test.
    """
    assert module_class in hook_calls


@pytest.mark.parametrize(
    "scenario",
    list(ParityScenario),
    ids=[member.name for member in ParityScenario],
    indirect=True,
)
def test_hybrid_matches_ddp(
    scenario: ParityScenario,
    workload: Workload,
    gloo: Gloo,
) -> None:
    """Hybrid sharding must take the same optimizer steps as DDP.

    Arrange: identical seeded init, batches, and dropout streams.
    Act: four real processes run ``optimizer_step`` under DDP, then under a
    2x2 mesh.
    Assert: weights, EMA, losses, and component gradient norms agree on every
    rank. This covers Adam, clipping, accumulation, activation checkpointing,
    and the EMA update.

    Args:
        scenario: Decides whether hybrid syncs once per window.
        workload: The scenario's run.
        gloo: Rendezvous details.
    """
    reports = _queue()
    _ = spawn(
        parity_rank,
        args=(gloo, workload, scenario.window_sync, reports),
        nprocs=_WORLD_SIZE,
    )
    gaps = [cast(ParityGaps, reports.get()) for _ in range(_WORLD_SIZE)]
    for gap in gaps:
        assert max(dataclasses.astuple(gap)) < _PARITY_TOLERANCE


@pytest.mark.parametrize(
    ("seeded", "expected_distinct_digests"),
    [(True, 2), (False, _WORLD_SIZE)],
    ids=["seeded_replicas_agree", "unseeded_replicas_diverge"],
)
def test_init_seed_decides_whether_replicas_agree(
    workload: Workload,
    gloo: Gloo,
    seeded: bool,  # noqa: FBT001
    expected_distinct_digests: int,
) -> None:
    """FSDP does not broadcast init, so the seed alone keeps replicas equal.

    Arrange: seed identically on all ranks, or differently as ``main()``
    effectively does today by never seeding.
    Act: build and shard the model on four ranks.
    Assert: with a seed, the two shards are each held by two replicas, giving
    two distinct digests. Without one, all four ranks hold different shards,
    and no error is raised.

    Args:
        workload: The run description.
        gloo: Rendezvous details.
        seeded: Whether to seed identically.
        expected_distinct_digests: Distinct shard digests across four ranks.
    """
    reports = _queue()
    _ = spawn(
        replica_rank,
        args=(gloo, workload, seeded, reports),
        nprocs=_WORLD_SIZE,
    )
    rows = [
        cast(tuple[int, float, float], reports.get())
        for _ in range(_WORLD_SIZE)
    ]
    assert len({row[1] for row in rows}) == expected_distinct_digests


@pytest.mark.xfail(
    strict=True,
    raises=ProcessRaisedException,
    reason="required change 2: EMA.swap assumes plain named_parameters",
)
def test_ema_swap_after_no_grad_forward(
    workload: Workload,
    gloo: Gloo,
) -> None:
    """``EMA.swap`` must survive the root group staying unsharded.

    Arrange: a trained sharded model.
    Act: evaluate raw weights, swap, evaluate, swap back, as ``train()`` does.
    Assert: the swapped model holds the shadow, and the raw weights return.

    Args:
        workload: The run description.
        gloo: Rendezvous details.
    """
    reports = _queue()
    _ = spawn(
        ema_swap_rank,
        args=(gloo, workload, reports),
        nprocs=_WORLD_SIZE,
    )
    results = [
        cast(tuple[float, float], reports.get()) for _ in range(_WORLD_SIZE)
    ]
    assert all(result == (0.0, 0.0) for result in results)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="required change 4: save_checkpoint stores one rank's shards",
)
def test_checkpoint_loads_into_plain_model(
    workload: Workload,
    gloo: Gloo,
) -> None:
    """A checkpoint saved under sharding must load the way sampling does.

    Arrange: a trained sharded model.
    Act: the real ``save_checkpoint`` on all ranks, then the same
    ``torch.load`` and ``load_state_dict`` that ``sampling.py`` performs.
    Assert: the plain model holds the true, gathered EMA weights.

    Args:
        workload: The run description.
        gloo: Rendezvous details.
    """
    truth_path = workload.jsonl_path.parent / "truth.pt"
    _ = spawn(
        checkpoint_rank,
        args=(gloo, workload, truth_path),
        nprocs=_WORLD_SIZE,
    )
    truth = cast(
        dict[str, torch.Tensor],
        torch.load(truth_path, map_location="cpu", weights_only=True),
    )
    saved = cast(
        dict[str, dict[str, torch.Tensor]],
        torch.load(
            workload.tcfg.checkpoint.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        ),
    )
    plain = build_model(workload.tcfg, torch.device("cpu"))
    _ = plain.load_state_dict(saved["ema"])
    assert state_gap(truth, plain.state_dict()) == 0.0


def test_gathered_state_resumes_exactly(
    workload: Workload,
    gloo: Gloo,
) -> None:
    """Gathered checkpoints must resume as if training never stopped.

    Arrange: a sharded model trained for ``n_windows`` steps.
    Act: save full tensors, take more steps as the reference, then rebuild,
    reshard, restore Adam and the scheduler, and take the same steps.
    Assert: both paths end at the same weights on every rank.

    Args:
        workload: The run description.
        gloo: Rendezvous details.
    """
    reports = _queue()
    _ = spawn(
        resume_rank,
        args=(gloo, workload, reports),
        nprocs=_WORLD_SIZE,
    )
    gaps = [cast(float, reports.get()) for _ in range(_WORLD_SIZE)]
    assert max(gaps) < _PARITY_TOLERANCE
