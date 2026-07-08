"""Tests for LossMetrics, ThroughputStatistics, and ComponentNorms.

Covers valid construction from 0-D float tensors, field finiteness checks,
jaxtyping shape-enforcement for each dataclass, and cross-rank averaging via
TensorAccumulatorMixin.all_reduce_mean (including a real multi-process test
proving genuine cross-rank communication, not just a single-rank no-op).
"""

import contextlib
import dataclasses
import multiprocessing as mp
import os
import socket
from collections.abc import Generator, Sequence
from multiprocessing.process import BaseProcess
from typing import TypeVar, cast

import pytest
import torch
import torch.distributed as dist
from helpers.useful_objects import (
    ComponentNorms,
    LossMetrics,
    TensorAccumulatorMixin,
    ThroughputStatistics,
)
from jaxtyping import Float, TypeCheckError

_ALL_REDUCE_TOLERANCE = 1e-6
_ResultT = TypeVar("_ResultT")


def _scalar() -> Float[torch.Tensor, ""]:
    """Return a 0-D float tensor.

    Creates a scalar tensor with value 1.0 for use as a valid field in the
    shape-checked dataclasses under test.
    """
    return torch.tensor(1.0)


# ---------------------------------------------------------------------------
# LossMetrics — valid construction
# ---------------------------------------------------------------------------


@pytest.fixture
def loss_metrics() -> LossMetrics:
    """Provide a valid LossMetrics with all scalar tensors set to 1.0.

    Constructs a LossMetrics instance where every field is a 0-D float tensor
    with value 1.0, suitable for shape-contract and finiteness tests.
    """
    s = _scalar()
    return LossMetrics(
        total_loss=s,
        Kabsch_aligned_MSE_loss=s,
        CE_loss=s,
        smooth_lddt_loss=s,
        res_distogram_loss=s,
        atom_distogram_loss=s,
        intermediate_loss=s,
        RMSD=s,
    )


def test_loss_metrics_constructs(loss_metrics: LossMetrics) -> None:
    """LossMetrics constructs successfully from 0-D float tensors.

    Verifies that a LossMetrics instance is created without error when all
    fields are valid scalar tensors.
    """
    assert isinstance(loss_metrics, LossMetrics)


def test_loss_metrics_total_loss_is_scalar(loss_metrics: LossMetrics) -> None:
    """LossMetrics.total_loss is a 0-D tensor.

    Verifies that the total_loss field has zero dimensions, confirming the
    scalar shape contract is preserved after construction.
    """
    assert loss_metrics.total_loss.ndim == 0


def test_loss_metrics_all_fields_finite(loss_metrics: LossMetrics) -> None:
    """All LossMetrics tensor fields are finite.

    Verifies that every tensor field in LossMetrics contains only finite values
    when initialised from the fixture.
    """
    for field in [
        loss_metrics.total_loss,
        loss_metrics.Kabsch_aligned_MSE_loss,
        loss_metrics.CE_loss,
        loss_metrics.smooth_lddt_loss,
        loss_metrics.res_distogram_loss,
        loss_metrics.atom_distogram_loss,
        loss_metrics.intermediate_loss,
        loss_metrics.RMSD,
    ]:
        assert torch.isfinite(field)


# ---------------------------------------------------------------------------
# LossMetrics — shape enforcement
# ---------------------------------------------------------------------------


def test_loss_metrics_rejects_1d_total_loss() -> None:
    """LossMetrics raises when total_loss is 1-D instead of scalar.

    Verifies that passing a rank-1 tensor for total_loss triggers a
    jaxtyping TypeCheckError or equivalent exception.
    """
    s = _scalar()
    with pytest.raises((TypeCheckError, Exception)):
        _ = LossMetrics(
            total_loss=torch.tensor([1.0]),
            Kabsch_aligned_MSE_loss=s,
            CE_loss=s,
            smooth_lddt_loss=s,
            res_distogram_loss=s,
            atom_distogram_loss=s,
            intermediate_loss=s,
            RMSD=s,
        )


# ---------------------------------------------------------------------------
# ThroughputStatistics — valid construction
# ---------------------------------------------------------------------------


@pytest.fixture
def throughput_statistics() -> ThroughputStatistics:
    """Provide a valid ThroughputStatistics with all scalar tensors set to 1.0.

    Constructs a ThroughputStatistics instance where every field is a 0-D float
    tensor with value 1.0, suitable for shape-contract and finiteness tests.
    """
    s = _scalar()
    return ThroughputStatistics(
        avg_batch_size=s,
        token_pack_rate=s,
        residues_per_sec=s,
        atoms_per_sec=s,
    )


def test_throughput_statistics_constructs(
    throughput_statistics: ThroughputStatistics,
) -> None:
    """ThroughputStatistics constructs successfully from 0-D float tensors.

    Verifies that a ThroughputStatistics instance is created without error when
    all fields are valid scalar tensors.
    """
    assert isinstance(throughput_statistics, ThroughputStatistics)


def test_throughput_statistics_avg_batch_size_is_scalar(
    throughput_statistics: ThroughputStatistics,
) -> None:
    """ThroughputStatistics.avg_batch_size is a 0-D tensor.

    Verifies that the avg_batch_size field has zero dimensions, confirming the
    scalar shape contract is preserved after construction.
    """
    assert throughput_statistics.avg_batch_size.ndim == 0


def test_throughput_statistics_all_fields_finite(
    throughput_statistics: ThroughputStatistics,
) -> None:
    """All ThroughputStatistics tensor fields are finite.

    Verifies every tensor field in ThroughputStatistics contains only finite
    values when initialised from the fixture.
    """
    for field in [
        throughput_statistics.avg_batch_size,
        throughput_statistics.token_pack_rate,
        throughput_statistics.residues_per_sec,
        throughput_statistics.atoms_per_sec,
    ]:
        assert torch.isfinite(field)


# ---------------------------------------------------------------------------
# ThroughputStatistics — shape enforcement
# ---------------------------------------------------------------------------


def test_throughput_statistics_rejects_1d_avg_batch_size() -> None:
    """ThroughputStatistics raises when avg_batch_size is 1-D instead of scalar.

    Verifies that passing a rank-1 tensor for avg_batch_size triggers a
    jaxtyping TypeCheckError or equivalent exception.
    """
    s = _scalar()
    with pytest.raises((TypeCheckError, Exception)):
        _ = ThroughputStatistics(
            avg_batch_size=torch.tensor([1.0]),
            token_pack_rate=s,
            residues_per_sec=s,
            atoms_per_sec=s,
        )


# ---------------------------------------------------------------------------
# ComponentNorms — valid construction
# ---------------------------------------------------------------------------


@pytest.fixture
def component_norms() -> ComponentNorms:
    """Provide a valid ComponentNorms with all scalar tensors set to 1.0.

    Constructs a ComponentNorms instance where every field is a 0-D float tensor
    with value 1.0, suitable for shape-contract and finiteness tests.
    """
    s = _scalar()
    return ComponentNorms(
        template_embedder=s,
        atom_encoder=s,
        atom_decoders=s,
        residue_distogram_head=s,
        atom_distogram_head=s,
        inter_proj_seq=s,
        inter_seq_logits=s,
        proj_seq=s,
        seq_logits=s,
    )


def test_component_norms_constructs(component_norms: ComponentNorms) -> None:
    """ComponentNorms constructs successfully from 0-D float tensors.

    Verifies that a ComponentNorms instance is created without error when all
    fields are valid scalar tensors.
    """
    assert isinstance(component_norms, ComponentNorms)


def test_component_norms_template_embedder_is_scalar(
    component_norms: ComponentNorms,
) -> None:
    """ComponentNorms.template_embedder is a 0-D tensor.

    Verifies the template_embedder field has zero dimensions, confirming the
    scalar shape contract is preserved after construction.
    """
    assert component_norms.template_embedder.ndim == 0


def test_component_norms_all_fields_finite(
    component_norms: ComponentNorms,
) -> None:
    """All ComponentNorms tensor fields are finite.

    Verifies every tensor field in ComponentNorms contains only finite values
    when initialised from the fixture.
    """
    for field in [
        component_norms.template_embedder,
        component_norms.atom_encoder,
        component_norms.atom_decoders,
        component_norms.residue_distogram_head,
        component_norms.atom_distogram_head,
        component_norms.inter_proj_seq,
        component_norms.inter_seq_logits,
        component_norms.proj_seq,
        component_norms.seq_logits,
    ]:
        assert torch.isfinite(field)


# ---------------------------------------------------------------------------
# ComponentNorms — shape enforcement
# ---------------------------------------------------------------------------


def test_component_norms_rejects_1d_template_embedder() -> None:
    """ComponentNorms raises when template_embedder is 1-D instead of scalar.

    Verifies that passing a rank-1 tensor for template_embedder triggers a
    jaxtyping TypeCheckError or equivalent exception.
    """
    s = _scalar()
    with pytest.raises((TypeCheckError, Exception)):
        _ = ComponentNorms(
            template_embedder=torch.tensor([1.0]),
            atom_encoder=s,
            atom_decoders=s,
            residue_distogram_head=s,
            atom_distogram_head=s,
            inter_proj_seq=s,
            inter_seq_logits=s,
            proj_seq=s,
            seq_logits=s,
        )


# ---------------------------------------------------------------------------
# TensorAccumulatorMixin.all_reduce_mean
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _single_process_gloo_group() -> Generator[None]:
    """Init and tear down a single-process gloo group for DDP-adjacent tests."""
    # HashStore is a private torch.distributed C-extension symbol that
    # isn't officially re-exported; getattr() sidesteps the static
    # private-import check, whose result is otherwise inconsistent
    # across torch's per-platform packaging.
    dist.init_process_group(
        backend="gloo",
        store=getattr(  # noqa: B009  # pyright: ignore[reportAny]
            dist,
            "HashStore",
        )(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "fixture_name",
    ["loss_metrics", "throughput_statistics", "component_norms"],
    ids=["LossMetrics", "ThroughputStatistics", "ComponentNorms"],
)
def test_all_reduce_mean_noop_without_process_group(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """all_reduce_mean() is a no-op when no distributed process group exists.

    Arrange: take a metrics instance (every field 1.0) with no active
    distributed process group -- the default state for ordinary
    single-device test execution.
    Act: call all_reduce_mean().
    Assert: every field is bit-for-bit unchanged, confirming single-device
    training never attempts a collective call that would hang or error
    without a process group.
    """
    metrics = cast(
        TensorAccumulatorMixin,
        request.getfixturevalue(fixture_name),
    )
    assert not dist.is_initialized()
    before = {
        f.name: cast(torch.Tensor, getattr(metrics, f.name)).clone()
        for f in dataclasses.fields(metrics)
    }

    metrics.all_reduce_mean()

    for f in dataclasses.fields(metrics):
        assert torch.equal(
            cast(torch.Tensor, getattr(metrics, f.name)),
            before[f.name],
        )


@pytest.mark.parametrize(
    "fixture_name",
    ["loss_metrics", "throughput_statistics", "component_norms"],
    ids=["LossMetrics", "ThroughputStatistics", "ComponentNorms"],
)
def test_all_reduce_mean_unchanged_with_single_rank_process_group(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """all_reduce_mean() runs the real collective path at world_size=1.

    Arrange: enter a real (single-process) gloo process group and take a
    metrics instance (every field 1.0).
    Act: call all_reduce_mean() with an actual process group active.
    Assert: every field is unchanged (the mean of one value is itself).
    This only proves the collective call path doesn't crash when a group
    is active -- it cannot by itself distinguish "correctly computed the
    mean of one rank" from "silently skipped the reduction", since both
    produce the same output at world_size=1. That distinction is what
    test_all_reduce_mean_averages_across_real_ranks below proves.
    """
    metrics = cast(
        TensorAccumulatorMixin,
        request.getfixturevalue(fixture_name),
    )
    before = {
        f.name: cast(torch.Tensor, getattr(metrics, f.name)).clone()
        for f in dataclasses.fields(metrics)
    }

    with _single_process_gloo_group():
        metrics.all_reduce_mean()

    for f in dataclasses.fields(metrics):
        assert torch.equal(
            cast(torch.Tensor, getattr(metrics, f.name)),
            before[f.name],
        )


def _free_port() -> int:
    """Return an OS-assigned free TCP port by briefly binding to it.

    Returns:
        A port number free at the moment of the call, for process-group
        rendezvous in the multiprocess test below.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


def _all_reduce_mean_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: "mp.Queue[float]",
) -> None:
    """Multiprocess worker for test_all_reduce_mean_averages_across_real_ranks.

    Builds a LossMetrics whose every field equals this process's own rank
    index, calls all_reduce_mean() inside a real gloo process group shared
    with the other worker processes, and reports the resulting total_loss
    back to the parent process so it can confirm every rank converged to
    the same, correctly-computed global mean.

    Args:
        rank: This process's rank.
        world_size: Total number of ranks in the process group.
        port: TCP port used for process-group rendezvous.
        result_queue: Multiprocessing queue to report the post-reduce value.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    rank_value = torch.tensor(float(rank))
    metrics = LossMetrics(
        total_loss=rank_value.clone(),
        Kabsch_aligned_MSE_loss=rank_value.clone(),
        CE_loss=rank_value.clone(),
        smooth_lddt_loss=rank_value.clone(),
        res_distogram_loss=rank_value.clone(),
        atom_distogram_loss=rank_value.clone(),
        intermediate_loss=rank_value.clone(),
        RMSD=rank_value.clone(),
    )
    metrics.all_reduce_mean()
    result_queue.put(metrics.total_loss.item())
    dist.destroy_process_group()


def _start_and_collect(
    processes: Sequence[BaseProcess],
    result_queue: "mp.Queue[_ResultT]",
    world_size: int,
) -> list[_ResultT]:
    """Start every worker process and collect their reported results.

    Args:
        processes: Worker processes to start.
        result_queue: Queue each worker reports its result to.
        world_size: Expected number of results to collect.

    Returns:
        One reported result per worker, in receipt order.
    """
    for p in processes:
        p.start()
    return [result_queue.get(timeout=60) for _ in range(world_size)]


def test_all_reduce_mean_averages_across_real_ranks() -> None:
    """all_reduce_mean() averages a field across genuinely separate processes.

    Arrange: spawn WORLD_SIZE=3 real OS processes, each building a
    LossMetrics with every field set to its own rank (0, 1, 2) -- so the
    correct global mean is (0+1+2)/3 = 1.0, a value none of the three
    ranks started with.
    Act: each process calls all_reduce_mean() inside a real gloo process
    group and reports its resulting total_loss back to this parent
    process via a multiprocessing queue.
    Assert: every rank's post-reduce total_loss equals 1.0 -- proving the
    reduction actually communicated across process boundaries, which a
    broken implementation (e.g. one that silently no-ops, or divides by
    the wrong denominator) cannot fake, since each rank started with a
    different value only real inter-process communication could unify.
    """
    world_size = 3
    port = _free_port()
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[float]" = ctx.Queue()
    processes = [
        ctx.Process(
            target=_all_reduce_mean_worker,
            args=(rank, world_size, port, result_queue),
        )
        for rank in range(world_size)
    ]
    try:
        results = _start_and_collect(processes, result_queue, world_size)
    finally:
        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

    expected_mean = sum(range(world_size)) / world_size
    for result in results:
        assert abs(result - expected_mean) < _ALL_REDUCE_TOLERANCE


_PRIOR_COLLECTIVE_CALLS = 3


def _multi_metric_all_reduce_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: "mp.Queue[tuple[float, float, float]]",
) -> None:
    """Multiprocess worker: all_reduce_mean() after prior collective calls.

    Simulates the realistic DDP pattern where other collectives (e.g.
    gradient all-reduces during backward) already exercised the process
    group before any epoch-metrics reduction runs: performs a few generic
    all_reduce calls on scratch tensors first, then builds a LossMetrics,
    ThroughputStatistics, and ComponentNorms -- every field set to this
    rank's own value -- and calls all_reduce_mean() on each in turn,
    reporting one representative field per class back to the parent.

    Args:
        rank: This process's rank.
        world_size: Total number of ranks in the process group.
        port: TCP port used for process-group rendezvous.
        result_queue: Multiprocessing queue to report the post-reduce
            ``(total_loss, avg_batch_size, template_embedder)`` triple.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    for _ in range(_PRIOR_COLLECTIVE_CALLS):
        scratch = torch.tensor(float(rank))
        _ = dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
            scratch,
            op=dist.ReduceOp.SUM,
        )

    rank_value = torch.tensor(float(rank))
    worker_loss_metrics = LossMetrics(
        total_loss=rank_value.clone(),
        Kabsch_aligned_MSE_loss=rank_value.clone(),
        CE_loss=rank_value.clone(),
        smooth_lddt_loss=rank_value.clone(),
        res_distogram_loss=rank_value.clone(),
        atom_distogram_loss=rank_value.clone(),
        intermediate_loss=rank_value.clone(),
        RMSD=rank_value.clone(),
    )
    worker_throughput_stats = ThroughputStatistics(
        avg_batch_size=rank_value.clone(),
        token_pack_rate=rank_value.clone(),
        residues_per_sec=rank_value.clone(),
        atoms_per_sec=rank_value.clone(),
    )
    worker_component_norms = ComponentNorms(
        template_embedder=rank_value.clone(),
        atom_encoder=rank_value.clone(),
        atom_decoders=rank_value.clone(),
        residue_distogram_head=rank_value.clone(),
        atom_distogram_head=rank_value.clone(),
        inter_proj_seq=rank_value.clone(),
        inter_seq_logits=rank_value.clone(),
        proj_seq=rank_value.clone(),
        seq_logits=rank_value.clone(),
    )

    worker_loss_metrics.all_reduce_mean()
    worker_throughput_stats.all_reduce_mean()
    worker_component_norms.all_reduce_mean()

    result_queue.put(
        (
            worker_loss_metrics.total_loss.item(),
            worker_throughput_stats.avg_batch_size.item(),
            worker_component_norms.template_embedder.item(),
        ),
    )
    dist.destroy_process_group()


def test_all_reduce_mean_after_prior_collective_calls() -> None:
    """all_reduce_mean() works correctly after other collectives already ran.

    Arrange: spawn WORLD_SIZE=3 real processes. Each rank first performs
    several generic all_reduce calls on scratch tensors -- mimicking the
    common DDP pattern where gradient synchronization (or other
    collectives) already exercised the process group before any
    epoch-metrics reduction runs -- then builds a LossMetrics,
    ThroughputStatistics, and ComponentNorms, every field set to its own
    rank value.
    Act: each rank calls all_reduce_mean() on all three metric instances,
    in the same order on every rank (collectives require every rank to
    issue the same sequence of calls, or the process group hangs or
    mismatches).
    Assert: every rank converges to the correct global mean (1.0) for all
    three metric types, proving prior collective traffic doesn't leave
    the process group in a state that corrupts, hangs, or desynchronizes
    subsequent all_reduce_mean() calls -- and that all_reduce_mean()
    itself behaves identically on all three metric dataclasses, not just
    LossMetrics.
    """
    world_size = 3
    port = _free_port()
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[tuple[float, float, float]]" = ctx.Queue()
    processes = [
        ctx.Process(
            target=_multi_metric_all_reduce_worker,
            args=(rank, world_size, port, result_queue),
        )
        for rank in range(world_size)
    ]
    try:
        results = _start_and_collect(processes, result_queue, world_size)
    finally:
        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()

    expected_mean = sum(range(world_size)) / world_size
    for loss_total, throughput_avg, norms_template in results:
        assert abs(loss_total - expected_mean) < _ALL_REDUCE_TOLERANCE
        assert abs(throughput_avg - expected_mean) < _ALL_REDUCE_TOLERANCE
        assert abs(norms_template - expected_mean) < _ALL_REDUCE_TOLERANCE
