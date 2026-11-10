"""Tests for TensorAccumulatorMixin and the metric dataclasses built on it.

Covers the mixin's field-wise arithmetic, ``weighted_avg``, ``zero_init`` and
``to_float_dict`` with exact values on ThroughputStatistics, whose four
fields hold distinct values; the scalar shape contract that
``@jaxtyped`` enforces on LossMetrics, ThroughputStatistics and
ComponentNorms; and ``all_reduce_mean`` with no process group, a one-rank
group, and three real processes.

Every expected value is exactly representable in float32, so results are
compared with ``==`` rather than a tolerance.
"""

import dataclasses
import enum
import multiprocessing as mp
import os
import socket
from collections.abc import Callable
from typing import cast

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


@dataclasses.dataclass(frozen=True)
class ScalarCase:
    """A mixin operator with a scalar operand and the values it must give.

    Attributes:
        operation: Unbound operator, called as ``operation(left, scalar)``
            with ``left = (1, 2, 3, 4)``.
        scalar: Scalar operand.
        expected: Field values of the result, in field order.
    """

    operation: Callable[[ThroughputStatistics, float], ThroughputStatistics]
    scalar: float
    expected: tuple[float, ...]


class ScalarScenario(enum.Enum):
    """Scalar operators of TensorAccumulatorMixin.

    Members:
        divide: ``left /= 2`` halves every field in place.
        multiply: ``left * 3`` scales every field into a new instance.
    """

    divide = ScalarCase(ThroughputStatistics.__itruediv__, 2, (0.5, 1, 1.5, 2))
    multiply = ScalarCase(ThroughputStatistics.__mul__, 3, (3, 6, 9, 12))


@dataclasses.dataclass(frozen=True)
class RankReport:
    """Reduced field values one spawned rank sent back, by reduction.

    Each field maps metric names to values, as ``to_float_dict`` returns
    them.

    Attributes:
        rank: Rank that sent the report; every field it reduced started
            at ``float(rank)``.
        fresh_loss: LossMetrics reduced on the freshly created group.
        loss: LossMetrics reduced after the scratch ``all_reduce`` calls.
        throughput: ThroughputStatistics reduced after the scratch calls.
        norms: ComponentNorms reduced after the scratch calls.
    """

    rank: int
    fresh_loss: dict[str, float]
    loss: dict[str, float]
    throughput: dict[str, float]
    norms: dict[str, float]

    def values(self) -> set[float]:
        """Return every reduced value in the report.

        Returns:
            The values of all four reductions, merged into one set.
        """
        reductions = (self.fresh_loss, self.loss, self.throughput, self.norms)
        return {v for reduction in reductions for v in reduction.values()}


@dataclasses.dataclass(frozen=True)
class AllReduceRun:
    """Reports from every spawned rank after reducing.

    Attributes:
        world_size: Number of ranks spawned.
        reports: One report per rank, in the order they reached the queue.
    """

    world_size: int
    reports: tuple[RankReport, ...]


@dataclasses.dataclass(frozen=True)
class ProcessGroupCase:
    """Process-group state a single-process ``all_reduce_mean`` runs under.

    Attributes:
        group_active: Whether a one-rank gloo group is started first.
    """

    group_active: bool


class ProcessGroup(enum.Enum):
    """Single-process settings for ``all_reduce_mean``.

    Members:
        no_group: No process group exists, so the method must return before
            any collective call.
        single_rank_group: A one-rank gloo group exists, so the real
            collective path runs over a world of one.
    """

    no_group = ProcessGroupCase(group_active=False)
    single_rank_group = ProcessGroupCase(group_active=True)


@dataclasses.dataclass(frozen=True)
class FieldOverride:
    """One field of a metric dataclass and a replacement value for it.

    Attributes:
        name: Name of the field to replace.
        value: Replacement tensor; 1-D, so it breaks the field's scalar
            shape contract.
    """

    name: str
    value: Float[torch.Tensor, " n"]


def metrics_filled_with(
    metric_class: type[TensorAccumulatorMixin],
    value: float,
) -> TensorAccumulatorMixin:
    """Build ``metric_class`` with every field a separate 0-D ``value``.

    The instance comes from the class's own ``zero_init``, and each field
    is then set to a fresh ``torch.tensor(value)``.

    Args:
        metric_class: Metric dataclass to build.
        value: Starting value of every field.

    Returns:
        Instance whose fields are all ``torch.tensor(value)``.
    """
    metrics = metric_class.zero_init(torch.device("cpu"))
    for field in dataclasses.fields(metrics):
        setattr(metrics, field.name, torch.tensor(value))
    return metrics


def reduce_at_rank(
    metric_class: type[TensorAccumulatorMixin],
    rank: int,
) -> dict[str, float]:
    """Build ``metric_class`` with every field at ``rank`` and reduce it.

    ``metrics_filled_with`` sets each field to ``float(rank)``, so every
    rank contributes its own
    number to the mean. Called by ``all_reduce_mean_worker`` inside a
    spawned rank, after the process group exists.

    Args:
        metric_class: Metric dataclass to build and reduce.
        rank: This process's rank, used as every field's starting value.

    Returns:
        The reduced instance's ``to_float_dict``.
    """
    metrics = metrics_filled_with(metric_class, float(rank))
    metrics.all_reduce_mean()
    return metrics.to_float_dict()


def all_reduce_mean_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: "mp.Queue[RankReport]",
) -> None:
    """Spawned rank for ``test_all_reduce_mean_averages_across_real_ranks``.

    A module-level function rather than a fixture because ``spawn`` pickles
    the target by its qualified name. The rank first reduces a LossMetrics
    on the fresh group, then runs three plain ``all_reduce`` calls on
    scratch tensors, as gradient syncs would during backward, and then
    reduces a LossMetrics, a ThroughputStatistics and a ComponentNorms.
    Every field of every instance starts at this rank's own value. The
    same sequence of collectives runs on every rank, as collectives
    require.

    Args:
        rank: This process's rank.
        world_size: Total number of ranks in the process group.
        port: TCP port used for process-group rendezvous.
        result_queue: Receives this rank's report.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    fresh_loss = reduce_at_rank(LossMetrics, rank)
    for _ in range(3):
        _ = dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
            torch.tensor(float(rank)),
            op=dist.ReduceOp.SUM,
        )
    loss = reduce_at_rank(LossMetrics, rank)
    throughput = reduce_at_rank(ThroughputStatistics, rank)
    norms = reduce_at_rank(ComponentNorms, rank)
    result_queue.put(RankReport(rank, fresh_loss, loss, throughput, norms))
    dist.destroy_process_group()


@pytest.fixture(
    params=[LossMetrics, ThroughputStatistics, ComponentNorms],
    ids=["LossMetrics", "ThroughputStatistics", "ComponentNorms"],
)
def metric_cls(request: pytest.FixtureRequest) -> type[TensorAccumulatorMixin]:
    """Run the requesting test once per real metric dataclass.

    Args:
        request: Pytest request carrying the class in ``param``.

    Returns:
        The metric dataclass under test.
    """
    return cast(type[TensorAccumulatorMixin], request.param)


@pytest.fixture
def metrics_of_ones(
    metric_cls: type[TensorAccumulatorMixin],
) -> TensorAccumulatorMixin:
    """Build ``metric_cls`` with every field a separate 0-D tensor of 1.0.

    Args:
        metric_cls: Metric dataclass to build.

    Returns:
        Instance whose fields are all ``torch.tensor(1.0)``.
    """
    return metrics_filled_with(metric_cls, 1.0)


@pytest.fixture
def vector_field(metric_cls: type[TensorAccumulatorMixin]) -> FieldOverride:
    """Return a 1-D tensor aimed at the first scalar field of ``metric_cls``.

    Only that one field is replaced, and every other field stays a valid
    0-D tensor, so the scalar shape of one field is the only contract the
    test breaks.

    Args:
        metric_cls: Metric dataclass whose first field is targeted.

    Returns:
        The first field's name with the 1-D tensor ``[1.0]``.
    """
    first_field = dataclasses.fields(metric_cls)[0]
    return FieldOverride(name=first_field.name, value=torch.tensor([1.0]))


@pytest.fixture
def process_group(request: pytest.FixtureRequest) -> ProcessGroup:
    """Start a one-rank gloo group when the scenario asks for one.

    Args:
        request: Pytest request carrying the ``ProcessGroup`` member,
            passed with ``indirect=True``.

    Returns:
        The scenario, with its group already started if it has one.
    """
    scenario = cast(ProcessGroup, request.param)
    if scenario.value.group_active:
        request.getfixturevalue("single_process_gloo_group")
    return scenario


@pytest.fixture
def left() -> ThroughputStatistics:
    """Return the left operand ``(1, 2, 3, 4)``.

    Distinct values per field catch an operation applied to the wrong
    field. Built by hand rather than from a factory fixture: ``left`` and
    ``right`` are the only two instances, both fixed, and naming each
    field beside its value is easier to check than a factory's positional
    arguments.

    Returns:
        Fresh metrics; operations may mutate them in place.
    """
    return ThroughputStatistics(
        avg_batch_size=torch.tensor(1.0),
        token_pack_rate=torch.tensor(2.0),
        residues_per_sec=torch.tensor(3.0),
        atoms_per_sec=torch.tensor(4.0),
    )


@pytest.fixture
def right() -> ThroughputStatistics:
    """Return the right operand ``(10, 20, 30, 40)``.

    Built by hand for the same reason as ``left``.

    Returns:
        Fresh metrics.
    """
    return ThroughputStatistics(
        avg_batch_size=torch.tensor(10.0),
        token_pack_rate=torch.tensor(20.0),
        residues_per_sec=torch.tensor(30.0),
        atoms_per_sec=torch.tensor(40.0),
    )


@pytest.fixture
def free_port() -> int:
    """Return an OS-assigned TCP port that was free when the fixture ran.

    Returns:
        Port number for process-group rendezvous.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])


@pytest.fixture
def all_reduce_run(free_port: int) -> AllReduceRun:
    """Spawn three real ranks that each run ``all_reduce_mean_worker``.

    Three ranks starting at 0, 1 and 2 have a mean of 1.0, a value no rank
    started with, so a reduction that silently no-ops or divides by the
    wrong count cannot produce it.

    Args:
        free_port: Port for the ranks' rendezvous.

    Returns:
        Every rank's report.
    """
    world_size = 3
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[RankReport]" = ctx.Queue()
    processes = [
        ctx.Process(
            target=all_reduce_mean_worker,
            args=(rank, world_size, free_port, result_queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    try:
        reports = tuple(result_queue.get(timeout=60) for _ in processes)
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
    return AllReduceRun(world_size=world_size, reports=reports)


def test_in_place_add_sums_field_wise(
    left: ThroughputStatistics,
    right: ThroughputStatistics,
) -> None:
    """``left += right`` adds each field of ``right`` to the same field.

    Not parametrized: every field holds a distinct value, so a field
    mix-up or a non-additive result would both show.

    Args:
        left: Left operand ``(1, 2, 3, 4)``, updated in place.
        right: Right operand ``(10, 20, 30, 40)``.
    """
    left += right
    assert tuple(left.to_float_dict().values()) == (11.0, 22.0, 33.0, 44.0)


@pytest.mark.parametrize(
    "scalar_scenario",
    list(ScalarScenario),
    ids=[case.name for case in ScalarScenario],
)
def test_scalar_operations_scale_every_field(
    left: ThroughputStatistics,
    scalar_scenario: ScalarScenario,
) -> None:
    """``/= scalar`` and ``* scalar`` apply the scalar to every field.

    Args:
        left: Operand ``(1, 2, 3, 4)``.
        scalar_scenario: Operator, scalar and expected field values.
    """
    case = scalar_scenario.value
    result = case.operation(left, case.scalar)
    assert tuple(result.to_float_dict().values()) == case.expected


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        pytest.param([1, 1], (5.5, 11.0, 16.5, 22.0), id="equal_weights"),
        pytest.param([2, 1], (4.0, 8.0, 12.0, 16.0), id="left_weighted_twice"),
        pytest.param([0, 1], (10.0, 20.0, 30.0, 40.0), id="left_weight_zero"),
    ],
)
def test_weighted_avg_weights_each_instance_by_protein_count(
    left: ThroughputStatistics,
    right: ThroughputStatistics,
    weights: list[int],
    expected: tuple[float, float, float, float],
) -> None:
    """``weighted_avg`` returns the protein-count-weighted mean per field.

    With left ``(1, 2, 3, 4)`` and right ``(10, 20, 30, 40)``, weights 2
    and 1 give ``(2 * 1 + 10) / 3 = 4`` in the first field; a zero weight
    drops that instance entirely. The result is built through the class's
    ``@jaxtyped`` constructor, so a wrongly shaped field would raise.

    Args:
        left: First instance ``(1, 2, 3, 4)``.
        right: Second instance ``(10, 20, 30, 40)``.
        weights: Protein count for each instance.
        expected: Field values of the weighted mean, in field order.
    """
    result = ThroughputStatistics.weighted_avg([left, right], weights)
    assert tuple(result.to_float_dict().values()) == expected


@pytest.mark.parametrize("prefix", ["", "train/"], ids=["no_prefix", "train"])
def test_to_float_dict_prefixes_every_key(
    left: ThroughputStatistics,
    prefix: str,
) -> None:
    """``to_float_dict(prefix)`` keys each field ``prefix + name``, in order.

    Args:
        left: Instance to convert.
        prefix: String prepended to each field name.
    """
    names = [f.name for f in dataclasses.fields(ThroughputStatistics)]
    assert list(left.to_float_dict(prefix)) == [prefix + n for n in names]


def test_metrics_zero_init_fields_are_float32_scalars(
    metric_cls: type[TensorAccumulatorMixin],
) -> None:
    """``zero_init`` gives every field a 0-D float32 zero.

    Parametrized over the real metric classes through ``metric_cls``; CPU
    is the only device available in every test environment.

    Args:
        metric_cls: Metric dataclass to build.
    """
    metrics = metric_cls.zero_init(torch.device("cpu"))
    tensors = [
        cast(torch.Tensor, getattr(metrics, f.name))
        for f in dataclasses.fields(metrics)
    ]
    assert {(t.shape, t.dtype) for t in tensors} == {
        (torch.Size([]), torch.float32),
    }
    assert set(metrics.to_float_dict().values()) == {0.0}


def test_metrics_reject_non_scalar_field(
    metrics_of_ones: TensorAccumulatorMixin,
    vector_field: FieldOverride,
) -> None:
    """``@jaxtyped`` rejects a 1-D tensor in a scalar field at construction.

    Parametrized over the real metric classes through ``metric_cls``.
    ``dataclasses.replace`` builds a new instance through the class's
    ``@jaxtyped`` ``__init__`` with one field swapped for the 1-D tensor,
    so the construction expected to raise happens inside ``pytest.raises``.

    Args:
        metrics_of_ones: Valid instance whose other fields are kept.
        vector_field: Name of the field to replace and its 1-D tensor.
    """
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            metrics_of_ones,
            **{vector_field.name: vector_field.value},
        )


@pytest.mark.parametrize(
    "process_group",
    list(ProcessGroup),
    ids=[case.name for case in ProcessGroup],
    indirect=True,
)
def test_all_reduce_mean_keeps_values_on_one_process(
    process_group: ProcessGroup,
    metrics_of_ones: TensorAccumulatorMixin,
) -> None:
    """With no group or a one-rank group, every field stays at 1.0.

    With no group, ``all_reduce_mean`` must return before any collective
    call, which would otherwise raise. With a one-rank group the real
    collective path runs and the mean of one value is itself. Neither case
    can tell a correct reduction from a skipped one; the three-process test
    below does that.

    Args:
        process_group: Whether a one-rank gloo group is active.
        metrics_of_ones: Instance with every field 1.0.
    """
    assert dist.is_initialized() == process_group.value.group_active
    metrics_of_ones.all_reduce_mean()
    assert set(metrics_of_ones.to_float_dict().values()) == {1.0}


def test_all_reduce_mean_averages_across_real_ranks(
    all_reduce_run: AllReduceRun,
) -> None:
    """Every field on every rank converges to the mean of the rank values.

    Not parametrized: spawning ranks re-imports torch in each, which takes
    seconds, so one process group covers every case. The reductions on a
    fresh group and after unrelated collectives are consecutive stages of
    that group, and each metric class is reduced in turn within it.

    Args:
        all_reduce_run: Field values every rank reported.
    """
    world_size = all_reduce_run.world_size
    expected = sum(range(world_size)) / world_size
    reports = all_reduce_run.reports
    assert sorted(report.rank for report in reports) == list(range(world_size))
    assert {v for report in reports for v in report.values()} == {expected}
