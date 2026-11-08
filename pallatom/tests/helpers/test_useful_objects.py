"""Tests for TensorAccumulatorMixin and the metric dataclasses built on it.

Covers the mixin's field-wise arithmetic, ``weighted_avg``, ``zero_init`` and
``to_float_dict`` with exact values; the scalar shape contract that
``@jaxtyped`` enforces on LossMetrics, ThroughputStatistics and
ComponentNorms; and ``all_reduce_mean`` with no process group, a one-rank
group, and three real processes.

Every expected value is exactly representable in float32, so results are
compared with ``==`` rather than a tolerance.
"""

import dataclasses
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


@dataclasses.dataclass
class PairMetrics(TensorAccumulatorMixin):
    """Two-field accumulator so mixin arithmetic can be checked by hand.

    The mixin's behaviour does not depend on which fields a subclass
    declares, so two fields holding distinct values are enough to catch a
    field mix-up.

    Attributes:
        first: First scalar field.
        second: Second scalar field.
    """

    first: Float[torch.Tensor, ""]
    second: Float[torch.Tensor, ""]


@dataclasses.dataclass(frozen=True)
class AllReduceRun:
    """Field values every spawned rank reported after reducing.

    Attributes:
        world_size: Number of ranks spawned; rank ``r`` started at ``r``.
        rank_reports: One tuple of reduced field values per rank.
    """

    world_size: int
    rank_reports: tuple[tuple[float, ...], ...]


def all_reduce_mean_worker(
    rank: int,
    world_size: int,
    port: int,
    metric_classes: tuple[type[TensorAccumulatorMixin], ...],
    result_queue: "mp.Queue[tuple[float, ...]]",
) -> None:
    """Spawned rank for ``test_all_reduce_mean_averages_across_real_ranks``.

    A module-level function rather than a fixture because ``spawn`` pickles
    the target by its qualified name. The rank first reduces a LossMetrics
    on the fresh group, then runs three plain ``all_reduce`` calls on
    scratch tensors, as gradient syncs would during backward, and then
    reduces one instance of each class in ``metric_classes``. Every field
    of every instance starts at this rank's own value. The same sequence
    of collectives runs on every rank, as collectives require.

    Args:
        rank: This process's rank.
        world_size: Total number of ranks in the process group.
        port: TCP port used for process-group rendezvous.
        metric_classes: Classes to build and reduce after the scratch calls.
        result_queue: Receives every reduced field value from this rank.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    reduced: list[float] = []

    def reduce_fields(metric_class: type[TensorAccumulatorMixin]) -> None:
        """Build ``metric_class`` at this rank's value, reduce, record fields.

        Args:
            metric_class: Metric dataclass to build and reduce.
        """
        constructor: Callable[..., TensorAccumulatorMixin] = metric_class
        names = [f.name for f in dataclasses.fields(metric_class)]
        metrics = constructor(**{n: torch.tensor(float(rank)) for n in names})
        metrics.all_reduce_mean()
        reduced.extend(metrics.to_float_dict().values())

    reduce_fields(LossMetrics)
    for _ in range(3):
        _ = dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
            torch.tensor(float(rank)),
            op=dist.ReduceOp.SUM,
        )
    for metric_class in metric_classes:
        reduce_fields(metric_class)
    result_queue.put(tuple(reduced))
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
    constructor: Callable[..., TensorAccumulatorMixin] = metric_cls
    names = [f.name for f in dataclasses.fields(metric_cls)]
    return constructor(**{name: torch.tensor(1.0) for name in names})


@pytest.fixture
def build_with_vector_field(
    metric_cls: type[TensorAccumulatorMixin],
) -> Callable[[], TensorAccumulatorMixin]:
    """Return a thunk building ``metric_cls`` with its first field 1-D.

    Every other field is a valid 0-D tensor, so the only contract broken is
    the scalar shape of one field.

    Args:
        metric_cls: Metric dataclass to build.

    Returns:
        Zero-argument callable that attempts the construction.
    """
    constructor: Callable[..., TensorAccumulatorMixin] = metric_cls
    names = [f.name for f in dataclasses.fields(metric_cls)]
    kwargs = {name: torch.tensor(1.0) for name in names}
    kwargs[names[0]] = torch.tensor([1.0])
    return lambda: constructor(**kwargs)


@pytest.fixture(params=[False, True], ids=["no_group", "single_rank_group"])
def process_group_active(request: pytest.FixtureRequest) -> bool:
    """Run the requesting test without and with a one-rank gloo group.

    Args:
        request: Pytest request carrying whether to start the group.

    Returns:
        Whether a process group is active for the test.
    """
    active = cast(bool, request.param)
    if active:
        request.getfixturevalue("single_process_gloo_group")
    return active


@pytest.fixture
def pair_left() -> PairMetrics:
    """Return the left operand ``(first=1, second=2)``.

    Returns:
        Fresh PairMetrics; operations may mutate it in place.
    """
    return PairMetrics(first=torch.tensor(1.0), second=torch.tensor(2.0))


@pytest.fixture
def pair_right() -> PairMetrics:
    """Return the right operand ``(first=10, second=20)``.

    Returns:
        Fresh PairMetrics.
    """
    return PairMetrics(first=torch.tensor(10.0), second=torch.tensor(20.0))


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
        Every rank's reported field values.
    """
    world_size = 3
    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[tuple[float, ...]]" = ctx.Queue()
    classes = (LossMetrics, ThroughputStatistics, ComponentNorms)
    processes = [
        ctx.Process(
            target=all_reduce_mean_worker,
            args=(rank, world_size, free_port, classes, result_queue),
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
    return AllReduceRun(world_size=world_size, rank_reports=reports)


def test_in_place_add_sums_field_wise(
    pair_left: PairMetrics,
    pair_right: PairMetrics,
) -> None:
    """``left += right`` adds each field of ``right`` to the same field.

    Not parametrized: the two operands already hold distinct values in each
    field, so a field mix-up or a non-additive result would both show.

    Args:
        pair_left: Left operand ``(1, 2)``, updated in place.
        pair_right: Right operand ``(10, 20)``.
    """
    pair_left += pair_right
    assert tuple(pair_left.to_float_dict().values()) == (11.0, 22.0)


@pytest.mark.parametrize(
    ("operation", "scalar", "expected"),
    [
        pytest.param(PairMetrics.__itruediv__, 2.0, (0.5, 1.0), id="divide"),
        pytest.param(PairMetrics.__mul__, 3.0, (3.0, 6.0), id="multiply"),
    ],
)
def test_scalar_operations_scale_every_field(
    pair_left: PairMetrics,
    operation: Callable[[PairMetrics, float], PairMetrics],
    scalar: float,
    expected: tuple[float, float],
) -> None:
    """``/= scalar`` and ``* scalar`` apply the scalar to every field.

    Args:
        pair_left: Operand ``(1, 2)``.
        operation: Unbound mixin operator, called as ``operation(left, s)``.
        scalar: Scalar operand.
        expected: ``(first, second)`` of the result.
    """
    result = operation(pair_left, scalar)
    assert tuple(result.to_float_dict().values()) == expected


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        pytest.param([1, 1], (5.5, 11.0), id="equal_weights"),
        pytest.param([2, 1], (4.0, 8.0), id="left_weighted_twice"),
        pytest.param([0, 1], (10.0, 20.0), id="left_weight_zero"),
    ],
)
def test_weighted_avg_weights_each_instance_by_protein_count(
    pair_left: PairMetrics,
    pair_right: PairMetrics,
    weights: list[int],
    expected: tuple[float, float],
) -> None:
    """``weighted_avg`` returns the protein-count-weighted mean per field.

    With left ``(1, 2)`` and right ``(10, 20)``, weights 2 and 1 give
    ``(2*1 + 10) / 3 = 4``; a zero weight drops that instance entirely.

    Args:
        pair_left: First instance ``(1, 2)``.
        pair_right: Second instance ``(10, 20)``.
        weights: Protein count for each instance.
        expected: ``(first, second)`` of the weighted mean.
    """
    result = PairMetrics.weighted_avg([pair_left, pair_right], weights)
    assert tuple(result.to_float_dict().values()) == expected


@pytest.mark.parametrize(
    ("prefix", "expected_keys"),
    [
        pytest.param("", ["first", "second"], id="no_prefix"),
        pytest.param("train/", ["train/first", "train/second"], id="train"),
    ],
)
def test_to_float_dict_prefixes_every_key(
    pair_left: PairMetrics,
    prefix: str,
    expected_keys: list[str],
) -> None:
    """``to_float_dict(prefix)`` namespaces every key, in field order.

    Args:
        pair_left: Instance to convert.
        prefix: String prepended to each field name.
        expected_keys: Keys the dict must have, in order.
    """
    assert list(pair_left.to_float_dict(prefix)) == expected_keys


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
    build_with_vector_field: Callable[[], TensorAccumulatorMixin],
) -> None:
    """``@jaxtyped`` rejects a 1-D tensor in a scalar field at construction.

    Parametrized over the real metric classes through ``metric_cls``.

    Args:
        build_with_vector_field: Thunk building an instance with one 1-D
            field.
    """
    with pytest.raises(TypeCheckError):
        _ = build_with_vector_field()


def test_all_reduce_mean_keeps_values_on_one_process(
    process_group_active: bool,  # noqa: FBT001
    metrics_of_ones: TensorAccumulatorMixin,
) -> None:
    """With no group or a one-rank group, every field stays at 1.0.

    With no group, ``all_reduce_mean`` must return before any collective
    call, which would otherwise raise. With a one-rank group the real
    collective path runs and the mean of one value is itself. Neither case
    can tell a correct reduction from a skipped one; the three-process test
    below does that.

    Args:
        process_group_active: Whether a one-rank gloo group is active.
        metrics_of_ones: Instance with every field 1.0.
    """
    assert dist.is_initialized() == process_group_active
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
    assert len(all_reduce_run.rank_reports) == world_size
    assert {v for r in all_reduce_run.rank_reports for v in r} == {expected}
