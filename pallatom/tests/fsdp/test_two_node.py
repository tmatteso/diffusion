"""Hybrid sharding on two real nodes; skipped unless launched for it.

Run through ``scripts/run_two_node_tests.sh`` on both nodes. These tests
answer what the CPU tier cannot: whether ranks land on the intended hardware,
how fast the links are, and what hybrid sharding costs on real GPUs. Every
check gathers from all ranks before asserting, so all ranks reach the same
verdict and a failure cannot leave a peer waiting in a collective.

Measurements go to ``PALLATOM_TWO_NODE_REPORT`` as JSON. They are recorded
rather than asserted, because no threshold is meaningful before the first run.
"""

import gc
import hashlib
import json
import os
import pathlib
import socket
import statistics
import time
from collections.abc import Callable, Generator
from datetime import timedelta
from typing import cast

import pytest
import torch
import torch.distributed as dist
from helpers.context_managers import DistProcessGroup
from helpers.useful_objects import ModelSetup
from tests.fsdp.harness import (
    ParityGaps,
    Workload,
    build_ddp_setup,
    build_hybrid_setup,
    build_mesh,
    parity_gaps,
    rank_batches,
    route_calls_through_module_hooks,
    run_windows,
    write_proteins,
)
from torch.distributed.device_mesh import DeviceMesh
from train.train_config import LoaderConfig, TrainConfig

_ENABLED = "1"
_RANK_ZERO = "0"

pytestmark = pytest.mark.skipif(
    os.environ.get("PALLATOM_TWO_NODE") != _ENABLED,
    reason="launch with scripts/run_two_node_tests.sh on two nodes",
)

_N_NODES = 2
_GPUS_PER_NODE = 2
_MESH_SHAPE = (_N_NODES, _GPUS_PER_NODE)
_WORLD_SIZE = _N_NODES * _GPUS_PER_NODE
_COLLECTIVE_TIMEOUT = timedelta(minutes=10)
_MIB = 2**20
_LINK_PAYLOAD_MIB = 256
_LINK_REPEATS = 5
_FLOAT32_BYTES = 4
_PARITY_TOLERANCE = 1e-3
_PARITY_RESIDUES = 64
_PARITY_BATCH_SIZE = 2
_PARITY_K_UNIT = 3
_BENCH_RESIDUES = 128
_BENCH_BATCH_SIZE = 3
_BENCH_WINDOWS = 5
_DEFAULT_REPORT = "two_node_report.json"


def _sized_config(n_res: int, batch_size: int, k_unit: int) -> TrainConfig:
    """Return a default TrainConfig resized for a run.

    Args:
        n_res: Residues per protein.
        batch_size: Proteins per micro-batch.
        k_unit: Decoder units.

    Returns:
        The configuration.
    """
    base = TrainConfig()
    return base.model_copy(
        update={
            "model": base.model.model_copy(
                update={"max_residues": n_res, "K_unit": k_unit},
            ),
            "test_loader": LoaderConfig(
                max_seq_length=n_res,
                batch_size=batch_size,
            ),
        },
    )


def _gather[T](value: T) -> list[T]:
    """Collect one picklable value from every rank, in rank order.

    Args:
        value: This rank's value.

    Returns:
        Every rank's value.
    """
    gathered: list[T | None] = [None] * dist.get_world_size()
    dist.all_gather_object(  # pyright: ignore[reportUnknownMemberType]
        gathered,
        value,
    )
    return [item for item in gathered if item is not None]


def _mesh_ranks(device_mesh: DeviceMesh, dimension: str) -> list[int]:
    """List the global ranks in this rank's group along a mesh dimension.

    Args:
        device_mesh: The device mesh.
        dimension: ``"shard"`` or ``"replicate"``.

    Returns:
        The group's global ranks.
    """
    ranks = device_mesh[
        dimension
    ].mesh.tolist()  # pyright: ignore[reportUnknownMemberType]
    return cast(list[int], ranks)


def _file_digest(path: pathlib.Path) -> str:
    """Return the SHA-256 of a file.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reduce_mib_per_second(group: dist.ProcessGroup, device: str) -> float:
    """Time all-reduces over a process group.

    Args:
        group: Group to measure.
        device: Device holding the payload.

    Returns:
        Payload MiB per second, median over the repeats.
    """
    payload = torch.ones(
        _LINK_PAYLOAD_MIB * _MIB // _FLOAT32_BYTES,
        device=device,
    )
    timings: list[float] = []
    for _ in range(_LINK_REPEATS):
        _ = dist.barrier(  # pyright: ignore[reportUnknownMemberType]
            group=group,
        )
        start = time.perf_counter()
        _ = dist.all_reduce(  # pyright: ignore[reportUnknownMemberType]
            payload,
            group=group,
        )
        torch.cuda.synchronize(payload.device)
        timings.append(time.perf_counter() - start)
    return _LINK_PAYLOAD_MIB / statistics.median(timings)


@pytest.fixture(scope="module")
def process_group() -> Generator[DistProcessGroup, None, None]:
    """Join the job's NCCL group through the production context manager.

    Yields:
        DistProcessGroup: The live process group.
    """
    route_calls_through_module_hooks()
    with DistProcessGroup("nccl", timeout=_COLLECTIVE_TIMEOUT) as group:
        yield group


@pytest.fixture(scope="module")
def report() -> Generator[dict[str, object], None, None]:
    """Provide the measurements dict and write it on rank 0 at teardown.

    Yields:
        dict[str, object]: Mapping that tests add measurements to.
    """
    measurements: dict[str, object] = {}
    yield measurements
    if os.environ.get("RANK", _RANK_ZERO) == _RANK_ZERO:
        path = pathlib.Path(
            os.environ.get("PALLATOM_TWO_NODE_REPORT", _DEFAULT_REPORT),
        )
        _ = path.write_text(
            json.dumps(measurements, indent=2) + "\n",
            encoding="utf-8",
        )


@pytest.fixture(scope="module")
def node_ids(process_group: DistProcessGroup) -> list[str]:
    """Identify each rank's node.

    Args:
        process_group: Live process group.

    Returns:
        Each rank's ``PALLATOM_NODE_ID``, or its hostname when unset.
    """
    assert process_group.world_size == _WORLD_SIZE
    return _gather(os.environ.get("PALLATOM_NODE_ID", socket.gethostname()))


@pytest.fixture
def parity_workload(tmp_path: pathlib.Path) -> Workload:
    """Build a small run for the GPU parity check.

    Args:
        tmp_path: Directory for this rank's proteins.

    Returns:
        Two windows of two micro-batches.
    """
    run = Workload(
        tcfg=_sized_config(
            _PARITY_RESIDUES,
            _PARITY_BATCH_SIZE,
            _PARITY_K_UNIT,
        ),
        jsonl_path=tmp_path / "proteins.jsonl",
        n_windows=2,
        micro_batches=2,
        batch_size=_PARITY_BATCH_SIZE,
        train_mode=True,
        mesh_shape=_MESH_SHAPE,
    )
    write_proteins(
        run.jsonl_path,
        _WORLD_SIZE * run.proteins_per_rank,
        _PARITY_RESIDUES,
    )
    return run


@pytest.fixture
def bench_workload(tmp_path: pathlib.Path) -> Workload:
    """Build a default-size model run for timing and memory.

    Args:
        tmp_path: Directory for this rank's proteins.

    Returns:
        Five single-micro-batch windows of the full model.
    """
    default_k = TrainConfig().model.K_unit
    run = Workload(
        tcfg=_sized_config(_BENCH_RESIDUES, _BENCH_BATCH_SIZE, default_k),
        jsonl_path=tmp_path / "proteins.jsonl",
        n_windows=_BENCH_WINDOWS,
        micro_batches=1,
        batch_size=_BENCH_BATCH_SIZE,
        train_mode=True,
        mesh_shape=_MESH_SHAPE,
    )
    write_proteins(
        run.jsonl_path,
        _WORLD_SIZE * run.proteins_per_rank,
        _BENCH_RESIDUES,
    )
    return run


@pytest.fixture
def mesh(
    process_group: DistProcessGroup,
    parity_workload: Workload,
) -> DeviceMesh:
    """Build the job's device mesh.

    Args:
        process_group: Live process group.
        parity_workload: Supplies the mesh shape.

    Returns:
        The two-dimensional mesh.
    """
    return build_mesh(parity_workload, torch.device(process_group.device))


def test_each_node_runs_two_ranks_on_distinct_gpus(
    process_group: DistProcessGroup,
    node_ids: list[str],
) -> None:
    """Ranks must land on the intended hardware.

    Arrange: the launched job.
    Act: gather each rank's node and GPU UUID.
    Assert: two nodes, two ranks each, and no GPU used twice on a node.

    Args:
        process_group: Live process group.
        node_ids: Each rank's node.
    """
    uuid = str(
        torch.cuda.get_device_properties(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            process_group.local_rank,
        ).uuid,
    )
    placements = list(zip(node_ids, _gather(uuid), strict=True))

    assert len(set(node_ids)) == _N_NODES
    assert all(node_ids.count(node) == _GPUS_PER_NODE for node in node_ids)
    assert len(set(placements)) == _WORLD_SIZE


def test_mesh_shards_within_nodes_and_replicates_across(
    mesh: DeviceMesh,
    node_ids: list[str],
) -> None:
    """Sharding must stay on the fast link.

    Arrange: the 2x2 mesh.
    Act: gather every rank's shard and replicate groups.
    Assert: each shard group sits on one node and each replicate group spans
    both. A wrong rank order would raise no error, and would instead send
    every all-gather over the network.

    Args:
        mesh: The device mesh.
        node_ids: Each rank's node.
    """
    shard_groups = _gather(_mesh_ranks(mesh, "shard"))
    replicate_groups = _gather(_mesh_ranks(mesh, "replicate"))

    for group in shard_groups:
        assert len({node_ids[rank] for rank in group}) == 1
    for group in replicate_groups:
        assert len({node_ids[rank] for rank in group}) == _N_NODES


def test_link_throughput_is_recorded(
    process_group: DistProcessGroup,
    mesh: DeviceMesh,
    report: dict[str, object],
) -> None:
    """Record the within-node and cross-node link speeds.

    Arrange: the shard group (within a node) and replicate group (across).
    Act: time large all-reduces on each.
    Assert: both finish with positive throughput on every rank. The ratio
    decides whether hybrid's cross-node savings are worth having.

    Args:
        process_group: Live process group.
        mesh: The device mesh.
        report: Measurements written at teardown.
    """
    within = _gather(
        _reduce_mib_per_second(
            mesh["shard"].get_group(),
            process_group.device,
        ),
    )
    across = _gather(
        _reduce_mib_per_second(
            mesh["replicate"].get_group(),
            process_group.device,
        ),
    )
    report["within_node_all_reduce_mib_per_s"] = within
    report["cross_node_all_reduce_mib_per_s"] = across

    assert min(within) > 0
    assert min(across) > 0


def test_dataset_matches_on_both_nodes(
    process_group: DistProcessGroup,
) -> None:
    """A training run needs byte-identical data on both nodes.

    Arrange: ``PALLATOM_DATASET_JSONL`` naming the real dataset.
    Act: hash the file on every rank.
    Assert: one digest. Differing files would silently give the nodes
    different proteins under the same names.

    Args:
        process_group: Live process group.
    """
    assert process_group.world_size == _WORLD_SIZE
    configured = os.environ.get("PALLATOM_DATASET_JSONL")
    if configured is None:
        pytest.skip("PALLATOM_DATASET_JSONL is not set")
    assert len(set(_gather(_file_digest(pathlib.Path(configured))))) == 1


def test_hybrid_matches_ddp_on_gpus(
    process_group: DistProcessGroup,
    parity_workload: Workload,
) -> None:
    """Hybrid must match DDP over real NCCL and real network links.

    Arrange: identical seeded init, batches, and dropout streams.
    Act: DDP, then the 2x2 mesh, on real GPUs with dropout active.
    Assert: every gap is within tolerance on every rank.

    Args:
        process_group: Live process group.
        parity_workload: The run description.
    """
    gaps: list[ParityGaps] = _gather(
        parity_gaps(
            parity_workload,
            torch.device(process_group.device),
            window_sync=False,
        ),
    )

    for gap in gaps:
        assert gap.weights < _PARITY_TOLERANCE
        assert gap.ema < _PARITY_TOLERANCE
        assert gap.loss < _PARITY_TOLERANCE
        assert gap.norms < _PARITY_TOLERANCE


def test_cost_of_hybrid_is_recorded(
    process_group: DistProcessGroup,
    bench_workload: Workload,
    report: dict[str, object],
) -> None:
    """Record step time and peak memory, DDP against hybrid.

    Arrange: the default-size model on identical batches.
    Act: five real optimizer steps per mode, dropping the first because it
    includes FSDP's lazy init.
    Assert: every step on every rank completed with memory allocated.

    Args:
        process_group: Live process group.
        bench_workload: The run description.
        report: Measurements written at teardown.
    """
    device = torch.device(process_group.device)
    batches = rank_batches(bench_workload)
    builders: dict[str, Callable[[Workload, torch.device], ModelSetup]] = {
        "ddp": build_ddp_setup,
        "hybrid": build_hybrid_setup,
    }
    peaks: list[float] = []
    for name, build in builders.items():
        record = run_windows(
            bench_workload,
            build(bench_workload, device),
            batches,
        )
        steps = _gather(record.step_seconds)
        peaks = _gather(record.peak_mib)
        report[name] = {
            "first_step_seconds": max(rank[0] for rank in steps),
            "median_step_seconds": statistics.median(
                max(rank[index] for rank in steps)
                for index in range(1, _BENCH_WINDOWS)
            ),
            "peak_allocated_mib": max(peaks),
        }
        _ = gc.collect()
        torch.cuda.empty_cache()

    assert min(peaks) > 0
