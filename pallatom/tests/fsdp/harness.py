"""Shared workload for the CPU and two-node hybrid-sharding tests.

Both test tiers run the same code against the repo's real ``optimizer_step``,
``EMA``, and ``take_step``, so a result on CPU gloo and a result on two real
nodes describe the same behaviour. Production has no FSDP code yet, so the
sharding recipe from ``docs/fsdp_device_mesh.md`` lives here until it does.
"""

import contextlib
import dataclasses
import functools
import json
import math
import pathlib
import time
from collections.abc import Generator, Mapping
from types import TracebackType
from typing import Self, cast

import structlog
import torch
import torch.distributed as dist
import torch.nn as nn
import train.train_loop as train_loop_module
from architecture.main_trunk import EMA, MainTrunk
from helpers.data import (
    FeaturizeCollate,
    FeaturizedBatch,
    ProteinDataset,
    build_distogram_module,
)
from helpers.useful_objects import ModelSetup, manual_seed
from jaxtyping import Float
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from train.train_config import TrainConfig
from train.train_loop import alphafold_lr_lambda, optimizer_step

StateDict = dict[str, Float[torch.Tensor, "..."]]

_INIT_SEED = 0
_BATCH_SEED_BASE = 1_000
_STEP_SEED_BASE = 2_000
_MIB = 2**20
_CUDA = "cuda"
_CALL_METHOD = "__call__"
_NO_SYNC_NAME = "DDPNoSync"
_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
_HELIX_RADIUS = 2.3
_HELIX_RISE = 1.5
_HELIX_TURN_RADIANS = math.radians(100.0)
_FIRST_PROJECT_MODULES = frozenset({"architecture", "helpers"})
_BACKBONE_OFFSETS = {
    "N": (-1.2, 0.0, 0.0),
    "CA": (0.0, 0.0, 0.0),
    "C": (1.2, 0.0, 0.0),
    "O": (1.2, 1.2, 0.0),
}


@dataclasses.dataclass(frozen=True)
class Workload:
    """A deterministic training run, identical for DDP and hybrid sharding.

    Attributes:
        tcfg: Training configuration, including model size and clipping.
        jsonl_path: Protein file written by ``write_proteins``.
        n_windows: Optimizer steps to take.
        micro_batches: Micro-batches per accumulation window.
        batch_size: Proteins per micro-batch.
        train_mode: Keep dropout active when True.
        mesh_shape: ``(n_nodes, gpus_per_node)``.
    """

    tcfg: TrainConfig
    jsonl_path: pathlib.Path
    n_windows: int
    micro_batches: int
    batch_size: int
    train_mode: bool
    mesh_shape: tuple[int, int]

    @property
    def proteins_per_rank(self) -> int:
        """Return how many proteins one rank consumes over the run."""
        return self.n_windows * self.micro_batches * self.batch_size


@dataclasses.dataclass(frozen=True)
class WindowRecord:
    """What one run produced on this rank.

    Attributes:
        losses: Total loss per optimizer step.
        norms: Component gradient norms from the last optimizer step.
        step_seconds: Wall time per optimizer step.
        peak_mib: Peak allocated GPU memory; ``0.0`` on CPU.
    """

    losses: list[float]
    norms: dict[str, float]
    step_seconds: list[float]
    peak_mib: float


@dataclasses.dataclass(frozen=True)
class ParityGaps:
    """Largest absolute DDP-versus-hybrid difference per quantity.

    Attributes:
        weights: Over every entry of every model weight.
        ema: Over every entry of every EMA shadow weight.
        loss: Over the per-step total losses.
        norms: Over the last step's component gradient norms.
    """

    weights: float
    ema: float
    loss: float
    norms: float


class WindowSync:
    """FSDP stand-in for ``DDPNoSync``: sync gradients once per window.

    ``DDPNoSync`` is a no-op under FSDP because ``no_sync`` does not exist
    there. Installed by ``window_sync_installed``.

    Args:
        model: The sharded model.
        is_last: Whether this is the window's final micro-batch.
    """

    def __init__(self, model: FSDPModule, *, is_last: bool) -> None:
        self._model: FSDPModule = model
        self._is_last: bool = is_last

    def __enter__(self) -> Self:
        """Sync gradients only on the final micro-batch.

        Returns:
            This ``WindowSync`` instance.
        """
        self._model.set_requires_gradient_sync(self._is_last)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore per-backward gradient sync.

        Args:
            exc_type: Exception class, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if an exception was raised.
        """
        self._model.set_requires_gradient_sync(True)


@contextlib.contextmanager
def window_sync_installed() -> Generator[None, None, None]:
    """Make ``process_accum_window`` use ``WindowSync`` for the block.

    Yields:
        None: The context only swaps the module attribute.
    """
    namespace = vars(train_loop_module)
    original = cast(object, namespace[_NO_SYNC_NAME])
    namespace[_NO_SYNC_NAME] = WindowSync
    try:
        yield
    finally:
        namespace[_NO_SYNC_NAME] = original


def route_calls_through_module_hooks(
    cls: type[nn.Module] = nn.Module,
) -> None:
    """Remove the typed ``__call__`` overrides for this process.

    They call ``self.forward`` directly, so FSDP's hooks never run. Removing
    them lets the other tests reach the defect they exist to catch. Call it
    in a spawned worker or under ``torchrun``, never in the pytest parent,
    and delete it once required change 1 lands.

    Args:
        cls: Class whose subclasses are visited.
    """
    for subclass in cls.__subclasses__():
        package = subclass.__module__.split(".", maxsplit=1)[0]
        if package in _FIRST_PROJECT_MODULES and _CALL_METHOD in vars(subclass):
            delattr(subclass, _CALL_METHOD)
        route_calls_through_module_hooks(subclass)


def write_proteins(path: pathlib.Path, n_proteins: int, n_res: int) -> None:
    """Write helical proteins with deterministic, distinct sequences.

    Args:
        path: Destination JSONL file.
        n_proteins: Number of proteins, named ``p0.A`` onwards.
        n_res: Residues per protein.
    """
    lines: list[str] = []
    for index in range(n_proteins):
        sequence = "".join(
            _SEQUENCE[(index + res) % len(_SEQUENCE)] for res in range(n_res)
        )
        trace = [
            (
                _HELIX_RADIUS * math.cos(res * _HELIX_TURN_RADIANS),
                _HELIX_RADIUS * math.sin(res * _HELIX_TURN_RADIANS),
                _HELIX_RISE * res,
            )
            for res in range(n_res)
        ]
        coords = {
            atom: [
                [
                    axis + delta
                    for axis, delta in zip(point, offset, strict=True)
                ]
                for point in trace
            ]
            for atom, offset in _BACKBONE_OFFSETS.items()
        }
        entry = {"name": f"p{index}.A", "seq": sequence, "coords": coords}
        lines.append(json.dumps(entry))
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rank_batches(workload: Workload) -> list[FeaturizedBatch]:
    """Build this rank's micro-batches with rank-seeded noise.

    Args:
        workload: The run description.

    Returns:
        ``n_windows * micro_batches`` CPU batches, distinct on every rank.
    """
    rank = dist.get_rank()
    tcfg = workload.tcfg
    first = rank * workload.proteins_per_rank
    names = [
        f"p{index}.A"
        for index in range(first, first + workload.proteins_per_rank)
    ]
    dataset = ProteinDataset(
        workload.jsonl_path,
        names,
        max_seq_length=tcfg.test_loader.max_seq_length,
    )
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_template=build_distogram_module(
            tcfg.distogram_template,
        ).eval(),
        distogram_residue=build_distogram_module(
            tcfg.distogram_residue,
        ).eval(),
        distogram_atom=build_distogram_module(tcfg.distogram_atom).eval(),
    )
    _ = manual_seed(_BATCH_SEED_BASE + rank)
    size = workload.batch_size
    return [
        collate([dataset[index] for index in range(start, start + size)])
        for start in range(0, len(names), size)
    ]


def build_mesh(workload: Workload, device: torch.device) -> DeviceMesh:
    """Build the ``(nodes, gpus)`` mesh named ``replicate`` and ``shard``.

    Args:
        workload: Supplies the mesh shape.
        device: Device whose type selects the mesh's device type.

    Returns:
        The two-dimensional device mesh.
    """
    return init_device_mesh(
        device.type,
        workload.mesh_shape,
        mesh_dim_names=("replicate", "shard"),
    )


def build_model(tcfg: TrainConfig, device: torch.device) -> MainTrunk:
    """Build an unwrapped MainTrunk on ``device``.

    Args:
        tcfg: Supplies the architecture parameters.
        device: Destination device.

    Returns:
        The model, with whatever weights the current RNG state produces.
    """
    return MainTrunk(
        model_params=tcfg.model,
        template_distogram_params=tcfg.distogram_template,
        residue_distogram_params=tcfg.distogram_residue,
        atom_distogram_params=tcfg.distogram_atom,
        noise_params=tcfg.noise,
    ).to(device)


def shard_model(model: MainTrunk, mesh: DeviceMesh) -> MainTrunk:
    """Apply ``fully_shard`` block by block, bottom-up, in place.

    Args:
        model: Unwrapped model.
        mesh: Two-dimensional mesh.

    Returns:
        The same model object, now sharded.
    """
    blocks: list[nn.Module] = [
        *model.node_updates,
        *model.atom_decoders,
        *model.pair_updates,
        model.template_embedder,
        model.atom_encoder,
    ]
    for block in blocks:
        _ = fully_shard(block, mesh=mesh)
    _ = fully_shard(model, mesh=mesh)
    return model


def build_setup(
    model: MainTrunk | DDP,
    tcfg: TrainConfig,
    device: torch.device,
) -> ModelSetup:
    """Build Adam, the scheduler, and EMA, after any sharding has happened.

    Args:
        model: Plain, DDP-wrapped, or sharded model.
        tcfg: Training configuration.
        device: Device the model lives on.

    Returns:
        The bundle ``optimizer_step`` expects.
    """
    tp = tcfg.training
    optimizer = Adam(model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=functools.partial(
            alphafold_lr_lambda,
            warmup_steps=tp.lr_warmup_steps,
            decay_steps=tp.lr_decay_steps,
            decay_factor=tp.lr_decay_factor,
        ),
    )
    return ModelSetup(
        model=model,
        tcfg=tcfg,
        distogram_template=build_distogram_module(tcfg.distogram_template),
        distogram_residue=build_distogram_module(tcfg.distogram_residue),
        distogram_atom=build_distogram_module(tcfg.distogram_atom),
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=EMA(model, decay=tp.ema_decay),
    )


def build_ddp_setup(workload: Workload, device: torch.device) -> ModelSetup:
    """Build the DDP baseline from a seeded init.

    Args:
        workload: The run description.
        device: Device for this rank.

    Returns:
        A ModelSetup holding a DDP-wrapped model.
    """
    _ = manual_seed(_INIT_SEED)
    model = build_model(workload.tcfg, device)
    wrapped = DDP(
        model,
        device_ids=[device.index] if device.type == _CUDA else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )
    _ = wrapped.train(workload.train_mode)
    return build_setup(wrapped, workload.tcfg, device)


def build_hybrid_setup(workload: Workload, device: torch.device) -> ModelSetup:
    """Build the hybrid-sharded model from the same seeded init.

    Args:
        workload: The run description.
        device: Device for this rank.

    Returns:
        A ModelSetup holding a sharded model.
    """
    _ = manual_seed(_INIT_SEED)
    mesh = build_mesh(workload, device)
    model = shard_model(build_model(workload.tcfg, device), mesh)
    _ = model.train(workload.train_mode)
    return build_setup(model, workload.tcfg, device)


def run_windows(
    workload: Workload,
    setup: ModelSetup,
    batches: list[FeaturizedBatch],
) -> WindowRecord:
    """Take ``n_windows`` real optimizer steps, reseeding dropout per rank.

    Args:
        workload: The run description.
        setup: Bundle from ``build_ddp_setup`` or ``build_hybrid_setup``.
        batches: This rank's micro-batches.

    Returns:
        Per-step losses and timing, final norms, and peak GPU memory.
    """
    is_cuda = setup.device.type == _CUDA
    _ = manual_seed(_STEP_SEED_BASE + dist.get_rank())
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(setup.device)
    losses: list[float] = []
    step_seconds: list[float] = []
    norms: dict[str, float] = {}
    size = workload.micro_batches
    for window in range(workload.n_windows):
        micro = batches[window * size : (window + 1) * size]
        start = time.perf_counter()
        loss_metrics, _, component_norms, _ = optimizer_step(
            micro_buffer=micro,
            n_proteins_buffer=[workload.batch_size] * size,
            model_params=setup,
            global_step=window,
        )
        if is_cuda:
            torch.cuda.synchronize(setup.device)
        step_seconds.append(time.perf_counter() - start)
        losses.append(float(loss_metrics.total_loss))
        norms = component_norms.to_float_dict()
    peak = torch.cuda.max_memory_allocated(setup.device) if is_cuda else 0
    return WindowRecord(
        losses=losses,
        norms=norms,
        step_seconds=step_seconds,
        peak_mib=peak / _MIB,
    )


def model_state(setup: ModelSetup) -> dict[str, torch.Tensor]:
    """Return the model's state dict, without any DDP wrapper.

    Args:
        setup: Bundle holding the model.

    Returns:
        The state dict; sharded entries remain DTensors.
    """
    model = setup.model
    inner = cast(nn.Module, getattr(model, "module", model))
    return cast(dict[str, torch.Tensor], inner.state_dict())


def gather_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collect a possibly sharded state dict into full CPU tensors.

    Collective when any entry is a DTensor: every rank must call it.

    Args:
        state: Name-to-tensor mapping.

    Returns:
        Full, plain CPU tensors under the same names.
    """
    return {
        name: (
            tensor.full_tensor().cpu()
            if isinstance(tensor, DTensor)
            else tensor.detach().cpu().clone()
        )
        for name, tensor in state.items()
    }


def reshard_like(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Shard a full optimizer slot the way its parameter is sharded.

    Args:
        value: Full tensor; a zero-dimensional slot such as Adam's ``step``
            stays as it is.
        reference: The parameter the slot belongs to.

    Returns:
        A DTensor matching ``reference``, or ``value`` when it cannot shard.
    """
    if value.ndim == 0 or not isinstance(reference, DTensor):
        return value
    return distribute_tensor(value, reference.device_mesh, reference.placements)


def state_gap(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    """Return the largest absolute entry-wise difference of two states.

    Args:
        left: First state dict.
        right: Second state dict, with the same keys.

    Returns:
        The largest absolute difference over all entries.
    """
    return max(float((left[name] - right[name]).abs().max()) for name in left)


def parity_gaps(
    workload: Workload,
    device: torch.device,
    *,
    window_sync: bool,
) -> ParityGaps:
    """Run DDP then hybrid sharding and measure how far they differ.

    Args:
        workload: The run description.
        device: Device for this rank.
        window_sync: Sync hybrid gradients once per window, not per backward.

    Returns:
        The largest differences between the two runs.
    """
    batches = rank_batches(workload)
    ddp_setup = build_ddp_setup(workload, device)
    ddp_record = run_windows(workload, ddp_setup, batches)
    ddp_model = gather_state(model_state(ddp_setup))
    ddp_ema = gather_state(ddp_setup.ema.state_dict())

    hybrid_setup = build_hybrid_setup(workload, device)
    sync_context = (
        window_sync_installed() if window_sync else contextlib.nullcontext()
    )
    with sync_context:
        hybrid_record = run_windows(workload, hybrid_setup, batches)
    hybrid_model = gather_state(model_state(hybrid_setup))
    hybrid_ema = gather_state(hybrid_setup.ema.state_dict())

    return ParityGaps(
        weights=state_gap(ddp_model, hybrid_model),
        ema=state_gap(ddp_ema, hybrid_ema),
        loss=max(
            abs(left - right)
            for left, right in zip(
                ddp_record.losses,
                hybrid_record.losses,
                strict=True,
            )
        ),
        norms=max(
            abs(value - hybrid_record.norms[name])
            for name, value in ddp_record.norms.items()
        ),
    )


def structlog_logger() -> structlog.typing.FilteringBoundLogger:
    """Return a bound logger for production helpers that require one."""
    return cast(
        structlog.typing.FilteringBoundLogger,
        structlog.get_logger(),
    )
