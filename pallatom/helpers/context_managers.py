"""Context manager helpers for distributed training and logging.

Provides ``DistProcessGroup``, ``FatalOnError``, ``StructlogConfig``,
``DDPNoSync``, and ``StepContext`` — reusable context managers that handle
setup and teardown boilerplate so training scripts remain concise.
"""

import contextlib
import io
import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt
import structlog
import torch
import torch.distributed as dist
from structlog.typing import EventDict, FilteringBoundLogger, Processor
from typing_extensions import Self

log: FilteringBoundLogger = cast("FilteringBoundLogger", structlog.get_logger())


class DistProcessGroup:
    """Context manager that inits and tears down distributed process group.

    Exposes rank, local_rank, world_size, device, and is_rank_zero after entry.

    Args:
        backend: Distributed backend for ``dist.init_process_group``.
    """

    def __init__(self, backend: str = "nccl") -> None:
        self.backend: str = backend
        self.rank: int = -1
        self.local_rank: int = -1
        self.world_size: int = -1
        self.device: str = ""

    def __enter__(self) -> Self:
        """Initialise the process group and populate distributed state.

        Returns:
            This ``DistProcessGroup`` instance with rank, local_rank,
            world_size, and device attributes set.
        """
        dist.init_process_group(backend=self.backend)
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(self.local_rank)
        return self

    @property
    def is_rank_zero(self) -> bool:
        """Return True only on the process with global rank 0.

        Returns:
            ``True`` when ``self.rank == 0``, ``False`` on all other ranks.
        """
        return self.rank == 0

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Destroy process group regardless of whether an exception occurred.

        Args:
            exc_type: Exception class if one was raised inside the block,
                else ``None``.
            exc_val: Exception instance if one was raised, else ``None``.
            exc_tb: Traceback if an exception was raised, else ``None``.
        """
        dist.destroy_process_group()


class FatalOnError:
    """Context manager that logs any exception via structlog.

    Intended to be used as the outermost context in a training script so that
    uncaught exceptions are always surfaced to the structured log before the
    process terminates.
    """

    def __enter__(self) -> None:
        """No setup required; all work happens in ``__exit__``.

        This method intentionally does nothing on entry.
        """

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Log the exception and raise SystemExit(1) if one occurred.

        Args:
            exc_type: Exception class if one was raised inside the block,
                else ``None``.
            exc_val: Exception instance if one was raised, else ``None``.
            exc_tb: Traceback if an exception was raised, else ``None``.

        Raises:
            SystemExit: If an exception was raised inside the block.
        """
        if exc_val is not None:
            log.exception(
                "fatal",
                error=str(exc_val),
                traceback=traceback.format_exc(),
            )
            raise SystemExit(1) from exc_val


class StructlogConfig:
    """Context manager that configures structlog sends log lines to JSON file.

    On non-rank-zero processes all output is suppressed; on rank zero, log
    events are printed to the console and, if ``log_file`` is given, written as
    JSON lines to that file.

    Args:
        is_rank_zero: Whether this process should emit console and file output.
        log_file: Path to write structured JSON log lines.
    """

    def __init__(
        self,
        *,
        is_rank_zero: bool,
        log_file: Path,
    ) -> None:
        self._is_rank_zero: bool = is_rank_zero
        self._log_file: Path = log_file
        self.f: io.TextIOWrapper | None = None  # what is this?

    def __enter__(self) -> Self:
        """Configure structlog and open the log file if requested.

        Returns:
            This ``StructlogConfig`` instance, ready for use as a log-file tee.
        """
        processors: list[Processor] = [
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]
        if self._is_rank_zero:
            self.f = self._log_file.open("w", encoding="utf-8", buffering=1)
            processors.append(self.write_log_line)
            processors.append(structlog.dev.ConsoleRenderer())
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(20),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        return self

    def write_log_line(  # pylint: disable=useless-param-doc
        self,
        _logger: object,
        _method: str | None,
        event_dict: EventDict,
    ) -> EventDict:
        """Structlog processor: write event dict as JSON line, pass through.

        Args:
            _logger: The structlog logger instance; required by the processor
                protocol but unused here.
            _method: The log method name, e.g. ``"info"``; required by the
                processor protocol but unused here.
            event_dict: The mutable event dictionary assembled by earlier
                processors.

        Returns:
            The unmodified ``event_dict`` so the processor chain continues.
        """
        if self.f is not None:
            _ = self.f.write(json.dumps(event_dict, default=str) + "\n")
        return event_dict

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the log file and reset structlog to a no-op configuration.

        Args:
            exc_type: Exception class if one was raised inside the block,
                else ``None``.
            exc_val: Exception instance if one was raised, else ``None``.
            exc_tb: Traceback if an exception was raised, else ``None``.
        """
        if self.f is not None:
            self.f.close()
        structlog.reset_defaults()


class ShardWorkerNotInitializedError(RuntimeError):
    """Raised when WorkerState.get() is called before WorkerState entered."""

    def __init__(self) -> None:
        super().__init__("No WorkerState is active for this process.")


class ShardWorkerState:
    """Context manager that initialises per-worker-process shard index arrays.

    Constructed once per worker subprocess via init_worker (the
    ProcessPoolExecutor initializer). Loads lengths.npy and shard_meta.npz
    from disk inside the subprocess so large arrays are never pickled across
    the process boundary. Exposes the arrays as public attributes and registers
    this instance as the process-wide singleton accessible via
    ShardWorkerState.get.

    Use init_worker as the ProcessPoolExecutor initializer so the executor
    subprocess enters the context automatically at startup.

    Args:
        lengths: Int16 array of seq_len for every protein, in shard order.
        shard_starts: Int32 start index into lengths for each shard.
        shard_sizes: Int32 protein count for each shard.
        log_file: Path forwarded to StructlogConfig for structured logging.
    """

    _current: ClassVar["ShardWorkerState | None"] = None

    def __init__(
        self,
        lengths: npt.NDArray[np.int16],
        shard_starts: npt.NDArray[np.int32],
        shard_sizes: npt.NDArray[np.int32],
        log_file: Path,
    ) -> None:
        self._lengths: npt.NDArray[np.int16] = lengths
        self._shard_starts: npt.NDArray[np.int32] = shard_starts
        self._shard_sizes: npt.NDArray[np.int32] = shard_sizes
        self._log_file: Path = log_file
        self.lengths: npt.NDArray[np.int16] | None = None
        self.shard_starts: npt.NDArray[np.int32] | None = None
        self.shard_sizes: npt.NDArray[np.int32] | None = None

    def __enter__(self) -> Self:
        """Configure structlog, populate ShardWorkerState instance.

        Returns:
            This ShardWorkerState instance with lengths, shard_starts, and
                shard_sizes set.
        """
        _ = StructlogConfig(
            is_rank_zero=True,
            log_file=self._log_file,
        ).__enter__()
        self.lengths = self._lengths
        self.shard_starts = self._shard_starts
        self.shard_sizes = self._shard_sizes
        ShardWorkerState._current = self
        log.info("worker_init")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Deregister ShardWorkerState instance and clear public attributes.

        Args:
            exc_type: Exception class if one was raised inside the block,
                else None.
            exc_val: Exception instance if one was raised, else None.
            exc_tb: Traceback if an exception was raised, else None.
        """
        ShardWorkerState._current = None
        self.lengths = None
        self.shard_starts = None
        self.shard_sizes = None

    @classmethod
    def get(cls) -> "ShardWorkerState":
        """Return the active ShardWorkerState for this process.

        Returns:
            The ShardWorkerState instance that is currently entered.

        Raises:
            ShardWorkerNotInitializedError: If no ShardWorkerState has been
                entered.
        """
        if cls._current is None:
            raise ShardWorkerNotInitializedError
        return cls._current

    @classmethod
    def init_worker(
        cls,
        lengths_path: Path,
        shard_meta_path: Path,
        log_file: Path,
    ) -> None:
        """Load shard metadata from disk and enter ShardWorkerState.

        This is the ProcessPoolExecutor initializer. Loading from disk avoids
        pickling large arrays across the process boundary.

        Args:
            lengths_path: Path to lengths.npy written by build_shards.
            shard_meta_path: Path to shard_meta.npz written by build_shards.
            log_file: Path forwarded to StructlogConfig for structured logging.
        """
        lengths = cast("npt.NDArray[np.int16]", np.load(lengths_path))
        shard_metadata = cast("np.lib.npyio.NpzFile", np.load(shard_meta_path))
        shard_starts = cast(
            "npt.NDArray[np.int32]",
            shard_metadata["shard_starts"],
        )
        shard_sizes = cast(
            "npt.NDArray[np.int32]",
            shard_metadata["shard_sizes"],
        )
        _ = cls(  # pylint: disable=unnecessary-dunder-call
            lengths,
            shard_starts,
            shard_sizes,
            log_file,
        ).__enter__()


# how does this differ from DDP's default "no_sync" context manager?
class DDPNoSync:
    """Suppresses DDP gradient all-reduces for non-final micro-batches.

    Wraps ``model.no_sync()`` on all but the last micro-batch in an
    accumulation
    window so the all-reduce fires exactly once, on the final backward pass.
    Falls back to a no-op when the model does not expose ``no_sync``
    (single-GPU).

    Args:
        model: The model (plain ``nn.Module`` or DDP-wrapped).
        is_last: Whether this is the final micro-batch in the accumulation
            window.
    """

    def __init__(self, model: torch.nn.Module, *, is_last: bool) -> None:
        maybe_no_sync: (
            Callable[[], contextlib.AbstractContextManager[None]] | None
        ) = getattr(model, "no_sync", None)
        if not is_last and callable(maybe_no_sync):
            self._ctx: contextlib.AbstractContextManager[None] = maybe_no_sync()
        else:
            self._ctx = contextlib.nullcontext()

    def __enter__(self) -> Self:
        """Enter the inner context manager.

        Returns:
            This ``DDPNoSync`` instance.
        """
        self._ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the inner context manager, propagating any exception.

        Args:
            exc_type: Exception class if one was raised inside the block,
                else ``None``.
            exc_val: Exception instance if one was raised, else ``None``.
            exc_tb: Traceback if an exception was raised, else ``None``.
        """
        _ = self._ctx.__exit__(exc_type, exc_val, exc_tb)


class StepContext:
    """Context manager that disables gradients and dropout for eval steps.

    On entry, switches the module to eval mode and disables autograd when
    ``train_mode=False``.  On exit, restores the module's original training
    state regardless of exceptions.

    Args:
        model: The model to manage (plain ``nn.Module`` or DDP-wrapped).
        train_mode: Pass ``True`` for a training step, ``False`` for eval.
    """

    def __init__(self, *, model: torch.nn.Module, train_mode: bool) -> None:
        self.model: torch.nn.Module = model
        self._train_mode: bool = train_mode  # desired state during the context.
        self._was_training: bool = False  # state to restore to on exit

    def __enter__(self) -> Self:
        """Switch model to eval mode and disable autograd if not training.

        Returns:
            This ``StepContext`` instance.
        """
        self._was_training = self.model.training
        if not self._train_mode:
            _ = self.model.eval()
        _ = torch.set_grad_enabled(self._train_mode)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore the model's original training state and re-enable autograd.

        Args:
            exc_type: Exception class if one was raised inside the block,
                else ``None``.
            exc_val: Exception instance if one was raised, else ``None``.
            exc_tb: Traceback if an exception was raised, else ``None``.
        """
        _ = self.model.train(self._was_training)
        _ = torch.set_grad_enabled(True)
