"""Context manager helpers for distributed training, structured logging, and error handling."""

import contextlib
import io
import json
import os
import traceback
from collections.abc import Callable
from types import TracebackType

import structlog
import torch
import torch.distributed as dist
from structlog.typing import EventDict, Processor, WrappedLogger

log = structlog.get_logger()


class DistProcessGroup:
    """Context manager that initialises and tears down a torch.distributed process group.

    Exposes rank, local_rank, world_size, device, and is_rank_zero after entry.
    """

    def __init__(self, backend: str = "nccl") -> None:
        """Initialise with the desired backend.

        Args:
            backend: Distributed backend passed to ``dist.init_process_group``.
        """
        self.backend = backend
        self.rank: int = -1
        self.local_rank: int = -1
        self.world_size: int = -1
        self.device: str = ""

    def __enter__(self) -> "DistProcessGroup":
        """Initialise the process group and populate distributed state."""
        dist.init_process_group(backend=self.backend)
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(self.local_rank)
        return self

    @property
    def is_rank_zero(self) -> bool:
        """Return True only on the process with global rank 0."""
        return self.rank == 0

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Destroy the process group regardless of whether an exception occurred."""
        dist.destroy_process_group()


class FatalOnError:
    """Context manager that logs any unhandled exception via structlog then exits with code 1."""

    def __enter__(self) -> None:
        """No setup required."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Log the exception and raise SystemExit(1) if one occurred."""
        if exc_val is not None:
            log.exception("fatal", error=str(exc_val), traceback=traceback.format_exc())
            raise SystemExit(1) from exc_val


class StructlogConfig:
    """Context manager that configures structlog and optionally tees log lines to a JSON file."""

    def __init__(self, is_rank_zero: bool, log_file: str | None = None) -> None:  # noqa: FBT001
        """Store configuration; no I/O happens until __enter__.

        Args:
            is_rank_zero: Whether this process should emit console and file output.
            log_file: Optional path to write structured JSON log lines.
        """
        self._is_rank_zero = is_rank_zero
        self._log_file = log_file
        self.f: io.TextIOWrapper | None = None

    def __enter__(self) -> "StructlogConfig":
        """Configure structlog and open the log file if requested."""
        processors: list[Processor] = [
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]
        if self._log_file and self._is_rank_zero:
            self.f = open(self._log_file, "w", buffering=1)
            processors.append(self.write_log_line)
        if self._is_rank_zero:
            processors.append(structlog.dev.ConsoleRenderer())
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(20),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        return self

    def write_log_line(
        self, _logger: WrappedLogger, _method: str | None, event_dict: EventDict
    ) -> EventDict:
        """Structlog processor: write the event dict as a JSON line and pass it through."""
        if self.f is not None:
            self.f.write(json.dumps(event_dict) + "\n")
        return event_dict

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the log file and reset structlog to a no-op configuration."""
        if self.f is not None:
            self.f.close()
        structlog.reset_defaults()


class DDPNoSync:
    """Suppresses DDP gradient all-reduces for non-final micro-batches.

    Wraps ``model.no_sync()`` on all but the last micro-batch in an accumulation
    window so the all-reduce fires exactly once, on the final backward pass.
    Falls back to a no-op when the model does not expose ``no_sync`` (single-GPU).
    """

    def __init__(self, model: torch.nn.Module, *, is_last: bool) -> None:
        """Select the inner context manager based on model type and micro-batch position.

        Args:
            model: The model (plain ``nn.Module`` or DDP-wrapped).
            is_last: Whether this is the final micro-batch in the accumulation window.
        """
        maybe_no_sync: Callable[[], contextlib.AbstractContextManager[None]] | None = getattr(
            model, "no_sync", None
        )
        if not is_last and callable(maybe_no_sync):
            self._ctx: contextlib.AbstractContextManager[None] = maybe_no_sync()
        else:
            self._ctx = contextlib.nullcontext()

    def __enter__(self) -> "DDPNoSync":
        """Enter the inner context manager."""
        self._ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the inner context manager."""
        self._ctx.__exit__(exc_type, exc_val, exc_tb)


class StepContext:
    """Context manager that disables gradients and dropout for eval steps.

    On entry, switches the module to eval mode and disables autograd when
    ``train=False``.  On exit, restores the module's original training state
    regardless of exceptions.
    """

    def __init__(self, *, model: torch.nn.Module, train: bool) -> None:
        """Store the module and desired mode; no side-effects until __enter__.

        Args:
            model: The model to manage (plain ``nn.Module`` or DDP-wrapped).
            train: Pass ``True`` for a training step, ``False`` for eval.
        """
        self.model = model
        self._train = train  # desired state during the context.
        self._was_training: bool = False  # state to restore to on exit

    def __enter__(self) -> "StepContext":
        """Switch model to eval mode and disable autograd if not training."""
        self._was_training = self.model.training
        if not self._train:
            self.model.eval()
        torch.set_grad_enabled(self._train)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore the model's original training state and re-enable autograd."""
        self.model.train(self._was_training)
        torch.set_grad_enabled(True)
