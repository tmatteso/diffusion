"""Context manager helpers for distributed training, structured logging, and error handling."""

import io
import json
import os
import traceback
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
        self._f: io.TextIOWrapper | None = None

    def __enter__(self) -> "StructlogConfig":
        """Configure structlog and open the log file if requested."""
        processors: list[Processor] = [
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]
        if self._log_file and self._is_rank_zero:
            self._f = open(self._log_file, "w", buffering=1)
            processors.append(self._write_log_line)
        if self._is_rank_zero:
            processors.append(structlog.dev.ConsoleRenderer())
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(20),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        return self

    def _write_log_line(
        self, _logger: WrappedLogger, _method: str | None, event_dict: EventDict
    ) -> EventDict:
        """Structlog processor: write the event dict as a JSON line and pass it through."""
        if self._f is not None:
            self._f.write(json.dumps(event_dict) + "\n")
        return event_dict

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the log file and reset structlog to a no-op configuration."""
        if self._f is not None:
            self._f.close()
        structlog.reset_defaults()
