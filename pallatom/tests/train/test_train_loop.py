"""Tests for the training loop."""

import dataclasses
import json
import os
import pathlib
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
import structlog
import torch
import torch.nn as nn
from architecture.main_trunk import MainTrunk
from einops import rearrange, reduce
from helpers.atom_utils import RESTYPE_NUM_NO_X
from helpers.bucketed_sampler import BucketedBatchSampler
from helpers.data import make_bucketed_data_loaders, to_protein_batch_dynamic
from helpers.featurize import Distogram, ProteinBatch
from helpers.useful_objects import (
    ComponentNorms,
    EpochMetrics,
    LossMetrics,
    ModelSetup,
    ThroughputStatistics,
    manual_seed,
)
from jaxtyping import Float
from pydantic import ValidationError
from structlog.typing import FilteringBoundLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from train.train_config import (
    CheckpointParams,
    LoggingParams,
    ModelParams,
    TrainConfig,
    TrainingParams,
)
from train.train_loop import (
    component_grad_norms,
    evaluate,
    load_checkpoint,
    log_epoch,
    optimizer_step,
    process_accum_window,
    save_checkpoint,
    take_step,
    train,
    wavg,
    wavg_component_norms,
    wavg_loss_metrics,
    wavg_throughput_stats,
)

manual_seed(42)


_N_KEEP = 16
_C_RES = 32
_C_ATOM = 32
_C_PAIR = 32
_C_ATOMPAIR = 16
_F_REF_DIM = 35
_N_BINS = 8
_N_ATOM_BINS = 5
_K_UNIT = 1
_BATCH_TOKENS: int = _N_KEEP

N_BLOCKS_ATOM_TRANSFORMER_ENCODER = 3
N_HEADS_ATOM_TRANSFORMER_ENCODER = 4
N_BLOCKS_ATOM_TRANSFORMER_DECODER = 3
N_HEADS_ATOM_TRANSFORMER_DECODER = 4
N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER = 2
N_PAIFORMER_HEADS_TEMPLATE_EMBEDDER = 16
SIGMA_DATA = 16.0
N_AMINO = RESTYPE_NUM_NO_X
RESIDUE_NUMBER = 50

EPOCH = 7
LEARNING_RATE = 1e-4
TOLERANCE = 1e-5
TIGHT_TOLERANCE = 1e-3
EXPECTED_CHECKPOINT_KEYS = frozenset({"model", "optimizer", "scheduler", "best_val_loss"})


class _ListDataset(torch.utils.data.Dataset[ProteinBatch]):
    """In-memory dataset wrapping a list of pre-built ProteinBatch-compatible mappings."""

    def __init__(self, items: list[Mapping[str, Float[torch.Tensor, "..."] | list[str]]]) -> None:
        """Store the items list.

        Args:
            items: Mappings with atom_positions, atom_mask, residue_index, seq keys.
        """
        self._items = items

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def __getitem__(self, idx: int) -> ProteinBatch:
        """Return item at idx as a ProteinBatch.

        Args:
            idx: Index into the item list.
        """
        item = self._items[idx]
        return ProteinBatch(
            atom_positions=cast("torch.Tensor", item["atom_positions"]),
            atom_mask=cast("torch.Tensor", item["atom_mask"]),
            residue_index=cast("torch.Tensor", item["residue_index"]),
            seq=cast("list[str]", item["seq"]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockSampler:
    """Batch sampler with set_epoch() for verifying epoch reseeding in train tests."""

    def __init__(self, data: list[Mapping[str, Float[torch.Tensor, "..."] | list[str]]]) -> None:
        """Store data and initialise set_epoch call log.

        Args:
            data: Dataset items (used only for length).
        """
        self._data = data
        self.set_epoch_calls: list[int] = []

    def __iter__(self):
        """Yield single-index lists for each item."""
        return iter([[i] for i in range(len(self._data))])

    def __len__(self) -> int:
        """Return number of items."""
        return len(self._data)

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch number for later assertion.

        Args:
            epoch: The epoch number passed by the training loop.
        """
        self.set_epoch_calls.append(epoch)


def _identity_collate(items: list[ProteinBatch]) -> ProteinBatch:
    """Return the single ProteinBatch item from a singleton batch list.

    Args:
        items: A one-element list produced by the DataLoader.

    Returns:
        The unwrapped ProteinBatch.
    """
    return items[0]


def _make_model_params(
    model: MainTrunk | DDP,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> ModelSetup:
    """Construct a ModelSetup with Adam optimizer and CosineAnnealingLR scheduler.

    Args:
        model: The MainTrunk to wrap.
        tcfg: Training configuration.
        distogram_res: Residue-level distogram module.
        distogram_atom: Atom-level distogram module.

    Returns:
        Fully initialised ModelSetup.
    """
    optimizer = Adam(model.parameters(), lr=tcfg.training.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=tcfg.training.num_epochs, eta_min=tcfg.training.lr * 0.01
    )
    return ModelSetup(
        model=model,
        tcfg=tcfg,
        distogram_res=distogram_res,
        distogram_atom=distogram_atom,
        device="cpu",
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _make_mock_loader(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    n_copies: int = 1,
) -> tuple[torch.utils.data.DataLoader[ProteinBatch], _MockSampler]:
    """Build a DataLoader with a _MockSampler containing n_copies of mini_batch.

    Args:
        mini_batch: The batch mapping to replicate.
        n_copies: How many copies to put in the dataset.

    Returns:
        Tuple of (DataLoader, _MockSampler) so callers can inspect set_epoch calls.
    """
    items = [mini_batch] * n_copies
    sampler = _MockSampler(items)
    loader: torch.utils.data.DataLoader[ProteinBatch] = torch.utils.data.DataLoader(
        _ListDataset(items), batch_sampler=sampler, collate_fn=_identity_collate
    )
    return loader, sampler


def _make_epoch_metrics(
    epoch: int,
    global_step: int = 0,
    zero_loss_metrics: LossMetrics | None = None,
    zero_throughput: ThroughputStatistics | None = None,
    zero_component_norms: ComponentNorms | None = None,
) -> EpochMetrics:
    """Build a minimal EpochMetrics with zero-valued fields for log_epoch tests.

    Args:
        epoch: Epoch number to embed.
        global_step: Global step count.
        zero_loss_metrics: LossMetrics with all-zero fields; built internally if None.
        zero_throughput: ThroughputStatistics with all-zero fields; built internally if None.
        zero_component_norms: ComponentNorms with all-zero fields; built internally if None.

    Returns:
        EpochMetrics suitable for passing to log_epoch.
    """
    z = torch.tensor(0.0)
    lm = zero_loss_metrics or LossMetrics(
        total_loss=z,
        Kabsch_aligned_MSE_loss=z,
        CE_loss=z,
        smooth_lddt_loss=z,
        res_distogram_loss=z,
        atom_distogram_loss=z,
        intermediate_loss=z,
        RMSD=z,
    )
    tp = zero_throughput or ThroughputStatistics(
        avg_batch_size=z, token_pack_rate=z, residues_per_sec=z, atoms_per_sec=z
    )
    cn = zero_component_norms or ComponentNorms(
        template_embedder=z,
        atom_encoder=z,
        atom_decoders=z,
        residue_distogram_head=z,
        atom_distogram_head=z,
        inter_proj_seq=z,
        inter_seq_logits=z,
        proj_seq=z,
        seq_logits=z,
    )
    return EpochMetrics(
        epoch=epoch,
        global_step=global_step,
        train_loss_metrics=lm,
        train_throughput_stats=tp,
        train_gradient_norms=cn,
        val_loss_metrics=lm,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zero_loss_metrics() -> LossMetrics:
    """Build a LossMetrics with all fields set to zero.

    Returns:
        LossMetrics with torch.tensor(0.0) in every field.
    """
    z = torch.tensor(0.0)
    return LossMetrics(
        total_loss=z,
        Kabsch_aligned_MSE_loss=z,
        CE_loss=z,
        smooth_lddt_loss=z,
        res_distogram_loss=z,
        atom_distogram_loss=z,
        intermediate_loss=z,
        RMSD=z,
    )


@pytest.fixture
def zero_throughput() -> ThroughputStatistics:
    """Build a ThroughputStatistics with all fields set to zero.

    Returns:
        ThroughputStatistics with torch.tensor(0.0) in every field.
    """
    z = torch.tensor(0.0)
    return ThroughputStatistics(
        avg_batch_size=z, token_pack_rate=z, residues_per_sec=z, atoms_per_sec=z
    )


@pytest.fixture
def zero_component_norms() -> ComponentNorms:
    """Build a ComponentNorms with all fields set to zero.

    Returns:
        ComponentNorms with torch.tensor(0.0) in every field.
    """
    z = torch.tensor(0.0)
    return ComponentNorms(
        template_embedder=z,
        atom_encoder=z,
        atom_decoders=z,
        residue_distogram_head=z,
        atom_distogram_head=z,
        inter_proj_seq=z,
        inter_seq_logits=z,
        proj_seq=z,
        seq_logits=z,
    )


@pytest.fixture
def distogram_res() -> Distogram:
    """Provide a residue-level Distogram in eval mode."""
    return Distogram(n_bins=_N_BINS, min_dist=3.25, max_dist=50.75, overflow_bin=False).eval()


@pytest.fixture
def distogram_atom() -> Distogram:
    """Provide an atom-level Distogram in eval mode."""
    return Distogram(n_bins=_N_ATOM_BINS, min_dist=0.0, max_dist=10.0, overflow_bin=False).eval()


@pytest.fixture
def model() -> MainTrunk:
    """Provide a small MainTrunk instance for training loop tests."""
    return MainTrunk(
        f_ref_dim=_F_REF_DIM,
        n_bins=_N_BINS,
        n_atom_bins=_N_ATOM_BINS,
        c_atom=_C_ATOM,
        c_pair=_C_PAIR,
        c_res=_C_RES,
        c_atompair=_C_ATOMPAIR,
        K_unit=_K_UNIT,
        n_blocks_atom_transformer_encoder=N_BLOCKS_ATOM_TRANSFORMER_ENCODER,
        n_heads_atom_transformer_encoder=N_HEADS_ATOM_TRANSFORMER_ENCODER,
        n_blocks_atom_transformer_decoder=N_BLOCKS_ATOM_TRANSFORMER_DECODER,
        n_heads_atom_transformer_decoder=N_HEADS_ATOM_TRANSFORMER_DECODER,
        n_pairformer_blocks_template_embedder=N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER,
        n_paiformer_heads_template_embedder=N_HEADS_ATOM_TRANSFORMER_DECODER,
        sigma_data=SIGMA_DATA,
        n_amino=N_AMINO,
        residue_number=RESIDUE_NUMBER,
    )


@pytest.fixture
def tcfg(tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a single-epoch TrainConfig with a temporary checkpoint path."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=LEARNING_RATE, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "best.pt"), save_every=100),
        logging=LoggingParams(use_wandb=False),
    )


@pytest.fixture
def tcfg3(model_params: ModelSetup, tmp_path: pathlib.Path) -> TrainConfig:
    """Provide a three-epoch TrainConfig sharing model params from model_params fixture."""
    return TrainConfig(
        training=TrainingParams(
            num_epochs=3, lr=LEARNING_RATE, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "best.pt"), save_every=100),
        logging=LoggingParams(use_wandb=False),
    )


@pytest.fixture
def model_params(
    model: MainTrunk,
    tcfg: TrainConfig,
    distogram_res: Distogram,
    distogram_atom: Distogram,
) -> ModelSetup:
    """Provide a ModelSetup bundling model, optimizer, scheduler, and config."""
    return _make_model_params(model, tcfg, distogram_res, distogram_atom)


@pytest.fixture
def single_sample() -> Mapping[str, Float[torch.Tensor, "..."] | str]:
    """Provide a single unbatched protein sample for to_protein_batch_dynamic tests."""
    return {
        "atom_positions": torch.randn(_N_KEEP, 37, 3),
        "atom_mask": torch.ones(_N_KEEP, 37),
        "residue_index": torch.arange(_N_KEEP, dtype=torch.float32),
        "seq": ("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP],
    }


@pytest.fixture
def mini_batch() -> Mapping[str, Float[torch.Tensor, "..."] | list[str]]:
    """Provide a single-item mini-batch of random tensors with _N_KEEP residues."""
    return {
        "atom_positions": torch.randn(1, _N_KEEP, 37, 3),
        "atom_mask": torch.ones(1, _N_KEEP, 37),
        "residue_index": rearrange(torch.arange(_N_KEEP, dtype=torch.float32), "n -> 1 n"),
        "seq": [("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP]],
    }


@pytest.fixture
def protein_batch() -> ProteinBatch:
    """Provide a single-item ProteinBatch with _N_KEEP residues for take_step tests."""
    return ProteinBatch(
        atom_positions=torch.randn(1, _N_KEEP, 37, 3),
        atom_mask=torch.ones(1, _N_KEEP, 37),
        residue_index=rearrange(torch.arange(_N_KEEP, dtype=torch.float32), "n -> 1 n"),
        seq=[("ACDEFGHIKLMNPQRSTVWY" * (_N_KEEP // 20 + 1))[:_N_KEEP]],
    )


@pytest.fixture
def plain_loader(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
) -> torch.utils.data.DataLoader[ProteinBatch]:
    """Provide a DataLoader that yields mini_batch directly (no set_epoch needed)."""
    return torch.utils.data.DataLoader(_ListDataset([mini_batch]), batch_size=None)


@pytest.fixture
def mock_loader(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
) -> torch.utils.data.DataLoader[ProteinBatch]:
    """Provide a DataLoader with _MockSampler (supports set_epoch) for train tests."""
    loader, _ = _make_mock_loader(mini_batch)
    return loader


@pytest.fixture
def mock_sampler(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
) -> BucketedBatchSampler:
    """Provide a BucketedBatchSampler-typed mock sampler for train() call sites."""
    _, sampler = _make_mock_loader(mini_batch)
    return cast(BucketedBatchSampler, sampler)


@pytest.fixture
def log() -> FilteringBoundLogger:
    """Provide a structlog logger for training loop tests."""
    return structlog.get_logger()


@pytest.fixture
def wandb_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch wandb.log and return the list of captured payload dicts."""
    captured: list[dict[str, object]] = []
    monkeypatch.setattr("train.train_loop.wandb.log", captured.append)
    return captured


@pytest.fixture
def wandb_call_counter(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch wandb.log and return a list that grows by one entry per call."""
    called: list[object] = []
    monkeypatch.setattr("train.train_loop.wandb.log", called.append)
    return called


# ---------------------------------------------------------------------------
# to_protein_batch_dynamic
# ---------------------------------------------------------------------------


def test_to_protein_batch_dynamic_returns_protein_batch(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic returns a ProteinBatch dataclass instance."""
    assert isinstance(to_protein_batch_dynamic([single_sample]), ProteinBatch)


def test_to_protein_batch_dynamic_atom_positions_shape(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic stacks atom_positions into batched tensor with leading batch dim."""
    result = to_protein_batch_dynamic([single_sample])
    assert result.atom_positions.shape == torch.Size(
        [1, *cast("torch.Tensor", single_sample["atom_positions"]).shape]
    )


def test_to_protein_batch_dynamic_atom_mask_shape(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic stacks atom_mask into a batched tensor with a leading batch dim."""
    result = to_protein_batch_dynamic([single_sample])
    assert result.atom_mask.shape == torch.Size(
        [1, *cast("torch.Tensor", single_sample["atom_mask"]).shape]
    )


def test_to_protein_batch_dynamic_residue_index_shape(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic stacks residue_index into batched tensor with leading batch dim."""
    result = to_protein_batch_dynamic([single_sample])
    assert result.residue_index.shape == torch.Size(
        [1, *cast("torch.Tensor", single_sample["residue_index"]).shape]
    )


def test_to_protein_batch_dynamic_seq_is_list(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic wraps the sequence field into a Python list."""
    result = to_protein_batch_dynamic([single_sample])
    assert isinstance(result.seq, list)


def test_to_protein_batch_dynamic_seq_elements_are_strings(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic seq list contains only str elements."""
    result = to_protein_batch_dynamic([single_sample])
    assert all(isinstance(s, str) for s in result.seq)


def test_to_protein_batch_dynamic_seq_length_matches_batch_size(
    single_sample: Mapping[str, Float[torch.Tensor, "..."] | str],
) -> None:
    """to_protein_batch_dynamic seq list has one entry per sample passed in."""
    result = to_protein_batch_dynamic([single_sample])
    assert len(result.seq) == 1


# ---------------------------------------------------------------------------
# wavg / wavg_loss_metrics / wavg_throughput_stats / wavg_component_norms
# ---------------------------------------------------------------------------


def testwavg_equal_weights_is_simple_mean() -> None:
    """Wavg with equal protein counts returns the simple mean."""
    vals = [torch.tensor(2.0), torch.tensor(4.0)]
    assert abs(wavg(vals, [1, 1]).item() - 3.0) < TOLERANCE


def testwavg_unequal_weights_biases_toward_larger_count() -> None:
    """Wavg with unequal counts biases the result toward the larger-count value."""
    vals = [torch.tensor(0.0), torch.tensor(10.0)]
    assert abs(wavg(vals, [1, 9]).item() - 9.0) < TOLERANCE


def testwavg_loss_metrics_equal_weights_is_mean() -> None:
    """wavg_loss_metrics with equal counts returns simple mean per field."""
    z2 = torch.tensor(2.0)
    z4 = torch.tensor(4.0)
    a = LossMetrics(
        total_loss=z2,
        Kabsch_aligned_MSE_loss=z2,
        CE_loss=z2,
        smooth_lddt_loss=z2,
        res_distogram_loss=z2,
        atom_distogram_loss=z2,
        intermediate_loss=z2,
        RMSD=z2,
    )
    b = LossMetrics(
        total_loss=z4,
        Kabsch_aligned_MSE_loss=z4,
        CE_loss=z4,
        smooth_lddt_loss=z4,
        res_distogram_loss=z4,
        atom_distogram_loss=z4,
        intermediate_loss=z4,
        RMSD=z4,
    )
    result = wavg_loss_metrics([a, b], [1, 1])
    assert abs(result.total_loss.item() - 3.0) < TOLERANCE


def testwavg_throughput_stats_equal_weights_is_mean() -> None:
    """wavg_throughput_stats with equal counts returns simple mean per field."""
    a = ThroughputStatistics(
        avg_batch_size=torch.tensor(2.0),
        token_pack_rate=torch.tensor(2.0),
        residues_per_sec=torch.tensor(2.0),
        atoms_per_sec=torch.tensor(2.0),
    )
    b = ThroughputStatistics(
        avg_batch_size=torch.tensor(6.0),
        token_pack_rate=torch.tensor(6.0),
        residues_per_sec=torch.tensor(6.0),
        atoms_per_sec=torch.tensor(6.0),
    )
    result = wavg_throughput_stats([a, b], [1, 1])
    assert abs(result.residues_per_sec.item() - 4.0) < TOLERANCE


def testwavg_component_norms_equal_weights_is_mean() -> None:
    """wavg_component_norms with equal counts returns simple mean per field."""
    a = ComponentNorms(
        template_embedder=torch.tensor(1.0),
        atom_encoder=torch.tensor(1.0),
        atom_decoders=torch.tensor(1.0),
        residue_distogram_head=torch.tensor(1.0),
        atom_distogram_head=torch.tensor(1.0),
        inter_proj_seq=torch.tensor(1.0),
        inter_seq_logits=torch.tensor(1.0),
        proj_seq=torch.tensor(1.0),
        seq_logits=torch.tensor(1.0),
    )
    b = ComponentNorms(
        template_embedder=torch.tensor(3.0),
        atom_encoder=torch.tensor(3.0),
        atom_decoders=torch.tensor(3.0),
        residue_distogram_head=torch.tensor(3.0),
        atom_distogram_head=torch.tensor(3.0),
        inter_proj_seq=torch.tensor(3.0),
        inter_seq_logits=torch.tensor(3.0),
        proj_seq=torch.tensor(3.0),
        seq_logits=torch.tensor(3.0),
    )
    result = wavg_component_norms([a, b], [1, 1])
    assert abs(result.atom_encoder.item() - 2.0) < TOLERANCE


# ---------------------------------------------------------------------------
# take_step — eval mode
# ---------------------------------------------------------------------------


def test_take_step_eval_returns_loss_metrics(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """take_step(train=False) returns a LossMetrics dataclass."""
    loss_metrics, _ = take_step(batch=protein_batch, model_params=model_params, train=False)
    assert isinstance(loss_metrics, LossMetrics)


def test_take_step_eval_returns_throughput_stats(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """take_step(train=False) returns a ThroughputStatistics dataclass."""
    _, tput = take_step(batch=protein_batch, model_params=model_params, train=False)
    assert isinstance(tput, ThroughputStatistics)


def test_take_step_eval_losses_are_finite(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """All LossMetrics fields from take_step(train=False) are finite."""
    loss_metrics, _ = take_step(batch=protein_batch, model_params=model_params, train=False)
    for f in dataclasses.fields(loss_metrics):
        v = getattr(loss_metrics, f.name)
        assert torch.isfinite(v), f"Field '{f.name}' is not finite: {v}"


def test_take_step_eval_token_pack_rate_in_range(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """token_pack_rate from take_step(train=False) is in (0, 1]."""
    _, tput = take_step(batch=protein_batch, model_params=model_params, train=False)
    assert 0.0 < tput.token_pack_rate.item() <= 1.0


def test_take_step_eval_throughput_positive(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """residues_per_sec and atoms_per_sec from take_step(train=False) are positive."""
    _, tput = take_step(batch=protein_batch, model_params=model_params, train=False)
    assert tput.residues_per_sec.item() > 0.0
    assert tput.atoms_per_sec.item() > 0.0


def test_take_step_eval_accumulates_no_gradients(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """take_step(train=False) leaves all model parameters gradient-free."""
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=False)
    for p in model_params.model.parameters():
        assert p.grad is None


def test_take_step_eval_rmsd_non_negative(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """RMSD from take_step(train=False) is non-negative."""
    loss_metrics, _ = take_step(batch=protein_batch, model_params=model_params, train=False)
    assert loss_metrics.RMSD.item() >= 0.0


# ---------------------------------------------------------------------------
# take_step — train mode
# ---------------------------------------------------------------------------


def test_take_step_train_produces_gradients(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """take_step(train=True) back-propagates gradients into model parameters."""
    model_params.model.train()
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=True)
    assert any(p.grad is not None for p in model_params.model.parameters())


def test_take_step_grad_scale_halves_gradient_norm(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """grad_scale=2 yields approximately half the gradient L2 norm of grad_scale=1."""
    model_params.model.train()

    manual_seed(0)
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=True, grad_scale=1.0)
    norm1: float = (
        sum(
            cast(float, reduce(p.grad**2, "... -> ", "sum").item())
            for p in model_params.model.parameters()
            if p.grad is not None
        )
        ** 0.5
    )

    manual_seed(0)
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=True, grad_scale=2.0)
    norm2: float = (
        sum(
            cast(float, reduce(p.grad**2, "... -> ", "sum").item())
            for p in model_params.model.parameters()
            if p.grad is not None
        )
        ** 0.5
    )

    assert abs(norm2 - norm1 / 2.0) < TIGHT_TOLERANCE


# ---------------------------------------------------------------------------
# process_accum_window
# ---------------------------------------------------------------------------


def test_process_accum_window_returns_loss_metrics(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """process_accum_window returns a LossMetrics instance."""
    model_params.model.train()
    model_params.model.zero_grad()
    loss_metrics, _ = process_accum_window([protein_batch], [1], model_params)
    assert isinstance(loss_metrics, LossMetrics)


def test_process_accum_window_returns_throughput_stats(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """process_accum_window returns a ThroughputStatistics instance."""
    model_params.model.train()
    model_params.model.zero_grad()
    _, tput = process_accum_window([protein_batch], [1], model_params)
    assert isinstance(tput, ThroughputStatistics)


def test_process_accum_window_losses_are_finite(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """All LossMetrics fields from process_accum_window are finite."""
    model_params.model.train()
    model_params.model.zero_grad()
    loss_metrics, _ = process_accum_window([protein_batch], [1], model_params)
    for f in dataclasses.fields(loss_metrics):
        v = getattr(loss_metrics, f.name)
        assert torch.isfinite(v), f"Field '{f.name}' is not finite: {v}"


def test_process_accum_window_protein_weighted_grad_scale(
    protein_batch: ProteinBatch,
    model_params: ModelSetup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_accum_window passes protein-count-weighted grad_scale to take_step.

    n_proteins=[1, 3] → total=4 → grad_scale 4/1=4.0 and 4/3≈1.333.
    """
    captured_scales: list[float] = []
    _real_take_step = take_step

    def _recording_take_step(
        *,
        batch: ProteinBatch,
        model_params: ModelSetup,
        train: bool,
        grad_scale: float = 1.0,
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        captured_scales.append(grad_scale)
        return _real_take_step(
            batch=batch, model_params=model_params, train=train, grad_scale=grad_scale
        )

    monkeypatch.setattr("train.train_loop.take_step", _recording_take_step)
    model_params.model.train()
    model_params.model.zero_grad()
    acummulated_batch = [protein_batch, protein_batch]
    process_accum_window(acummulated_batch, [1, 3], model_params)

    assert len(captured_scales) == len(acummulated_batch)
    assert abs(captured_scales[0] - 4.0) < TOLERANCE
    assert abs(captured_scales[1] - 4.0 / 3.0) < TOLERANCE


# ---------------------------------------------------------------------------
# component_grad_norms
# ---------------------------------------------------------------------------


def test_component_grad_norms_returns_component_norms(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """component_grad_norms returns a ComponentNorms dataclass after a backward pass."""
    model_params.model.train()
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=True)
    assert isinstance(component_grad_norms(model_params.model), ComponentNorms)


def test_component_grad_norms_are_positive(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """All per-component gradient norms are positive after a backward pass."""
    model_params.model.train()
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=True)
    norms = component_grad_norms(model_params.model)
    for f in dataclasses.fields(norms):
        v = getattr(norms, f.name)
        assert v.item() > 0.0, f"Component norm '{f.name}' is not positive: {v}"


# ---------------------------------------------------------------------------
# optimizer_step
# ---------------------------------------------------------------------------


def test_optimizer_step_increments_global_step(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """optimizer_step returns global_step + 1 as the fourth element."""
    model_params.model.train()
    model_params.model.zero_grad()
    global_step = 5
    _, _, _, new_step = optimizer_step([protein_batch], [1], model_params, global_step=global_step)
    assert new_step == global_step + 1


def test_optimizer_step_returns_component_norms(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """optimizer_step returns a ComponentNorms instance as the third element."""
    model_params.model.train()
    model_params.model.zero_grad()
    _, _, norms, _ = optimizer_step([protein_batch], [1], model_params, global_step=0)
    assert isinstance(norms, ComponentNorms)


def test_optimizer_step_returns_loss_metrics(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """optimizer_step returns a LossMetrics instance as the first element."""
    model_params.model.train()
    model_params.model.zero_grad()
    loss_metrics, _, _, _ = optimizer_step([protein_batch], [1], model_params, global_step=0)
    assert isinstance(loss_metrics, LossMetrics)


def test_optimizer_step_updates_model_parameters(
    protein_batch: ProteinBatch, model_params: ModelSetup
) -> None:
    """optimizer_step modifies at least one model parameter via gradient descent."""
    params_before = [p.clone().detach() for p in model_params.model.parameters()]
    model_params.model.train()
    model_params.model.zero_grad()
    optimizer_step([protein_batch], [1], model_params, global_step=0)
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, model_params.model.parameters(), strict=False)
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_loss_metrics(
    plain_loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> None:
    """Evaluate returns a LossMetrics instance as its first element."""
    loss_metrics, _ = evaluate(plain_loader, model_params)
    assert isinstance(loss_metrics, LossMetrics)


def test_evaluate_returns_throughput_stats(
    plain_loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> None:
    """Evaluate returns a ThroughputStatistics instance as its second element."""
    _, tput = evaluate(plain_loader, model_params)
    assert isinstance(tput, ThroughputStatistics)


def test_evaluate_losses_are_finite(
    plain_loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> None:
    """All LossMetrics fields returned by evaluate are finite."""
    loss_metrics, _ = evaluate(plain_loader, model_params)
    for f in dataclasses.fields(loss_metrics):
        v = getattr(loss_metrics, f.name)
        assert torch.isfinite(v), f"Field '{f.name}' is not finite: {v}"


def test_evaluate_total_loss_non_negative(
    plain_loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> None:
    """Total loss returned by evaluate is non-negative."""
    loss_metrics, _ = evaluate(plain_loader, model_params)
    assert loss_metrics.total_loss.item() >= 0.0


def test_evaluate_rmsd_non_negative(
    plain_loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> None:
    """RMSD returned by evaluate is non-negative."""
    loss_metrics, _ = evaluate(plain_loader, model_params)
    assert loss_metrics.RMSD.item() >= 0.0


def test_evaluate_accumulates_no_gradients(
    plain_loader: torch.utils.data.DataLoader[ProteinBatch], model_params: ModelSetup
) -> None:
    """Evaluate leaves all model parameters gradient-free."""
    evaluate(plain_loader, model_params)
    for p in model_params.model.parameters():
        assert p.grad is None


def test_evaluate_empty_loader_returns_zero_loss(model_params: ModelSetup) -> None:
    """Evaluate with an empty loader returns all-zero LossMetrics."""
    empty: torch.utils.data.DataLoader[ProteinBatch] = torch.utils.data.DataLoader(
        _ListDataset([]), batch_size=None
    )
    loss_metrics, _ = evaluate(empty, model_params)
    for f in dataclasses.fields(loss_metrics):
        assert getattr(loss_metrics, f.name).item() == 0.0


def test_evaluate_multi_batch_losses_finite(
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    model_params: ModelSetup,
) -> None:
    """Evaluate with a 3-batch loader still returns finite loss metrics."""
    loader3: torch.utils.data.DataLoader[ProteinBatch] = torch.utils.data.DataLoader(
        _ListDataset([mini_batch] * 3), batch_size=None
    )
    loss_metrics, _ = evaluate(loader3, model_params)
    for f in dataclasses.fields(loss_metrics):
        v = getattr(loss_metrics, f.name)
        assert torch.isfinite(v), f"Field '{f.name}' is not finite: {v}"


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


def test_save_checkpoint_rank0_writes_file(
    model_params: ModelSetup, log: FilteringBoundLogger
) -> None:
    """save_checkpoint writes a file on rank 0."""
    save_checkpoint(model_params=model_params, rank=0, log=log, best_val_loss=torch.tensor(1.0))
    assert os.path.exists(model_params.tcfg.checkpoint.checkpoint_path)


def test_save_checkpoint_rank1_skips_write(
    model_params: ModelSetup, log: FilteringBoundLogger
) -> None:
    """save_checkpoint does not write a file on rank != 0."""
    save_checkpoint(model_params=model_params, rank=1, log=log, best_val_loss=torch.tensor(1.0))
    assert not os.path.exists(model_params.tcfg.checkpoint.checkpoint_path)


def test_save_checkpoint_has_expected_keys(
    model_params: ModelSetup, log: FilteringBoundLogger
) -> None:
    """Checkpoint written by save_checkpoint has model, optimizer, scheduler, best_val_loss."""
    save_checkpoint(model_params=model_params, rank=0, log=log, best_val_loss=torch.tensor(0.5))
    ckpt = torch.load(model_params.tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert isinstance(ckpt, dict)
    assert ckpt.keys() >= EXPECTED_CHECKPOINT_KEYS


def test_load_checkpoint_restores_model_state(
    model_params: ModelSetup, log: FilteringBoundLogger
) -> None:
    """load_checkpoint restores model weights matching the saved state dict."""
    save_checkpoint(model_params=model_params, rank=0, log=log, best_val_loss=torch.tensor(0.42))
    original_sd = torch.load(model_params.tcfg.checkpoint.checkpoint_path, weights_only=True)[
        "model"
    ]
    for p in model_params.model.parameters():
        nn.init.zeros_(p)
    restored, _ = load_checkpoint(model_params=model_params, rank=0, log=log)
    for k, v in restored.model.state_dict().items():
        assert torch.allclose(v, original_sd[k]), f"Param '{k}' not restored"


def test_load_checkpoint_returns_best_val_loss(
    model_params: ModelSetup, log: FilteringBoundLogger
) -> None:
    """load_checkpoint returns the best_val_loss stored in the checkpoint."""
    save_checkpoint(model_params=model_params, rank=0, log=log, best_val_loss=torch.tensor(0.314))
    _, loaded_loss = load_checkpoint(model_params=model_params, rank=0, log=log)
    assert abs(loaded_loss.item() - 0.314) < TOLERANCE


def test_checkpoint_round_trip_preserves_weights_and_loss(
    model_params: ModelSetup, log: FilteringBoundLogger
) -> None:
    """save→load→save is lossless: second checkpoint matches first for weights and best_val_loss."""
    best_val_loss = torch.tensor(0.271)
    path = model_params.tcfg.checkpoint.checkpoint_path

    save_checkpoint(model_params=model_params, rank=0, log=log, best_val_loss=best_val_loss)
    first_sd = {k: v.clone() for k, v in model_params.model.state_dict().items()}

    for p in model_params.model.parameters():
        nn.init.zeros_(p)

    restored, loaded_loss = load_checkpoint(model_params=model_params, rank=0, log=log)
    save_checkpoint(model_params=restored, rank=0, log=log, best_val_loss=loaded_loss)

    second_ckpt = torch.load(path, weights_only=True)
    for k, v in first_sd.items():
        assert torch.allclose(v, second_ckpt["model"][k]), f"Param '{k}' differs after round trip"
    assert abs(second_ckpt["best_val_loss"].item() - best_val_loss.item()) < TOLERANCE


# ---------------------------------------------------------------------------
# log_epoch
# ---------------------------------------------------------------------------


def _tcfg_with_wandb(*, base: TrainConfig, use_wandb: bool) -> TrainConfig:
    """Return a copy of base with use_wandb overridden.

    Args:
        base: Base TrainConfig to copy from.
        use_wandb: Whether to enable W&B logging.

    Returns:
        New TrainConfig with logging.use_wandb set to use_wandb.
    """
    return TrainConfig(
        training=base.training,
        model=base.model,
        checkpoint=base.checkpoint,
        logging=LoggingParams(use_wandb=use_wandb),
    )


def test_log_epoch_do_log_false_skips_wandb(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_call_counter: list[object],
) -> None:
    """log_epoch with do_log=False does not call wandb.log even if use_wandb=True."""
    mp = dataclasses.replace(
        model_params, tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True)
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=1, global_step=1),
        model_params=mp,
        log=log,
        do_log=False,
    )
    assert len(wandb_call_counter) == 0


def test_log_epoch_wandb_disabled_skips_call(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_call_counter: list[object],
) -> None:
    """log_epoch with use_wandb=False does not call wandb.log even if do_log=True."""
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=1, global_step=1),
        model_params=model_params,
        log=log,
        do_log=True,
    )
    assert len(wandb_call_counter) == 0


def test_log_epoch_wandb_enabled_calls_once(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """log_epoch with do_log=True and use_wandb=True calls wandb.log exactly once."""
    mp = dataclasses.replace(
        model_params, tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True)
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=3, global_step=1),
        model_params=mp,
        log=log,
        do_log=True,
    )
    assert len(wandb_payloads) == 1


def test_log_epoch_wandb_payload_has_correct_epoch(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """log_epoch W&B payload 'epoch' key equals the epoch embedded in EpochMetrics."""
    mp = dataclasses.replace(
        model_params, tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True)
    )
    log_epoch(epoch_metrics=_make_epoch_metrics(epoch=EPOCH), model_params=mp, log=log, do_log=True)
    assert wandb_payloads[0]["epoch"] == EPOCH


def test_log_epoch_wandb_payload_has_train_prefix(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """log_epoch W&B payload contains at least one key prefixed with 'train/'."""
    mp = dataclasses.replace(
        model_params, tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True)
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=1, global_step=1),
        model_params=mp,
        log=log,
        do_log=True,
    )
    assert any(k.startswith("train/") for k in wandb_payloads[0])


def test_log_epoch_wandb_payload_has_val_prefix(
    model_params: ModelSetup,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """log_epoch W&B payload contains at least one key prefixed with 'val/'."""
    mp = dataclasses.replace(
        model_params, tcfg=_tcfg_with_wandb(base=model_params.tcfg, use_wandb=True)
    )
    log_epoch(
        epoch_metrics=_make_epoch_metrics(epoch=1, global_step=1),
        model_params=mp,
        log=log,
        do_log=True,
    )
    assert any(k.startswith("val/") for k in wandb_payloads[0])


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_returns_none(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
) -> None:
    """Train returns None (side-effect-only function)."""
    result = train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    assert result is None


def test_train_saves_checkpoint_when_val_improves(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
) -> None:
    """Train writes a checkpoint file at the configured path when val loss improves."""
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    assert os.path.exists(model_params.tcfg.checkpoint.checkpoint_path)


def test_train_checkpoint_has_expected_keys(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
) -> None:
    """Checkpoint written by train has model, optimizer, scheduler, best_val_loss."""
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    ckpt = torch.load(model_params.tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert ckpt.keys() >= EXPECTED_CHECKPOINT_KEYS


def test_train_updates_model_parameters(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
) -> None:
    """Train modifies at least one model parameter via gradient descent."""
    params_before = [p.clone().detach() for p in model_params.model.parameters()]
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, model_params.model.parameters(), strict=False)
    )


def test_train_calls_set_epoch_each_epoch(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tcfg3: TrainConfig,
    log: FilteringBoundLogger,
) -> None:
    """Train calls sampler.set_epoch(epoch) in ascending order once per epoch."""
    mp3 = _make_model_params(model_params.model, tcfg3, distogram_res, distogram_atom)
    loader, sampler = _make_mock_loader(mini_batch)
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, sampler),
        test_loader=loader,
        model_params=mp3,
        log=log,
    )
    assert sampler.set_epoch_calls == [1, 2, 3]


def test_train_no_grad_clip_completes(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    tmp_path: pathlib.Path,
) -> None:
    """Train completes without error when grad_clip is None."""
    tcfg_nc = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=None, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "no_clip.pt"), save_every=100),
        logging=LoggingParams(use_wandb=False),
    )
    mp_nc = _make_model_params(model_params.model, tcfg_nc, distogram_res, distogram_atom)
    loader, _sampler_nc = _make_mock_loader(mini_batch)
    assert (
        train(
            best_val_loss=torch.tensor(float("inf")),
            train_loader=loader,
            train_sampler=cast(BucketedBatchSampler, _sampler_nc),
            test_loader=loader,
            model_params=mp_nc,
            log=log,
        )
        is None
    )


def test_train_partial_window_at_epoch_end_updates_params(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    tmp_path: pathlib.Path,
) -> None:
    """Partial window is flushed at epoch end even when the token budget is not reached.

    With 1 batch of _BATCH_TOKENS and budget=2*_BATCH_TOKENS, no pre-flush fires,
    but the end-of-epoch flush still triggers one optimizer step so params must change.
    """
    tcfg_acc = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=2 * _BATCH_TOKENS
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "accum.pt"), save_every=100),
        logging=LoggingParams(use_wandb=False),
    )
    mp_acc = _make_model_params(model_params.model, tcfg_acc, distogram_res, distogram_atom)
    loader, _sampler_acc = _make_mock_loader(mini_batch)
    params_before = [p.clone().detach() for p in mp_acc.model.parameters()]
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, _sampler_acc),
        test_loader=loader,
        model_params=mp_acc,
        log=log,
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, mp_acc.model.parameters(), strict=False)
    )


def test_train_token_budget_preflush_fires_before_oversized_batch(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Pre-flush fires when the next batch would push tokens over the budget.

    Budget=24, two batches of 16 tokens each:
    - batch1 (buffer empty): added, no pre-flush
    - batch2: 16+16=32 > 24 → pre-flush batch1, then batch2 added
    - epoch end: batch2 flushed
    Result: two optimizer windows each of size 1.
    """
    window_sizes: list[int] = []
    _real_process = process_accum_window

    def _tracking_process(
        micro_buffer: list[ProteinBatch],
        n_proteins_per_batch: list[int],
        model_params: ModelSetup,
    ) -> tuple[LossMetrics, ThroughputStatistics]:
        window_sizes.append(len(micro_buffer))
        return _real_process(micro_buffer, n_proteins_per_batch, model_params)

    monkeypatch.setattr("train.train_loop.process_accum_window", _tracking_process)

    budget: int = _BATCH_TOKENS + _BATCH_TOKENS // 2  # 24
    tcfg_b = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=budget
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "flush.pt"), save_every=100),
        logging=LoggingParams(use_wandb=False),
    )
    mp_b = _make_model_params(model_params.model, tcfg_b, distogram_res, distogram_atom)
    loader, _sampler_b = _make_mock_loader(mini_batch, n_copies=2)
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, _sampler_b),
        test_loader=loader,
        model_params=mp_b,
        log=log,
    )
    assert window_sizes == [1, 1]


def test_train_wandb_called_once_per_epoch_when_enabled(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
    tmp_path: pathlib.Path,
) -> None:
    """Train calls wandb.log exactly once per epoch when use_wandb=True."""
    tcfg_w = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=_BATCH_TOKENS
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "wandb.pt"), save_every=100),
        logging=LoggingParams(use_wandb=True),
    )
    mp_w = _make_model_params(model_params.model, tcfg_w, distogram_res, distogram_atom)
    loader, _sampler_w = _make_mock_loader(mini_batch)
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, _sampler_w),
        test_loader=loader,
        model_params=mp_w,
        log=log,
    )
    assert len(wandb_payloads) == 1


def test_train_wandb_not_called_when_disabled(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
    wandb_call_counter: list[object],
) -> None:
    """Train does not call wandb.log when use_wandb=False."""
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    assert len(wandb_call_counter) == 0


def test_train_metrics_nonzero_when_budget_exceeds_total_tokens(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
    tmp_path: pathlib.Path,
) -> None:
    """train/total_loss logged to W&B is non-zero when the dataset fits under the token budget.

    Regression: if the end-of-epoch flush is dropped, no training occurs and loss = 0.
    """
    tcfg_large = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=10 * _BATCH_TOKENS
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(checkpoint_path=str(tmp_path / "large.pt"), save_every=100),
        logging=LoggingParams(use_wandb=True),
    )
    mp_large = _make_model_params(model_params.model, tcfg_large, distogram_res, distogram_atom)
    loader, _sampler_large = _make_mock_loader(mini_batch)
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, _sampler_large),
        test_loader=loader,
        model_params=mp_large,
        log=log,
    )
    assert len(wandb_payloads) == 1
    train_loss = wandb_payloads[0]["train/total_loss"]
    assert isinstance(train_loss, float)
    assert train_loss > 0.0, "train/total_loss is 0 — end-of-epoch flush was dropped"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_accumulated_token_budget_rejects_zero() -> None:
    """TrainingParams raises ValidationError when accumulated_token_budget is zero."""
    with pytest.raises(ValidationError):
        TrainingParams(accumulated_token_budget=0)


def test_train_config_accepts_any_positive_token_budget() -> None:
    """TrainConfig accepts any positive accumulated_token_budget."""
    cfg = TrainConfig(training=TrainingParams(accumulated_token_budget=1))
    assert cfg.training.accumulated_token_budget == 1


# ---------------------------------------------------------------------------
# BucketedBatchSampler integration
# ---------------------------------------------------------------------------

_N_RES_BUCKET = 6
_ENTRY_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
_TRAIN_NAMES_BUCKET = ["1aa.A", "2bb.A", "3cc.A"]
_VAL_NAMES_BUCKET = ["4dd.A"]
_TEST_NAMES_BUCKET = ["5ee.A"]


@pytest.fixture
def jsonl_path(tmp_path: pathlib.Path) -> str:
    """Write a temporary JSONL file with synthetic protein entries and return its path."""
    path = tmp_path / "proteins.jsonl"
    with open(path, "w") as f:
        for name in _ENTRY_NAMES_BUCKET:
            entry = {
                "name": name,
                "seq": "ACDEFG"[:_N_RES_BUCKET],
                "coords": {
                    atom: np.zeros((_N_RES_BUCKET, 3)).tolist() for atom in ("N", "CA", "C", "O")
                },
            }
            f.write(json.dumps(entry) + "\n")
    return str(path)


@pytest.fixture
def splits_path(tmp_path: pathlib.Path) -> str:
    """Write a temporary splits JSON with train/val/test name lists and return its path."""
    path = tmp_path / "splits.json"
    with open(path, "w") as f:
        json.dump(
            {
                "train": _TRAIN_NAMES_BUCKET,
                "validation": _VAL_NAMES_BUCKET,
                "test": _TEST_NAMES_BUCKET,
            },
            f,
        )
    return str(path)


def test_train_one_epoch_with_bucketed_loader(
    model_params: ModelSetup,
    jsonl_path: str,
    splits_path: str,
    log: FilteringBoundLogger,
) -> None:
    """train() completes one epoch with a real BucketedBatchSampler DataLoader."""
    train_loader, val_loader, _, train_sampler = make_bucketed_data_loaders(
        cfg=model_params.tcfg,
        jsonl_path=jsonl_path,
        splits_path=splits_path,
        num_workers=0,
        debug_run=False,
    )
    assert isinstance(train_sampler, BucketedBatchSampler)
    result = train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=train_loader,
        train_sampler=train_sampler,
        test_loader=val_loader,
        model_params=model_params,
        log=log,
    )
    assert result is None


def test_train_accum_full_window_updates_params(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    log: FilteringBoundLogger,
    tmp_path: pathlib.Path,
) -> None:
    """With accum budget=2x batch tokens and a 2-batch loader, train completes one window."""
    manual_seed(0)
    tcfg_acc = TrainConfig(
        training=TrainingParams(
            num_epochs=1, lr=1e-4, grad_clip=1.0, accumulated_token_budget=2 * _BATCH_TOKENS
        ),
        model=model_params.tcfg.model,
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "full_accum.pt"), save_every=100
        ),
        logging=LoggingParams(use_wandb=False),
    )
    mp_acc = _make_model_params(model_params.model, tcfg_acc, distogram_res, distogram_atom)
    loader, _sampler_acc2 = _make_mock_loader(mini_batch, n_copies=2)
    params_before = [p.clone().detach() for p in mp_acc.model.parameters()]
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, _sampler_acc2),
        test_loader=loader,
        model_params=mp_acc,
        log=log,
    )
    assert any(
        not torch.equal(b, a)
        for b, a in zip(params_before, mp_acc.model.parameters(), strict=False)
    )


def test_train_optimizer_state_non_zero_after_one_step(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
) -> None:
    """Adam exp_avg in the saved checkpoint is non-zero after one training step."""
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    ckpt = torch.load(model_params.tcfg.checkpoint.checkpoint_path, weights_only=True)
    opt_state = ckpt["optimizer"]["state"]
    assert len(opt_state) > 0
    first_state = next(iter(opt_state.values()))
    assert "exp_avg" in first_state
    assert first_state["exp_avg"].abs().max().item() > 0


def test_train_scheduler_state_reflects_completed_epoch(
    model_params: ModelSetup,
    mock_loader: torch.utils.data.DataLoader[ProteinBatch],
    mock_sampler: BucketedBatchSampler,
    log: FilteringBoundLogger,
) -> None:
    """Scheduler last_epoch in checkpoint equals the number of epochs completed."""
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=mock_loader,
        train_sampler=mock_sampler,
        test_loader=mock_loader,
        model_params=model_params,
        log=log,
    )
    ckpt = torch.load(model_params.tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert ckpt["scheduler"]["last_epoch"] == model_params.tcfg.training.num_epochs


def test_train_wandb_epoch_increments_across_epochs(
    model_params: ModelSetup,
    mini_batch: Mapping[str, Float[torch.Tensor, "..."] | list[str]],
    distogram_res: Distogram,
    distogram_atom: Distogram,
    tcfg3: TrainConfig,
    log: FilteringBoundLogger,
    wandb_payloads: list[dict[str, object]],
) -> None:
    """W&B epoch values logged across a 3-epoch run are [1, 2, 3]."""
    mp3 = _make_model_params(
        model_params.model,
        _tcfg_with_wandb(base=tcfg3, use_wandb=True),
        distogram_res,
        distogram_atom,
    )
    loader, _sampler_w3 = _make_mock_loader(mini_batch)
    train(
        best_val_loss=torch.tensor(float("inf")),
        train_loader=loader,
        train_sampler=cast(BucketedBatchSampler, _sampler_w3),
        test_loader=loader,
        model_params=mp3,
        log=log,
    )
    assert [p["epoch"] for p in wandb_payloads] == [1, 2, 3]


def test_integration_gradient_flow_through_all_submodules(
    protein_batch: ProteinBatch,
    model_params: ModelSetup,
) -> None:
    """take_step(train=True) back-propagates finite nonzero grads to every MainTrunk submodule."""
    model_params.model.train()
    model_params.model.zero_grad()
    take_step(batch=protein_batch, model_params=model_params, train=True)

    buckets: dict[str, list[torch.Tensor]] = {}
    for name, param in model_params.model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)

    assert buckets, "no parameters have gradients — backward was not called"
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0 for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"
