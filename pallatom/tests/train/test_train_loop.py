import json
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from architecture.main_trunk import MainTrunk
from helpers.featurize import Distogram, ProteinBatch
from train.train_config import (
    CheckpointParams,
    LoggingParams,
    ModelParams,
    TrainConfig,
    TrainingParams,
)
import torch.distributed as dist_module
import torch.nn.parallel
from train.train_loop import _FileLogProcessor, _to_protein_batch, evaluate, evaluate_ddp, train, train_ddp

torch.manual_seed(42)

_TEST_DICT = os.path.join(os.path.dirname(__file__), "..", "..", "test_dict.pt")

_N_KEEP     = 16
_C_RES      = 32
_C_ATOM     = 32
_C_PAIR     = 32
_C_ATOMPAIR = 16
_F_REF_DIM  = 35
_N_BINS     = 8
_N_ATOM_BINS = 5
_K_UNIT     = 1

EXPECTED_EVAL_KEYS = frozenset({
    "total loss",
    "Kabsch aligned MSE loss",
    "Cross Entropy loss",
    "Smooth LDDT loss",
    "Residue Distogram loss",
    "Atom Distogram loss",
    "Intermediate loss",
    "RMSD",
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_batch() -> dict:
    return torch.load(_TEST_DICT, weights_only=False)


@pytest.fixture
def distogram_res() -> Distogram:
    return Distogram(n_bins=_N_BINS, min_dist=3.25, max_dist=50.75, overflow_bin=False).eval()


@pytest.fixture
def distogram_atom() -> Distogram:
    return Distogram(n_bins=_N_ATOM_BINS, min_dist=0.0, max_dist=10.0, overflow_bin=False).eval()


@pytest.fixture
def index_embedding() -> nn.Embedding:
    return nn.Embedding(256, _C_RES)


@pytest.fixture
def model() -> MainTrunk:
    return MainTrunk(
        f_ref_dim=_F_REF_DIM,
        n_bins=_N_BINS,
        n_atom_bins=_N_ATOM_BINS,
        c_atom=_C_ATOM,
        c_pair=_C_PAIR,
        c_res=_C_RES,
        c_atompair=_C_ATOMPAIR,
        K_unit=_K_UNIT,
    )


@pytest.fixture
def tcfg(tmp_path) -> TrainConfig:
    return TrainConfig(
        training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def mini_batch(raw_batch) -> dict:
    return {
        "atom_positions": raw_batch["atom_positions"][:1, :_N_KEEP],
        "atom_mask":      raw_batch["atom_mask"][:1, :_N_KEEP],
        "residue_index":  raw_batch["residue_index"][:1, :_N_KEEP],
        "seq":            [raw_batch["seq"][0][:_N_KEEP]],
    }


@pytest.fixture
def loader(mini_batch) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader([mini_batch], batch_size=None, collate_fn=lambda x: x)


@pytest.fixture
def tcfg_multi(tmp_path) -> TrainConfig:
    return TrainConfig(
        training=TrainingParams(num_epochs=3, lr=1e-3, grad_clip=1.0),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_multi.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def tcfg_no_clip(tmp_path) -> TrainConfig:
    return TrainConfig(
        training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=None),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_no_clip.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


@pytest.fixture
def tcfg_wandb(tmp_path) -> TrainConfig:
    return TrainConfig(
        training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_wandb.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=True, log_interval=1),
    )


@pytest.fixture
def tcfg_wandb_3ep(tmp_path) -> TrainConfig:
    return TrainConfig(
        training=TrainingParams(num_epochs=3, lr=1e-4, grad_clip=1.0),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_wandb_3ep.pt"),
            save_every=100,
        ),
        logging=LoggingParams(use_wandb=True, log_interval=1),
    )


@pytest.fixture
def tcfg_save(tmp_path) -> TrainConfig:
    return TrainConfig(
        training=TrainingParams(num_epochs=1, lr=1e-4, grad_clip=1.0),
        model=ModelParams(
            f_ref_dim=_F_REF_DIM,
            n_bins=_N_BINS,
            c_atom=_C_ATOM,
            c_pair=_C_PAIR,
            c_res=_C_RES,
            c_atompair=_C_ATOMPAIR,
            K_unit=_K_UNIT,
        ),
        checkpoint=CheckpointParams(
            checkpoint_path=str(tmp_path / "best_save_every.pt"),
            save_every=1,
        ),
        logging=LoggingParams(use_wandb=False, log_interval=1),
    )


# ---------------------------------------------------------------------------
# _to_protein_batch
# ---------------------------------------------------------------------------

def test_to_protein_batch_returns_protein_batch(mini_batch):
    assert isinstance(_to_protein_batch(mini_batch), ProteinBatch)


def test_to_protein_batch_atom_positions_shape(mini_batch):
    result = _to_protein_batch(mini_batch)
    assert result.atom_positions.shape == mini_batch["atom_positions"].shape


def test_to_protein_batch_atom_mask_shape(mini_batch):
    result = _to_protein_batch(mini_batch)
    assert result.atom_mask.shape == mini_batch["atom_mask"].shape


def test_to_protein_batch_residue_index_shape(mini_batch):
    result = _to_protein_batch(mini_batch)
    assert result.residue_index.shape == mini_batch["residue_index"].shape


def test_to_protein_batch_seq_is_list(mini_batch):
    result = _to_protein_batch(mini_batch)
    assert isinstance(result.seq, list)


def test_to_protein_batch_seq_elements_are_strings(mini_batch):
    result = _to_protein_batch(mini_batch)
    assert all(isinstance(s, str) for s in result.seq)


def test_to_protein_batch_seq_length_matches_batch_size(mini_batch):
    result = _to_protein_batch(mini_batch)
    assert len(result.seq) == mini_batch["atom_positions"].shape[0]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_returns_dict(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert isinstance(result, dict)


def test_evaluate_returns_expected_keys(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_all_values_are_floats(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert isinstance(v, float), f"'{k}' is {type(v)}, expected float"


def test_evaluate_all_losses_finite(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert v == v, f"NaN in '{k}'"
        assert v != float("inf"), f"Inf in '{k}'"


def test_evaluate_total_loss_non_negative(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result["total loss"] >= 0.0


def test_evaluate_rmsd_non_negative(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result["RMSD"] >= 0.0


def test_evaluate_sets_model_to_eval_mode(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    model.train()
    evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert not model.training


def test_evaluate_empty_loader_returns_zero_dict(model, tcfg, distogram_res, distogram_atom, index_embedding):
    empty = torch.utils.data.DataLoader([], batch_size=None, collate_fn=lambda x: x)
    result = evaluate(model, empty, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert all(v == 0.0 for v in result.values())


def test_evaluate_uses_no_grad(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for p in model.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_returns_none(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    assert train(model, tcfg, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu") is None


def test_train_saves_best_checkpoint(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_checkpoint_is_valid_state_dict(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert isinstance(ckpt, dict)
    assert "model" in ckpt and "index_embedding" in ckpt


def test_train_updates_model_parameters(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    params_before = [p.clone().detach() for p in model.parameters()]
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    params_after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


def test_train_model_in_train_mode_after(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    model.eval()
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert model.training


def test_train_runs_all_epochs(model, loader, tcfg_multi, distogram_res, distogram_atom, index_embedding, monkeypatch):
    losses: list[float] = []
    _real_evaluate = evaluate

    def _patched_evaluate(m, ldr, cfg, dr, da, emb, dev):
        result = _real_evaluate(m, ldr, cfg, dr, da, emb, dev)
        losses.append(result["total loss"])
        return result

    monkeypatch.setattr("train.train_loop.evaluate", _patched_evaluate)
    train(model, tcfg_multi, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")

    assert len(losses) == 3
    assert all(v == v and v != float("inf") for v in losses)


def test_train_no_grad_clip_runs(model, loader, tcfg_no_clip, distogram_res, distogram_atom, index_embedding):
    result = train(model, tcfg_no_clip, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result is None


def test_train_no_grad_clip_saves_checkpoint(model, loader, tcfg_no_clip, distogram_res, distogram_atom, index_embedding):
    train(model, tcfg_no_clip, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert os.path.exists(tcfg_no_clip.checkpoint.checkpoint_path)


def test_train_save_every_creates_epoch_checkpoint(model, loader, tcfg_save, distogram_res, distogram_atom, index_embedding, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train(model, tcfg_save, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert os.path.exists(tmp_path / "checkpoint_epoch_001.pt")


def test_train_save_every_checkpoint_is_valid_state_dict(model, loader, tcfg_save, distogram_res, distogram_atom, index_embedding, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train(model, tcfg_save, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    ckpt = torch.load(tmp_path / "checkpoint_epoch_001.pt", weights_only=True)
    assert isinstance(ckpt, dict)
    assert "model" in ckpt and "index_embedding" in ckpt


# ---------------------------------------------------------------------------
# evaluate – multi-batch averaging
# ---------------------------------------------------------------------------

def test_evaluate_multi_batch_returns_expected_keys(model, mini_batch, tcfg, distogram_res, distogram_atom, index_embedding):
    loader2 = torch.utils.data.DataLoader([mini_batch, mini_batch], batch_size=None, collate_fn=lambda x: x)
    result = evaluate(model, loader2, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_multi_batch_n_batches_counted(model, mini_batch, tcfg, distogram_res, distogram_atom, index_embedding):
    loader3 = torch.utils.data.DataLoader([mini_batch] * 3, batch_size=None, collate_fn=lambda x: x)
    result = evaluate(model, loader3, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert v == v and v != float("inf"), f"bad value for '{k}'"


# ---------------------------------------------------------------------------
# _FileLogProcessor
# ---------------------------------------------------------------------------

def test_file_log_processor_creates_file(tmp_path):
    path = str(tmp_path / "run.jsonl")
    _FileLogProcessor(path)
    assert os.path.exists(path)


def test_file_log_processor_returns_event_dict(tmp_path):
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    event = {"event": "test", "value": 1}
    returned = proc(None, None, event)
    assert returned is event


def test_file_log_processor_writes_json_line(tmp_path):
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    event = {"event": "train", "loss": 0.5}
    proc(None, None, event)
    proc._f.flush()
    with open(path) as fh:
        written = json.loads(fh.readline())
    assert written == event


def test_file_log_processor_writes_multiple_lines(tmp_path):
    path = str(tmp_path / "run.jsonl")
    proc = _FileLogProcessor(path)
    events = [{"event": "a", "x": 1}, {"event": "b", "x": 2}]
    for ev in events:
        proc(None, None, ev)
    proc._f.flush()
    with open(path) as fh:
        lines = fh.readlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == events


def test_file_log_processor_truncates_existing_file(tmp_path):
    path = str(tmp_path / "run.jsonl")
    with open(path, "w") as fh:
        fh.write('{"stale": true}\n' * 5)
    proc = _FileLogProcessor(path)
    proc(None, None, {"event": "fresh"})
    proc._f.flush()
    with open(path) as fh:
        lines = fh.readlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "fresh"}


# ---------------------------------------------------------------------------
# train – wandb logging branch
# ---------------------------------------------------------------------------

def test_train_calls_wandb_log_when_enabled(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    logged: list = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: logged.append((data, step)))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert len(logged) == 1


def test_train_wandb_log_not_called_when_disabled(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, monkeypatch):
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train(model, tcfg, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    mock_log.assert_not_called()


def test_train_wandb_log_payload_has_epoch_key(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert "epoch" in payloads[0]
    assert payloads[0]["epoch"] == 1


def test_train_wandb_log_payload_has_train_keys(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert any(k.startswith("train/") for k in payloads[0])


def test_train_wandb_log_payload_has_val_keys(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert any(k.startswith("val/") for k in payloads[0])


def test_train_wandb_log_step_is_positive(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    steps: list[int] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: steps.append(step))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert steps[0] >= 1


def test_train_wandb_log_called_once_per_epoch(model, loader, tcfg_wandb_3ep, distogram_res, distogram_atom, index_embedding, monkeypatch):
    logged: list = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: logged.append(data))
    train(model, tcfg_wandb_3ep, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert len(logged) == 3


def test_train_wandb_log_epoch_increments(model, loader, tcfg_wandb_3ep, distogram_res, distogram_atom, index_embedding, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: payloads.append(data))
    train(model, tcfg_wandb_3ep, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    epochs = [p["epoch"] for p in payloads]
    assert epochs == [1, 2, 3]


def test_train_wandb_log_train_total_loss_is_float(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert isinstance(payloads[0]["train/total loss"], float)


def test_train_wandb_log_val_total_loss_is_float(model, loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, monkeypatch):
    payloads: list[dict] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: payloads.append(data))
    train(model, tcfg_wandb, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert isinstance(payloads[0]["val/total loss"], float)


def test_train_wandb_log_steps_increase_across_epochs(model, loader, tcfg_wandb_3ep, distogram_res, distogram_atom, index_embedding, monkeypatch):
    steps: list[int] = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: steps.append(step))
    train(model, tcfg_wandb_3ep, loader, loader, distogram_res, distogram_atom, index_embedding, "cpu")
    assert steps == sorted(steps)
    assert len(set(steps)) == 3


# ---------------------------------------------------------------------------
# evaluate_ddp
# ---------------------------------------------------------------------------

def test_evaluate_ddp_returns_dict(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert isinstance(result, dict)


def test_evaluate_ddp_returns_expected_keys(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS


def test_evaluate_ddp_all_values_are_floats(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert isinstance(v, float), f"'{k}' is {type(v)}, expected float"


def test_evaluate_ddp_all_losses_finite(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k, v in result.items():
        assert v == v, f"NaN in '{k}'"
        assert v != float("inf"), f"Inf in '{k}'"


def test_evaluate_ddp_total_loss_non_negative(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result["total loss"] >= 0.0


def test_evaluate_ddp_rmsd_non_negative(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    result = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert result["RMSD"] >= 0.0


def test_evaluate_ddp_sets_model_to_eval_mode(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    model.train()
    evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert not model.training


def test_evaluate_ddp_empty_loader_returns_zero_dict(model, tcfg, distogram_res, distogram_atom, index_embedding):
    empty = torch.utils.data.DataLoader([], batch_size=None, collate_fn=lambda x: x)
    result = evaluate_ddp(0, 1, model, empty, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert all(v == 0.0 for v in result.values())


def test_evaluate_ddp_calls_all_reduce_once_when_world_size_gt_1(
    model, loader, tcfg, distogram_res, distogram_atom, index_embedding, monkeypatch
):
    calls = []
    monkeypatch.setattr(dist_module, "all_reduce", lambda t, op=None: calls.append(1))
    evaluate_ddp(0, 2, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert len(calls) == 1


def test_evaluate_ddp_skips_all_reduce_when_world_size_is_1(
    model, loader, tcfg, distogram_res, distogram_atom, index_embedding, monkeypatch
):
    calls = []
    monkeypatch.setattr(dist_module, "all_reduce", lambda t, op=None: calls.append(1))
    evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    assert len(calls) == 0


class _FakeDDP(nn.Module):
    """Minimal DDP stand-in that exposes .module and works on CPU."""

    def __init__(self, module, device_ids=None):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _MockSampler:
    """DataLoader sampler with set_epoch() for verifying epoch reseeding."""

    def __init__(self, data):
        self._data = data
        self.set_epoch_calls: list[int] = []

    def __iter__(self):
        return iter(range(len(self._data)))

    def __len__(self):
        return len(self._data)

    def set_epoch(self, epoch: int) -> None:
        self.set_epoch_calls.append(epoch)


@pytest.fixture
def ddp_loader(mini_batch):
    sampler = _MockSampler([mini_batch])
    loader = torch.utils.data.DataLoader([mini_batch], batch_size=None, sampler=sampler, collate_fn=lambda x: x)
    return loader


@pytest.fixture
def patch_ddp(monkeypatch):
    monkeypatch.setattr("train.train_loop.DDP", _FakeDDP)


# ---------------------------------------------------------------------------
# train_ddp
# ---------------------------------------------------------------------------

def test_train_ddp_returns_none(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    result = train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert result is None


def test_train_ddp_rank0_saves_checkpoint(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_ddp_checkpoint_has_correct_keys(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    ckpt = torch.load(tcfg.checkpoint.checkpoint_path, weights_only=True)
    assert "model" in ckpt and "index_embedding" in ckpt


def test_train_ddp_rank1_does_not_save_checkpoint(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(1, 0, 2, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert not os.path.exists(tcfg.checkpoint.checkpoint_path)


def test_train_ddp_calls_set_epoch_each_epoch(model, ddp_loader, tcfg_multi, distogram_res, distogram_atom, index_embedding, patch_ddp):
    train_ddp(0, 0, 1, model, tcfg_multi, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert ddp_loader.sampler.set_epoch_calls == [1, 2, 3]


def test_train_ddp_updates_model_parameters(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp):
    params_before = [p.clone().detach() for p in model.parameters()]
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert any(not torch.equal(b, a) for b, a in zip(params_before, model.parameters()))


def test_train_ddp_wandb_not_called_when_disabled(model, ddp_loader, tcfg, distogram_res, distogram_atom, index_embedding, patch_ddp, monkeypatch):
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train_ddp(0, 0, 1, model, tcfg, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    mock_log.assert_not_called()


def test_train_ddp_wandb_called_when_rank0_and_enabled(model, ddp_loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, patch_ddp, monkeypatch):
    logged = []
    monkeypatch.setattr("train.train_loop.wandb.log", lambda data, step: logged.append(data))
    train_ddp(0, 0, 1, model, tcfg_wandb, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    assert len(logged) == 1


def test_train_ddp_wandb_not_called_when_rank_nonzero(model, ddp_loader, tcfg_wandb, distogram_res, distogram_atom, index_embedding, patch_ddp, monkeypatch):
    mock_log = MagicMock()
    monkeypatch.setattr("train.train_loop.wandb.log", mock_log)
    train_ddp(1, 0, 2, model, tcfg_wandb, ddp_loader, ddp_loader, distogram_res, distogram_atom, index_embedding, device="cpu")
    mock_log.assert_not_called()


def test_evaluate_ddp_world_size1_matches_evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding):
    torch.manual_seed(42)
    r1 = evaluate(model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    torch.manual_seed(42)
    r2 = evaluate_ddp(0, 1, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, "cpu")
    for k in r1:
        assert abs(r1[k] - r2[k]) < 1e-5, f"Mismatch for '{k}': evaluate={r1[k]}, evaluate_ddp={r2[k]}"
