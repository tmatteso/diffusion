"""Tests for SampleRequest validation."""

import os
import sys

# Ensure pallatom is importable (api.py does this at module level too)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "pallatom"))

import asyncio  # noqa: E402
import glob  # noqa: E402
import tempfile as _tf  # noqa: E402
from collections.abc import MutableMapping  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from helpers.atom_utils import Protein, to_pdb  # noqa: E402
from helpers.featurize import Distogram, FeaturizedBatch  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from train.train_config import ModelParams, NoiseScheduleParams  # noqa: E402

from REST_APIs.api import SampleRequest, _AppState, _run_sampling  # noqa: E402

# ---------------------------------------------------------------------------
# SampleRequest validation
# ---------------------------------------------------------------------------


def _import_request() -> type[SampleRequest]:
    """Import SampleRequest lazily so errors surface per-test."""
    return SampleRequest


def test_sample_request_minimal_valid() -> None:
    """SampleRequest with only n_res is valid and optional fields default to None."""
    SampleRequest = _import_request()
    req = SampleRequest(n_res=10)
    assert req.n_res == 10
    assert req.sequence is None
    assert req.structure_pdb is None
    assert req.template_pdb is None


def test_sample_request_sequence_length_must_match_n_res() -> None:
    """Sequence whose length differs from n_res raises a ValidationError."""
    SampleRequest = _import_request()
    with pytest.raises(ValidationError, match="sequence length"):
        SampleRequest(n_res=10, sequence="ACDE")  # len 4 ≠ 10


def test_sample_request_sequence_exact_length_accepted() -> None:
    """Sequence whose length exactly equals n_res is accepted."""
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="ACDE")
    assert req.sequence == "ACDE"


def test_sample_request_sequence_x_characters_accepted() -> None:
    """Sequence containing 'X' (unknown residue) characters is accepted."""
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="AXXA")
    assert req.sequence == "AXXA"


def test_sample_request_sequence_all_x_accepted() -> None:
    """All-'X' sequence (fully unknown) is accepted."""
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="XXXX")
    assert req.sequence == "XXXX"


def test_sample_request_sequence_invalid_character_rejected() -> None:
    """Sequence with characters outside the valid amino-acid alphabet raises ValidationError."""
    SampleRequest = _import_request()
    with pytest.raises(ValidationError, match="Invalid characters"):
        SampleRequest(n_res=3, sequence="ABZ")  # B and Z are not valid


def test_sample_request_n_res_must_be_positive() -> None:
    """n_res of zero raises ValidationError."""
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=0)


def test_sample_request_n_res_max_512() -> None:
    """n_res above 512 raises ValidationError."""
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=513)


def test_sample_request_n_samples_max_10() -> None:
    """n_samples above 10 raises ValidationError."""
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=10, n_samples=11)


def test_sample_request_defaults() -> None:
    """SampleRequest default field values match the documented API spec."""
    SampleRequest = _import_request()
    req = SampleRequest(n_res=50)
    assert req.n_samples == 1
    assert req.ddim_steps == 40
    assert req.rho == 7.0
    assert req.S_churn == 0.0
    assert req.S_noise == 1.003


# ---------------------------------------------------------------------------
# _run_sampling integration tests (mocked model, no checkpoint required)
# ---------------------------------------------------------------------------

N_RES_TEST = 4


def _make_trunk_mock() -> MagicMock:
    mock = MagicMock()

    def _forward(
        batch: FeaturizedBatch,
    ) -> tuple[torch.Tensor, None, None, None, list[object], list[object]]:
        B = batch.r_input.shape[0]
        N_atom = batch.r_input.shape[1]
        return (
            torch.zeros(B, N_atom, 3),
            None,
            None,
            None,
            [],
            [],
        )

    mock.side_effect = _forward
    return mock


def _make_mock_state() -> _AppState:
    mp = ModelParams()
    noise = NoiseScheduleParams()
    return _AppState(
        semaphore=asyncio.Semaphore(1),
        model=_make_trunk_mock(),
        mp=mp,
        noise=noise,
        atom_disto=Distogram(n_bins=22, min_dist=2.0, max_dist=22.0, overflow_bin=False),
        templ_disto=Distogram(
            n_bins=mp.n_bins - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True
        ),
    )


def _make_pdb_string(n_res: int) -> str:
    """Produce a minimal valid PDB string with n_res alanine residues."""
    rng = np.random.RandomState(42)
    pos = rng.randn(n_res, 37, 3).astype(np.float64)
    mask = np.ones((n_res, 37), dtype=np.float64)
    prot = Protein(
        atom_positions=pos,
        aatype=np.zeros(n_res, dtype=np.intp),
        atom_mask=mask,
        residue_index=np.arange(n_res, dtype=np.intp),
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.ones((n_res, 37), dtype=np.float64),
    )
    return to_pdb(prot)


def _run(req_kwargs: MutableMapping[str, object]) -> list[str]:
    req_kwargs.setdefault("ddim_steps", 2)
    req = SampleRequest.model_validate(req_kwargs)
    mock_state = _make_mock_state()
    return _run_sampling(req, mock_state)


def test_run_sampling_unconditional_returns_pdb_strings() -> None:
    """Unconditional sampling returns a list of PDB-format strings."""
    result = _run({"n_res": N_RES_TEST})
    assert isinstance(result, list)
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_unconditional_n_samples_controls_output_count() -> None:
    """n_samples controls how many PDB strings are returned."""
    result = _run({"n_res": N_RES_TEST, "n_samples": 3})
    assert len(result) == 3


def test_run_sampling_sequence_only_returns_pdb_strings() -> None:
    """Sequence-conditional sampling returns PDB-format strings."""
    result = _run({"n_res": N_RES_TEST, "sequence": "ACDE"})
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_sequence_with_x_returns_pdb_strings() -> None:
    """Sequence containing X (unknown residue) is accepted and returns PDB strings."""
    result = _run({"n_res": N_RES_TEST, "sequence": "AXXE"})
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_seq_partial_atoms_returns_pdb_strings() -> None:
    """Sampling with sequence and a partial structure_pdb returns PDB strings."""
    partial_pdb = _make_pdb_string(N_RES_TEST // 2)
    result = _run(
        {
            "n_res": N_RES_TEST,
            "sequence": "ACDE",
            "structure_pdb": partial_pdb,
        }
    )
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_partial_template_returns_pdb_strings() -> None:
    """Sampling with a partial template_pdb returns PDB strings."""
    partial_pdb = _make_pdb_string(N_RES_TEST // 2)
    result = _run({"n_res": N_RES_TEST, "template_pdb": partial_pdb})
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_full_template_returns_pdb_strings() -> None:
    """Sampling with a full-length template_pdb returns PDB strings."""
    full_pdb = _make_pdb_string(N_RES_TEST)
    result = _run({"n_res": N_RES_TEST, "template_pdb": full_pdb})
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_full_atoms_partial_template_returns_pdb_strings() -> None:
    """Sampling with full structure_pdb and a partial template_pdb returns PDB strings."""
    full_struct = _make_pdb_string(N_RES_TEST)
    partial_templ = _make_pdb_string(N_RES_TEST // 2)
    result = _run(
        {
            "n_res": N_RES_TEST,
            "structure_pdb": full_struct,
            "template_pdb": partial_templ,
        }
    )
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_structure_pdb_too_many_residues_raises() -> None:
    """A structure_pdb with more residues than n_res raises ValueError."""
    oversized_pdb = _make_pdb_string(N_RES_TEST + 2)
    with pytest.raises(ValueError, match="structure_pdb has"):
        _run({"n_res": N_RES_TEST, "structure_pdb": oversized_pdb})


def test_run_sampling_non_contiguous_residue_index_does_not_raise() -> None:
    """PDB with non-contiguous residue numbers (gap between residues)."""
    n_res = 4
    pos = np.zeros((n_res, 37, 3), dtype=np.float64)
    mask = np.ones((n_res, 37), dtype=np.float64)
    prot = Protein(
        atom_positions=pos,
        aatype=np.zeros(n_res, dtype=np.intp),
        atom_mask=mask,
        residue_index=np.array([1, 2, 10, 11], dtype=np.intp),
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.ones((n_res, 37), dtype=np.float64),
    )
    pdb_str = to_pdb(prot)
    result = _run({"n_res": N_RES_TEST, "structure_pdb": pdb_str})
    assert len(result) == 1
    assert "ATOM" in result[0]


def test_run_sampling_tempfile_deleted_after_sampling() -> None:
    """Temporary PDB files written during sampling are deleted on completion."""
    tmpdir = _tf.gettempdir()
    pdb_before = set(glob.glob(os.path.join(tmpdir, "*.pdb")))
    full_pdb = _make_pdb_string(N_RES_TEST)
    _run({"n_res": N_RES_TEST, "template_pdb": full_pdb})
    pdb_after = set(glob.glob(os.path.join(tmpdir, "*.pdb")))
    assert pdb_after == pdb_before
