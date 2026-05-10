"""Tests for SampleRequest validation and _protein_from_pdb_string."""

import os
import sys

# Ensure pallatom is importable (api.py does this at module level too)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "pallatom"))

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

# ---------------------------------------------------------------------------
# SampleRequest validation
# ---------------------------------------------------------------------------


def _import_request():
    """Import SampleRequest lazily so errors surface per-test."""
    from REST_APIs.api import SampleRequest

    return SampleRequest


def test_sample_request_minimal_valid():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=10)
    assert req.n_res == 10
    assert req.sequence is None
    assert req.structure_pdb is None
    assert req.template_pdb is None


def test_sample_request_sequence_length_must_match_n_res():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError, match="sequence length"):
        SampleRequest(n_res=10, sequence="ACDE")  # len 4 ≠ 10


def test_sample_request_sequence_exact_length_accepted():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="ACDE")
    assert req.sequence == "ACDE"


def test_sample_request_sequence_x_characters_accepted():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="AXXA")
    assert req.sequence == "AXXA"


def test_sample_request_sequence_all_x_accepted():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=4, sequence="XXXX")
    assert req.sequence == "XXXX"


def test_sample_request_sequence_invalid_character_rejected():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError, match="Invalid characters"):
        SampleRequest(n_res=3, sequence="ABZ")  # B and Z are not valid


def test_sample_request_n_res_must_be_positive():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=0)


def test_sample_request_n_res_max_512():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=513)


def test_sample_request_n_samples_max_10():
    SampleRequest = _import_request()
    with pytest.raises(ValidationError):
        SampleRequest(n_res=10, n_samples=11)


def test_sample_request_defaults():
    SampleRequest = _import_request()
    req = SampleRequest(n_res=50)
    assert req.n_samples == 1
    assert req.ddim_steps == 40
    assert req.rho == 7.0
    assert req.S_churn == 0.0
    assert req.S_noise == 1.003
