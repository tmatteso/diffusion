"""Tests for LossMetrics, ThroughputStatistics, and ComponentNorms.

Covers valid construction from 0-D float tensors, field finiteness checks, and
jaxtyping shape-enforcement for each dataclass.
"""

import pytest
import torch
from helpers.useful_objects import (
    ComponentNorms,
    LossMetrics,
    ThroughputStatistics,
)
from jaxtyping import Float, TypeCheckError


def _scalar() -> Float[torch.Tensor, ""]:
    """Return a 0-D float tensor.

    Creates a scalar tensor with value 1.0 for use as a valid field in the
    shape-checked dataclasses under test.
    """
    return torch.tensor(1.0)


# ---------------------------------------------------------------------------
# LossMetrics — valid construction
# ---------------------------------------------------------------------------


@pytest.fixture
def loss_metrics() -> LossMetrics:
    """Provide a valid LossMetrics with all scalar tensors set to 1.0.

    Constructs a LossMetrics instance where every field is a 0-D float tensor
    with value 1.0, suitable for shape-contract and finiteness tests.
    """
    s = _scalar()
    return LossMetrics(
        total_loss=s,
        Kabsch_aligned_MSE_loss=s,
        CE_loss=s,
        smooth_lddt_loss=s,
        res_distogram_loss=s,
        atom_distogram_loss=s,
        intermediate_loss=s,
        RMSD=s,
    )


def test_loss_metrics_constructs(loss_metrics: LossMetrics) -> None:
    """LossMetrics constructs successfully from 0-D float tensors.

    Verifies that a LossMetrics instance is created without error when all
    fields are valid scalar tensors.
    """
    assert isinstance(loss_metrics, LossMetrics)


def test_loss_metrics_total_loss_is_scalar(loss_metrics: LossMetrics) -> None:
    """LossMetrics.total_loss is a 0-D tensor.

    Verifies that the total_loss field has zero dimensions, confirming the
    scalar shape contract is preserved after construction.
    """
    assert loss_metrics.total_loss.ndim == 0


def test_loss_metrics_all_fields_finite(loss_metrics: LossMetrics) -> None:
    """All LossMetrics tensor fields are finite.

    Verifies that every tensor field in LossMetrics contains only finite values
    when initialised from the fixture.
    """
    for field in [
        loss_metrics.total_loss,
        loss_metrics.Kabsch_aligned_MSE_loss,
        loss_metrics.CE_loss,
        loss_metrics.smooth_lddt_loss,
        loss_metrics.res_distogram_loss,
        loss_metrics.atom_distogram_loss,
        loss_metrics.intermediate_loss,
        loss_metrics.RMSD,
    ]:
        assert torch.isfinite(field)


# ---------------------------------------------------------------------------
# LossMetrics — shape enforcement
# ---------------------------------------------------------------------------


def test_loss_metrics_rejects_1d_total_loss() -> None:
    """LossMetrics raises when total_loss is 1-D instead of scalar.

    Verifies that passing a rank-1 tensor for total_loss triggers a
    jaxtyping TypeCheckError or equivalent exception.
    """
    s = _scalar()
    with pytest.raises((TypeCheckError, Exception)):
        _ = LossMetrics(
            total_loss=torch.tensor([1.0]),
            Kabsch_aligned_MSE_loss=s,
            CE_loss=s,
            smooth_lddt_loss=s,
            res_distogram_loss=s,
            atom_distogram_loss=s,
            intermediate_loss=s,
            RMSD=s,
        )


# ---------------------------------------------------------------------------
# ThroughputStatistics — valid construction
# ---------------------------------------------------------------------------


@pytest.fixture
def throughput_statistics() -> ThroughputStatistics:
    """Provide a valid ThroughputStatistics with all scalar tensors set to 1.0.

    Constructs a ThroughputStatistics instance where every field is a 0-D float
    tensor with value 1.0, suitable for shape-contract and finiteness tests.
    """
    s = _scalar()
    return ThroughputStatistics(
        avg_batch_size=s,
        token_pack_rate=s,
        residues_per_sec=s,
        atoms_per_sec=s,
    )


def test_throughput_statistics_constructs(
    throughput_statistics: ThroughputStatistics,
) -> None:
    """ThroughputStatistics constructs successfully from 0-D float tensors.

    Verifies that a ThroughputStatistics instance is created without error when
    all fields are valid scalar tensors.
    """
    assert isinstance(throughput_statistics, ThroughputStatistics)


def test_throughput_statistics_avg_batch_size_is_scalar(
    throughput_statistics: ThroughputStatistics,
) -> None:
    """ThroughputStatistics.avg_batch_size is a 0-D tensor.

    Verifies that the avg_batch_size field has zero dimensions, confirming the
    scalar shape contract is preserved after construction.
    """
    assert throughput_statistics.avg_batch_size.ndim == 0


def test_throughput_statistics_all_fields_finite(
    throughput_statistics: ThroughputStatistics,
) -> None:
    """All ThroughputStatistics tensor fields are finite.

    Verifies every tensor field in ThroughputStatistics contains only finite
    values when initialised from the fixture.
    """
    for field in [
        throughput_statistics.avg_batch_size,
        throughput_statistics.token_pack_rate,
        throughput_statistics.residues_per_sec,
        throughput_statistics.atoms_per_sec,
    ]:
        assert torch.isfinite(field)


# ---------------------------------------------------------------------------
# ThroughputStatistics — shape enforcement
# ---------------------------------------------------------------------------


def test_throughput_statistics_rejects_1d_avg_batch_size() -> None:
    """ThroughputStatistics raises when avg_batch_size is 1-D instead of scalar.

    Verifies that passing a rank-1 tensor for avg_batch_size triggers a
    jaxtyping TypeCheckError or equivalent exception.
    """
    s = _scalar()
    with pytest.raises((TypeCheckError, Exception)):
        _ = ThroughputStatistics(
            avg_batch_size=torch.tensor([1.0]),
            token_pack_rate=s,
            residues_per_sec=s,
            atoms_per_sec=s,
        )


# ---------------------------------------------------------------------------
# ComponentNorms — valid construction
# ---------------------------------------------------------------------------


@pytest.fixture
def component_norms() -> ComponentNorms:
    """Provide a valid ComponentNorms with all scalar tensors set to 1.0.

    Constructs a ComponentNorms instance where every field is a 0-D float tensor
    with value 1.0, suitable for shape-contract and finiteness tests.
    """
    s = _scalar()
    return ComponentNorms(
        template_embedder=s,
        atom_encoder=s,
        atom_decoders=s,
        residue_distogram_head=s,
        atom_distogram_head=s,
        inter_proj_seq=s,
        inter_seq_logits=s,
        proj_seq=s,
        seq_logits=s,
    )


def test_component_norms_constructs(component_norms: ComponentNorms) -> None:
    """ComponentNorms constructs successfully from 0-D float tensors.

    Verifies that a ComponentNorms instance is created without error when all
    fields are valid scalar tensors.
    """
    assert isinstance(component_norms, ComponentNorms)


def test_component_norms_template_embedder_is_scalar(
    component_norms: ComponentNorms,
) -> None:
    """ComponentNorms.template_embedder is a 0-D tensor.

    Verifies the template_embedder field has zero dimensions, confirming the
    scalar shape contract is preserved after construction.
    """
    assert component_norms.template_embedder.ndim == 0


def test_component_norms_all_fields_finite(
    component_norms: ComponentNorms,
) -> None:
    """All ComponentNorms tensor fields are finite.

    Verifies every tensor field in ComponentNorms contains only finite values
    when initialised from the fixture.
    """
    for field in [
        component_norms.template_embedder,
        component_norms.atom_encoder,
        component_norms.atom_decoders,
        component_norms.residue_distogram_head,
        component_norms.atom_distogram_head,
        component_norms.inter_proj_seq,
        component_norms.inter_seq_logits,
        component_norms.proj_seq,
        component_norms.seq_logits,
    ]:
        assert torch.isfinite(field)


# ---------------------------------------------------------------------------
# ComponentNorms — shape enforcement
# ---------------------------------------------------------------------------


def test_component_norms_rejects_1d_template_embedder() -> None:
    """ComponentNorms raises when template_embedder is 1-D instead of scalar.

    Verifies that passing a rank-1 tensor for template_embedder triggers a
    jaxtyping TypeCheckError or equivalent exception.
    """
    s = _scalar()
    with pytest.raises((TypeCheckError, Exception)):
        _ = ComponentNorms(
            template_embedder=torch.tensor([1.0]),
            atom_encoder=s,
            atom_decoders=s,
            residue_distogram_head=s,
            atom_distogram_head=s,
            inter_proj_seq=s,
            inter_seq_logits=s,
            proj_seq=s,
            seq_logits=s,
        )
