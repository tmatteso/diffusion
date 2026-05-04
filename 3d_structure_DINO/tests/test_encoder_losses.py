"""
Comprehensive tests for encoder loss functions.

Tests for numerical stability, gradient flow, and correctness of DINO, KoLeo, and iBOT losses.
"""

import torch

from diffusion.encoder_losses import (
    DINOLoss,
    DINOv2Loss,
    GramAnchoringLoss,
    KoLeoLoss,
    iBOTPatchLoss,
)


class TestDINOLoss:
    """Tests for DINO loss."""

    def test_forward_shape(self):
        """Test that loss returns a scalar."""
        out_dim = 256
        dino_loss = DINOLoss(out_dim=out_dim)

        student_output = [torch.randn(4, out_dim)]
        teacher_output = [torch.randn(4, out_dim)]

        # Process teacher output
        teacher_softmaxed = [
            dino_loss.softmax_center_teacher(t, teacher_temp=0.07) for t in teacher_output
        ]

        loss = dino_loss(student_output, teacher_softmaxed)

        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_center_update(self):
        """Test that center updates correctly."""
        out_dim = 256
        dino_loss = DINOLoss(out_dim=out_dim, center_momentum=0.9)

        # Initial center should be zeros
        assert torch.allclose(dino_loss.center, torch.zeros(1, out_dim))

        teacher_output = torch.randn(4, out_dim)
        dino_loss.update_center(teacher_output)

        # Center should have updated
        assert not torch.allclose(dino_loss.center, torch.zeros(1, out_dim))

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        out_dim = 256
        dino_loss = DINOLoss(out_dim=out_dim)

        student_output = [torch.randn(4, out_dim, requires_grad=True)]
        teacher_output = [torch.randn(4, out_dim)]

        teacher_softmaxed = [
            dino_loss.softmax_center_teacher(t, teacher_temp=0.07) for t in teacher_output
        ]

        loss = dino_loss(student_output, teacher_softmaxed)
        loss.backward()

        assert student_output[0].grad is not None
        assert not torch.isnan(student_output[0].grad).any()


class TestKoLeoLoss:
    """Tests for KoLeo loss."""

    def test_forward_shape(self):
        """Test that loss returns a scalar."""
        koleo_loss = KoLeoLoss()

        student_output = torch.randn(10, 256)
        loss = koleo_loss(student_output)

        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_normalized_inputs(self):
        """Test that KoLeo normalizes inputs internally."""
        koleo_loss = KoLeoLoss()

        # Create unnormalized inputs
        student_output = torch.randn(10, 256) * 10.0
        loss = koleo_loss(student_output)

        assert not torch.isnan(loss), "Loss is NaN with unnormalized inputs"
        assert not torch.isinf(loss), "Loss is Inf with unnormalized inputs"

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        koleo_loss = KoLeoLoss()

        student_output = torch.randn(10, 256, requires_grad=True)
        loss = koleo_loss(student_output)
        loss.backward()

        assert student_output.grad is not None
        assert not torch.isnan(student_output.grad).any()

    def test_loss_decreases_with_uniform_distribution(self):
        """Test that loss is lower for more uniformly distributed points."""
        koleo_loss = KoLeoLoss()

        # Clustered points
        clustered = torch.randn(100, 64) * 0.1  # Small variance
        loss_clustered = koleo_loss(clustered)

        # Spread out points
        spread = torch.randn(100, 64) * 1.0  # Large variance
        loss_spread = koleo_loss(spread)

        # Spread points should have lower loss (more entropy)
        assert (
            loss_spread < loss_clustered
        ), f"Spread loss {loss_spread} should be < clustered loss {loss_clustered}"


class TestiBOTPatchLoss:
    """Tests for iBOT patch loss."""

    def test_forward_shape(self):
        """Test that loss returns a scalar."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        masks = torch.rand(batch_size, n_patches) < 0.15  # 15% masked

        # Process teacher
        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)

        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_masked_positions_only(self):
        """Test that loss only considers masked positions."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 2, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        # No masks - loss should be 0
        masks_none = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss_none = ibot_loss(student_patches, teacher_softmaxed, masks_none)

        # All masks
        masks_all = torch.ones(batch_size, n_patches, dtype=torch.bool)
        loss_all = ibot_loss(student_patches, teacher_softmaxed, masks_all)

        # All masked should have higher loss magnitude than none
        assert loss_all.abs() > loss_none.abs()

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim, requires_grad=True)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        masks = torch.rand(batch_size, n_patches) < 0.15

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)
        loss.backward()

        assert student_patches.grad is not None
        assert not torch.isnan(student_patches.grad).any()

    def test_center_update(self):
        """Test that center updates correctly."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim, center_momentum=0.9)

        # Initial center should be zeros
        assert torch.allclose(ibot_loss.center, torch.zeros(1, 1, patch_out_dim))

        teacher_patches = torch.randn(4, 10, patch_out_dim)
        ibot_loss.update_center(teacher_patches)

        # Center should have updated
        assert not torch.allclose(ibot_loss.center, torch.zeros(1, 1, patch_out_dim))


class TestGramAnchoringLoss:
    """Tests for Gram Anchoring loss (DINOv3)."""

    def test_forward_shape(self):
        """Test that loss returns a scalar."""
        gram_loss = GramAnchoringLoss()

        batch_size, n_patches, embed_dim = 4, 16, 256
        student_patches = torch.randn(batch_size, n_patches, embed_dim)
        gram_teacher_patches = torch.randn(batch_size, n_patches, embed_dim)

        loss = gram_loss(student_patches, gram_teacher_patches)

        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_gram_matrix_shape(self):
        """Test that Gram matrix has correct shape."""
        gram_loss = GramAnchoringLoss()

        batch_size, n_patches, embed_dim = 4, 16, 256
        patches = torch.randn(batch_size, n_patches, embed_dim)

        gram = gram_loss.compute_gram_matrix(patches)

        assert gram.shape == (batch_size, n_patches, n_patches)

    def test_gram_matrix_symmetry(self):
        """Test that Gram matrix is symmetric."""
        gram_loss = GramAnchoringLoss()

        batch_size, n_patches, embed_dim = 2, 8, 64
        patches = torch.randn(batch_size, n_patches, embed_dim)

        gram = gram_loss.compute_gram_matrix(patches)

        # Check symmetry: G[i,j] == G[j,i]
        assert torch.allclose(gram, gram.transpose(1, 2), atol=1e-6)

    def test_identical_inputs_zero_loss(self):
        """Test that identical inputs give zero loss."""
        gram_loss = GramAnchoringLoss()

        batch_size, n_patches, embed_dim = 4, 16, 256
        patches = torch.randn(batch_size, n_patches, embed_dim)

        # Same input for both should give zero loss
        loss = gram_loss(patches, patches.clone())

        assert loss.item() < 1e-6, f"Loss should be ~0 for identical inputs, got {loss.item()}"

    def test_gradient_flow(self):
        """Test that gradients flow correctly (only to student)."""
        gram_loss = GramAnchoringLoss()

        batch_size, n_patches, embed_dim = 4, 16, 256
        student_patches = torch.randn(batch_size, n_patches, embed_dim, requires_grad=True)
        gram_teacher_patches = torch.randn(batch_size, n_patches, embed_dim)

        loss = gram_loss(student_patches, gram_teacher_patches)
        loss.backward()

        assert student_patches.grad is not None
        assert not torch.isnan(student_patches.grad).any()

    def test_different_loss_types(self):
        """Test both MSE and cosine loss types."""
        batch_size, n_patches, embed_dim = 4, 16, 256
        student_patches = torch.randn(batch_size, n_patches, embed_dim)
        gram_teacher_patches = torch.randn(batch_size, n_patches, embed_dim)

        # MSE loss
        gram_loss_mse = GramAnchoringLoss(loss_type="mse")
        loss_mse = gram_loss_mse(student_patches, gram_teacher_patches)
        assert not torch.isnan(loss_mse), "MSE loss is NaN"

        # Cosine loss
        gram_loss_cosine = GramAnchoringLoss(loss_type="cosine")
        loss_cosine = gram_loss_cosine(student_patches, gram_teacher_patches)
        assert not torch.isnan(loss_cosine), "Cosine loss is NaN"
        assert 0 <= loss_cosine <= 2, f"Cosine loss should be in [0, 2], got {loss_cosine}"

    def test_normalization_option(self):
        """Test that normalization option works."""
        batch_size, n_patches, embed_dim = 4, 16, 256
        patches = torch.randn(batch_size, n_patches, embed_dim) * 10.0  # Large values

        # With normalization (default)
        gram_loss_norm = GramAnchoringLoss(normalize=True)
        gram_norm = gram_loss_norm.compute_gram_matrix(patches)
        # Diagonal should be ~1 after normalization
        diag = torch.diagonal(gram_norm, dim1=1, dim2=2)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

        # Without normalization
        gram_loss_no_norm = GramAnchoringLoss(normalize=False)
        gram_no_norm = gram_loss_no_norm.compute_gram_matrix(patches)
        # Diagonal should NOT be ~1
        diag_no_norm = torch.diagonal(gram_no_norm, dim1=1, dim2=2)
        assert not torch.allclose(diag_no_norm, torch.ones_like(diag_no_norm), atol=1e-5)

    def test_loss_sensitive_to_structure_changes(self):
        """Test that loss increases when patch relationships change."""
        gram_loss = GramAnchoringLoss()

        batch_size, n_patches, embed_dim = 4, 16, 256
        gram_teacher_patches = torch.randn(batch_size, n_patches, embed_dim)

        # Student with same structure as teacher
        student_similar = gram_teacher_patches + torch.randn_like(gram_teacher_patches) * 0.1
        loss_similar = gram_loss(student_similar, gram_teacher_patches)

        # Student with shuffled patches (different structure)
        student_shuffled = gram_teacher_patches[:, torch.randperm(n_patches), :]
        loss_shuffled = gram_loss(student_shuffled, gram_teacher_patches)

        # Shuffled should have higher loss
        assert loss_shuffled > loss_similar, "Shuffled patches should have higher loss"


class TestDINOv2Loss:
    """Tests for combined DINOv2 loss."""

    def test_forward_with_all_losses(self):
        """Test that combined loss works with all components."""
        out_dim = 256
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=1.0,
            patch_out_dim=out_dim,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        # Create dummy data
        student_cls = [torch.randn(4, out_dim)]
        teacher_cls = [torch.randn(4, out_dim)]
        student_patches = torch.randn(4, 10, out_dim)
        teacher_patches = torch.randn(4, 10, out_dim)
        masks = torch.rand(4, 10) < 0.15

        losses = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=0,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
        )

        assert "total" in losses
        assert "dino" in losses
        assert "koleo" in losses
        assert "ibot" in losses

        for key, loss in losses.items():
            assert not torch.isnan(loss), f"{key} loss is NaN"
            assert not torch.isinf(loss), f"{key} loss is Inf"

    def test_teacher_temp_schedule(self):
        """Test that teacher temperature schedule works."""
        loss_fn = DINOv2Loss(
            out_dim=256,
            warmup_teacher_temp=0.04,
            teacher_temp=0.07,
            warmup_teacher_temp_epochs=30,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        # Temperature should start at warmup value
        temp_0 = loss_fn.get_teacher_temp(0)
        assert abs(temp_0 - 0.04) < 1e-6

        # Temperature should end at final value
        temp_50 = loss_fn.get_teacher_temp(50)
        assert abs(temp_50 - 0.07) < 1e-6

        # Temperature should increase during warmup
        temp_15 = loss_fn.get_teacher_temp(15)
        assert 0.04 < temp_15 < 0.07

    def test_gradient_flow_through_combined_loss(self):
        """Test that gradients flow through all loss components."""
        out_dim = 256
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=1.0,
            patch_out_dim=out_dim,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        student_cls = [torch.randn(4, out_dim, requires_grad=True)]
        teacher_cls = [torch.randn(4, out_dim)]
        student_patches = torch.randn(4, 10, out_dim, requires_grad=True)
        teacher_patches = torch.randn(4, 10, out_dim)
        masks = torch.rand(4, 10) < 0.15

        losses = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=0,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
        )

        losses["total"].backward()

        assert student_cls[0].grad is not None
        assert not torch.isnan(student_cls[0].grad).any()
        assert student_patches.grad is not None
        assert not torch.isnan(student_patches.grad).any()

    def test_loss_without_ibot(self):
        """Test that loss works without iBOT component."""
        out_dim = 256
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=0.0,  # Disable iBOT
            lambda_gram=0.0,  # Disable Gram
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        student_cls = [torch.randn(4, out_dim)]
        teacher_cls = [torch.randn(4, out_dim)]

        losses = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=0,
        )

        assert losses["ibot"].item() == 0.0
        assert losses["total"] == losses["dino"] + 0.1 * losses["koleo"]

    def test_gram_anchoring_integration(self):
        """Test that Gram Anchoring loss integrates correctly."""
        out_dim = 256
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=1.0,
            lambda_gram=0.5,
            enable_gram_after_epoch=0,
            patch_out_dim=out_dim,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        student_cls = [torch.randn(4, out_dim)]
        teacher_cls = [torch.randn(4, out_dim)]
        student_patches = torch.randn(4, 10, out_dim)
        teacher_patches = torch.randn(4, 10, out_dim)
        gram_teacher_patches = torch.randn(4, 10, out_dim)
        masks = torch.rand(4, 10) < 0.15

        losses = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=0,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
            gram_teacher_patch_tokens=gram_teacher_patches,
        )

        assert "gram" in losses
        assert losses["gram"].item() > 0, "Gram loss should be > 0 for different inputs"
        assert not torch.isnan(losses["gram"]), "Gram loss is NaN"

    def test_gram_anchoring_delayed_start(self):
        """Test that Gram Anchoring is disabled before enable_gram_after_epoch."""
        out_dim = 256
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=1.0,
            lambda_gram=0.5,
            enable_gram_after_epoch=10,  # Enable after epoch 10
            patch_out_dim=out_dim,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        student_cls = [torch.randn(4, out_dim)]
        teacher_cls = [torch.randn(4, out_dim)]
        student_patches = torch.randn(4, 10, out_dim)
        teacher_patches = torch.randn(4, 10, out_dim)
        gram_teacher_patches = torch.randn(4, 10, out_dim)
        masks = torch.rand(4, 10) < 0.15

        # Before enable epoch - gram loss should be 0
        losses_early = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=5,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
            gram_teacher_patch_tokens=gram_teacher_patches,
        )
        assert losses_early["gram"].item() == 0.0

        # After enable epoch - gram loss should be > 0
        losses_late = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=15,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
            gram_teacher_patch_tokens=gram_teacher_patches,
        )
        assert losses_late["gram"].item() > 0.0

    def test_is_gram_anchoring_enabled(self):
        """Test the is_gram_anchoring_enabled helper method."""
        loss_fn = DINOv2Loss(
            out_dim=256,
            lambda_gram=0.5,
            enable_gram_after_epoch=10,
        )

        assert not loss_fn.is_gram_anchoring_enabled(5)
        assert not loss_fn.is_gram_anchoring_enabled(9)
        assert loss_fn.is_gram_anchoring_enabled(10)
        assert loss_fn.is_gram_anchoring_enabled(50)

    def test_should_update_gram_teacher(self):
        """Test the should_update_gram_teacher helper method."""
        loss_fn = DINOv2Loss(
            out_dim=256,
            lambda_gram=0.5,
            enable_gram_after_epoch=10,
        )

        # Before gram is enabled
        assert not loss_fn.should_update_gram_teacher(5, update_interval=5)

        # At enable epoch (should update)
        assert loss_fn.should_update_gram_teacher(10, update_interval=5)

        # Between intervals (should not update)
        assert not loss_fn.should_update_gram_teacher(12, update_interval=5)

        # At next interval (should update)
        assert loss_fn.should_update_gram_teacher(15, update_interval=5)

    def test_gram_gradient_flow(self):
        """Test that gradients flow through Gram Anchoring loss."""
        out_dim = 256
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=1.0,
            lambda_gram=0.5,
            enable_gram_after_epoch=0,
            patch_out_dim=out_dim,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        student_cls = [torch.randn(4, out_dim, requires_grad=True)]
        teacher_cls = [torch.randn(4, out_dim)]
        student_patches = torch.randn(4, 10, out_dim, requires_grad=True)
        teacher_patches = torch.randn(4, 10, out_dim)
        gram_teacher_patches = torch.randn(4, 10, out_dim)
        masks = torch.rand(4, 10) < 0.15

        losses = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=0,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
            gram_teacher_patch_tokens=gram_teacher_patches,
        )

        losses["total"].backward()

        assert student_patches.grad is not None
        assert not torch.isnan(student_patches.grad).any()


class TestNumericalStabilityInLosses:
    """Tests specifically for numerical stability in losses."""

    def test_dino_loss_with_large_values(self):
        """Test DINO loss with large magnitude inputs."""
        out_dim = 4096  # High dimensional like in training
        dino_loss = DINOLoss(out_dim=out_dim)

        # Create large magnitude inputs
        student_output = [torch.randn(4, out_dim) * 10.0]
        teacher_output = [torch.randn(4, out_dim) * 10.0]

        teacher_softmaxed = [
            dino_loss.softmax_center_teacher(t, teacher_temp=0.07) for t in teacher_output
        ]

        loss = dino_loss(student_output, teacher_softmaxed)

        assert not torch.isnan(loss), "DINO loss is NaN with large values"
        assert not torch.isinf(loss), "DINO loss is Inf with large values"
        assert loss < 1000, f"DINO loss too large: {loss}"

    def test_ibot_loss_with_low_temperature(self):
        """Test iBOT loss with low temperature (as used in training)."""
        patch_out_dim = 4096
        ibot_loss = iBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,  # Low temperature
        )

        student_patches = torch.randn(4, 10, patch_out_dim)
        teacher_patches = torch.randn(4, 10, patch_out_dim)
        masks = torch.ones(4, 10, dtype=torch.bool)  # All masked

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        assert not torch.isnan(loss), "iBOT loss is NaN with low temp"
        assert not torch.isinf(loss), "iBOT loss is Inf with low temp"

    def test_high_dimensional_softmax(self):
        """Test that high-dimensional softmax doesn't overflow."""
        out_dim = 4096
        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1)

        # Extreme case: all values large
        # student = torch.randn(4, out_dim) * 5.0
        teacher = torch.randn(4, out_dim) * 5.0

        teacher_softmaxed = dino_loss.softmax_center_teacher(teacher, 0.04)

        # Check softmax didn't overflow
        assert not torch.isnan(teacher_softmaxed).any()
        assert not torch.isinf(teacher_softmaxed).any()
        assert torch.allclose(teacher_softmaxed.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_gradient_magnitude_in_high_dim(self):
        """Test that gradients don't explode in high dimensions."""
        out_dim = 4096
        loss_fn = DINOv2Loss(
            out_dim=out_dim,
            lambda_koleo=0.1,
            lambda_ibot=1.0,
            patch_out_dim=out_dim,
            student_temp=0.1,
            teacher_temp=0.07,
        )
        loss_fn.setup_teacher_temp_schedule(num_epochs=100)

        student_cls = [torch.randn(4, out_dim, requires_grad=True)]
        teacher_cls = [torch.randn(4, out_dim)]
        student_patches = torch.randn(4, 10, out_dim, requires_grad=True)
        teacher_patches = torch.randn(4, 10, out_dim)
        masks = torch.rand(4, 10) < 0.15

        losses = loss_fn(
            student_output=student_cls,
            teacher_output=teacher_cls,
            epoch=0,
            student_patch_tokens=student_patches,
            teacher_patch_tokens=teacher_patches,
            student_masks_flat=masks,
        )

        losses["total"].backward()

        # Check gradient magnitudes
        grad_norm_cls = student_cls[0].grad.norm().item()
        grad_norm_patches = student_patches.grad.norm().item()

        assert grad_norm_cls < 10000, f"CLS gradient explosion: {grad_norm_cls}"
        assert grad_norm_patches < 10000, f"Patch gradient explosion: {grad_norm_patches}"
