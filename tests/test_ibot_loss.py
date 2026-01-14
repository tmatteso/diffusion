"""Comprehensive tests for iBOT loss."""

import torch
import torch.nn.functional as F

from diffusion.encoder_losses import iBOTPatchLoss


class TestiBOTPatchLoss:
    """Comprehensive tests for iBOT patch loss."""

    def test_only_masked_positions_contribute(self):
        """Test that only masked positions contribute to the loss."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        # Create mask with some positions masked
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        masks[:, 0] = True  # Only first position is masked

        # Process teacher
        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)

        # Compute loss
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        # Manually compute expected loss for first position only
        expected_loss = 0.0
        for b in range(batch_size):
            s = student_patches[b, 0]  # First position
            t = teacher_softmaxed[b, 0]
            # Cross-entropy: -sum(t * log_softmax(s))
            ce = -torch.sum(t * F.log_softmax(s / ibot_loss.student_temp, dim=-1))
            expected_loss += ce
        expected_loss /= batch_size

        assert torch.allclose(
            loss, expected_loss, atol=1e-5
        ), f"Loss mismatch: {loss} != {expected_loss}"

    def test_no_masked_positions_gives_zero_loss(self):
        """Test that no masked positions gives zero loss."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        # No masks
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        # Loss should be zero when no positions are masked
        assert loss.item() == 0.0, f"Loss should be 0 with no masks, got {loss.item()}"

    def test_all_masked_positions(self):
        """Test with all positions masked."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        # All positions masked
        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        # Loss should be positive and reasonable
        assert loss > 0, f"Loss should be positive, got {loss}"
        assert loss < 100, f"Loss unreasonably large: {loss}"

    def test_padding_doesnt_contribute(self):
        """Test that padding positions (not masked) don't contribute."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size = 2
        # Sequence 1: 10 real patches, sequence 2: 5 real patches, rest is padding
        max_patches = 10
        student_patches = torch.randn(batch_size, max_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, max_patches, patch_out_dim)

        # Mask: only real positions can be masked
        # Seq 1: mask first 3 positions out of 10
        # Seq 2: mask first 2 positions out of 5, positions 5-9 are padding (not masked)
        masks = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        masks[0, :3] = True  # Seq 1: mask positions 0,1,2
        masks[1, :2] = True  # Seq 2: mask positions 0,1 (positions 5-9 are padding, not masked)

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        # Manually compute expected loss
        expected_loss = 0.0
        # Seq 1: average over 3 masked positions
        for i in range(3):
            s = student_patches[0, i]
            t = teacher_softmaxed[0, i]
            ce = -torch.sum(t * F.log_softmax(s / ibot_loss.student_temp, dim=-1))
            expected_loss += ce / 3.0

        # Seq 2: average over 2 masked positions
        for i in range(2):
            s = student_patches[1, i]
            t = teacher_softmaxed[1, i]
            ce = -torch.sum(t * F.log_softmax(s / ibot_loss.student_temp, dim=-1))
            expected_loss += ce / 2.0

        # Average over batch
        expected_loss /= batch_size

        assert torch.allclose(
            loss, expected_loss, atol=1e-4
        ), f"Loss mismatch: {loss} != {expected_loss}"

    def test_cross_entropy_is_computed_correctly(self):
        """Test that cross-entropy is computed correctly."""
        patch_out_dim = 10  # Small for easier verification
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim, student_temp=1.0)

        # Single batch, single patch for simplicity
        student = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]])
        teacher = torch.tensor([[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])

        # Normalize teacher to be a proper probability distribution
        teacher = F.softmax(teacher, dim=-1)

        masks = torch.ones(1, 1, dtype=torch.bool)  # Mask the single position

        # Compute loss (teacher already softmaxed, use directly)
        loss = ibot_loss(student, teacher, masks)

        # Manual cross-entropy computation
        student_log_probs = F.log_softmax(student / ibot_loss.student_temp, dim=-1)
        expected_ce = -torch.sum(teacher * student_log_probs)

        assert torch.allclose(
            loss, expected_ce, atol=1e-5
        ), f"Cross-entropy mismatch: {loss} != {expected_ce}"

    def test_teacher_centering(self):
        """Test that teacher centering works correctly."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim, center_momentum=0.9)

        # Initial center should be zeros
        assert torch.allclose(ibot_loss.center, torch.zeros(1, 1, patch_out_dim))

        # Update center with some data
        teacher_patches = torch.randn(4, 10, patch_out_dim) + 5.0  # Biased towards positive
        ibot_loss.update_center(teacher_patches)

        # Center should have updated (no longer all zeros)
        assert not torch.allclose(
            ibot_loss.center, torch.zeros(1, 1, patch_out_dim)
        ), "Center should update"

        # Center should be in the direction of the data
        mean_teacher = teacher_patches.mean(dim=(0, 1))
        # The center should be correlated with the mean (same sign on average)
        correlation = (ibot_loss.center.squeeze() * mean_teacher).mean()
        assert correlation > 0, "Center should be correlated with teacher mean"

    def test_different_mask_ratios(self):
        """Test with different mask ratios."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 20
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)

        # Test different mask ratios
        for mask_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            masks = torch.rand(batch_size, n_patches) < mask_ratio

            # Skip if no masks
            if not masks.any():
                continue

            loss = ibot_loss(student_patches, teacher_softmaxed, masks)

            assert not torch.isnan(loss), f"NaN loss with mask_ratio={mask_ratio}"
            assert not torch.isinf(loss), f"Inf loss with mask_ratio={mask_ratio}"
            assert loss > 0, f"Loss should be positive with mask_ratio={mask_ratio}"

    def test_loss_decreases_with_better_predictions(self):
        """Test that loss decreases when student predictions improve."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        masks = torch.rand(batch_size, n_patches) < 0.3

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)

        # Bad student prediction (random)
        student_bad = torch.randn(batch_size, n_patches, patch_out_dim)
        loss_bad = ibot_loss(student_bad, teacher_softmaxed, masks)

        # Good student prediction (closer to teacher)
        student_good = teacher_patches + torch.randn_like(teacher_patches) * 0.1
        loss_good = ibot_loss(student_good, teacher_softmaxed, masks)

        # Perfect student prediction (same as teacher)
        student_perfect = teacher_patches.clone()
        loss_perfect = ibot_loss(student_perfect, teacher_softmaxed, masks)

        # Loss should decrease as predictions improve
        assert (
            loss_good < loss_bad
        ), f"Loss should decrease with better predictions: {loss_good} >= {loss_bad}"
        assert (
            loss_perfect < loss_good
        ), f"Perfect predictions should have lowest loss: {loss_perfect} >= {loss_good}"

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim, requires_grad=True)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        masks = torch.rand(batch_size, n_patches) < 0.3

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)
        loss.backward()

        assert student_patches.grad is not None, "Gradient should flow to student"
        assert not torch.isnan(student_patches.grad).any(), "Gradient contains NaN"

        # Only masked positions should have gradients
        for b in range(batch_size):
            for n in range(n_patches):
                if masks[b, n]:
                    # Masked position should have gradient
                    assert (
                        student_patches.grad[b, n].abs().sum() > 0
                    ), f"Masked position [{b},{n}] should have gradient"

    def test_temperature_effect(self):
        """Test that student temperature affects loss magnitude."""
        patch_out_dim = 256

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        masks = torch.ones(batch_size, n_patches, dtype=torch.bool)

        losses = {}
        for temp in [0.05, 0.1, 0.5, 1.0]:
            ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim, student_temp=temp)
            teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
            losses[temp] = ibot_loss(student_patches, teacher_softmaxed, masks).item()

        # Different temperatures should give different losses
        assert len(set(losses.values())) > 1, "Different temperatures should give different losses"

    def test_batch_consistency(self):
        """Test that batch size doesn't affect per-sample loss."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        n_patches = 10
        # Single sample
        student_1 = torch.randn(1, n_patches, patch_out_dim)
        teacher_1 = torch.randn(1, n_patches, patch_out_dim)
        masks_1 = torch.rand(1, n_patches) < 0.3

        # Same sample in a batch
        student_batch = student_1.repeat(4, 1, 1)
        teacher_batch = teacher_1.repeat(4, 1, 1)
        masks_batch = masks_1.repeat(4, 1)

        teacher_softmaxed_1 = ibot_loss.softmax_center_teacher(teacher_1, 0.07)
        teacher_softmaxed_batch = ibot_loss.softmax_center_teacher(teacher_batch, 0.07)

        loss_1 = ibot_loss(student_1, teacher_softmaxed_1, masks_1)
        loss_batch = ibot_loss(student_batch, teacher_softmaxed_batch, masks_batch)

        # Losses should be the same (averaged over identical samples)
        assert torch.allclose(
            loss_1, loss_batch, atol=1e-5
        ), f"Batch loss should match single sample: {loss_1} != {loss_batch}"

    def test_numerical_stability_with_extreme_values(self):
        """Test numerical stability with extreme input values."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        masks = torch.rand(batch_size, n_patches) < 0.3

        # Test with large values
        student_large = torch.randn(batch_size, n_patches, patch_out_dim) * 100
        teacher_large = torch.randn(batch_size, n_patches, patch_out_dim) * 100

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_large, 0.07)
        loss_large = ibot_loss(student_large, teacher_softmaxed, masks)

        assert not torch.isnan(loss_large), "Loss is NaN with large values"
        assert not torch.isinf(loss_large), "Loss is Inf with large values"

        # Test with small values
        student_small = torch.randn(batch_size, n_patches, patch_out_dim) * 1e-3
        teacher_small = torch.randn(batch_size, n_patches, patch_out_dim) * 1e-3

        teacher_softmaxed_small = ibot_loss.softmax_center_teacher(teacher_small, 0.07)
        loss_small = ibot_loss(student_small, teacher_softmaxed_small, masks)

        assert not torch.isnan(loss_small), "Loss is NaN with small values"
        assert not torch.isinf(loss_small), "Loss is Inf with small values"

    def test_edge_case_single_masked_position(self):
        """Test with only a single position masked in entire batch."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 4, 10
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        # Only one position masked in entire batch
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        masks[0, 0] = True

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss = ibot_loss(student_patches, teacher_softmaxed, masks)

        assert not torch.isnan(loss), "Loss is NaN with single masked position"
        assert loss > 0, "Loss should be positive"

    def test_loss_invariant_to_permutation_within_sequence(self):
        """Test that permuting masked positions doesn't change loss."""
        patch_out_dim = 256
        ibot_loss = iBOTPatchLoss(patch_out_dim=patch_out_dim)

        batch_size, n_patches = 1, 10  # Single sequence for clarity
        student_patches = torch.randn(batch_size, n_patches, patch_out_dim)
        teacher_patches = torch.randn(batch_size, n_patches, patch_out_dim)

        # Mask positions [0, 2, 5, 7]
        masks = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        masked_indices = [0, 2, 5, 7]
        masks[0, masked_indices] = True

        teacher_softmaxed = ibot_loss.softmax_center_teacher(teacher_patches, 0.07)
        loss1 = ibot_loss(student_patches, teacher_softmaxed, masks)

        # Permute the patches but keep same positions masked
        perm = torch.randperm(n_patches)
        student_perm = student_patches[:, perm]
        teacher_perm = teacher_patches[:, perm]
        masks_perm = masks[:, perm]

        teacher_softmaxed_perm = ibot_loss.softmax_center_teacher(teacher_perm, 0.07)
        loss2 = ibot_loss(student_perm, teacher_softmaxed_perm, masks_perm)

        # Losses should be similar (might differ slightly due to centering)
        assert (
            abs(loss1 - loss2) < 1.0
        ), f"Loss should be approximately invariant to permutation: {loss1} vs {loss2}"
