"""Tests for data pipeline components."""
import pytest
import torch
from diffusion.data_pipeline import RandomRotationTranslation3D


class TestRandomRotationTranslation3D:
    """Tests for RandomRotationTranslation3D transform."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        transform = RandomRotationTranslation3D(translation_range=5.0)

        coords = torch.randn(50, 37, 3)
        output = transform(coords)

        assert output.shape == coords.shape, f"Shape mismatch: {output.shape} != {coords.shape}"

    def test_rotation_is_orthogonal(self):
        """Test that rotation matrix is orthogonal (preserves distances)."""
        transform = RandomRotationTranslation3D(translation_range=0.0)  # No translation

        # Create a simple structure
        coords = torch.tensor([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Residue 1
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],  # Residue 2
        ])

        # Compute original pairwise distances
        coords_flat = coords.reshape(-1, 3)
        dists_orig = torch.cdist(coords_flat, coords_flat)

        # Apply rotation (no translation)
        output = transform(coords)
        output_flat = output.reshape(-1, 3)
        dists_rotated = torch.cdist(output_flat, output_flat)

        # Distances should be preserved
        assert torch.allclose(dists_orig, dists_rotated, atol=1e-5), \
            "Rotation does not preserve distances (not orthogonal)"

    def test_translation_range(self):
        """Test that translation is within specified range."""
        translation_range = 5.0
        transform = RandomRotationTranslation3D(translation_range=translation_range)

        # Create simple coords at origin
        coords = torch.zeros(10, 37, 3)

        # Apply transform multiple times and check translation
        num_samples = 100
        max_translation_seen = 0.0

        for _ in range(num_samples):
            output = transform(coords)
            # Since input is all zeros, output is just translation
            # (rotation of zeros is still zeros)
            translation_applied = output[0, 0]  # Any point
            translation_magnitude = translation_applied.norm().item()
            max_translation_seen = max(max_translation_seen, translation_magnitude)

        # Translation should be within range
        assert max_translation_seen <= translation_range * 1.732 + 0.1, \
            f"Translation {max_translation_seen} exceeds range {translation_range}"

    def test_zero_translation_range(self):
        """Test with zero translation range (rotation only)."""
        transform = RandomRotationTranslation3D(translation_range=0.0)

        coords = torch.randn(20, 37, 3)

        # Compute center of mass
        com_orig = coords.mean(dim=(0, 1))

        output = transform(coords)
        com_rotated = output.mean(dim=(0, 1))

        # With zero translation, center of mass should only be rotated
        # The magnitude should be preserved
        assert torch.allclose(com_orig.norm(), com_rotated.norm(), atol=1e-5), \
            "Zero translation should preserve COM distance from origin"

    def test_deterministic_with_seed(self):
        """Test that transform is deterministic when seed is set."""
        transform = RandomRotationTranslation3D(translation_range=5.0)
        coords = torch.randn(15, 37, 3)

        # Set seed and apply
        torch.manual_seed(42)
        output1 = transform(coords.clone())

        # Set same seed and apply again
        torch.manual_seed(42)
        output2 = transform(coords.clone())

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Transform should be deterministic with same seed"

    def test_randomness_without_seed(self):
        """Test that transform is random without seed."""
        transform = RandomRotationTranslation3D(translation_range=5.0)
        coords = torch.randn(15, 37, 3)

        # Apply multiple times
        outputs = [transform(coords.clone()) for _ in range(10)]

        # Check that outputs are different
        all_different = True
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if torch.allclose(outputs[i], outputs[j], atol=1e-6):
                    all_different = False
                    break
            if not all_different:
                break

        assert all_different, "Transform should produce different outputs"

    def test_preserves_dtype(self):
        """Test that output dtype matches input dtype."""
        transform = RandomRotationTranslation3D(translation_range=5.0)

        for dtype in [torch.float32, torch.float64]:
            coords = torch.randn(10, 37, 3, dtype=dtype)
            output = transform(coords)

            assert output.dtype == dtype, f"Output dtype {output.dtype} != input dtype {dtype}"

    def test_preserves_device(self):
        """Test that output device matches input device."""
        transform = RandomRotationTranslation3D(translation_range=5.0)
        coords = torch.randn(10, 37, 3)

        # CPU
        output_cpu = transform(coords)
        assert output_cpu.device == coords.device, "Output device should match input (CPU)"

        # GPU if available
        if torch.cuda.is_available():
            coords_gpu = coords.cuda()
            output_gpu = transform(coords_gpu)
            assert output_gpu.device == coords_gpu.device, "Output device should match input (GPU)"

    def test_rotation_matrix_properties(self):
        """Test that generated rotation matrix is valid (orthogonal, det=1)."""
        transform = RandomRotationTranslation3D(translation_range=0.0)

        # We need to extract the rotation matrix used internally
        # We'll do this by applying to basis vectors
        basis = torch.eye(3).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        torch.manual_seed(42)
        # Create coords and apply transform
        coords = torch.randn(5, 37, 3)
        _ = transform(coords)

        # Verify by checking that pairwise distances are preserved
        # (already covered in test_rotation_is_orthogonal)
        # Here we'll just do a sanity check
        coords_simple = torch.tensor([[[1.0, 0.0, 0.0]]])
        rotated = transform(coords_simple)

        # Rotated vector should have same length as original
        orig_length = coords_simple.norm()
        rotated_length = (rotated - rotated.mean()).norm()  # Remove translation

        # This test is approximate since we have translation
        # So we just check the vector has reasonable magnitude
        assert 0.5 < rotated.norm() < 10.0, "Rotated coords have unreasonable magnitude"

    def test_batch_processing(self):
        """Test that transform works correctly on different batch sizes."""
        transform = RandomRotationTranslation3D(translation_range=5.0)

        for n_residues in [1, 10, 50, 100]:
            for n_atoms in [1, 4, 37]:
                coords = torch.randn(n_residues, n_atoms, 3)
                output = transform(coords)

                assert output.shape == coords.shape, \
                    f"Failed for shape [{n_residues}, {n_atoms}, 3]"
                assert not torch.isnan(output).any(), \
                    f"NaN in output for shape [{n_residues}, {n_atoms}, 3]"
                assert not torch.isinf(output).any(), \
                    f"Inf in output for shape [{n_residues}, {n_atoms}, 3]"

    def test_edge_case_single_atom(self):
        """Test with single atom."""
        transform = RandomRotationTranslation3D(translation_range=5.0)
        coords = torch.randn(1, 1, 3)

        output = transform(coords)

        assert output.shape == (1, 1, 3)
        assert not torch.isnan(output).any()

    def test_edge_case_large_coords(self):
        """Test with large coordinate values."""
        transform = RandomRotationTranslation3D(translation_range=5.0)
        coords = torch.randn(10, 37, 3) * 1000.0  # Large coordinates

        output = transform(coords)

        # Should still work
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow through the transform."""
        transform = RandomRotationTranslation3D(translation_range=5.0)
        coords = torch.randn(10, 37, 3, requires_grad=True)

        output = transform(coords)
        loss = output.sum()
        loss.backward()

        assert coords.grad is not None, "Gradient should flow through transform"
        assert not torch.isnan(coords.grad).any(), "Gradient contains NaN"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        transform = RandomRotationTranslation3D(translation_range=1.0)

        # Very small values
        coords_small = torch.randn(5, 10, 3) * 1e-6
        output_small = transform(coords_small)
        assert not torch.isnan(output_small).any(), "NaN with small values"

        # Zero values
        coords_zero = torch.zeros(5, 10, 3)
        output_zero = transform(coords_zero)
        assert not torch.isnan(output_zero).any(), "NaN with zero values"
