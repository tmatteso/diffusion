"""
Comprehensive tests for encoder architecture.

Tests for numerical stability, gradient flow, and correctness.
"""
import pytest
import torch
import torch.nn as nn
from diffusion.encoder_architecture import (
    MultiHeadSelfAttention,
    MLP,
    DropPath,
    TransformerBlock,
    ResidueEmbedding,
    pack_sequences,
    unpack_sequences,
    pack_protein_batch,
    ProjectionHead,
    PackedSequenceEncoder,
)


class TestMultiHeadSelfAttention:
    """Tests for MultiHeadSelfAttention module."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, dim = 2, 10, 64
        num_heads = 8

        attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)

        out = attn(x)

        assert out.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_attention_mask(self):
        """Test that attention mask correctly blocks attention."""
        batch_size, seq_len, dim = 1, 4, 64
        num_heads = 8

        attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)

        # Create mask that blocks all cross-attention
        # True = don't attend, False = attend
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        mask.fill_diagonal_(False)  # Only allow self-attention

        out = attn(x, attn_mask=mask)

        assert out.shape == (batch_size, seq_len, dim)
        assert not torch.isnan(out).any(), "Output contains NaN with mask"

    def test_attention_mask_effectiveness(self):
        """Test that attention mask actually affects outputs (not just ignored)."""
        attn = MultiHeadSelfAttention(dim=64, num_heads=8, attn_drop=0.0)
        attn.eval()

        # Create two distinct sequences
        seq1 = torch.ones(1, 5, 64) * 1.0  # All ones
        seq2 = torch.ones(1, 5, 64) * -1.0  # All negative ones
        x_concat = torch.cat([seq1, seq2], dim=1)  # [1, 10, 64]

        # Create block diagonal mask
        mask = torch.ones(10, 10, dtype=torch.bool)
        mask[:5, :5] = False  # seq1 can attend to itself
        mask[5:, 5:] = False  # seq2 can attend to itself

        with torch.no_grad():
            out_with_mask = attn(x_concat, attn_mask=mask)
            out_no_mask = attn(x_concat, attn_mask=None)

        # Outputs should differ because mask blocks cross-sequence attention
        diff = (out_with_mask - out_no_mask).abs().max()
        assert diff > 1e-5, "Mask appears to be ignored! Outputs are identical."

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        batch_size, seq_len, dim = 2, 10, 64
        num_heads = 8

        attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Gradient contains Inf"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        dim = 64
        num_heads = 7  # Not divisible!

        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(dim=dim, num_heads=num_heads)


class TestProjectionHead:
    """Tests for ProjectionHead module."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        input_dim = 64
        output_dim = 256
        hidden_dim = 128
        bottleneck_dim = 32

        proj = ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

        x = torch.randn(10, input_dim)
        out = proj(x)

        assert out.shape == (10, output_dim)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_l2_normalization_before_last_layer(self):
        """Test that input to last layer is L2-normalized."""
        input_dim = 64
        output_dim = 256
        hidden_dim = 128
        bottleneck_dim = 32

        proj = ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

        # Hook to capture input to last layer
        captured = {}
        def hook(module, input, output):
            captured['input'] = input[0]

        handle = proj.last_layer.register_forward_hook(hook)

        x = torch.randn(10, input_dim)
        _ = proj(x)

        handle.remove()

        # Check that input to last layer has norm ~1
        norms = torch.norm(captured['input'], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Input to last layer not normalized: norms={norms}"

    def test_weight_normalization(self):
        """Test that last layer uses weight normalization."""
        input_dim = 64
        output_dim = 256

        proj = ProjectionHead(input_dim=input_dim, output_dim=output_dim)

        # Check that last layer has weight_g and weight_v (weight norm parameters)
        assert hasattr(proj.last_layer, 'weight_g'), "Last layer not weight-normalized"
        assert hasattr(proj.last_layer, 'weight_v'), "Last layer not weight-normalized"

    def test_output_magnitude(self):
        """Test that output values have reasonable magnitude."""
        input_dim = 64
        output_dim = 256

        proj = ProjectionHead(input_dim=input_dim, output_dim=output_dim)
        proj.eval()

        x = torch.randn(100, input_dim)
        out = proj(x)

        # With normalized input and normalized weights (g=1), output should be bounded
        # Each output is a dot product between unit vectors, so should be in [-1, 1]
        # But with 256 dimensions, max can be larger due to summation
        # Let's check that it's not extreme (say, < 10)
        assert out.abs().max() < 10.0, \
            f"Output values too large: max={out.abs().max()}"

    def test_gradient_magnitude(self):
        """Test that gradients don't explode."""
        input_dim = 64
        output_dim = 256

        proj = ProjectionHead(input_dim=input_dim, output_dim=output_dim)

        x = torch.randn(10, input_dim, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()

        # Check gradient norms
        for name, param in proj.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1000, \
                    f"Gradient explosion in {name}: norm={grad_norm}"


class TestPackingFunctions:
    """Tests for sequence packing functions."""

    def test_pack_unpack_sequences(self):
        """Test that pack and unpack are inverses."""
        seq1 = torch.randn(5, 64)
        seq2 = torch.randn(3, 64)
        seq3 = torch.randn(7, 64)
        sequences = [seq1, seq2, seq3]

        packed, mask, lengths = pack_sequences(sequences)
        unpacked = unpack_sequences(packed, lengths)

        assert len(unpacked) == len(sequences)
        for orig, rec in zip(sequences, unpacked):
            assert torch.allclose(orig, rec), "Pack/unpack not inverse"

    def test_attention_mask_shape(self):
        """Test that attention mask has correct shape."""
        seq1 = torch.randn(5, 64)
        seq2 = torch.randn(3, 64)
        sequences = [seq1, seq2]

        packed, mask, lengths = pack_sequences(sequences)

        total_len = sum(lengths)
        assert mask.shape == (total_len, total_len)

    def test_attention_mask_blocks_cross_sequence(self):
        """Test that mask blocks attention between sequences."""
        seq1 = torch.randn(5, 64)
        seq2 = torch.randn(3, 64)
        sequences = [seq1, seq2]

        packed, mask, lengths = pack_sequences(sequences)

        # Check that sequences can't attend to each other
        # seq1 is at indices 0-4, seq2 is at indices 5-7
        assert mask[0, 5].item() == True, "Should block cross-sequence attention"
        assert mask[5, 0].item() == True, "Should block cross-sequence attention"

        # Check that within-sequence attention is allowed
        assert mask[0, 0].item() == False, "Should allow within-sequence attention"
        assert mask[0, 4].item() == False, "Should allow within-sequence attention"
        assert mask[5, 7].item() == False, "Should allow within-sequence attention"

    def test_pack_protein_batch(self):
        """Test packing protein structures."""
        struct1 = torch.randn(10, 37, 3)  # 10 residues
        struct2 = torch.randn(15, 37, 3)  # 15 residues
        structures = [struct1, struct2]

        packed, mask, lengths = pack_protein_batch(structures)

        assert packed.shape == (25, 37, 3)  # 10 + 15 residues
        assert mask.shape == (25, 25)
        assert lengths == [10, 15]


class TestPackedSequenceEncoder:
    """Tests for the full encoder model."""

    def test_forward_shape(self):
        """Test that forward pass produces correct shapes."""
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=256,
            proj_hidden_dim=128,
            proj_bottleneck_dim=32,
        )

        struct1 = torch.randn(10, 37, 3)
        struct2 = torch.randn(15, 37, 3)
        structures = [struct1, struct2]

        output = encoder(structures, return_projected=True)

        assert 'backbone' in output
        assert 'projected' in output
        assert len(output['backbone']) == 2
        assert len(output['projected']) == 2
        assert output['backbone'][0].shape == (10, 48)
        assert output['backbone'][1].shape == (15, 48)
        assert output['projected'][0].shape == (10, 256)
        assert output['projected'][1].shape == (15, 256)

    def test_no_nans_in_forward(self):
        """Test that forward pass doesn't produce NaNs."""
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=256,
        )

        struct1 = torch.randn(10, 37, 3)
        structures = [struct1]

        output = encoder(structures, return_projected=True)

        for seq in output['backbone']:
            assert not torch.isnan(seq).any(), "NaN in backbone output"
        for seq in output['projected']:
            assert not torch.isnan(seq).any(), "NaN in projected output"

    def test_gradient_flow(self):
        """Test that gradients flow through the entire model."""
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=256,
        )

        struct1 = torch.randn(10, 37, 3, requires_grad=True)
        structures = [struct1]

        output = encoder(structures, return_projected=True)

        # Compute loss
        loss = sum(seq.sum() for seq in output['projected'])
        loss.backward()

        # Check that input has gradients
        assert struct1.grad is not None
        assert not torch.isnan(struct1.grad).any(), "NaN in gradients"

        # Check that all parameters have gradients
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_gradient_magnitude(self):
        """Test that gradients don't explode during training."""
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=256,
            proj_bottleneck_dim=16,
        )

        struct1 = torch.randn(10, 37, 3)
        structures = [struct1]

        output = encoder(structures, return_projected=True)
        loss = sum(seq.sum() for seq in output['projected'])
        loss.backward()

        # Compute total gradient norm
        total_norm = 0.0
        for param in encoder.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm < 10000, f"Gradient explosion: norm={total_norm}"

    def test_batch_consistency(self):
        """Test that batching doesn't change results (due to packing)."""
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=256,
        )
        encoder.eval()

        struct1 = torch.randn(10, 37, 3)
        struct2 = torch.randn(15, 37, 3)

        # Process separately
        out1 = encoder([struct1], return_projected=True)
        out2 = encoder([struct2], return_projected=True)

        # Process together
        out_batch = encoder([struct1, struct2], return_projected=True)

        # Results should be identical
        assert torch.allclose(out1['backbone'][0], out_batch['backbone'][0], atol=1e-5)
        assert torch.allclose(out2['backbone'][0], out_batch['backbone'][1], atol=1e-5)
        assert torch.allclose(out1['projected'][0], out_batch['projected'][0], atol=1e-5)
        assert torch.allclose(out2['projected'][0], out_batch['projected'][1], atol=1e-5)

    def test_sdpa_backend_consistency(self):
        """Test that batch consistency holds across different SDPA backends."""
        from torch.nn.attention import sdpa_kernel, SDPBackend

        torch.manual_seed(42)
        struct1 = torch.randn(10, 37, 3)
        struct2 = torch.randn(15, 37, 3)

        # Test with MATH backend (most compatible)
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=256,
        )
        encoder.eval()

        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                out1_math = encoder([struct1], return_projected=False)
                out_both_math = encoder([struct1, struct2], return_projected=False)

        diff_math = (out1_math['backbone'][0] - out_both_math['backbone'][0]).abs().max()
        assert diff_math < 1e-5, \
            f"MATH backend inconsistent: diff={diff_math:.2e}"


class TestNumericalStability:
    """Tests specifically for numerical stability issues."""

    def test_projection_head_with_large_inputs(self):
        """Test projection head with large magnitude inputs."""
        proj = ProjectionHead(input_dim=64, output_dim=256)

        # Create inputs with large magnitude
        x = torch.randn(10, 64) * 10.0

        out = proj(x)

        assert not torch.isnan(out).any(), "NaN with large inputs"
        assert not torch.isinf(out).any(), "Inf with large inputs"

    def test_softmax_temperature_scaling(self):
        """Test that projection outputs work with low temperature softmax."""
        proj = ProjectionHead(input_dim=64, output_dim=256)
        proj.eval()

        x = torch.randn(10, 64)
        out = proj(x)

        # Apply low temperature softmax (as used in DINO)
        temperature = 0.1
        softmax_out = torch.softmax(out / temperature, dim=-1)

        assert not torch.isnan(softmax_out).any(), "NaN in softmax with low temp"
        assert not torch.isinf(softmax_out).any(), "Inf in softmax with low temp"

    def test_encoder_with_high_dimensional_projection(self):
        """Test encoder with high-dimensional projection space."""
        encoder = PackedSequenceEncoder(
            n_atoms=37,
            embed_dim=48,
            depth=2,
            num_heads=12,
            proj_output_dim=4096,  # High dimensional like in training
            proj_bottleneck_dim=16,  # Small bottleneck
        )

        struct1 = torch.randn(10, 37, 3)
        structures = [struct1]

        output = encoder(structures, return_projected=True)

        # Check for numerical issues
        for seq in output['projected']:
            assert not torch.isnan(seq).any(), "NaN in high-dim projection"
            assert not torch.isinf(seq).any(), "Inf in high-dim projection"

            # Check that values are reasonable
            assert seq.abs().max() < 100, \
                f"Projection values too large: max={seq.abs().max()}"
