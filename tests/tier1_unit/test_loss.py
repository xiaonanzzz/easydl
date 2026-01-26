"""
Tier 1 Unit Tests: Loss Functions

Fast tests for loss functions with synthetic data.
"""
import pytest
import torch
import torch.nn.functional as F

from easydl.dml.loss import ArcFace


@pytest.mark.unit
class TestArcFace:
    """Unit tests for ArcFace loss function."""

    def test_basic_forward(self):
        """Test basic forward pass of ArcFace loss."""
        batch_size = 4
        embedding_dim = 128
        num_classes = 10

        arcface = ArcFace(in_features=embedding_dim, out_features=num_classes)

        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, num_classes, (batch_size,))

        output = arcface(embeddings, labels)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_output_shape(self):
        """Test output shape matches expected dimensions."""
        for batch_size in [1, 4, 16]:
            for num_classes in [5, 10, 100]:
                arcface = ArcFace(in_features=64, out_features=num_classes)
                embeddings = torch.randn(batch_size, 64)
                labels = torch.randint(0, num_classes, (batch_size,))

                output = arcface(embeddings, labels)
                assert output.shape == (batch_size, num_classes)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through ArcFace."""
        arcface = ArcFace(in_features=64, out_features=5)
        embeddings = torch.randn(2, 64, requires_grad=True)
        labels = torch.randint(0, 5, (2,))

        output = arcface(embeddings, labels)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        assert embeddings.grad is not None
        assert arcface.weight.grad is not None
        assert not torch.allclose(embeddings.grad, torch.zeros_like(embeddings.grad))

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        arcface = ArcFace(in_features=64, out_features=10, s=64.0, m=0.50)

        # Test with normalized embeddings
        embeddings = F.normalize(torch.randn(4, 64), p=2, dim=1)
        labels = torch.randint(0, 10, (4,))

        output = arcface(embeddings, labels)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test with very small embeddings
        small_embeddings = torch.randn(4, 64) * 1e-6
        output_small = arcface(small_embeddings, labels)
        assert not torch.isnan(output_small).any()

    def test_different_margins(self):
        """Test ArcFace with different margin values."""
        for m in [0.1, 0.3, 0.5, 0.7]:
            arcface = ArcFace(in_features=64, out_features=5, s=64.0, m=m)
            embeddings = F.normalize(torch.randn(2, 64), p=2, dim=1)
            labels = torch.randint(0, 5, (2,))

            output = arcface(embeddings, labels)
            assert output.shape == (2, 5)
            assert not torch.isnan(output).any()

    def test_scale_factor(self):
        """Test that scale factor is applied correctly."""
        for s in [32.0, 64.0, 128.0]:
            arcface = ArcFace(in_features=32, out_features=3, s=s, m=0.50)
            embeddings = F.normalize(torch.randn(2, 32), p=2, dim=1)
            labels = torch.randint(0, 3, (2,))

            output = arcface(embeddings, labels)

            # For non-target classes, output should be exactly s * cosine
            normalized_weight = F.normalize(arcface.weight, p=2, dim=1)
            cosine_sim = F.linear(embeddings, normalized_weight)

            for i, label in enumerate(labels):
                for j in range(3):
                    if j != label:
                        expected = s * cosine_sim[i, j]
                        actual = output[i, j]
                        assert torch.allclose(actual, expected, atol=1e-5)
