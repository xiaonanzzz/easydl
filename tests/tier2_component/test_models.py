"""
Tier 2 Component Tests: Model Components

Tests for model forward passes and architecture.
May require GPU for larger models.
"""

import pytest
import torch

from easydl.dml.pytorch_models import Resnet18MetricModel, Resnet50MetricModel


@pytest.mark.component
class TestResnet18MetricModel:
    """Component tests for ResNet18 metric model."""

    def test_model_loads_pretrained(self):
        """Test that model loads pretrained weights."""
        model = Resnet18MetricModel(embedding_dim=128)

        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "backbone")
        assert hasattr(model, "embedding")

    def test_forward_pass(self, sample_batch_tensor):
        """Test forward pass with batch of images."""
        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch_tensor)

        assert output.shape == (8, 128)

    def test_output_is_normalized(self, sample_batch_tensor):
        """Test that output embeddings are L2 normalized."""
        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch_tensor)

        norms = output.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_different_embedding_dims(self, sample_batch_tensor):
        """Test model with different embedding dimensions."""
        for dim in [32, 64, 128, 256, 512]:
            model = Resnet18MetricModel(embedding_dim=dim)
            model.eval()

            with torch.no_grad():
                output = model(sample_batch_tensor)

            assert output.shape == (8, dim)

    def test_single_image_forward(self, sample_image_tensor):
        """Test forward pass with single image."""
        model = Resnet18MetricModel(embedding_dim=64)
        model.eval()

        with torch.no_grad():
            output = model(sample_image_tensor)

        assert output.shape == (1, 64)


@pytest.mark.component
class TestResnet50MetricModel:
    """Component tests for ResNet50 metric model."""

    def test_model_loads_pretrained(self):
        """Test that model loads pretrained weights."""
        model = Resnet50MetricModel(embedding_dim=128)

        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "backbone")
        assert hasattr(model, "embedding")

    def test_forward_pass(self, sample_batch_tensor):
        """Test forward pass with batch of images."""
        model = Resnet50MetricModel(embedding_dim=128)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch_tensor)

        assert output.shape == (8, 128)

    def test_output_is_normalized(self, sample_batch_tensor):
        """Test that output embeddings are L2 normalized."""
        model = Resnet50MetricModel(embedding_dim=128)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch_tensor)

        norms = output.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


@pytest.mark.component
@pytest.mark.gpu
class TestModelsOnGPU:
    """Component tests for models on GPU."""

    def test_resnet18_on_gpu(self, device, sample_batch_tensor):
        """Test ResNet18 forward pass on GPU."""
        if device.type != "cuda":
            pytest.skip("GPU not available")

        model = Resnet18MetricModel(embedding_dim=128).to(device)
        model.eval()

        input_tensor = sample_batch_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.device.type == "cuda"
        assert output.shape == (8, 128)

    def test_resnet50_on_gpu(self, device, sample_batch_tensor):
        """Test ResNet50 forward pass on GPU."""
        if device.type != "cuda":
            pytest.skip("GPU not available")

        model = Resnet50MetricModel(embedding_dim=128).to(device)
        model.eval()

        input_tensor = sample_batch_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.device.type == "cuda"
        assert output.shape == (8, 128)
