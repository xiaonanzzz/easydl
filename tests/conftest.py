"""
Shared pytest fixtures and configuration for EasyDL tests.

Test Tiers:
- tier1_unit: Fast unit tests (<1s per test, no external deps)
- tier2_component: Component tests (<30s, may need GPU)
- tier3_integration: Integration tests (<5min, uses real data)
- tier4_e2e: End-to-end tests (>5min, full pipeline)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ============== Configuration ==============


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests (<1s)")
    config.addinivalue_line("markers", "component: component tests (<30s)")
    config.addinivalue_line("markers", "integration: integration tests (<5min)")
    config.addinivalue_line("markers", "e2e: end-to-end tests (>5min)")
    config.addinivalue_line("markers", "slow: slow tests (>1min)")
    config.addinivalue_line("markers", "gpu: tests requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ============== Device Fixtures ==============


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== Data Fixtures ==============


@pytest.fixture
def random_embeddings():
    """Generate random normalized embeddings."""

    def _generate(num_samples=100, dim=64, seed=42):
        np.random.seed(seed)
        emb = np.random.randn(num_samples, dim)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    return _generate


@pytest.fixture
def random_labels():
    """Generate random labels."""

    def _generate(num_samples=100, num_classes=10, seed=42):
        np.random.seed(seed)
        return np.random.randint(0, num_classes, num_samples)

    return _generate


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_tensor():
    """Create a batch of image tensors."""
    return torch.randn(8, 3, 224, 224)


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def temp_image_file(sample_pil_image):
    """Create a temporary image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        sample_pil_image.save(f.name)
        yield f.name
    os.unlink(f.name)


# ============== Cluster Data Fixtures ==============


@pytest.fixture
def sample_cluster_data():
    """Fixture providing sample cluster data for testing."""
    return {
        "cluster_id": [1, 1, 1, 1, 2, 2, 3, 3],
        "label": ["A", "A", "A", "B", "A", "C", "B", "B"],
    }


@pytest.fixture
def sample_cluster_df(sample_cluster_data):
    """Fixture providing a DataFrame with sample cluster data."""
    import pandas as pd

    return pd.DataFrame(sample_cluster_data)


# ============== Model Fixtures ==============


@pytest.fixture
def resnet18_model():
    """Create ResNet18 metric model."""
    from easydl.dml.pytorch_models import Resnet18MetricModel

    return Resnet18MetricModel(embedding_dim=128)


@pytest.fixture
def trained_model_path(tmp_path):
    """Create a temporary trained model checkpoint."""
    from easydl.dml.pytorch_models import Resnet18MetricModel

    model = Resnet18MetricModel(embedding_dim=128)
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    return path


# ============== Directory Fixtures ==============


@pytest.fixture
def exp_dir(tmp_path):
    """Create temporary experiment directory."""
    exp = tmp_path / "experiment"
    exp.mkdir()
    return exp


# ============== Dataset Fixtures (Lazy Loading) ==============


@pytest.fixture
def cub_train_small():
    """Small CUB training dataset."""
    try:
        from easydl.public_dataset.cub import (
            get_small_train_dataset_with_image_and_encoded_labels,
        )

        return get_small_train_dataset_with_image_and_encoded_labels(num_samples=100)
    except Exception:
        pytest.skip("CUB dataset not available")


@pytest.fixture
def cub_test_small():
    """Small CUB test dataset."""
    try:
        from easydl.public_dataset.cub import (
            get_small_train_dataset_with_image_and_encoded_labels,
        )

        return get_small_train_dataset_with_image_and_encoded_labels(num_samples=50)
    except Exception:
        pytest.skip("CUB dataset not available")
