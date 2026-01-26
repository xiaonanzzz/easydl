"""
Tier 2 Component Tests: Regression Tests for Embeddings

These tests verify that pairwise distances remain consistent across code changes.
Golden data stores only pairwise distances (not raw embeddings).
"""
import pytest
import torch
import numpy as np
import json
from pathlib import Path

from easydl.dml.pytorch_models import Resnet18MetricModel


# Path to store golden regression data
GOLDEN_DATA_DIR = Path(__file__).parent / "golden_data"


def generate_golden_data(model_name: str = 'resnet18', seed: int = 42, num_samples: int = 10):
    """
    Generate golden pairwise distance data.

    Usage:
        python -c "from tests.tier2_component.test_regression import generate_golden_data; generate_golden_data()"
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create deterministic input images
    input_images = torch.randn(num_samples, 3, 224, 224)

    # Generate embeddings
    model = Resnet18MetricModel(embedding_dim=128)
    model.eval()

    with torch.no_grad():
        embeddings = model(input_images).numpy()

    # Calculate pairwise cosine distances (1 - cosine_similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    pairwise_distances = 1 - (normalized @ normalized.T)

    # Save only distances and metadata
    golden_data = {
        'seed': seed,
        'num_samples': num_samples,
        'model_name': model_name,
        'pairwise_distances': pairwise_distances.tolist(),
    }

    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GOLDEN_DATA_DIR / f"{model_name}_distances.json"

    with open(output_path, 'w') as f:
        json.dump(golden_data, f, indent=2)

    print(f"Golden data saved to {output_path}")
    return golden_data


def load_golden_data(model_name: str = 'resnet18') -> dict:
    """Load golden regression data."""
    path = GOLDEN_DATA_DIR / f"{model_name}_distances.json"
    if not path.exists():
        raise FileNotFoundError(f"Golden data not found. Run generate_golden_data() first.")
    with open(path, 'r') as f:
        return json.load(f)


@pytest.mark.component
class TestEmbeddingRegression:
    """Regression tests for embedding pairwise distances."""

    @pytest.fixture
    def golden_data(self):
        """Load or generate golden data."""
        try:
            return load_golden_data('resnet18')
        except FileNotFoundError:
            return generate_golden_data()

    @pytest.fixture
    def current_distances(self, golden_data):
        """Generate current pairwise distances with same seed."""
        torch.manual_seed(golden_data['seed'])
        input_images = torch.randn(golden_data['num_samples'], 3, 224, 224)

        model = Resnet18MetricModel(embedding_dim=128)
        model.eval()

        with torch.no_grad():
            embeddings = model(input_images).numpy()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        return 1 - (normalized @ normalized.T)

    def test_pairwise_distances_match(self, golden_data, current_distances):
        """Test that pairwise distances match golden values."""
        golden_distances = np.array(golden_data['pairwise_distances'])

        np.testing.assert_allclose(
            current_distances,
            golden_distances,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Pairwise distances changed from baseline!"
        )

    def test_nearest_neighbor_order_preserved(self, golden_data, current_distances):
        """Test that nearest neighbor ordering is preserved."""
        golden_distances = np.array(golden_data['pairwise_distances'])

        for i in range(len(current_distances)):
            current_order = np.argsort(current_distances[i])
            golden_order = np.argsort(golden_distances[i])

            np.testing.assert_array_equal(
                current_order,
                golden_order,
                err_msg=f"Nearest neighbor order changed for sample {i}"
            )


if __name__ == "__main__":
    generate_golden_data()
