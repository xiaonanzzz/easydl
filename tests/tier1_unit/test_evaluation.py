"""
Tier 1 Unit Tests: Evaluation Metrics

Fast tests for evaluation functions with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from easydl.dml.evaluation import (
    calculate_cosine_similarity_matrix,
    create_pairwise_similarity_ground_truth_matrix,
    evaluate_embedding_top1_accuracy_ignore_self,
)


def _create_embeddings_dataframe(embeddings, labels):
    """Helper to create dataframe from embeddings and labels."""
    return pd.DataFrame({"embedding": list(embeddings), "label": labels})


@pytest.mark.unit
class TestEvaluationMetrics:
    """Unit tests for evaluation metrics."""

    def test_perfect_embeddings_give_high_accuracy(self, random_embeddings):
        """Test that perfectly clustered embeddings give high accuracy."""
        # Create embeddings where same-class items are identical
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        embeddings = np.zeros((9, 64))

        # Same-class items have identical embeddings
        for i, label in enumerate(labels):
            embeddings[i, label * 20 : (label + 1) * 20] = 1.0

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        df = _create_embeddings_dataframe(embeddings, labels)
        result = evaluate_embedding_top1_accuracy_ignore_self(df)
        accuracy = result["avg_top1_accuracy"]
        assert accuracy == 1.0

    def test_random_embeddings_give_low_accuracy(self):
        """Test that random embeddings give approximately chance-level accuracy."""
        np.random.seed(42)
        num_samples = 500
        num_classes = 10

        labels = np.random.randint(0, num_classes, num_samples)
        embeddings = np.random.randn(num_samples, 64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        df = _create_embeddings_dataframe(embeddings, labels)
        result = evaluate_embedding_top1_accuracy_ignore_self(df)
        accuracy = result["avg_top1_accuracy"]

        # Chance accuracy for 10 classes is ~10%
        assert 0.05 < accuracy < 0.25

    def test_accuracy_in_valid_range(self, random_embeddings, random_labels):
        """Test that accuracy is always between 0 and 1."""
        embeddings = random_embeddings(num_samples=100, dim=64)
        labels = random_labels(num_samples=100, num_classes=10)

        df = _create_embeddings_dataframe(embeddings, labels)
        result = evaluate_embedding_top1_accuracy_ignore_self(df)
        accuracy = result["avg_top1_accuracy"]

        assert 0.0 <= accuracy <= 1.0


@pytest.mark.unit
class TestCosineSimilarityMatrix:
    """Unit tests for cosine similarity matrix computation."""

    def test_similarity_matrix_is_symmetric(self):
        """Test that cosine similarity matrix is symmetric."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        sim_matrix = calculate_cosine_similarity_matrix(embeddings)

        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T)

    def test_self_similarity_is_one(self):
        """Test that self-similarity (diagonal) is 1.0."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        sim_matrix = calculate_cosine_similarity_matrix(embeddings)

        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(50))

    def test_similarity_in_valid_range(self):
        """Test that all similarity values are between -1 and 1."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        sim_matrix = calculate_cosine_similarity_matrix(embeddings)

        assert np.all(sim_matrix >= -1.0 - 1e-6)
        assert np.all(sim_matrix <= 1.0 + 1e-6)

    def test_identical_embeddings_have_similarity_one(self):
        """Test that identical embeddings have similarity 1.0."""
        embeddings = np.ones((5, 64))
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        sim_matrix = calculate_cosine_similarity_matrix(embeddings)

        np.testing.assert_array_almost_equal(sim_matrix, np.ones((5, 5)))


@pytest.mark.unit
class TestGroundTruthMatrix:
    """Unit tests for ground truth matrix creation."""

    def test_ground_truth_matrix_correctness(self):
        """Test that ground truth matrix marks same-class pairs correctly."""
        labels = np.array([0, 0, 1, 1, 2])
        gt_matrix = create_pairwise_similarity_ground_truth_matrix(labels)

        expected = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

        np.testing.assert_array_equal(gt_matrix, expected)

    def test_ground_truth_matrix_is_symmetric(self):
        """Test that ground truth matrix is symmetric."""
        labels = np.random.randint(0, 5, 20)
        gt_matrix = create_pairwise_similarity_ground_truth_matrix(labels)

        np.testing.assert_array_equal(gt_matrix, gt_matrix.T)

    def test_ground_truth_matrix_diagonal_is_one(self):
        """Test that diagonal is all ones (self is same class)."""
        labels = np.array([0, 1, 2, 3, 4])
        gt_matrix = create_pairwise_similarity_ground_truth_matrix(labels)

        np.testing.assert_array_equal(np.diag(gt_matrix), np.ones(5))

    def test_all_same_class(self):
        """Test when all samples are same class."""
        labels = np.zeros(10, dtype=int)
        gt_matrix = create_pairwise_similarity_ground_truth_matrix(labels)

        np.testing.assert_array_equal(gt_matrix, np.ones((10, 10)))

    def test_all_different_classes(self):
        """Test when all samples are different classes."""
        labels = np.arange(5)
        gt_matrix = create_pairwise_similarity_ground_truth_matrix(labels)

        np.testing.assert_array_equal(gt_matrix, np.eye(5))
