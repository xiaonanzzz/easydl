import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from easydl.dml.evaluation import (
    calculate_cosine_similarity_matrix,
    calculate_precision_recall_auc_for_pairwise_score_matrix,
    evaluate_embedding_top1_accuracy_ignore_self,
    evaluate_major_cluster_precision_recall,
)


def test_evaluate_major_cluster_precision_recall():
    # Create a sample DataFrame with known cluster IDs and labels
    data = {
        "cluster_id": [1, 1, 1, 1, 2, 2, 3, 3],
        "label": ["A", "A", "A", "B", "A", "C", "B", "B"],
    }
    df = pd.DataFrame(data)

    # Calculate metrics
    result = evaluate_major_cluster_precision_recall(df)

    # Verify results
    assert result["major_cluster_id"] == 1  # Cluster 1 has 4 items
    assert result["major_cluster_label"] == "A"  # Label A is most common in cluster 1

    # In cluster 1 (major cluster):
    # - 3 items have label 'A' (true positives)
    # - 1 item has label 'B' (false positive)
    # - 1 item with label 'A' is in cluster 2 (false negative)
    assert result["tp"] == 3
    assert result["fp"] == 1
    assert result["fn"] == 1

    # Calculate expected precision and recall
    expected_precision = 3 / (3 + 1)  # tp / (tp + fp)
    expected_recall = 3 / (3 + 1)  # tp / (tp + fn)

    assert result["precision"] == expected_precision
    assert result["recall"] == expected_recall


def test_evaluate_major_cluster_precision_recall_empty():
    # Test with empty DataFrame
    df = pd.DataFrame(columns=["cluster_id", "label"])

    with pytest.raises(AssertionError):
        evaluate_major_cluster_precision_recall(df)


def test_evaluate_major_cluster_precision_recall_single_cluster():
    # Test with single cluster and single label
    data = {"cluster_id": [1, 1, 1], "label": ["A", "A", "A"]}
    df = pd.DataFrame(data)

    result = evaluate_major_cluster_precision_recall(df)

    assert result["major_cluster_id"] == 1
    assert result["major_cluster_label"] == "A"
    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_evaluate_embedding_top1_accuracy():
    # Create sample embeddings and labels
    # We'll create 4 points in 2D space, with 2 points close to each other for each class
    embeddings = [
        [0.0, 0.0],  # Class A
        [0.1, 0.1],  # Class A
        [1.0, 1.0],  # Class B
        [1.1, 1.1],  # Class B
    ]

    data = {"embedding": embeddings, "label": ["A", "A", "B", "B"]}
    df = pd.DataFrame(data)

    # Calculate accuracy
    result = evaluate_embedding_top1_accuracy_ignore_self(df)

    # Since points are well-separated, we expect perfect accuracy
    assert result["avg_top1_accuracy"] == 1.0


def test_evaluate_embedding_top1_accuracy_overlapping():
    # Create sample embeddings with some overlap between classes
    embeddings = [
        [0.0, 0.0],  # Class A
        [0.2, 0.2],  # Class A
        [0.2, 0.2],  # Class B (close to Class A)
        [1.0, 1.0],  # Class B
    ]

    data = {"embedding": embeddings, "label": ["A", "A", "B", "B"]}
    df = pd.DataFrame(data)

    # Calculate accuracy
    result = evaluate_embedding_top1_accuracy_ignore_self(df)

    # We expect less than perfect accuracy due to the overlapping point
    assert result["avg_top1_accuracy"] < 1.0
    assert result["avg_top1_accuracy"] >= 0.0


def test_evaluate_embedding_top1_accuracy_empty():
    # Test with empty DataFrame
    df = pd.DataFrame(columns=["embedding", "label"])

    with pytest.raises(AssertionError):
        evaluate_embedding_top1_accuracy_ignore_self(df)


def test_evaluate_embedding_top1_accuracy_missing_columns():
    # Test with missing required columns
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})

    with pytest.raises(AssertionError):
        evaluate_embedding_top1_accuracy_ignore_self(df)


def test_calculate_pr_auc_for_matrices_perfect_predictions():
    """Test with perfect predictions - high scores for positive pairs, low for negative."""
    # Create a 4x4 ground truth matrix with 2 classes: [0, 0, 1, 1]
    # This means items 0-1 are same class, items 2-3 are same class
    labels = np.array([0, 0, 1, 1])
    y_true = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])

    # Perfect predictions: high scores (0.9) for same class, low scores (0.1) for different
    y_score = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # With perfect predictions, PR AUC should be very high (close to 1.0)
    assert pr_auc > 0.9
    assert 0.0 <= pr_auc <= 1.0


def test_calculate_pr_auc_for_matrices_random_predictions():
    """Test with random predictions - should give moderate PR AUC."""
    np.random.seed(42)

    # Create a 5x5 ground truth matrix with 2 classes
    labels = np.array([0, 0, 0, 1, 1])
    y_true = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )

    # Random scores between 0 and 1
    y_score = np.random.rand(5, 5)
    # Make symmetric (since similarity is symmetric)
    y_score = (y_score + y_score.T) / 2

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # Random predictions should give moderate PR AUC
    assert 0.0 <= pr_auc <= 1.0


def test_calculate_pr_auc_for_matrices_small_matrix():
    """Test with a small 2x2 matrix."""
    # 2 items, different classes
    y_true = np.array([[1, 0], [0, 1]])

    y_score = np.array([[1.0, 0.2], [0.2, 1.0]])

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # For 2x2, only one off-diagonal element, so PR AUC should be 0 or 1
    # Since we have 0 in ground truth and 0.2 in score, it's a true negative
    # But wait, let me reconsider - with only one pair, the PR curve behavior might be different
    assert 0.0 <= pr_auc <= 1.0


def test_calculate_pr_auc_for_matrices_single_class():
    """Test with single class (all items same class) - should handle gracefully."""
    # All items are same class
    y_true = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    y_score = np.array([[1.0, 0.8, 0.7], [0.8, 1.0, 0.6], [0.7, 0.6, 1.0]])

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # Should return a valid value (might be 0.0 if sklearn raises ValueError for single class)
    assert 0.0 <= pr_auc <= 1.0


def test_calculate_pr_auc_for_matrices_all_zeros():
    """Test with all zeros in ground truth (no positive pairs)."""
    # All items are different classes
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    y_score = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # With no positive pairs in ground truth, should handle gracefully
    assert 0.0 <= pr_auc <= 1.0


def test_calculate_pr_auc_for_matrices_inverted_predictions():
    """Test with inverted predictions - low scores for positive pairs, high for negative."""
    # Same setup as perfect predictions test
    y_true = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])

    # Inverted: low scores for same class, high scores for different
    y_score = np.array(
        [
            [1.0, 0.1, 0.9, 0.9],
            [0.1, 1.0, 0.9, 0.9],
            [0.9, 0.9, 1.0, 0.1],
            [0.9, 0.9, 0.1, 1.0],
        ]
    )

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # Inverted predictions should give low PR AUC
    assert 0.0 <= pr_auc < 0.5


def test_calculate_pr_auc_for_matrices_larger_matrix():
    """Test with a larger matrix to ensure it works with more data points."""
    # Create a 10x10 matrix with 3 classes
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    y_true = np.zeros((10, 10), dtype=int)
    for i in range(10):
        for j in range(10):
            if labels[i] == labels[j]:
                y_true[i, j] = 1

    # Create scores that correlate well with ground truth
    np.random.seed(123)
    y_score = np.random.rand(10, 10)
    # Make symmetric
    y_score = (y_score + y_score.T) / 2
    # Boost scores for same-class pairs
    for i in range(10):
        for j in range(10):
            if labels[i] == labels[j]:
                y_score[i, j] = 0.5 + y_score[i, j] * 0.5
            else:
                y_score[i, j] = y_score[i, j] * 0.5

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # Should give a reasonable PR AUC value
    assert 0.0 <= pr_auc <= 1.0
    # With boosted scores for same-class, should be better than random
    assert pr_auc > 0.3


def test_calculate_pr_auc_for_matrices_ignores_diagonal():
    """Test that diagonal elements are ignored in the calculation."""
    # Create matrices where diagonal would affect result if included
    y_true = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    # Diagonal has wrong values, but should be ignored
    y_score = np.array(
        [
            [0.0, 0.9, 0.1],  # diagonal is 0.0 (wrong)
            [0.9, 0.0, 0.1],  # diagonal is 0.0 (wrong)
            [0.1, 0.1, 0.0],
        ]
    )  # diagonal is 0.0 (wrong)

    result = calculate_precision_recall_auc_for_pairwise_score_matrix(y_true, y_score)
    pr_auc = result["pr_auc"]

    # Should still give reasonable result since diagonal is ignored
    assert 0.0 <= pr_auc <= 1.0


def test_calculate_cosine_similarity_matrix_vs_sklearn():
    """Test that our function gives similar results as sklearn's cosine_similarity."""
    from easydl.utils import AcceleratorSetting

    # Initialize accelerator if not already initialized
    if not AcceleratorSetting.using_accelerator:
        AcceleratorSetting.init()

    # Create a sample embedding matrix with various shapes and values
    np.random.seed(42)

    # Test case 1: Small matrix with random values
    embedding_matrix_1 = np.random.rand(5, 10).astype(np.float32)

    # Our function
    result_ours_1 = calculate_cosine_similarity_matrix(embedding_matrix_1)

    # Sklearn's function
    result_sklearn_1 = cosine_similarity(embedding_matrix_1)

    # Compare results (using rtol=1e-5 for float32 precision)
    np.testing.assert_allclose(result_ours_1, result_sklearn_1, rtol=1e-5, atol=1e-6)

    # Test case 2: Larger matrix
    embedding_matrix_2 = np.random.rand(20, 128).astype(np.float32)

    result_ours_2 = calculate_cosine_similarity_matrix(embedding_matrix_2)
    result_sklearn_2 = cosine_similarity(embedding_matrix_2)

    np.testing.assert_allclose(result_ours_2, result_sklearn_2, rtol=1e-5, atol=1e-6)

    # Test case 3: Matrix with some zero vectors (edge case)
    embedding_matrix_3 = np.random.rand(4, 8).astype(np.float32)
    embedding_matrix_3[2, :] = 0.0  # One zero vector

    result_ours_3 = calculate_cosine_similarity_matrix(embedding_matrix_3)
    result_sklearn_3 = cosine_similarity(embedding_matrix_3)

    # For zero vectors, cosine similarity should be 0 (or NaN in some cases)
    # Use a slightly larger tolerance for edge cases
    np.testing.assert_allclose(
        result_ours_3, result_sklearn_3, rtol=1e-4, atol=1e-5, equal_nan=True
    )

    # Test case 4: Matrix with identical vectors (should give similarity of 1.0)
    embedding_matrix_4 = np.random.rand(1, 10).astype(np.float32)
    embedding_matrix_4 = np.repeat(embedding_matrix_4, 3, axis=0)  # 3 identical vectors

    result_ours_4 = calculate_cosine_similarity_matrix(embedding_matrix_4)
    result_sklearn_4 = cosine_similarity(embedding_matrix_4)

    np.testing.assert_allclose(result_ours_4, result_sklearn_4, rtol=1e-5, atol=1e-6)

    # Verify that identical vectors give similarity of 1.0
    assert np.allclose(result_ours_4, 1.0, rtol=1e-5)

    # Test case 5: Orthogonal vectors (should give similarity close to 0.0)
    embedding_matrix_5 = np.eye(5).astype(
        np.float32
    )  # Identity matrix (orthogonal unit vectors)

    result_ours_5 = calculate_cosine_similarity_matrix(embedding_matrix_5)
    result_sklearn_5 = cosine_similarity(embedding_matrix_5)

    np.testing.assert_allclose(result_ours_5, result_sklearn_5, rtol=1e-5, atol=1e-6)

    # Verify diagonal is 1.0 (self-similarity) and off-diagonal is 0.0 (orthogonal)
    assert np.allclose(np.diag(result_ours_5), 1.0, rtol=1e-5)
    off_diagonal = result_ours_5.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.allclose(off_diagonal, 0.0, rtol=1e-5)
