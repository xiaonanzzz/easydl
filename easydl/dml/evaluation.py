import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve
from sklearn.neighbors import NearestNeighbors

from easydl.common_infer import infer_x_dataset_with_simple_stacking
from easydl.common_trainer import model_file_default_name_given_epoch
from easydl.data import GenericXYLambdaAutoLabelEncoderDataset
from easydl.dml.pytorch_models import DMLModelManager
from easydl.utils import (
    AcceleratorSetting,
    smart_any_to_torch_tensor,
    torch_load_with_prefix_removal,
)


def calculate_cosine_similarity_matrix(embedding_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the cosine similarity matrix for a given embedding matrix using PyTorch and accelerator.

    This function efficiently computes the pairwise cosine similarity between all embeddings
    in the matrix using GPU acceleration via the accelerator framework.

    Args:
        embedding_matrix: A numpy array of shape (N, D) where N is the number
                         of embeddings and D is the embedding dimension.

    Returns:
        A numpy array of shape (N, N) containing the cosine similarity matrix.
        Each element (i, j) represents the cosine similarity between embedding i and embedding j.
    """
    # Initialize accelerator if not already initialized
    if not AcceleratorSetting.using_accelerator:
        AcceleratorSetting.init()

    device = AcceleratorSetting.device

    # Convert to torch tensor and move to device
    embeddings = smart_any_to_torch_tensor(embedding_matrix).float().to(device)

    # Normalize embeddings for efficient cosine similarity computation
    # Cosine similarity = dot product of normalized vectors
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())

    # Convert back to numpy array
    return similarity_matrix.cpu().numpy()


def create_pairwise_similarity_ground_truth_matrix(labels: np.ndarray) -> np.ndarray:
    """
    Creates a symmetric, binary N x N matrix where matrix[i, j] is 1 if
    item i and item j share the same class label, and 0 otherwise.

    Args:
        labels: A 1D numpy array of class labels (e.g., [1, 2, 1, 3, 2]).

    Returns:
        An N x N numpy array (binary similarity matrix).
    """
    # Reshape the 1D labels array to (N, 1) and (1, N) for broadcasting
    labels = np.array(labels)
    labels_row = labels.reshape(-1, 1)
    labels_col = labels.reshape(1, -1)

    # Use broadcasting to check for equality across all pairs
    # The result is a boolean matrix, which is then converted to an integer (1 or 0)
    pairwise_matrix = (labels_row == labels_col).astype(int)

    return pairwise_matrix


def evaluate_pairwise_score_matrix_with_true_label(
    y_true_matrix: np.ndarray, y_score_matrix: np.ndarray
) -> dict:
    """
    Evaluate a pairwise score matrix against a ground truth label matrix.

    This function calculates two metrics:
    1. Top1 accuracy (ignoring itself): For each item, find its nearest neighbor (excluding itself)
       and check if they share the same class label.
    2. PR AUC: Precision-Recall Area Under Curve using off-diagonal elements.

    Args:
        y_true_matrix: The N x N ground truth matrix (binary labels, 0 or 1).
                      y_true_matrix[i, j] = 1 if items i and j share the same class, 0 otherwise.
        y_score_matrix: The N x N prediction score matrix (floats, typically 0 to 1).
                       Higher scores indicate higher similarity.

    Returns:
        A dictionary with the following keys:
        - 'top1_accuracy': The 1-nearest neighbor accuracy (float between 0.0 and 1.0)
        - 'pr_auc': The Precision-Recall AUC score (float between 0.0 and 1.0)
    """
    n = y_true_matrix.shape[0]
    assert y_score_matrix.shape == (
        n,
        n,
    ), f"Shape mismatch: y_true_matrix is {y_true_matrix.shape}, y_score_matrix is {y_score_matrix.shape}"

    # Calculate 1NN accuracy ignoring itself
    # For each row, find the index with the highest score (excluding diagonal/itself)
    correct_predictions = 0
    total_predictions = 0

    for i in range(n):
        # Get scores for row i, excluding the diagonal element (itself)
        row_scores = y_score_matrix[i, :].copy()
        row_scores[i] = -np.inf  # Set diagonal to -inf so it won't be selected

        # Find the index of the nearest neighbor (highest score)
        nearest_neighbor_idx = np.argmax(row_scores)

        # Check if the nearest neighbor has the same label (y_true_matrix[i, nearest_neighbor_idx] == 1)
        if y_true_matrix[i, nearest_neighbor_idx] == 1:
            correct_predictions += 1
        total_predictions += 1

    nn_accuracy = (
        correct_predictions / total_predictions if total_predictions > 0 else 0.0
    )

    # Calculate PR AUC using the existing function
    metrics = calculate_precision_recall_auc_for_pairwise_score_matrix(
        y_true_matrix, y_score_matrix
    )

    metrics["top1_accuracy"] = nn_accuracy
    return metrics


def calculate_precision_recall_auc_for_pairwise_score_matrix(
    y_true_matrix: np.ndarray, y_score_matrix: np.ndarray
) -> dict:
    """
    Calculate the Precision-Recall Area Under Curve (PR AUC) between two N x N matrices.

    This function is used to evaluate the quality of similarity scores (y_score_matrix) against ground truth labels (y_true_matrix)
    for all possible pairs in a dataset (excluding self-pairs on the diagonal). This is common in pairwise metric learning evaluation.

    Args:
        y_true_matrix: numpy.ndarray of shape (N, N)
            Binary ground truth matrix. y_true_matrix[i, j] = 1 if i and j belong to the same class, else 0.
        y_score_matrix: numpy.ndarray of shape (N, N)
            Score matrix where higher values indicate stronger predicted similarity between item i and j.

    Returns:
        dict:
            - precision_list: list or numpy array of precision values for different thresholds
            - recall_list: list or numpy array of recall values for different thresholds
            - pr_auc: calculated PR AUC value (float)
            - thresholds: thresholds used to compute precision and recall

    Note:
        Only the off-diagonal upper-triangular part of the matrices is used (each pair once, no duplicates, no self-pair).
    """

    # 1. Flatten the unique, off-diagonal elements
    # Get the indices of the upper triangle (k=1 excludes the diagonal)
    n = y_true_matrix.shape[0]
    # Create an index mask for the upper triangle (excluding the main diagonal)
    # Note: np.triu_mask(n, k=1) is only available in specific numpy versions;
    # we use the more standard np.triu indices check below.

    # Standard way to get indices for the upper triangle (excluding diagonal)
    i_upper = np.triu_indices(n, k=1)

    # Flatten the true labels and predicted scores using the indices
    y_true_flattened = y_true_matrix[i_upper]
    y_score_flattened = y_score_matrix[i_upper]

    # Check for empty or singular data which would cause errors
    if len(y_true_flattened) == 0:
        print(
            "Error: Input matrices are too small or empty after filtering the diagonal."
        )
        return {
            "precision_list": [],
            "recall_list": [],
            "pr_auc": 0.0,
            "threshold_list": [],
        }

    # precision_recall_curve computes precision and recall for all possible thresholds.
    precision, recall, thresholds = precision_recall_curve(
        y_true_flattened, y_score_flattened
    )

    # auc computes the area under the curve using the trapezoidal rule.
    pr_auc = float(auc(recall, precision))
    return {
        "precision_list": precision,
        "recall_list": recall,
        "pr_auc": pr_auc,
        "threshold_list": thresholds,
    }


def evaluate_embedding_top1_accuracy_ignore_self(embeddings_dataframe):
    """
    Assuming the embeddings_dataframe is a dataframe with the following columns:
    - embedding: the embedding of the image
    - label: the label of the image

    The return value is a dictionary with the following keys:
    - avg_top1_accuracy: the average top1 accuracy, a float between 0 and 1
    - accuracy_upper_bound: the upper bound of the top1 accuracy, a float between 0 and 1
    - result_dataframe: the dataframe with the following columns:
        - top1_label: the label of the top1 nearest neighbor
        - top1_distance: the distance to the top1 nearest neighbor
        - top1_index: the index of the top1 nearest neighbor
        - same_group_count: the number of samples in the same group

    """
    embeddings_dataframe = embeddings_dataframe.copy()
    assert "embedding" in embeddings_dataframe.columns
    assert "label" in embeddings_dataframe.columns
    assert (
        embeddings_dataframe.shape[0] > 0
    ), "The embeddings_dataframe is empty, at least one row is required for evaluation"

    # build a nearest neighbor index using scikit-learn
    embeddings = np.array(embeddings_dataframe["embedding"].tolist())
    assert (
        embeddings.ndim == 2
    ), f"The embeddings should be a 2D array, but got a {embeddings.ndim}D array, type: {type(embeddings)}, shape: {embeddings.shape}"

    labels = np.array(embeddings_dataframe["label"].tolist())
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    top1_label = labels[indices[:, 1]]
    top1_accuracy = (top1_label == labels).mean()
    embeddings_dataframe["top1_label"] = top1_label
    embeddings_dataframe["top1_distance"] = distances[:, 1]
    embeddings_dataframe["top1_index"] = indices[:, 1]

    # for each label, count the number of samples, and for each sample, show the number of samples in the same label
    label_counts = embeddings_dataframe["label"].value_counts()
    embeddings_dataframe["same_group_count"] = embeddings_dataframe["label"].map(
        label_counts
    )

    accuracy_upper_bound = np.mean(embeddings_dataframe["same_group_count"] > 1)

    return {
        "avg_top1_accuracy": top1_accuracy,
        "accuracy_upper_bound": accuracy_upper_bound,
        "result_dataframe": embeddings_dataframe,
    }


def evaluate_major_cluster_precision_recall(embeddings_dataframe):
    """
    Assuming the embeddings_dataframe is a dataframe with the following columns:
    - cluster_id: the id of the cluster
    - label: the label of the cluster

    We want to evaluate the precision and recall of the major cluster.

    """
    assert "cluster_id" in embeddings_dataframe.columns
    assert "label" in embeddings_dataframe.columns
    assert (
        embeddings_dataframe.shape[0] > 0
    ), "The embeddings_dataframe is empty, at least one row is required for evaluation"

    # get the major cluster id
    major_cluster_id = embeddings_dataframe["cluster_id"].value_counts().idxmax()

    # get the major cluster label
    major_cluster_label = (
        embeddings_dataframe[embeddings_dataframe["cluster_id"] == major_cluster_id][
            "label"
        ]
        .value_counts()
        .idxmax()
    )

    # get tp, fp, fn
    tp = embeddings_dataframe[
        (embeddings_dataframe["cluster_id"] == major_cluster_id)
        & (embeddings_dataframe["label"] == major_cluster_label)
    ].shape[0]
    fn = embeddings_dataframe[
        (embeddings_dataframe["cluster_id"] != major_cluster_id)
        & (embeddings_dataframe["label"] == major_cluster_label)
    ].shape[0]
    fp = embeddings_dataframe[
        (embeddings_dataframe["cluster_id"] == major_cluster_id)
        & (embeddings_dataframe["label"] != major_cluster_label)
    ].shape[0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "major_cluster_id": major_cluster_id,
        "major_cluster_label": major_cluster_label,
    }


class StandardEmbeddingEvaluationV1:
    @staticmethod
    def evaluate_given_dataset(
        dataset: GenericXYLambdaAutoLabelEncoderDataset, model
    ) -> dict:
        evaluator = StandardEmbeddingEvaluationV1(dataset)
        return evaluator.evaluate(model)

    def __init__(
        self,
        test_dataset: GenericXYLambdaAutoLabelEncoderDataset,
        save_pairwise_score_matrix_in_metric_dict: bool = False,
    ):
        self.test_dataset = test_dataset
        self.pairwise_similarity_ground_truth_matrix = (
            create_pairwise_similarity_ground_truth_matrix(
                self.test_dataset.get_y_list_with_encoded_labels()
            )
        )
        self.save_pairwise_score_matrix_in_metric_dict = (
            save_pairwise_score_matrix_in_metric_dict
        )

    def evaluate_given_embeddings(self, embeddings: np.ndarray) -> dict:

        pairwise_similarity_score_matrix = calculate_cosine_similarity_matrix(
            embeddings
        )
        metrics = evaluate_pairwise_score_matrix_with_true_label(
            self.pairwise_similarity_ground_truth_matrix,
            pairwise_similarity_score_matrix,
        )
        if self.save_pairwise_score_matrix_in_metric_dict:
            metrics["pairwise_similarity_score_matrix"] = (
                pairwise_similarity_score_matrix
            )
        return metrics

    def evaluate(self, model) -> dict:
        all_embeddings = infer_x_dataset_with_simple_stacking(self.test_dataset, model)
        return self.evaluate_given_embeddings(all_embeddings)


class StandardEmbeddingEvaluationReportV2:
    def __init__(self):
        self.metrics = {}
        self.ground_truth_matrix = None
        self.pairwise_similarity_score_matrix = None


def standard_embedding_evaluation_v2(
    embeddings: np.ndarray, labels: np.ndarray
) -> StandardEmbeddingEvaluationReportV2:
    ground_truth_matrix = create_pairwise_similarity_ground_truth_matrix(labels)
    pairwise_similarity_score_matrix = calculate_cosine_similarity_matrix(embeddings)
    metrics = evaluate_pairwise_score_matrix_with_true_label(
        ground_truth_matrix, pairwise_similarity_score_matrix
    )
    report = StandardEmbeddingEvaluationReportV2()
    report.metrics = metrics
    report.ground_truth_matrix = ground_truth_matrix
    report.pairwise_similarity_score_matrix = pairwise_similarity_score_matrix
    return report


class DeepMetricLearningImageEvaluatorOnEachEpoch:
    def __init__(
        self,
        test_dataset: GenericXYLambdaAutoLabelEncoderDataset,
        model_name: str,
        embedding_dim: int,
        model_epoch_params_dir: str,
        num_epochs: int,
        evaluation_report_dir: str = None,
    ):
        if evaluation_report_dir is None:
            evaluation_report_dir = ""
        Path(evaluation_report_dir).mkdir(parents=True, exist_ok=True)
        standard_embedding_evaluator = StandardEmbeddingEvaluationV1(test_dataset)

        results_summary_of_each_epoch = []

        model = DMLModelManager.get_model(model_name, embedding_dim)
        # for each file in the model_param_dir, load the model and evaluate the PR AUC

        for epoch in range(1, num_epochs + 1):
            model_file_name = model_file_default_name_given_epoch(epoch)
            model_path = os.path.join(model_epoch_params_dir, model_file_name)
            if not os.path.exists(model_path):
                print(
                    f"Model file {model_path} does not exist, skipping evaluation for epoch {epoch}"
                )
                continue
            model.load_state_dict(torch_load_with_prefix_removal(model_path))

            metrics = standard_embedding_evaluator.evaluate(model)
            results_summary_of_each_epoch.append(
                {
                    "epoch": epoch,
                    "top1_accuracy": metrics["top1_accuracy"],
                    "pr_auc": metrics["pr_auc"],
                }
            )
        if len(results_summary_of_each_epoch) == 0:
            print(
                "No model files found in the model_epoch_params_dir, skipping evaluation"
            )
            return

        self.report_csv_path = os.path.join(
            evaluation_report_dir,
            "deep_metric_learning_image_evaluator_on_each_epoch.csv",
        )
        results_summary_of_each_epoch_df = pd.DataFrame(results_summary_of_each_epoch)
        results_summary_of_each_epoch_df.to_csv(self.report_csv_path, index=False)


class EmbeddingEvaluationAgglomerativeClustering:

    @staticmethod
    def evaluate_given_threshold(embeddings_dataframe, threshold=2.0) -> dict:
        from easydl.clustering.functions import (
            agglomerative_clustering_pairwise_cosine_distance_with_threshold,
        )
        from easydl.clustering.metrics import calculate_clustering_metrics_all_in_one

        df_to_be_clustered = embeddings_dataframe.copy()

        df_clustered = agglomerative_clustering_pairwise_cosine_distance_with_threshold(
            df_to_be_clustered, threshold=threshold
        )

        results = calculate_clustering_metrics_all_in_one(df_clustered)
        return results
