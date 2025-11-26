from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import torch
import torch.nn.functional as F
from easydl.utils import AcceleratorSetting



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
    embeddings = torch.from_numpy(embedding_matrix).float().to(device)
    
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

def calculate_pr_auc_for_matrices(y_true_matrix: np.ndarray, y_score_matrix: np.ndarray) -> float:
    """
    Calculates the Area Under the Precision-Recall Curve (PR AUC) for a
    predicted score matrix against a ground truth binary similarity matrix.

    It only considers the unique off-diagonal elements (the upper triangle)
    to calculate the metric, as the diagonal is usually trivial (i.e.,
    an item is always similar to itself).

    Args:
        y_true_matrix: The N x N ground truth matrix (binary labels, 0 or 1).
        y_score_matrix: The N x N prediction score matrix (floats, typically 0 to 1).

    Returns:
        The PR AUC score (a float between 0.0 and 1.0).
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
        print("Error: Input matrices are too small or empty after filtering the diagonal.")
        return 0.0

    # 2. Calculate Precision, Recall, and AUC
    try:
        # precision_recall_curve computes precision and recall for all possible thresholds.
        precision, recall, _ = precision_recall_curve(y_true_flattened, y_score_flattened)

        # auc computes the area under the curve using the trapezoidal rule.
        pr_auc = auc(recall, precision)

        return pr_auc
    except ValueError as e:
        # This usually happens if the number of positive or negative samples is zero.
        # This is a good sanity check for highly imbalanced/degenerate data.
        print(f"Error during AUC calculation (likely due to single-class data): {e}")
        return 0.0

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
    assert 'embedding' in embeddings_dataframe.columns
    assert 'label' in embeddings_dataframe.columns
    assert embeddings_dataframe.shape[0] > 0, "The embeddings_dataframe is empty, at least one row is required for evaluation"
    
    # build a nearest neighbor index using scikit-learn
    embeddings = np.array(embeddings_dataframe['embedding'].tolist())
    assert embeddings.ndim == 2, f"The embeddings should be a 2D array, but got a {embeddings.ndim}D array, type: {type(embeddings)}, shape: {embeddings.shape}"

    labels = np.array(embeddings_dataframe['label'].tolist())
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    top1_label = labels[indices[:, 1]]
    top1_accuracy = (top1_label == labels).mean()
    embeddings_dataframe['top1_label'] = top1_label
    embeddings_dataframe['top1_distance'] = distances[:, 1]
    embeddings_dataframe['top1_index'] = indices[:, 1]

    # for each label, count the number of samples, and for each sample, show the number of samples in the same label
    label_counts = embeddings_dataframe['label'].value_counts()
    embeddings_dataframe['same_group_count'] = embeddings_dataframe['label'].map(label_counts)

    accuracy_upper_bound = np.mean(embeddings_dataframe['same_group_count'] > 1)

    return {'avg_top1_accuracy': top1_accuracy, 'accuracy_upper_bound': accuracy_upper_bound, 'result_dataframe': embeddings_dataframe}


def evaluate_major_cluster_precision_recall(embeddings_dataframe):
    """
    Assuming the embeddings_dataframe is a dataframe with the following columns:
    - cluster_id: the id of the cluster
    - label: the label of the cluster

    We want to evaluate the precision and recall of the major cluster.
    
    """
    assert 'cluster_id' in embeddings_dataframe.columns
    assert 'label' in embeddings_dataframe.columns
    assert embeddings_dataframe.shape[0] > 0, "The embeddings_dataframe is empty, at least one row is required for evaluation"

    # get the major cluster id
    major_cluster_id = embeddings_dataframe['cluster_id'].value_counts().idxmax()

    # get the major cluster label
    major_cluster_label = embeddings_dataframe[embeddings_dataframe['cluster_id'] == major_cluster_id]['label'].value_counts().idxmax()

    # get tp, fp, fn
    tp = embeddings_dataframe[(embeddings_dataframe['cluster_id'] == major_cluster_id) & (embeddings_dataframe['label'] == major_cluster_label)].shape[0]
    fn = embeddings_dataframe[(embeddings_dataframe['cluster_id'] != major_cluster_id) & (embeddings_dataframe['label'] == major_cluster_label)].shape[0]
    fp = embeddings_dataframe[(embeddings_dataframe['cluster_id'] == major_cluster_id) & (embeddings_dataframe['label'] != major_cluster_label)].shape[0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return {'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn, 'major_cluster_id': major_cluster_id, 'major_cluster_label': major_cluster_label}



class EmbeddingEvaluationAgglomerativeClustering:

    @staticmethod
    def evaluate_given_threshold(embeddings_dataframe, threshold=2.0) -> dict:
        from easydl.clustering.functions import agglomerative_clustering_pairwise_cosine_distance_with_threshold
        from easydl.clustering.metrics import calculate_clustering_metrics_all_in_one
        df_to_be_clustered = embeddings_dataframe.copy()

        df_clustered = agglomerative_clustering_pairwise_cosine_distance_with_threshold(df_to_be_clustered, threshold=threshold)

        results = calculate_clustering_metrics_all_in_one(df_clustered)
        return results