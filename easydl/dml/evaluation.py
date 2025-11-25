from sklearn.neighbors import NearestNeighbors
import numpy as np


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