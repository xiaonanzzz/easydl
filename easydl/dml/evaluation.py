from sklearn.neighbors import NearestNeighbors
import numpy as np

def evaluate_embedding_top1_accuracy(embeddings_dataframe):
    """
    Assuming the embeddings_dataframe is a dataframe with the following columns:
    - embedding: the embedding of the image
    - label: the label of the image

    We want to evaluate the top1 accuracy of the embeddings.
    
    """
    assert 'embedding' in embeddings_dataframe.columns
    assert 'label' in embeddings_dataframe.columns
    assert embeddings_dataframe.shape[0] > 0, "The embeddings_dataframe is empty, at least one row is required for evaluation"
    
    # build a nearest neighbor index using scikit-learn
    embeddings = np.array(embeddings_dataframe['embedding'].tolist())
    assert embeddings.ndim == 2, f"The embeddings should be a 2D array, but got a {embeddings.ndim}D array, type: {type(embeddings)}, shape: {embeddings.shape}"

    labels = np.array(embeddings_dataframe['label'].tolist())
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    top1_label = labels[indices[:, 0]]
    top1_accuracy = (top1_label == labels).mean()
    return top1_accuracy


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
