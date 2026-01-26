import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


def agglomerative_clustering_pairwise_cosine_distance_with_threshold(
    df_embeddings, threshold
) -> pd.DataFrame:
    """
    Run agglomerative clustering on the embeddings using pairwise cosine distance
    Args:
        df_embeddings: dataframe with required columns: embedding
        threshold: threshold for the clustering
    Returns:
        dataframe with cluster_label column
    """

    assert (
        "embedding" in df_embeddings.columns
    ), "df_detections must have embedding column"

    # Get embeddings and image paths
    df_embeddings = df_embeddings.copy()
    embeddings = df_embeddings["embedding"].tolist()

    # Compute pairwise distances
    distances = pairwise_distances(embeddings, metric="cosine")
    # Plotting the distribution of distances

    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    clustering.fit(distances)
    # Add cluster labels to the dataframe
    df_embeddings["cluster_label"] = clustering.labels_
    return df_embeddings
