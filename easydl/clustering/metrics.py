from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score, calinski_harabasz_score, rand_score, davies_bouldin_score, v_measure_score, fowlkes_mallows_score



def calculate_clustering_metrics_all_in_one(df_clustered, save_input_df_clustered=True) -> dict:
    """
    Calculate all clustering metrics in one function
    Args:
        df_clustered: dataframe with required columns: cluster_label, label, embedding
    Returns:
        results: dictionary with all clustering metrics
    """

    assert 'cluster_label' in df_clustered.columns, "df_clustered must have cluster_label column"
    assert 'label' in df_clustered.columns, "df_clustered must have label column, which represents the true label of one embedding"
    assert 'embedding' in df_clustered.columns, "df_clustered must have embedding column"

    # step 1, keep only the primary embeddings
    results = {
        "df_clustered": None,
    }
    
    embeddings = df_clustered['embedding'].tolist()
    results['number_of_labels'] = df_clustered['label'].nunique()
    results['number_of_clusters'] = df_clustered['cluster_label'].nunique()
    results['rand_score'] = rand_score(df_clustered['label'], df_clustered['cluster_label'])
    results['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, df_clustered['cluster_label'].tolist())
    results['davies_bouldin_score'] = davies_bouldin_score(embeddings, df_clustered['cluster_label'].tolist())
    results['adjusted_mutual_info_score'] = adjusted_mutual_info_score(df_clustered['label'], df_clustered['cluster_label'])
    results['homogeneity_score'] = homogeneity_score(df_clustered['label'], df_clustered['cluster_label'])
    results['completeness_score'] = completeness_score(df_clustered['label'], df_clustered['cluster_label'])
    results['v_measure_score'] = v_measure_score(df_clustered['label'], df_clustered['cluster_label'])
    results['fowlkes_mallows_score'] = fowlkes_mallows_score(df_clustered['label'], df_clustered['cluster_label'])
    if save_input_df_clustered:
        results['df_clustered'] = df_clustered
    return results