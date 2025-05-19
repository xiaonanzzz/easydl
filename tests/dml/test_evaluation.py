import pytest
import pandas as pd
from easydl.dml.evaluation import evaluate_major_cluster_precision_recall

def test_evaluate_major_cluster_precision_recall():
    # Create a sample DataFrame with known cluster IDs and labels
    data = {
        'cluster_id': [1, 1, 1, 1, 
                       2, 2, 3, 3],
        'label': ['A', 'A', 'A', 'B', 
                  'A', 'C', 'B', 'B']
    }
    df = pd.DataFrame(data)
    
    # Calculate metrics
    result = evaluate_major_cluster_precision_recall(df)
    
    # Verify results
    assert result['major_cluster_id'] == 1  # Cluster 1 has 4 items
    assert result['major_cluster_label'] == 'A'  # Label A is most common in cluster 1
    
    # In cluster 1 (major cluster):
    # - 3 items have label 'A' (true positives)
    # - 1 item has label 'B' (false positive)
    # - 1 item with label 'A' is in cluster 2 (false negative)
    assert result['tp'] == 3
    assert result['fp'] == 1
    assert result['fn'] == 1
    
    # Calculate expected precision and recall
    expected_precision = 3 / (3 + 1)  # tp / (tp + fp)
    expected_recall = 3 / (3 + 1)     # tp / (tp + fn)
    
    assert result['precision'] == expected_precision
    assert result['recall'] == expected_recall

def test_evaluate_major_cluster_precision_recall_empty():
    # Test with empty DataFrame
    df = pd.DataFrame(columns=['cluster_id', 'label'])
    
    with pytest.raises(AssertionError):
        evaluate_major_cluster_precision_recall(df)

def test_evaluate_major_cluster_precision_recall_single_cluster():
    # Test with single cluster and single label
    data = {
        'cluster_id': [1, 1, 1],
        'label': ['A', 'A', 'A']
    }
    df = pd.DataFrame(data)
    
    result = evaluate_major_cluster_precision_recall(df)
    
    assert result['major_cluster_id'] == 1
    assert result['major_cluster_label'] == 'A'
    assert result['tp'] == 3
    assert result['fp'] == 0
    assert result['fn'] == 0
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0 