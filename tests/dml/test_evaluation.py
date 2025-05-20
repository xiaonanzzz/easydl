import pytest
import pandas as pd
from easydl.dml.evaluation import evaluate_major_cluster_precision_recall
import numpy as np
from easydl.dml.evaluation import evaluate_embedding_top1_accuracy_ignore_self

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

def test_evaluate_embedding_top1_accuracy():
    # Create sample embeddings and labels
    # We'll create 4 points in 2D space, with 2 points close to each other for each class
    embeddings = [
        [0.0, 0.0],  # Class A
        [0.1, 0.1],  # Class A
        [1.0, 1.0],  # Class B
        [1.1, 1.1],  # Class B
    ]
    
    data = {
        'embedding': embeddings,
        'label': ['A', 'A', 'B', 'B']
    }
    df = pd.DataFrame(data)
    
    # Calculate accuracy
    result = evaluate_embedding_top1_accuracy_ignore_self(df)
    
    # Since points are well-separated, we expect perfect accuracy
    assert result['avg_top1_accuracy'] == 1.0

def test_evaluate_embedding_top1_accuracy_overlapping():
    # Create sample embeddings with some overlap between classes
    embeddings = [
        [0.0, 0.0],  # Class A
        [0.2, 0.2],  # Class A
        [0.2, 0.2],  # Class B (close to Class A)
        [1.0, 1.0],  # Class B
    ]
    
    data = {
        'embedding': embeddings,
        'label': ['A', 'A', 'B', 'B']
    }
    df = pd.DataFrame(data)
    
    # Calculate accuracy
    result = evaluate_embedding_top1_accuracy_ignore_self(df)
    
    # We expect less than perfect accuracy due to the overlapping point
    assert result['avg_top1_accuracy'] < 1.0
    assert result['avg_top1_accuracy'] >= 0.0

def test_evaluate_embedding_top1_accuracy_empty():
    # Test with empty DataFrame
    df = pd.DataFrame(columns=['embedding', 'label'])
    
    with pytest.raises(AssertionError):
        evaluate_embedding_top1_accuracy_ignore_self(df)

def test_evaluate_embedding_top1_accuracy_missing_columns():
    # Test with missing required columns
    df = pd.DataFrame({'wrong_column': [1, 2, 3]})
    
    with pytest.raises(AssertionError):
        evaluate_embedding_top1_accuracy_ignore_self(df) 