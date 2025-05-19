import pytest
import pandas as pd

@pytest.fixture
def sample_cluster_data():
    """Fixture providing sample cluster data for testing."""
    return {
        'cluster_id': [1, 1, 1, 1, 2, 2, 3, 3],
        'label': ['A', 'A', 'A', 'B', 'A', 'C', 'B', 'B']
    }

@pytest.fixture
def sample_cluster_df(sample_cluster_data):
    """Fixture providing a DataFrame with sample cluster data."""
    return pd.DataFrame(sample_cluster_data) 