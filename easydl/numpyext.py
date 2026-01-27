"""
NumPy extension utilities for common matrix operations.

This module provides helper functions for NumPy operations that are
frequently used in deep metric learning, such as extracting values
from pairwise similarity/distance matrices.

Functions:
    get_upper_triangle_values: Extract upper triangle values from a square matrix,
        excluding the diagonal. Useful for pairwise comparisons where (i,j) and (j,i)
        are equivalent.

Example:
    >>> import numpy as np
    >>> from easydl.numpyext import get_upper_triangle_values
    >>> matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> values = get_upper_triangle_values(matrix)
    >>> print(values)  # [2, 3, 6]
"""
import numpy as np


def get_upper_triangle_values(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]
