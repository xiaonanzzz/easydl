import numpy as np


def get_upper_triangle_values(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]
