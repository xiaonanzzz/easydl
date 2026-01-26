import numpy as np


def generate_2d_gaussian_points(
    N: int, mean: np.ndarray, cov: np.ndarray, random_seed: int = None
) -> np.ndarray:
    """
    Generate N random points from a 2D Gaussian distribution.

    Args:
        N: Number of points to generate.
        mean: Mean vector of the 2D Gaussian distribution. Should be a 1D array of shape (2,)
              or a list/tuple of 2 elements [mean_x, mean_y].
        cov: Covariance matrix of the 2D Gaussian distribution. Should be a 2x2 array.
             Can also be specified as:
             - A 2x2 matrix: [[var_x, cov_xy], [cov_xy, var_y]]
             - A list/tuple of 2 elements [var_x, var_y] for diagonal covariance (no correlation)
             - A single float for isotropic covariance (var_x = var_y, no correlation)
        random_seed: Optional random seed for reproducibility.

    Returns:
        A numpy array of shape (N, 2) containing the generated points.
        Each row represents a point [x, y].

    Examples:
        >>> # Generate 100 points from a Gaussian with mean [0, 0] and identity covariance
        >>> points = generate_2d_gaussian_points(100, [0, 0], [[1, 0], [0, 1]])

        >>> # Generate points with diagonal covariance (no correlation)
        >>> points = generate_2d_gaussian_points(100, [1, 2], [0.5, 1.0])

        >>> # Generate points with isotropic covariance
        >>> points = generate_2d_gaussian_points(100, [0, 0], 1.0)
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert mean to numpy array and ensure it's 1D with 2 elements
    mean = np.array(mean, dtype=np.float64)
    if mean.ndim == 0:
        raise ValueError("mean must be a 1D array or list/tuple with 2 elements")
    if mean.shape != (2,):
        mean = mean.flatten()
        if mean.shape[0] != 2:
            raise ValueError(f"mean must have 2 elements, got {mean.shape[0]}")

    # Convert cov to numpy array and handle different input formats
    cov = np.array(cov, dtype=np.float64)

    if cov.ndim == 0:
        # Single scalar: isotropic covariance (diagonal matrix with same variance)
        cov = np.eye(2) * float(cov)
    elif cov.ndim == 1:
        # 1D array: diagonal covariance (no correlation)
        if cov.shape[0] != 2:
            raise ValueError(f"cov must have 2 elements when 1D, got {cov.shape[0]}")
        cov = np.diag(cov)
    elif cov.ndim == 2:
        # 2D array: full covariance matrix
        if cov.shape != (2, 2):
            raise ValueError(f"cov must be a 2x2 matrix, got {cov.shape}")
    else:
        raise ValueError(
            f"cov must be a scalar, 1D array, or 2D array, got {cov.ndim}D array"
        )

    # Ensure covariance matrix is symmetric and positive semi-definite
    if not np.allclose(cov, cov.T):
        # Make it symmetric by taking the average
        cov = (cov + cov.T) / 2

    # Generate random points using multivariate normal distribution
    points = np.random.multivariate_normal(mean, cov, size=N)

    return points
