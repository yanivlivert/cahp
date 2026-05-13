import numpy as np
    
def get_distance(metric: str, mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> np.ndarray:
    """
    Computes the distance between two distributions based on the specified metric 
    using pre-calculated mean and variance.

    This function serves as a dispatcher that selects the appropriate distance calculation
    method according to the specified metric. It operates element-wise across the feature 
    dimensions (D).

    Supported metrics:
        - 'jm': Jeffries-Matusita (JM) distance
        - 'bhattacharyya': Bhattacharyya distance

    Args:
        metric (str): The name of the metric to use for computing the distance.
                      Options are 'jm' and 'bhattacharyya'.
        mu1 (np.ndarray): Mean vector of the first distribution.
        var1 (np.ndarray): Variance vector of the first distribution.
        mu2 (np.ndarray): Mean vector of the second distribution.
        var2 (np.ndarray): Variance vector of the second distribution.

    Returns:
        np.ndarray: A vector of shape (D,) containing the computed distances for each feature.

    Raises:
        ValueError: If an unsupported metric name is provided.
    """
    if metric == 'jm':
        return jm_distance(mu1, var1, mu2, var2)
    elif metric == 'bhattacharyya':
        return bhattacharyya_distance(mu1, var1, mu2, var2)
    else:
        raise ValueError(f"Unsupported metric '{metric}' provided.")


def jm_distance(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> np.ndarray:
    """
    Computes the Jeffries-Matusita (JM) distance vector between two distributions.

    Args:
        mu1 (np.ndarray): Mean vector of the first distribution.
        var1 (np.ndarray): Variance vector of the first distribution.
        mu2 (np.ndarray): Mean vector of the second distribution.
        var2 (np.ndarray): Variance vector of the second distribution.

    Returns:
        np.ndarray: A vector of shape (D,) containing the JM distance for each feature dimension.
    """
    B = bhattacharyya_distance(mu1, var1, mu2, var2)
    JM = 2.0 * (1.0 - np.exp(-B))
    return JM.astype(np.float32)
    
    
def bhattacharyya_distance(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes the Bhattacharyya distance vector between two normal distributions.

    Args:
        mu1 (np.ndarray): Mean vector of the first distribution.
        var1 (np.ndarray): Variance vector of the first distribution.
        mu2 (np.ndarray): Mean vector of the second distribution.
        var2 (np.ndarray): Variance vector of the second distribution.
        eps (float, optional): Small constant to ensure numerical stability by 
                               preventing division by zero or log(0). Defaults to 1e-6.

    Returns:
        np.ndarray: A vector of shape (D,) containing the Bhattacharyya distance for each feature.
    """
    s1 = np.maximum(var1, eps)
    s2 = np.maximum(var2, eps)
    diff2 = (mu1 - mu2) ** 2
    
    # Bhattacharyya formula for univariate normals:
    # B = 1/4 * (diff^2 / (s1+s2)) + 1/2 * log((s1+s2) / (2*sqrt(s1*s2)))
    B = 0.25 * diff2 / (s1 + s2) + 0.5 * np.log((s1 + s2) / (2.0 * np.sqrt(s1 * s2)))
    
    return B
