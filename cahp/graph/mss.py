from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calc_mss_value(space: np.ndarray, clustering: dict) -> Optional[float]:
    """
    Computes the Mean Simplified Silhouette (MSS) value to evaluate clustering quality.

    This function calculates the MSS value, which assesses clustering quality based on intra-cluster
    and nearest-cluster distances for each data point.

    Args:
        space (np.ndarray): A 2D array where each row represents a data point in the feature space.
        clustering (dict): Clustering information containing the following keys:
            - 'labels' (np.ndarray): Array of cluster labels for each data point in `space`.
            - 'medoids_loc' (np.ndarray): A 2D array where each row is the centroid of a cluster in the feature space.

    Returns:
        Optional[float]: The mean silhouette score, with higher scores indicating better clustering quality.
            Returns 1 if all intra-cluster distances are zero, indicating perfect clustering.
    """
    cluster_labels = clustering['labels']
    cluster_centers = clustering['medoids_loc']

    # Calculate intra-cluster distances for each data point
    intra_cluster_distances = euclidean_distances(
        space, cluster_centers[cluster_labels]
    ).diagonal()

    # Initialize nearest-cluster distances array
    nearest_cluster_distances = np.zeros_like(intra_cluster_distances)

    # Calculate the nearest-cluster distances
    for cluster_index, _ in enumerate(cluster_centers):
        cluster_members = space[cluster_labels == cluster_index]
        if cluster_members.size == 0:
            continue

        # Exclude the current cluster center
        non_current_centers = np.delete(cluster_centers, cluster_index, axis=0)
        distances_to_non_current_centers = euclidean_distances(
            cluster_members, non_current_centers
        )
        nearest_cluster_distances[cluster_labels == cluster_index] = (
            distances_to_non_current_centers.mean(axis=1)
        )

    # Check for non-zero intra-cluster distances
    valid_distances_mask = intra_cluster_distances != 0
    if not np.any(valid_distances_mask):
        return 1

    # Calculate silhouette scores for each valid point
    a = intra_cluster_distances[valid_distances_mask]
    b = nearest_cluster_distances[valid_distances_mask]
    silhouette_scores = (b - a) / np.maximum(a, b)

    # Return the mean silhouette score
    return np.mean(silhouette_scores)
