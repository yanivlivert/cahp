from typing import Dict, List
import numpy as np
from kmedoids import fasterpam
from kneed import KneeLocator
from sklearn.metrics.pairwise import pairwise_distances
from cahp.graph.mss import calc_mss_value
from cahp.utils.plot_utils import plot_mss_curve

def kmedoids_fasterpam(data: np.ndarray, k: int, seed: int = 42) -> dict:
    """
    Performs KMedoids clustering using the FasterPAM algorithm on the provided data.

    Args:
        data (np.ndarray): The dataset to cluster, as a NumPy array.
        k (int): The number of clusters to use for KMedoids clustering.
        seed (int): Seed for the fasterpam() function.

    Returns:
        dict: A dictionary containing:
            - 'labels' (np.ndarray): Cluster labels for each data point.
            - 'medoids' (np.ndarray): Indices of the medoids.
            - 'medoids_loc' (np.ndarray): Locations of the medoids in the data.
    """
    distances = pairwise_distances(data)
    k_medoids = fasterpam(diss=distances, medoids=k, random_state=seed)
    medoids_loc = data[k_medoids.medoids]

    return {
        'labels': k_medoids.labels,
        'medoids': k_medoids.medoids,
        'medoids_loc': medoids_loc,
    }


def get_knee(x: list, y: list, poly_deg: int = 6) -> int:
    """
    Identifies the 'knee' point in a curve using the KneeLocator.

    Args:
        x (list): The x-values of the curve.
        y (list): The y-values of the curve.
        poly_deg (int, optional): The degree of the polynomial used for interpolation. Defaults to 6.

    Returns:
        int: The x-value of the knee point if found; otherwise, None.
    """
    kn = KneeLocator(
        x,
        y,
        curve='concave',
        direction='increasing',
        interp_method='polynomial',
        polynomial_degree=poly_deg,
    )

    return kn.knee


def select_optimal_components(graph_space: Dict[str, np.ndarray], weights: np.ndarray, num_components: int, weight_form: bool = True, seed: int = 42, poly_deg: int = 6) -> List[int]:
    """
    Selects optimal components based on the k-medoids clustering and MSS value.

    This function performs k-medoids clustering on the reduced matrix from the graph space
    for different values of k, calculates the MSS value for each clustering, and identifies
    the optimal number of components k* using the knee point of the MSS curve. Finally, it
    selects representative components to retain: if weight_form is True, it performs a weighted
    selection within each cluster using the provided weights; otherwise, it returns the
    standard medoids identified by the clustering algorithm.

    Args:
        graph_space (Dict[str, np.ndarray]): Graph space containing the reduced matrix as a NumPy array.
        weights (np.ndarray): Weights of the components.
        num_components (int): The number of components to consider for selection.
        weight_form (bool, optional): Whether to consider weights in the clustering process. Defaults to True.
        seed (int, optional): Seed value for the random number generator to ensure reproducibility during the k-medoids clustering. Defaults to 42.
        poly_deg (int, optional): The degree of the polynomial used for fitting the MSS curve in the knee-point detection algorithm. Defaults to 6.

    Returns:
        List[int]: A list of indices representing the optimal components.
    """
    mss_values = []
    k_values = range(2, num_components)

    for i, k in enumerate(k_values):
        k_medoids = kmedoids_fasterpam(graph_space['reduced_matrix'], k, seed)
        mss_value = calc_mss_value(clustering=k_medoids, space=graph_space['reduced_matrix'])
        mss_values.append(mss_value)

        if mss_value == 1.0 or mss_value == 1:
            extension_length = len(k_values) - i - 1
            mss_values.extend([1.0] * extension_length)
            break

    knee = get_knee(list(k_values), y=mss_values, poly_deg=poly_deg)
    
    plot_mss_curve(
        k_values=list(k_values),
        mss_values=mss_values,
        knee=knee,
        folder_name="MSS(k)",
        file_name="MSS vs k (with knee)"
    )
        
    optimal_kmedoids = kmedoids_fasterpam(graph_space['reduced_matrix'], int(knee), seed)
    
    if weight_form:
        return find_optimal_weighted_medoids(weights, optimal_kmedoids['labels'], optimal_kmedoids['medoids'])
    else:
        return optimal_kmedoids['medoids'].tolist()


def find_optimal_weighted_medoids(weights: np.ndarray, labels: np.ndarray, medoids: np.ndarray) -> List[int]:
    """
    Finds the optimal weighted medoids based on component weights within clusters.

    Args:
        weights (np.ndarray): Weights of all components.
        labels (np.ndarray): Cluster labels for each data point.
        medoids (np.ndarray): Indices of the initial medoids.

    Returns:
        list[int]: Indices of the data points with the highest weight in each cluster.
    """
    highest_weight_indices = np.empty_like(medoids)

    for i, medoid in enumerate(medoids):
        cluster_indices = np.where(labels == labels[medoid])[0]
        cluster_weights = weights[cluster_indices]
        max_weight_index = cluster_indices[np.argmax(cluster_weights)]
        highest_weight_indices[i] = max_weight_index

    return highest_weight_indices.tolist()
