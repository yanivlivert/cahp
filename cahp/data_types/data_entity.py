from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader


@dataclass
class DataEntity:
    """
    A Data Transfer Object (DTO) that encapsulates information related to the dataset.

    Attributes:
        data_loader (DataLoader): A PyTorch DataLoader that manages data loading and batching.
        num_classes (int): The number of unique classes within the dataset.
        data_path (str): The path to the dataset.
        total_samples (int): The total count of samples present in the dataset.
        labels (np.ndarray): An array containing the labels for each sample in the dataset.
        unique_label_indices (np.ndarray): An array of indices representing unique labels in the dataset.
    """

    data_loader: DataLoader
    num_classes: int
    data_path: str
    total_samples: int
    labels: np.ndarray
    unique_label_indices: np.ndarray
