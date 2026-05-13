import os
import numpy as np
import torch
import random
from typing import Tuple
from cahp.data_types.data_entity import DataEntity
from cahp.data_types.model_entity import ModelEntity
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, set_seed
 
    
def load_model(model_path: str) -> ModelEntity:
    """
    Loads a HuggingFace model for sequence classification.

    Args:
        model_path (str): Path to a HuggingFace model folder.

    Returns:
        ModelEntity: A DTO wrapping the model and its path.
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            attn_implementation="eager"
        )
    except Exception as e:
        raise ValueError(f"Failed to load HuggingFace model: {e}")

    return ModelEntity(model=model, model_path=model_path)

    
def load_dataset(dataset_path: str, batch_size: int = 32, num_workers: int = 0) -> Tuple[DataEntity, DataEntity]:
    """
    Loads a HuggingFace dataset (saved with `dataset.save_to_disk`) and wraps
    the train/test splits into DataEntity objects.

    Args:
        dataset_path (str): Path to the HuggingFace dataset folder on disk.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 0.

    Returns:
        Tuple[DataEntity, DataEntity]: DataEntity instances for training and test splits.
    """
    dataset_dict = load_from_disk(dataset_path)

    if "train" not in dataset_dict or "test" not in dataset_dict:
        raise ValueError(f"Expected 'train' and 'test' splits in dataset at {dataset_path}")

    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_labels = np.array(train_dataset["labels"])
    test_labels = np.array(test_dataset["labels"])

    train_entity = DataEntity(
        data_loader=train_loader,
        num_classes=len(np.unique(train_labels)),
        data_path=dataset_path,
        total_samples=len(train_dataset),
        labels=train_labels,
        unique_label_indices=np.unique(train_labels),
    )

    test_entity = DataEntity(
        data_loader=test_loader,
        num_classes=len(np.unique(test_labels)),
        data_path=dataset_path,
        total_samples=len(test_dataset),
        labels=test_labels,
        unique_label_indices=np.unique(test_labels),
    )

    return train_entity, test_entity
    
    
def force_set_seed(seed: int = 42):
    """
    Sets global random seeds and configures backends to ensure experiment reproducibility.

    Args:
        seed (int, optional): The integer value used to seed all random number 
                              generators. Defaults to 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
