from dataclasses import dataclass
import torch.nn as nn

@dataclass
class ModelEntity:
    """
    A Data Transfer Object (DTO) that encapsulates information related to the model.

    Attributes:
        model (nn.Module): The PyTorch model.
        model_path (str): The path to the complete model file.
    """

    model: nn.Module
    model_path: str
