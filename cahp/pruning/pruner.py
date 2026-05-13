from loguru import logger
import torch
from cahp.data_types.model_entity import ModelEntity
from cahp.data_types.data_entity import DataEntity
from cahp.utils.data_utils import force_set_seed
from cahp.utils.model_utils import get_transformer_blocks, train_model, evaluate_model, save_model
from cahp.pruning.pruners import AttentionHeadPruner

class Pruner:
    """
    High-level orchestrator for the model pruning pipeline.

    This class manages the end-to-end workflow of the pruning experiment, 
    including device management, baseline evaluation, global head pruning 
    via CAHP, post-pruning fine-tuning, and final performance verification.
    """
      
    def __init__(self, model: ModelEntity, train_data: DataEntity, test_data: DataEntity, retrain_epochs: int):
        """
        Initializes the Pruner with necessary entities and identifies the model backbone.

        Args:
            model (ModelEntity): The neural network model wrapper to be pruned.
            train_data (DataEntity): Entity containing the training DataLoader and metadata.
            test_data (DataEntity): Entity containing the test DataLoader for evaluation.
            retrain_epochs (int): Number of epochs to fine-tune the model after pruning.
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.retrain_epochs = retrain_epochs
        
        # Identify the transformer backbone (e.g., bert, roberta, vit)
        prefix = getattr(self.model.model, "base_model_prefix", None)
        self.backbone = getattr(self.model.model, prefix, None) if prefix else None
        
        if self.backbone is None:
            raise ValueError(f"Model type {type(self.model.model).__name__} is not a recognized transformer.")

        # Extract transformer blocks for pruning
        self.blocks = get_transformer_blocks(self.backbone)

        logger.debug(
            f"Pruner initialized for {type(self.model.model).__name__}: "
            f"{len(self.blocks)} blocks identified, retrain_epochs={self.retrain_epochs}."
        )      
            
   
    def prune(self, pool_size: int = 32, seed: int = 42, poly_deg: int = 6) -> None:
        """
        Executes the full pruning and recovery pipeline.

        This method performs the following sequence:
        1. Resets global seeds and sets the computation device.
        2. Evaluates pre-pruning baseline accuracy.
        3. Executes the CAHP global head selection and pruning.
        4. Evaluates post-pruning accuracy to measure immediate impact.
        5. Fine-tunes the model to recover performance.
        6. Records final post-recovery accuracy.

        Args:
            pool_size (int, optional): Resolution for attention map pooling. Defaults to 32.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            poly_deg (int, optional): Polynomial degree for knee detection. Defaults to 6.
        """
        # 1. Setup
        force_set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.model.to(device)

        logger.info(f"Starting pruning run. Model: {type(self.model.model).__name__} | Device: {device}")
        logger.info(f"Configuration: seed={seed}, poly_deg={poly_deg}, pool_size={pool_size}")

        # 2. Baseline Evaluation
        baseline_accuracy = evaluate_model(self.model.model, self.test_data.data_loader)
        logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")

        # 3. Execution of CAHP
        head_pruner = AttentionHeadPruner(
            model=self.model,
            train_data=self.train_data,
            transformer_blocks=self.blocks,
            pool_size=pool_size
        )

        # Update model and record pruning magnitude
        self.model, prune_pct = head_pruner.prune(seed=seed, poly_deg=poly_deg)
        
        # 4. Post-Pruning Evaluation
        post_prune_accuracy = evaluate_model(self.model.model, self.test_data.data_loader)
        logger.info(f"Post-prune accuracy (ratio {prune_pct:.2%}): {post_prune_accuracy:.4f}")

        # 5. Recovery (Fine-tuning)
        logger.info(f"Recovering performance for {self.retrain_epochs} epochs.")
        train_model(self.model.model, self.train_data.data_loader, num_epochs=self.retrain_epochs, seed=seed)

        # 6. Final Evaluation
        final_accuracy = evaluate_model(self.model.model, self.test_data.data_loader)
        logger.info(f"Post-recovery accuracy: {final_accuracy:.4f}")

        save_model(self.model.model, "final_pruned_model.pth")
        
        logger.success("Pruning and recovery pipeline completed successfully.")
