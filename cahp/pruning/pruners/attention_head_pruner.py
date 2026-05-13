from loguru import logger
from typing import Dict, List, Tuple
import numpy as np
import torch
from cahp.data_types.decorators import timing_decorator
from cahp.data_types.model_entity import ModelEntity
from cahp.data_types.data_entity import DataEntity
from cahp.utils.model_utils import get_head_mappings
from cahp.graph.clustering import select_optimal_components
from cahp.utils.head_utils import compute_head_importance, collect_head_signatures, build_global_feature_matrix, reduce_feature_dimension, execute_head_pruning
from cahp.utils.data_utils import force_set_seed

class AttentionHeadPruner():
    """
    A class to handle the pruning of attention heads in a Transformer model.
    """
    def __init__(self, model: ModelEntity, train_data: DataEntity, transformer_blocks: List[torch.nn.Module], pool_size: int = 32, normalize_rows: bool = False):
        self.model = model
        self.train_data = train_data
        self.blocks = transformer_blocks
        self.pool_size = int(pool_size)
        self.num_classes = self.train_data.num_classes
        self._global_to_local, self._layer_heads = get_head_mappings(self.blocks)
        
    
    @property
    def layer_type(self) -> str:
        return "AttentionHead"
            
            
    @timing_decorator
    def create_head_feature_space(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Generates high-dimensional and reduced feature representations of the attention heads.

        This function orchestrates a three-stage pipeline:
        1. Extraction: Collects class-specific attention signatures using streaming statistics.
        2. Construction: Builds a global feature matrix by calculating pairwise 
           Jeffries-Matusita (JM) distances between classes for each head.
        3. Projection: Reduces the feature matrix to a 2D space via t-SNE to facilitate
           subsequent clustering.

        Args:
            seed (int, optional): Random seed for the dimensionality reduction 
                                  algorithm to ensure reproducible projections. 
                                  Defaults to 42.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing:
                - 'matrix': The high-dimensional feature matrix X of shape [N, D * P].
                - 'reduced_matrix': The 2D projected matrix Y of shape [N, 2] used for clustering.
        """
        logger.debug("Collecting per-head signatures from attentions")
        stats = collect_head_signatures(self.model.model, self.train_data.data_loader, self._layer_heads, self.pool_size, self.num_classes)
        
        logger.debug("Building global JM feature matrix")
        X = build_global_feature_matrix(
            stats=stats,
            num_heads=len(self._global_to_local),
            num_classes=self.num_classes,
            pool_size=self.pool_size
        )
        Y = reduce_feature_dimension(X, seed=seed) if X.shape[0] >= 2 else np.zeros((X.shape[0], 2), dtype=np.float32)
        
        return {"matrix": X, "reduced_matrix": Y}


    def prune(self, seed: int = 42, poly_deg: int = 6) -> Tuple[ModelEntity, float]:
        """
        Executes the global attention-head pruning pipeline (CAHP).
    
        The pruning process follows a five-step pipeline:
        1. Construction of a global feature space based on JM separability.
        2. Calculation of head importance weights.
        3. Selection of optimal representative heads using k-medoids and MSS.
        4. Enforcement of architectural stability constraints (minimum one head per layer).
        5. Physical removal of redundant heads from the model.
    
        Args:
            seed (int, optional): Random seed for clustering and projection. Defaults to 42.
            poly_deg (int, optional): Polynomial degree for MSS knee detection. Defaults to 6.
    
        Returns:
            Tuple[ModelEntity, float]: A tuple containing the pruned model and the 
                                       final pruning ratio (1 - kept_ratio).
        """
        # --- Step 1: Feature Space Construction ---
        logger.debug("Step 1: Building feature graph space.")
        graph_space = self.create_head_feature_space(seed=seed)
    
        num_heads_in_matrix = int(graph_space["matrix"].shape[0])
        total_mapped_heads = len(self._global_to_local)
    
        if num_heads_in_matrix != total_mapped_heads:
            logger.warning(f"Matrix mismatch: {num_heads_in_matrix} rows vs {total_mapped_heads} mapped heads.")
    
        if num_heads_in_matrix < 2:
            logger.warning("Insufficient heads for clustering; aborting pruning.")
            return self.model, 0.0
      
            
        # --- Step 2: Importance Weighting ---
        logger.debug("Step 2: Calculating salience weights.")
        force_set_seed(seed)
        weights = compute_head_importance(self.model.model, self.train_data.data_loader, self._layer_heads)
        
        if weights.shape[0] != num_heads_in_matrix:
            logger.warning("Weight vector size mismatch; falling back to uniform weights.")
            weights = np.ones((num_heads_in_matrix,), dtype=np.float32)
    
    
        # --- Step 3: Optimal Selection ---
        logger.debug("Step 3: Selecting optimal components via MSS.")
        kept_global_indices = select_optimal_components(
            graph_space=graph_space,
            weights=weights,
            num_components=num_heads_in_matrix,
            seed=seed,
            poly_deg=poly_deg
        )
               
        # --- Step 4: Index Mapping & Stability Constraints ---
        logger.debug("Step 4: Mapping indices and enforcing architectural constraints.")
        kept_by_layer: Dict[int, List[int]] = {}
        for gi in kept_global_indices:
            layer_idx, head_idx = self._global_to_local[gi]
            kept_by_layer.setdefault(layer_idx, []).append(int(head_idx))
    
        # Stability Guardrail: Ensure no layer is completely empty
        num_layers = len(self.blocks)
        empty_layers = [i for i in range(num_layers) if len(kept_by_layer.get(i, [])) == 0]
        
        if empty_layers:
            logger.warning(f"Empty layers detected: {empty_layers}. Enforcing min-keep constraint.")
            for layer_idx in empty_layers:
                # Identify global indices belonging to this specific layer
                layer_gi = [i for i, (l, _) in enumerate(self._global_to_local) if l == layer_idx]
                
                # Rescue the head with the highest salience weight in this layer
                best_gi = layer_gi[int(np.argmax([weights[i] for i in layer_gi]))]
                _, head_idx = self._global_to_local[best_gi]
                
                kept_by_layer.setdefault(layer_idx, []).append(int(head_idx))
                kept_by_layer[layer_idx].sort()
    
                
        # --- Step 5: Physical Execution ---
        logger.debug("Step 5: Executing physical head removal")
        execute_head_pruning(
            model=self.model.model,
            blocks=self.blocks,
            layer_heads=self._layer_heads,
            kept_by_layer=kept_by_layer
        )
        
        # Finalize results
        total_kept = sum(len(h) for h in kept_by_layer.values())
        kept_ratio = total_kept / total_mapped_heads
        pruning_ratio = 1.0 - kept_ratio
    
        logger.info(f"CAHP Completed: Kept {total_kept}/{total_mapped_heads} heads ({kept_ratio*100:.2f}%).")
        return self.model, pruning_ratio
