from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm
from itertools import combinations
from sklearn.manifold import TSNE
from cahp.data_types.running_stats import RunningStats
from cahp.graph.distance import get_distance
from cahp.utils.model_utils import get_num_heads
    
    
def compute_head_importance(model: nn.Module, train_loader: DataLoader, layer_heads: List[int]) -> np.ndarray:
    """
    Computes importance scores for all attention heads based on loss sensitivity.
    Args:
        model (nn.Module): The transformer model to evaluate. Must support the `head_mask` argument in its forward pass.
        train_loader (DataLoader): The data loader used to provide batches for sensitivity calculation.
        layer_heads (List[int]): A list containing the number of attention heads for each layer.
    Returns:
        np.ndarray: A 1D NumPy array of shape (Total_Heads,) containing the normalized importance scores.
    """
    device = next(model.parameters()).device
    num_layers = len(layer_heads)
    unique_heads = set(layer_heads)
    if len(unique_heads) != 1:
        raise ValueError(f"Expected a uniform number of heads per layer, got {unique_heads}.")
    num_heads = unique_heads.pop()

    head_mask = torch.ones(num_layers, num_heads, device=device, requires_grad=True)
    head_importance = torch.zeros_like(head_mask)

    # Freeze model weights
    original_requires_grad = {}
    for param in model.parameters():
        original_requires_grad[param] = param.requires_grad
        param.requires_grad = False

    model.eval()
    model.zero_grad(set_to_none=True)

    # --- Register hooks that multiply the mask into each layer's attention output ---
    # BertAttention output is (context_layer, attention_probs) where context_layer
    # is [B, seq, hidden]. We intercept it and scale by the per-head mask.
    hooks = []

    # Collect the self-attention modules in layer order
    # For BertForSequenceClassification: model.bert.encoder.layer[i].attention.self
    try:
        encoder_layers = model.bert.encoder.layer
    except AttributeError:
        raise RuntimeError("Could not find model.bert.encoder.layer - adjust the path for your model architecture.")

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is a tuple: (context_layer, attention_probs) or just (context_layer,)
            # context_layer shape: [B, seq_len, num_heads * head_dim]
            # We need to reshape, scale, then reshape back.
            context = output[0]  # [B, seq, hidden]
            B, S, H = context.shape
            head_dim = H // num_heads

            # [B, seq, num_heads, head_dim]
            context = context.view(B, S, num_heads, head_dim)

            # head_mask[layer_idx]: [num_heads] -> [1, 1, num_heads, 1]
            mask = head_mask[layer_idx].view(1, 1, num_heads, 1)
            context = context * mask

            # Reshape back
            context = context.view(B, S, H)

            return (context,) + output[1:]
        return hook

    for i, enc_layer in enumerate(encoder_layers):
        h = enc_layer.attention.self.register_forward_hook(make_hook(i))
        hooks.append(h)

    target_batches = max(1, int(0.25 * len(train_loader)))
    n_batches = 0

    try:
        for batch in tqdm(train_loader, total=target_batches, desc="Calculating head weights"):
            if n_batches >= target_batches:
                break
            n_batches += 1

            if head_mask.grad is not None:
                head_mask.grad.zero_()

            with torch.enable_grad():
                if isinstance(batch, dict):
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                    labels = batch["labels"].to(device)
                    # No head_mask argument needed � hooks handle it
                    outputs = model(**inputs, labels=labels)
                else:
                    inputs, labels = batch
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        labels = labels.to(device)
                        outputs = model(**inputs, labels=labels)
                    else:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs, labels=labels)

                loss = getattr(outputs, "loss", None)
                if loss is None:
                    raise RuntimeError("Model forward did not return a loss.")

                loss.backward()

            head_importance += head_mask.grad.abs().detach()

    finally:
        # Remove hooks
        for h in hooks:
            h.remove()
        # Unfreeze weights
        for param in model.parameters():
            param.requires_grad = original_requires_grad.get(param, True)
        model.zero_grad(set_to_none=True)

    if n_batches == 0:
        return np.ones(num_layers * num_heads, dtype=np.float32)

    head_importance /= float(n_batches)
    weights = head_importance.reshape(-1).detach().cpu().numpy().astype(np.float32)
    w_min, w_max = float(weights.min()), float(weights.max())
    if w_max > w_min:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        weights = np.ones_like(weights, dtype=np.float32)
    return weights
    

def collect_head_signatures(model: nn.Module, train_loader: DataLoader, layer_heads: List[int], pool_size: int, num_classes: int) -> Dict[int, Dict[int, "RunningStats"]]:
    """
    Extracts and aggregates pooled attention signatures for all heads across the dataset.

    Args:
        model (nn.Module): The transformer model to probe. Must support 'output_attentions=True'.
        train_loader (DataLoader): DataLoader containing the training samples and labels.
        layer_heads (List[int]): A list of the number of attention heads per layer.
        pool_size (int): The spatial resolution (B) to which attention maps are pooled.
        num_classes (int): The total number of unique classes in the dataset.

    Returns:
        Dict[int, Dict[int, RunningStats]]: A nested dictionary mapping [global_head_id] 
                                            to [class_label] to its accumulated 
                                            RunningStats object.
    """
    device = next(model.parameters()).device
    model.eval()

    D = pool_size * pool_size
    total_heads = sum(layer_heads)

    stats = {
        head_id: {c: RunningStats(D) for c in range(num_classes)}
        for head_id in range(total_heads)
    }

    with torch.no_grad():
        total_batches = len(train_loader)

        for batch in tqdm(train_loader, total=total_batches, desc="Collecting per-head signatures"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            attn_mask = inputs.get("attention_mask", None)

            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # tuple of [B, H_layer, Seq, Seq]
            
            Bsize = labels.shape[0]
            
            batch_signatures = [] 

            for layer_idx, layer_attn in enumerate(attentions):
                n_heads = layer_heads[layer_idx]
                
                layer_pooled_list = []
                
                for b in range(Bsize):
                    L_real = int(attn_mask[b].sum().item()) if attn_mask is not None else layer_attn.size(-1)
                    
                    # Slice: [n_heads, L_real, L_real]
                    # We treat the heads as the "Batch" dimension for F.interpolate that expects [N, C, H, W].
                    heads_raw = layer_attn[b, :, :L_real, :L_real].unsqueeze(1)
                    
                    pooled = F.interpolate(
                        heads_raw, 
                        size=(pool_size, pool_size), 
                        mode="bilinear", 
                        align_corners=False
                    )
                    
                    # Flatten -> [n_heads, D]
                    layer_pooled_list.append(pooled.flatten(1)) 
                
                # Stack samples -> [B, n_heads, D]
                layer_batch_signatures = torch.stack(layer_pooled_list, dim=0)
                batch_signatures.append(layer_batch_signatures)

            # Concatenate all layers -> [B, Total_Heads, D]
            batch_signatures = torch.cat(batch_signatures, dim=1)

            batch_signatures_np = batch_signatures.cpu().numpy()
            labels_np = labels.cpu().numpy()

            unique_labels = np.unique(labels_np)
            
            for lbl in unique_labels:
                sample_idxs = np.where(labels_np == lbl)[0]
                
                subset = batch_signatures_np[sample_idxs]
                
                for global_id in range(total_heads):
                    stats[global_id][lbl].update_batch(subset[:, global_id, :])

    return stats
    

def build_global_feature_matrix(stats: Dict[int, Dict[int, "RunningStats"]], num_heads: int, num_classes: int, pool_size: int) -> np.ndarray:
    """
    Constructs a global feature matrix by computing pairwise separability vectors for all heads.

    For each attention head and every possible unique pair of classes, this function 
    calculates the Jeffries-Matusita (JM) distance vector using the aggregated 
    class-wise statistics.

    Args:
        stats (Dict[int, Dict[int, RunningStats]]): Nested dictionary containing 
                                                    RunningStats per head and per class.
        num_heads (int): Total number of attention heads across all layers.
        num_classes (int): Total number of unique classes in the dataset.
        pool_size (int): The spatial resolution ($B$) used during the pooling stage.

    Returns:
        np.ndarray: A feature matrix X of shape [N, D * P], where N is the 
                    number of heads, D is the pooled dimension (pool_size^2), 
                    and P is the number of unique class pairs.
    """
    D = pool_size * pool_size

    class_labels = list(range(num_classes))
    pairs = list(combinations(class_labels, 2))
    P = len(pairs)

    if P == 0:
        # Degenerate: 0 or 1 class -> no separability signal.
        return np.zeros((num_heads, D), dtype=np.float32)

    X = np.zeros((num_heads, D * P), dtype=np.float32)

    for gi in range(num_heads):
        per_class = stats.get(gi, None)
        col = 0
        for (c1, c2) in pairs:
            if (
                per_class is None
                or c1 not in per_class or c2 not in per_class
                or per_class[c1].n == 0 or per_class[c2].n == 0
            ):
                jm_vec = np.zeros((D,), dtype=np.float32)
            else:
                s1 = per_class[c1]
                s2 = per_class[c2]
                jm_vec = get_distance('jm', s1.mean, s1.var, s2.mean, s2.var)

            X[gi, col: col + D] = jm_vec
            col += D

    return X
    
    
def reduce_feature_dimension(data: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Reduces high-dimensional feature vectors to a 2D space using t-SNE.

    Args:
        data (np.ndarray): The high-dimensional feature matrix X of shape 
                           [N, D * P].
        seed (int, optional): Random seed for the t-SNE algorithm to ensure 
                              reproducible projections. Defaults to 42.

    Returns:
        np.ndarray: A reduced matrix of shape [N, 2] representing the 2D 
                    projection of the feature space.
    """
    logger.debug(f"seed={seed}")
    return TSNE(n_components=2, max_iter=1000, perplexity=10, random_state=seed).fit_transform(data)


def execute_head_pruning(model: nn.Module, blocks: List[nn.Module], layer_heads: List[int], kept_by_layer: Dict[int, List[int]]) -> None:
    """
    Physically removes attention heads from a BERT-style transformer by performing
    in-place weight surgery on each layer's Q, K, V, and output projection matrices.
    
    For each layer with heads to prune, the function slices the Q/K/V linear layers
    along the output dimension (rows) and the attention output dense layer along the
    input dimension (cols), retaining only the neurons corresponding to kept heads.
    Head bookkeeping (num_attention_heads, all_head_size, pruned_heads) is updated
    accordingly.
    
    Args:
        model (nn.Module): The top-level transformer model (used only for device resolution).
        blocks (List[nn.Module]): Ordered list of transformer encoder blocks (BertLayer instances).
        layer_heads (List[int]): Number of attention heads per layer before pruning.
        kept_by_layer (Dict[int, List[int]]): Mapping of layer index to the list of
                                              local head indices to retain.
    
    Raises:
        AttributeError: If a block does not expose the expected BertAttention structure
                        (block.attention.self / block.attention.output).
    """
    heads_to_prune: Dict[int, List[int]] = {}
    for layer_idx, n_heads in enumerate(layer_heads):
        kept = set(kept_by_layer.get(layer_idx, []))
        to_prune = sorted(set(range(n_heads)) - kept)
        if to_prune:
            heads_to_prune[layer_idx] = to_prune

    if not heads_to_prune:
        logger.debug("No heads to prune.")
        return

    pre_total = sum(get_num_heads(b) for b in blocks)
    logger.debug(f"Sanity check - total heads BEFORE pruning: {pre_total}")

    for layer_idx, prune_list in heads_to_prune.items():
        kept_list = sorted(set(kept_by_layer.get(layer_idx, [])))
        block = blocks[layer_idx]

        self_attn = block.attention.self
        attn_out  = block.attention.output

        n_heads   = self_attn.num_attention_heads
        head_size = self_attn.attention_head_size

        # Build the index of individual neurons to keep across Q, K, V
        # Each head occupies `head_size` consecutive neurons
        keep_idx = torch.cat([
            torch.arange(h * head_size, (h + 1) * head_size)
            for h in kept_list
        ]).to(next(model.parameters()).device)

        def prune_rows(linear: nn.Linear, idx: torch.Tensor) -> nn.Linear:
            """Keep only rows `idx` of weight; keep full bias subset."""
            new_linear = nn.Linear(linear.in_features, len(idx), bias=linear.bias is not None)
            new_linear.weight = nn.Parameter(linear.weight[idx].clone())
            if linear.bias is not None:
                new_linear.bias = nn.Parameter(linear.bias[idx].clone())
            return new_linear.to(linear.weight.device)

        def prune_cols(linear: nn.Linear, idx: torch.Tensor) -> nn.Linear:
            """Keep only columns `idx` of weight; bias unchanged."""
            new_linear = nn.Linear(len(idx), linear.out_features, bias=linear.bias is not None)
            new_linear.weight = nn.Parameter(linear.weight[:, idx].clone())
            if linear.bias is not None:
                new_linear.bias = nn.Parameter(linear.bias.clone())
            return new_linear.to(linear.weight.device)

        # Q, K, V: prune output neurons (rows) corresponding to dropped heads
        self_attn.query = prune_rows(self_attn.query, keep_idx)
        self_attn.key   = prune_rows(self_attn.key,   keep_idx)
        self_attn.value = prune_rows(self_attn.value, keep_idx)

        # Output dense: prune input neurons (cols) corresponding to dropped heads
        attn_out.dense = prune_cols(attn_out.dense, keep_idx)

        # Update bookkeeping
        new_n_heads = len(kept_list)
        self_attn.num_attention_heads = new_n_heads
        self_attn.all_head_size = new_n_heads * head_size
        if hasattr(self_attn, 'pruned_heads'):
            self_attn.pruned_heads = self_attn.pruned_heads.union(set(prune_list))

        logger.debug(f"Pruned layer {layer_idx}: removed heads {prune_list}, kept {kept_list}")

    post_total = sum(get_num_heads(b) for b in blocks)
    logger.debug(f"Sanity check - total heads AFTER pruning: {post_total}")