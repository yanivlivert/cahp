from loguru import logger
import os
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, default_data_collator

def get_transformer_blocks(backbone: nn.Module) -> List[nn.Module]:
    """
    Retrieves the sequence of Transformer blocks from a HuggingFace backbone.

    Args:
        backbone (nn.Module): The model’s backbone submodule, e.g.
            - `model.distilbert` for DistilBERT
            - `model.bert` for BERT/RoBERTa
            - `model.gpt2` for GPT-2/GPT-Neo

    Returns:
        List[nn.Module]: A list of transformer block modules (e.g. `TransformerBlock`,
        `BertLayer`, or `GPT2Block`), in their original order.

    Raises:
        ValueError: If the backbone does not expose any of `.transformer.layer`,
                    `.encoder.layer`, or `.h` attributes.
    """
    if hasattr(backbone, "transformer"):
        return list(backbone.transformer.layer)
    elif hasattr(backbone, "encoder"):
        return list(backbone.encoder.layer)
    elif hasattr(backbone, "h"):
        return list(backbone.h)
    else:
        raise ValueError(f"Unknown transformer family: {type(backbone)}")
        
        
def get_num_heads(block: nn.Module) -> int:
    """
    Extracts the number of attention heads from a transformer block.

    Args:
        block (nn.Module): The transformer block or layer to be inspected for 
                           attention head information.

    Returns:
        int: The number of attention heads identified within the block.

    Raises:
        ValueError: If an attention module cannot be located or if the number of 
                    heads is not defined within the block's attributes or configuration.
    """
    attn = getattr(block, "attention", None) or getattr(block, "attn", None) or getattr(block, "self_attn", None)
    if attn is None:
        raise ValueError(f"No attention module found in block type {type(block)}")

    for name in ("num_attention_heads", "n_heads", "num_heads"):
        if hasattr(attn, name):
            return int(getattr(attn, name))

    if hasattr(attn, "self"):
        for name in ("num_attention_heads", "num_heads", "n_heads"):
            if hasattr(attn.self, name):
                return int(getattr(attn.self, name))

    if hasattr(block, "config") and hasattr(block.config, "num_attention_heads"):
        return int(block.config.num_attention_heads)

    raise ValueError(f"Could not find num_heads for block type {type(block)}")
        
  
def get_head_mappings(blocks: List[nn.Module]) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Analyzes the model architecture to map global head indices and record per-layer head counts.

    This utility performs a single pass over the transformer blocks to extract two key 
    pieces of information: a mapping from a flattened global index to (layer, local_head) 
    coordinates, and a list of the total number of heads present in each layer. Combining 
     these operations ensures architectural consistency and optimizes model inspection.

    Args:
        blocks (List[nn.Module]): A list of transformer blocks or layers to be analyzed.

    Returns:
        Tuple[List[Tuple[int, int]], List[int]]: A tuple containing:
            - global_to_local: List of (layer_idx, local_head_idx) for every global head.
            - layer_head_counts: List of integers representing the head count per layer.
    """
    global_to_local = []
    layer_head_counts = []
    
    for layer_idx, block in enumerate(blocks):
        num_heads = get_num_heads(block)
        layer_head_counts.append(num_heads)
        
        for head_idx in range(num_heads):
            global_to_local.append((layer_idx, head_idx))
            
    return global_to_local, layer_head_counts


def save_model(model: nn.Module, file_name: str = 'pruned_model.pth') -> None:
    """
    Saves the model using both Hugging Face's save_pretrained and standard PyTorch serialization.

    This function identifies the latest timestamped directory in the 'outputs' folder 
    and persists the model there. It prioritizes Hugging Face's native saving 
    mechanism for compatibility with the Transformers library, while also 
    providing a standard .pth file as a secondary backup.

    Args:
        model (nn.Module): The transformer or neural network model to persist.
        file_name (str, optional): Filename for the PyTorch serialized object. 
                                   Defaults to 'pruned_model.pth'.
    """
    output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))

    all_subfolders = [
        os.path.join(output_base, f) 
        for f in os.listdir(output_base) 
        if os.path.isdir(os.path.join(output_base, f))
    ]
    latest_subfolder = max(all_subfolders, key=os.path.getmtime)

    if hasattr(model, "save_pretrained"):
        hf_save_path = os.path.join(latest_subfolder, "hf_model")
        model.save_pretrained(hf_save_path)
        logger.info(f"Hugging Face weights and config saved to: {hf_save_path}")

    pth_save_path = os.path.join(latest_subfolder, file_name)
    torch.save(model.state_dict(), pth_save_path)
    logger.info(f"Full PyTorch model object saved to: {pth_save_path}")


def train_model(model: nn.Module, train_loader: DataLoader, num_epochs: int = 3, seed: int = 42, learning_rate: float = 2e-5, output_dir: str = "./tmp_head_prune_ft") -> None:
    """
    Fine-tunes the provided model using the Hugging Face Trainer API.

    Args:
        model (nn.Module): The transformer model or neural network to be fine-tuned.
        train_loader (DataLoader): A PyTorch DataLoader containing the training dataset.
        num_epochs (int, optional): Total number of training epochs. Defaults to 3.
        seed (int, optional): Random seed for both the trainer and data sampling to 
                              ensure reproducibility. Defaults to 42.
        learning_rate (float, optional): Initial learning rate for the AdamW optimizer. 
                                         Defaults to 2e-5.
        output_dir (str, optional): Local directory path for saving training logs and 
                                    checkpoints. Defaults to "./tmp_head_prune_ft".
    """
    logger.info(f"Starting fine-tuning for {num_epochs} epochs.")
    
    train_dataset = train_loader.dataset

    train_bs = getattr(train_loader, "batch_size", 8) or 8

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=int(0.1 * num_epochs * len(train_loader)),  # warmup_steps replaced warmup_ratio
        lr_scheduler_type="linear",
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    
    
def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Evaluates the model's performance on a given dataset using the Hugging Face accuracy metric.

    Args:
        model (nn.Module): The neural network to be evaluated.
        data_loader (DataLoader): A PyTorch DataLoader containing the evaluation dataset.

    Returns:
        float: The computed accuracy score, bounded between 0.0 and 1.0.
    """
    metric = evaluate.load("accuracy")
    
    device = next(model.parameters()).device
    
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating the accuracy"):
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)
            else:
                inputs, labels = batch
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                else:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                labels = labels.to(device)

            logits = getattr(outputs, "logits", outputs)
            preds = torch.argmax(logits, dim=-1)

            metric.add_batch(
                predictions=preds.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )

    return float(metric.compute()["accuracy"])
