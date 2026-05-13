# coding=utf-8
import time
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
from datasets import load_from_disk

from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    PASSTrainer,
    TrainingArguments,
    set_seed,
    GatedBertForSequenceClassification,
    AdamW,
)
from pruning_utils import print_2d_tensor, convert_gate_to_mask

logger = logging.getLogger(__name__)

# ============================================================
# 1. CUSTOM MNLI DATASET WRAPPER
# ============================================================
class MNLIDataset(torch.utils.data.Dataset):
    """
    Wrapper for MNLI. Handles the sentence-pair format (requires token_type_ids).
    """
    def __init__(self, hf_split):
        self.input_ids = hf_split["input_ids"]
        self.attention_mask = hf_split["attention_mask"]
        # MNLI uses BERT sentence pairs, so token_type_ids are mandatory
        self.token_type_ids = hf_split["token_type_ids"]
        
        # Ensure labels are standard ints
        self.labels = [int(l) for l in hf_split["labels"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "token_type_ids": torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item

# ============================================================
# 2. ARGUMENTS (Mirrors run_pass_sst5.py)
# ============================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)

@dataclass
class PruningArguments:
    num_of_heads: Optional[int] = field(default=None, metadata={"help": "Heads to keep"})
    joint_pruning: Optional[bool] = field(default=False)
    pruning_lr: Optional[float] = field(default=0.5)

# ============================================================
# 3. MAIN
# ============================================================
def main():
    start_time = time.time()

    parser = HfArgumentParser((ModelArguments, PruningArguments, DataTrainingArguments, TrainingArguments))
    model_args, pruning_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Unified Seeding
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    # Task Configuration for MNLI
    # MNLI has 3 classes: Entailment, Neutral, Contradiction
    num_labels = 3
    metric = "eval_accuracy"

    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
    )

    # Load Dataset from disk
    logger.info(f"Loading dataset from {data_args.data_dir}")
    
    try:
        hf_dataset = load_from_disk(data_args.data_dir)
    except ValueError:
        raise
        
    # User specified: 'test' key contains the dev mismatched set
    train_dataset = MNLIDataset(hf_dataset["train"]) if training_args.do_train else None
    eval_dataset  = MNLIDataset(hf_dataset["test"])  if training_args.do_eval else None

    model = GatedBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Robust Sparsity Calculation
    total_heads = config.num_hidden_layers * config.num_attention_heads
    
    target_sparsity = 1 - pruning_args.num_of_heads / total_heads
    logger.info(f"Target Sparsity: {target_sparsity:.4f} ({pruning_args.num_of_heads}/{total_heads} heads)")
    model.apply_gates(target_sparsity)

    # Optimizer (PASS Logic)
    if pruning_args.joint_pruning:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if "log_a" not in n], "lr": training_args.learning_rate},
            {"params": [p for n, p in model.named_parameters() if "log_a" in n], "lr": pruning_args.pruning_lr},
        ]
    else:
        for n, p in model.named_parameters():
            if "log_a" not in n: p.requires_grad = False
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if "log_a" in n], "lr": pruning_args.pruning_lr},
        ]

    optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)

    trainer = PASSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        optimizers=(optimizer, None)
    )

    if training_args.do_train:
        trainer.train()

    # Final Masking (Multi-GPU Unwrap)
    m = model.module if hasattr(model, "module") else model
    gates = m.get_gate_values()
    head_mask = convert_gate_to_mask(gates, pruning_args.num_of_heads)
    
    m.remove_gates()
    m.apply_masks(head_mask)

    if training_args.do_eval:
        results = trainer.evaluate(eval_dataset=eval_dataset)
        final_score = results[metric]
        print("\n" + "="*50)
        print(f"FINAL EVALUATION ON TARGET ARCHITECTURE (MNLI)")
        print(f"Target Heads: {pruning_args.num_of_heads}")
        print(f"Actual Remaining Heads: {head_mask.sum().item()}")
        
        print("Final Head Mask (1=Kept, 0=Pruned):")
        print_2d_tensor(head_mask)
        
        print(head_mask)
        
        print(f"Final Accuracy: {final_score * 100:.2f}%")
        print("="*50 + "\n")
        
        total_time = time.time() - start_time
        print(f"[Timer] Total Execution Time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")

if __name__ == "__main__":
    main()