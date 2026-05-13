# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import time

import numpy as np
import torch
from datasets import load_from_disk

from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    PASSTrainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
    GatedBertForSequenceClassification,
    AdamW,
)
from pruning_utils import print_2d_tensor, convert_gate_to_mask

logger = logging.getLogger(__name__)

# ============================================================
# 1. CUSTOM SST-5 DATASET WRAPPER
# ============================================================
class SST5Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_split):
        self.input_ids = hf_split["input_ids"]
        self.attention_mask = hf_split["attention_mask"]
        self.has_token_type = "token_type_ids" in hf_split.column_names
        if self.has_token_type:
            self.token_type_ids = hf_split["token_type_ids"]
            
        #self.labels = hf_split["labels"]
        self.labels = [int(l) for l in hf_split["labels"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.has_token_type:
            item["token_type_ids"] = torch.tensor(self.token_type_ids[idx])
        return item

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

def main():
    start_time = time.time()
    parser = HfArgumentParser((ModelArguments, PruningArguments, DataTrainingArguments, TrainingArguments))
    model_args, pruning_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    os.environ["PYTHONHASHSEED"] = str(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Task Configuration for SST-5
    is_sst5 = "sst5" in data_args.task_name.lower()
    if is_sst5:
        num_labels = 5
        metric = "eval_accuracy"
        output_mode = "classification"
    else:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        metric = "eval_acc"

    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        if is_sst5:
            return {"accuracy": (preds == p.label_ids).mean()}
        return glue_compute_metrics(data_args.task_name, preds, p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
    )

    # Load Dataset from disk (DSP logic)
    logger.info(f"Loading dataset from {data_args.data_dir}")
    
    try:
        hf_dataset = load_from_disk(data_args.data_dir)
    except ValueError:
        raise
        
    train_dataset = SST5Dataset(hf_dataset["train"]) if training_args.do_train else None
    eval_dataset  = SST5Dataset(hf_dataset["test"])  if training_args.do_eval else None

    model = GatedBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Robust Sparsity Calculation
    total_heads = config.num_hidden_layers * config.num_attention_heads
    
    target_sparsity = 1 - pruning_args.num_of_heads / total_heads
    logger.info(f"Target Sparsity: {target_sparsity:.4f} ({pruning_args.num_of_heads}/{total_heads} heads)")
    model.apply_gates(target_sparsity)

    # Optimizer
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

    trainer.train()

    # Final Masking (Multi-GPU Unwrap)
    m = model.module if hasattr(model, "module") else model
    gates = m.get_gate_values()
    head_mask = convert_gate_to_mask(gates, pruning_args.num_of_heads)
    
    m.remove_gates()
    m.apply_masks(head_mask)

    results = trainer.evaluate(eval_dataset=eval_dataset)
    final_score = results[metric]
    print("\n" + "="*50)
    print(f"FINAL EVALUATION ON TARGET ARCHITECTURE")
    print(f"Target Heads: {pruning_args.num_of_heads}")
    print(f"Actual Remaining Heads: {head_mask.sum().item()}")
    print(f"Final Accuracy: {final_score:.4f}")
    print("Head Mask:")
    print(head_mask)
    print()
    print(f"Time it took: {time.time() - start_time}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()