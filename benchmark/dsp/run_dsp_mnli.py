# coding=utf-8
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from transformers import AutoConfig, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    DSPTrainer,
    TrainingArguments,
    GatedBertForSequenceClassification,
    AdamW,
    set_seed
)

from pruning_utils import print_2d_tensor, convert_gate_to_mask

logger = logging.getLogger(__name__)

# ============================================================
# 1. ARGUMENT CLASSES (Remain the same as SST-5)
# ============================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)

@dataclass
class PruningArguments:
    num_of_heads: int = field(metadata={"help": "number of heads to be kept."})
    pruning_lr: Optional[float] = field(default=0.5)
    annealing: Optional[bool] = field(default=False)
    initial_temperature: Optional[float] = field(default=1000)
    final_temperature: Optional[float] = field(default=1e-8)
    cooldown_steps: Optional[int] = field(default=25000)
    joint_pruning: Optional[bool] = field(default=False)
    use_ste: Optional[bool] = field(default=False)
    intermediate_masks: Optional[bool] = field(default=False)

# ============================================================
# 2. CUSTOM MNLI DATASET WRAPPER
# ============================================================
class MNLIDataset(torch.utils.data.Dataset):
    """
    Wrapper for MNLI. Handles the sentence-pair format.
    Expects hf_split to contain: input_ids, attention_mask, token_type_ids, labels
    """
    def __init__(self, hf_split):
        self.input_ids = hf_split["input_ids"]
        self.attention_mask = hf_split["attention_mask"]
        self.token_type_ids = hf_split["token_type_ids"]
        self.labels = hf_split["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # MNLI always requires token_type_ids to distinguish premise/hypothesis
        item = {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "token_type_ids": torch.tensor(self.token_type_ids[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }
        return item

# ============================================================
# 3. MAIN
# ============================================================
def main():
    parser = HfArgumentParser((ModelArguments, PruningArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, pruning_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, pruning_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # MNLI Constants
    num_labels = 3  # MNLI has 3 classes: entailment, neutral, contradiction
    metric_name = "eval_accuracy"

    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    # ============================================================
    # LOAD MNLI DATASET (Based on your custom save format)
    # ============================================================
    from datasets import load_from_disk
    hf_dataset = load_from_disk(data_args.data_dir)

    # We use 'validation' (matched) for dev and 'test' (mismatched) for final reporting
    train_dataset = MNLIDataset(hf_dataset["train"]) if training_args.do_train else None
    eval_dataset  = MNLIDataset(hf_dataset["test"])  if training_args.do_eval else None

    # Set seed
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    # Load Gated-BERT
    model = GatedBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # Optimizer Logic
    if pruning_args.joint_pruning:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if n != "w"], "lr": training_args.learning_rate},
            {"params": [p for n, p in model.named_parameters() if n == "w"], "lr": pruning_args.pruning_lr},
        ]
    else:
        for n, p in model.named_parameters():
            if n != "w": p.requires_grad = False
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if n == "w"], "lr": pruning_args.pruning_lr},
        ]

    optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)

    # DSP Trainer
    training_args.max_steps = -1
    trainer = DSPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        num_of_heads=pruning_args.num_of_heads,
        final_temperature=pruning_args.final_temperature,
        cooldown_steps=pruning_args.cooldown_steps,
        annealing=pruning_args.annealing,
        initial_temperature=pruning_args.initial_temperature,
        optimizers=(optimizer, None),
        intermediate_masks=pruning_args.intermediate_masks,
        use_ste=pruning_args.use_ste,
    )

    trainer.train()
    
    # Final Masking + Eval on Mismatched set
    model.use_dsp = False
    head_mask = convert_gate_to_mask(model.get_w(), pruning_args.num_of_heads)
    torch.save(head_mask, os.path.join(training_args.output_dir, f"mask_{pruning_args.num_of_heads}.pt"))
    model.apply_masks(head_mask)

    score = trainer.evaluate(eval_dataset=eval_dataset)[metric_name]
    logger.info(f"Final Mismatched Accuracy: {score:.4f} with {head_mask.sum()} heads.")

if __name__ == "__main__":
    main()