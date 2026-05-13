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
# 1. MODEL ARGUMENTS
# ============================================================
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


# ============================================================
# 2. PRUNING ARGUMENTS
# ============================================================
@dataclass
class PruningArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    num_of_heads: int = field(
        metadata={"help": "number of heads to be kept."}
    )
    pruning_lr: Optional[float] = field(
        default=0.5, metadata={"help": "learning rate for head importance variables."}
    )
    annealing: Optional[bool] = field(
        default=False, metadata={"help": "if set, anneal the temperature of DSP."}
    )
    initial_temperature: Optional[float] = field(
        default=1000, metadata={"help": "intial temperature of annealing."}
    )
    final_temperature: Optional[float] = field(
        default=1e-8, metadata={"help": "final temperature of annealing."}
    )
    cooldown_steps: Optional[int] = field(
        default=25000, metadata={"help": "Number of training steps for the temperature to cooldown."}
    )
    joint_pruning: Optional[bool] = field(
        default=False, metadata={"help": "if set, train head importance variables and other parameters in the original model together."}
    )
    use_ste: Optional[bool] = field(
        default=False, metadata={"help": "if set, use straight-through estimator."}
    )
    intermediate_masks: Optional[bool] = field(
        default=False, metadata={"help": "if set, save the intermediate head masks during training."}
    )


# ============================================================
# 3. CUSTOM SST-5 DATASET WRAPPER
# ============================================================
class SST5Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_split):
        self.input_ids = hf_split["input_ids"]
        self.attention_mask = hf_split["attention_mask"]
        
        # token_type_ids exists ONLY for some tokenizers
        self.has_token_type = "token_type_ids" in hf_split.column_names
        if self.has_token_type:
            self.token_type_ids = hf_split["token_type_ids"]
        
        self.labels = hf_split["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }
        if self.has_token_type:
            item["token_type_ids"] = torch.tensor(self.token_type_ids[idx])
        return item


# ============================================================
# 4. MAIN
# ============================================================
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, PruningArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, pruning_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, pruning_args, data_args, training_args = parser.parse_args_into_dataclasses()
        

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # ============================================================
    # LOAD MODEL CONFIG + TOKENIZER (tokenizer is unused, but required)
    # ============================================================
    num_labels = 5  # SST-5 has 5 classes
    
    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": (preds == p.label_ids).mean()}

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    # ============================================================
    # LOAD YOUR EXISTING SST-5 DATASET FROM DISK
    # ============================================================
    from datasets import load_from_disk
    hf_dataset = load_from_disk(data_args.data_dir)   # same argument name as original code uses

    train_dataset = SST5Dataset(hf_dataset["train"]) if training_args.do_train else None
    eval_dataset  = SST5Dataset(hf_dataset["test"])  if training_args.do_eval else None

    # We evaluate on TEST set (ACSP consistency)
    metric_name = "eval_accuracy"
    
    logger.info(
        "{}: use_ste = {}, cooldown_steps = {}, initial_temperature = {}, " \
        "final_temperature = {}, pruning_lr = {}".format(
            "Joint pruning" if pruning_args.joint_pruning else "Pipelined Pruning",
            "True" if pruning_args.use_ste else "False",
            pruning_args.cooldown_steps if pruning_args.annealing and not pruning_args.use_ste else "N.A.", 
            pruning_args.initial_temperature if pruning_args.annealing and not pruning_args.use_ste else "N.A.", 
            pruning_args.final_temperature if not pruning_args.use_ste else "N.A.",
            pruning_args.pruning_lr,
    ))
    
    # Set seed
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    # ============================================================
    # LOAD GATED-BERT (DSP MODEL)
    # ============================================================
    model = GatedBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # ============================================================
    # OPTIMIZER LOGIC (same as original DSP)
    # ============================================================
    if pruning_args.joint_pruning:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if n != "w"],
             "lr": training_args.learning_rate},
            {"params": [p for n, p in model.named_parameters() if n == "w"],
             "lr": pruning_args.pruning_lr},
        ]
    else:
        for n, p in model.named_parameters():
            if n != "w":
                p.requires_grad = False

        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if n == "w"],
             "lr": pruning_args.pruning_lr},
        ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ============================================================
    # DSP TRAINER
    # ============================================================
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

    # ============================================================
    # TRAIN DSP
    # ============================================================
    trainer.train()
    trainer.save_model()

    # ============================================================
    # APPLY FINAL MASK + EVALUATE
    # ============================================================
    model.use_dsp = False
    head_mask = convert_gate_to_mask(model.get_w(), pruning_args.num_of_heads)
    torch.save(head_mask, os.path.join(training_args.output_dir, "mask" + str(pruning_args.num_of_heads) + ".pt"))
    print_2d_tensor(head_mask)

    model.apply_masks(head_mask)

    score = trainer.evaluate(eval_dataset=eval_dataset)[metric_name]
    sparsity = 100 - head_mask.sum() / head_mask.numel() * 100
    logger.info(
        "Masking finished. Accuracy: %f, Remaining heads %d (%.1f%%)",
        score,
        head_mask.sum(),
        100 - sparsity,
    )


# ============================================================
def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
