import os
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)

# ==========================================
# 1. GLOBAL CONFIGURATION & PATHS
# ==========================================
SAVE_ROOT = "./mnli_reproducibility"
DATASET_HF_PATH = os.path.join(SAVE_ROOT, "dataset_hf_tokenized")
TSV_PATH = os.path.join(SAVE_ROOT, "dataset_tsvs")
FRESH_CKPT_DIR = os.path.join(SAVE_ROOT, "fresh_checkpoints")
TRAINED_CKPT_DIR = os.path.join(SAVE_ROOT, "trained_checkpoints")

MODELS_CONFIG = {
    "bert-base-cased": {"batch_size": 32, "lr": 2e-5},
    "bert-large-cased": {"batch_size": 8, "lr": 2e-5}
}

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(TSV_PATH, exist_ok=True)
os.makedirs(FRESH_CKPT_DIR, exist_ok=True)
os.makedirs(TRAINED_CKPT_DIR, exist_ok=True)

# ==========================================
# 2. DATASET PREPARATION & SAVING
# ==========================================
def prepare_and_save_dataset():
    print("\n" + "="*40)
    print("PHASE 1: DATASET PREPARATION")
    print("="*40)
    
    print("Loading raw MNLI dataset...")
    raw_datasets = load_dataset("glue", "mnli")

    dataset_dict = DatasetDict({
        "train": raw_datasets["train"],
        "validation": raw_datasets["validation_matched"],
        "test": raw_datasets["validation_mismatched"]
    })

    # Save as TSV
    print("Saving splits as TSV files...")
    for split_name, dataset in dataset_dict.items():
        df = dataset.to_pandas()
        df = df.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"})
        tsv_file = os.path.join(TSV_PATH, f"{split_name}.tsv")
        df.to_csv(tsv_file, sep='\t', index=True, index_label="index")
        print(f"  Saved {tsv_file}")

    # Tokenize and format for PyTorch
    print("Tokenizing dataset using bert-base-cased tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_fn(example):
        return tokenizer(
            example["premise"], 
            example["hypothesis"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )

    tokenized_dataset = dataset_dict.map(tokenize_fn, batched=True)
    
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels", "token_type_ids"]
    )

    print(f"Saving fully prepared HuggingFace dataset to {DATASET_HF_PATH}...")
    tokenized_dataset.save_to_disk(DATASET_HF_PATH)
    return tokenized_dataset

# ==========================================
# 3. FRESH CHECKPOINTS PREPARATION
# ==========================================
def save_fresh_checkpoints():
    print("\n" + "="*40)
    print("PHASE 2: SAVING FRESH CHECKPOINTS")
    print("="*40)

    for model_name in MODELS_CONFIG.keys():
        print(f"\nProcessing fresh checkpoint for: {model_name}")
        save_path = os.path.join(FRESH_CKPT_DIR, model_name.replace("-", "_"))
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        # 1. HuggingFace standard save
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        # 2. Legacy .bin save
        legacy_bin_path = os.path.join(save_path, "weights_only.bin")
        torch.save(model.state_dict(), legacy_bin_path)
        print(f"  Saved fresh HF and .bin formats to {save_path}")

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train_and_save_models(tokenized_dataset):
    print("\n" + "="*40)
    print("PHASE 3: TRAINING MODELS")
    print("="*40)

    for model_name, config in MODELS_CONFIG.items():
        print(f"\n--- Starting fine-tuning for {model_name} ---")
        save_path = os.path.join(TRAINED_CKPT_DIR, model_name.replace("-", "_"))
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

        training_args = TrainingArguments(
            output_dir=save_path,
            eval_strategy="no",
            save_strategy="no",
            learning_rate=config["lr"],
            per_device_train_batch_size=config["batch_size"],
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            logging_steps=500,
            report_to=[]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics
        )

        print(f"Training {model_name} (Batch Size: {config['batch_size']})...")
        trainer.train()

        print(f"Evaluating {model_name} on Mismatched Test Set...")
        eval_results = trainer.evaluate(tokenized_dataset["test"])
        print(f"--> Final Accuracy for {model_name}: {eval_results['eval_accuracy']:.4f}")

        # Save Trained Formats
        print(f"Saving trained models to {save_path}...")
        trainer.save_model(save_path) # HF Format
        torch.save(model.state_dict(), os.path.join(save_path, "weights_only.bin")) # Legacy .bin

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Prep data (or load if already prepped to save time)
    if os.path.exists(DATASET_HF_PATH):
        print(f"Found existing dataset at {DATASET_HF_PATH}. Loading directly...")
        from datasets import load_from_disk
        dataset = load_from_disk(DATASET_HF_PATH)
    else:
        dataset = prepare_and_save_dataset()

    # 2. Save fresh architectures
    save_fresh_checkpoints()

    # 3. Train both models
    train_and_save_models(dataset)
    
    print("\nAll MNLI reproducibility tasks completed successfully!")