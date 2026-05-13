# CAHP: Complementary Attention Head Pruning

Welcome to the official implementation of **CAHP (Complementary Attention Head Pruning)**.

This repository accompanies the research article **"Complementary Attention Head Pruning for Efficient Transformers"**.

### 🌟 Project Overview
Transformer-based models are powerful but computationally expensive, making them difficult to deploy on edge devices or in real-time applications. **CAHP** is an automated, post-hoc framework designed to bridge the gap between theoretical prunability and practical deployment.

Instead of looking at attention heads in isolation or relying on unstable gradient rankings, CAHP treats the entire model as a **unified graph space**. By measuring the "complementarity" between heads using information-theoretic distances, it identifies groups of heads that perform similar tasks and ensures the model retains a diverse set of functional "representatives." 

**Key Features:**
* **Fully Automated:** No need to manually set a pruning ratio or sparsity target. CAHP uses a "knee-finding" algorithm (Kneedle) on the Mean Simplified Silhouette (MSS) curve to find the optimal number of heads.
* **Post-Hoc:** Works on pre-trained models without requiring expensive retraining from scratch.
* **Structurally Aware:** Preserves the "functional core" of the model, typically found in the intermediate layers, avoiding the "proximity bias" seen in traditional methods.

### 🤝 Shoutout to ACSP
CAHP builds upon the foundational principles of **Automatic Complementary Separation Pruning (ACSP)**. While CAHP is specifically evolved for the unique structural dependencies of self-attention in Transformers, the original ACSP logic pioneered the automated, graph-based pruning approach.

To gain a broader understanding of this methodology and to see how this logic is applied to **Convolutional (CNN)** and **Linear** settings, we highly recommend checking out the original work:
> [**Automatic Complementary Separation Pruning Toward Lightweight CNNs**](https://arxiv.org/abs/2505.13225) by David Levin and Gonen Singer.

---

## 🚀 Installation

It is highly recommended to use a new virtual environment to avoid dependency conflicts. You can set this up using either **venv** or **Conda**.

### Option 1: Using venv
1.  **Create and activate the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Using Conda
1.  **Create and activate the environment:**
    ```bash
    conda create --name cahp python=3.11 -y
    conda activate cahp
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

> **Note:** The `requirements.txt` file contains the necessary packages without hardcoded version numbers to ensure maximum compatibility with your specific hardware and environment.

---

## 🛠 How to Run CAHP

The algorithm is executed via the `pruning_algorithm.py` script. This orchestrates the three stages: signature extraction, graph-based selection, and lightweight fine-tuning.

### Basic Command
```bash
python pruning_algorithm.py --model_path /path/to/your/hf_model --dataset_path /path/to/dataset
```

### Command-Line Arguments
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model_path` | (Required) | Path to the HuggingFace model directory (containing `config.json`). |
| `--dataset_path` | (Required) | Path to the directory containing `train` and `test` splits. |
| `--pool_size` | `32` | Spatial resolution (B) for attention map pooling. |
| `--poly_deg` | `6` | Polynomial degree for the MSS knee-finding algorithm. |
| `--retrain_epochs`| `3` | Number of fine-tuning epochs after pruning. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--verbose` | `True` | Enables detailed logging and visualization of the pruning curves. |

---

## 📂 Dataset & Model Preparation

To reproduce the specific datasets and models used in our research, you can utilize the scripts provided in the `use_case/` directory: `prepare_sst5.py` and `prepare_mnli.py`.

Running these files will generate the following assets:
* **Datasets:** Preprocesses and saves the data in two formats:
    * **HuggingFace format:** Required for compatibility with the CAHP framework.
    * **TSV format:** Required for compatibility with certain benchmark algorithms.
* **Base Models:** Downloads and saves "fresh" (pre-trained) instances of both **BERT-base-cased** and **BERT-large-cased**.
* **Trained Variants:** Executes the fine-tuning process for both architectures on the respective task, saving the trained models used as the starting point for pruning.

By executing these scripts, you ensure that your local environment precisely matches the experimental setup described in the article.

---

## 📊 Benchmarking & Reproduction

As detailed in our article, we benchmarked CAHP against three state-of-the-art approaches. Their original implementations can be found here:
* [**DSP**](https://github.com/rycolab/differentiable-subset-pruning)
* [**PASS**](https://github.com/DujianDing/PASS)
* [**AttAttr**](https://github.com/YRdddream/attattr)

### Using Our Modifications
We have provided a `benchmark/` folder containing modified files to support **SST-5** and improved **MNLI** handling. To reproduce our results, first install the original repositories as per their authors' instructions, then "plug in" our files as follows:

#### 1. DSP
* **Files:** `run_dsp_sst5.py`, `run_dsp_mnli.py`
* **Placement:** `/transformers/examples/pruning/`
* **Usage:** Run these files instead of the original `run_dsp.py`.
* **CLI Parameter Note:** When executing, ensure the `cooldown_steps` parameter is correctly specified based on the dataset and DSP variant:
    * **SST-5:** Use **90** for pipelined DSP and **270** for joint DSP.
    * **MNLI:** Use **4090** for pipelined DSP and **12272** for joint DSP.

#### 2. PASS
* **Execution Files:** Place `run_pass_sst5.py` and `run_pass_mnli.py` in `/transformers/examples/pruning/`.
* **Model Logic:** Place `modeling_gated_bert.py` inside `/transformers/src/transformers/`.
* **Trainer Logic:** Inside `/transformers/src/transformers/`, select the dataset-specific trainer (`trainer_pass_sst5.py` or `trainer_pass_mnli.py`), move it to the directory, and **rename it to `trainer_pass.py`** to override the original file.
* **Usage:** Execute these scripts following the original repository's instructions, using one of our provided run files in place of the original `run_pass.py`.

#### 3. AttAttr
* **Files:** `prune_attr_sst5.py`, `prune_attr_mnli.py`
* **Placement:** Inside the `examples` folder of the AttAttr repo.
* **Usage:** Run these as alternative entry points for each dataset.

---

## 📂 Repository Structure
```text
cahp/               # Core engine (data_types, graph clustering, pruner logic)
benchmark/          # Modified baseline files for DSP, PASS, and AttAttr
use_case/           # Scripts for dataset preparation and model initialization
pruning_algorithm.py # Main entry point for the CAHP algorithm
config.py           # Global configuration and logger setup
requirements.txt    # Project dependencies
```
---

## 📝 Citation
If you find this work useful for your research, please cite our paper:
*TBA*