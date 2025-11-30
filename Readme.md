## Residualized Similarity Prediction using Attention

This repository contains the code used in:

- **Paper**: *Residualized Similarity for Faithfully Explainable Authorship Verification*  
  ([Zeng et al., Findings 2025](https://aclanthology.org/2025.findings-emnlp.856/))

The core idea is to combine an interpretable similarity model (Gram2Vec-style features) with a neural encoder (e.g., LUAR) by learning a **residual** that corrects the interpretable similarity score.

### Setup

You can use either **Pixi (recommended)** or **conda**.

#### Option 1: Pixi (fast, reproducible)

From the repo root:

- Install Pixi (once per machine)
  - `curl -fsSL https://pixi.sh/install.sh | bash`
- Create the environment and install dependencies
  - `pixi install`

Pixi will:

- Install Python 3.10 and the packages from `requirements.txt` (via `pixi.toml`)
- Install `gram2vec` from GitHub via pip

#### Option 2: Conda

- `conda create -n rsp python=3.10`
- `conda activate rsp`
- `git clone https://github.com/eric-sclafani/gram2vec`
- `pip install gram2vec/`
- `pip install -r requirements.txt`

### Data

- With respect to the Amazon and Fanfiction datasets, we've included post-processing scripts of the data after they have been downloaded according to the LUAR paper 
- We share the raw and processed files of the Reddit dataset as described in the Style Embedding paper.
- We share the post processed data for the Amazon and Fanfiction datasets.

All three datasets are expected in the `data/<dataset>` folders as:

- `train.csv`, `dev.csv`, `test.csv`

### One-command LUAR-based Residualized Similarity (Reddit, Amazon, Fanfiction)

To train the **Residualized Similarity (RS)** model with **LUAR** as the neural base encoder on all three English datasets (Reddit, Amazon, Fanfiction), run **from the repo root**:

- With Pixi:
  - `pixi run train-rs-luar-all`
- With an activated conda environment:
  - `bash scripts/train_rs_luar_all.sh`

This script simply calls `src/train_attention_residual.py` with:

- `-m luar`
- `-d reddit | amazon | fanfiction`

and logs each run to `training_logs/` (created automatically).

### Training individual systems (advanced)

If you want to run individual training jobs or use other encoders, you can call the training scripts directly from the repo root:

- **Residualized Similarity with attention (RS + neural + interpretable)**
  - `python src/train_attention_residual.py -m <model_type> -d <dataset>`
  - **Examples**
    - `python src/train_attention_residual.py -m luar -d reddit`
    - `python src/train_attention_residual.py -m luar -d amazon`
    - `python src/train_attention_residual.py -m luar -d fanfiction`
- **Neural residual-only baseline (no interpretable features)**
  - `python src/train_residual.py -m <model_type> -d <dataset>`
  - **Example**
    - `python src/train_residual.py -m roberta -d reddit`

Supported `model_type` values (subject to GPU memory and availability) are those defined in `src/train_attention_residual.py` and `src/train_residual.py` (e.g., `roberta`, `roberta-large`, `luar`, `style`, etc.).


