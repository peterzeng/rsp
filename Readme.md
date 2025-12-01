## Residualized Similarity Prediction using Attention

This repository contains the code used in:

- **Paper**: *Residualized Similarity for Faithfully Explainable Authorship Verification*  
  ([Zeng et al., Findings 2025](https://aclanthology.org/2025.findings-emnlp.856.pdf))

The core idea is to combine an interpretable similarity model (Gram2Vec-style features) with a neural encoder (e.g., LUAR) by learning a **residual** that corrects the interpretable similarity score.

### What this model does

- **Task**: Authorship verification framed as a **similarity prediction** problem for pairs of documents.
- **Interpretable base model (Gram2Vec)**: Computes a cosine similarity score from linguistically interpretable, traceable features (POS n-grams, function words, dependency labels, morphology, etc.) using `gram2vec`.  
- **Neural residual model**: A neural encoder (e.g., LUAR) observes the same document pair and predicts a **residual similarity**: the error in the Gram2Vec similarity score.
- **Final similarity**:  
  - `final_score = gram2vec_score + predicted_residual`  
  - If the base model is already correct, the residual tends to be small, so the prediction is largely based on interpretable features.  
  - If the base model is wrong, the residual is larger in magnitude and corrects the score using information from the neural encoder.
- **Interpretability confidence (IC)**: During testing, we compute `ic = 1 - |predicted_residual|`.  
  - `ic` close to 1 ⇒ the residual correction is small and the prediction is **highly faithful to the interpretable system**.  
  - `ic` closer to 0 ⇒ the neural model made a large correction and the prediction is **less interpretable**, matching the notion of interpretability confidence in the paper ([Zeng et al., Findings 2025](https://aclanthology.org/2025.findings-emnlp.856.pdf)).

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

### How to use the trained model and outputs

- **Datasets and CSV format**
  - Place your data under `data/<dataset>` with files: `train.csv`, `dev.csv`, `test.csv`.
  - Each CSV is expected to contain at least:
    - `document1`, `document2`: the two texts being compared.
    - `documentID1`, `documentID2`: stable IDs used to cache Gram2Vec feature vectors.
    - `same_author_label`: binary label (`1` = same author, `0` = different author).
- **Running on your own dataset**
  - Create a new folder `data/<your_dataset>` with the CSVs above.
  - Run (from repo root), e.g.:  
    - `python src/train_attention_residual.py -m luar -d <your_dataset>`
  - Optional useful flags:
    - `-p <float>`: subsample a percentage of the data (e.g., `-p 0.25` for 25%).  
    - `-r <run_id>`: tag the run for easier experiment tracking (used in output paths).  
    - `-k <fold>`: run on a specific fold if you organize data as `<dataset>_kfold/fold_<k>`.  
    - `-n/--normalized`: use normalized Gram2Vec caches (on by default in this script).
- **Where results are saved**
  - For a run with `-m luar -d reddit -r 2025-12-01 -n` you will get an experiment folder:  
    - `experiments/luar_reddit_normalized_no_ln/2025-12-01/`
  - Inside:
    - `models/`: best model checkpoint (`model.pt` and `checkpoint.pt`).  
    - `results/`:  
      - `predictions.csv`: per-pair outputs with columns:
        - `gram2vec_score`, `predicted_residual`, `final_score`, and `ic` (interpretability confidence).  
      - `metrics.csv`: accuracy/precision/recall/F1 across thresholds for Gram2Vec vs. residualized similarity.  
      - `training_config.txt`: hyperparameters and training configuration.  
      - `attention_weights.txt`: epoch-wise average attention over `hidden1`, `hidden2`, `features1`, `features2`.  
    - `graphs/`:  
      - `auc_curve.png`: ROC curves for Gram2Vec vs. residualized similarity.  
      - `loss_curves.png`: training/validation losses over epochs.  
      - `predicted_labels.png`: histogram of predicted residual values.
- **Using the model for analysis**
  - For a given document pair in `predictions.csv`:
    - **Use `final_score`** to decide same/different author by applying a threshold (e.g., selected from `metrics.csv`).  
    - **Inspect `gram2vec_score` and feature categories** (via `gram2vec`) to see which linguistic patterns drive the interpretable similarity.  
    - **Check `predicted_residual` and `ic`** to understand how much the neural model corrected the interpretable system, as discussed in the paper ([Zeng et al., Findings 2025](https://aclanthology.org/2025.findings-emnlp.856.pdf)).

