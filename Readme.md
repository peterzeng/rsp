## Residualized Similarity Prediction using Attention

This repository contains the code used in:

- **Paper**: *Residualized Similarity for Faithfully Explainable Authorship Verification*  
  ([Zeng et al., Findings 2025](https://aclanthology.org/2025.findings-emnlp.856/))

The core idea is to combine an interpretable similarity model (Gram2Vec-style features) with a neural encoder (e.g., LUAR) by learning a **residual** that corrects the interpretable similarity score.

### Setup

- **Create environment**
  - `conda create -n rsp python=3.10`
  - `conda activate rsp`
- **Install interpretable features**
  - `git clone https://github.com/eric-sclafani/gram2vec`
  - `pip install gram2vec/`
- **Install remaining dependencies**
  - `pip install -r requirements.txt`

### Data

- **Reddit**
  - We share raw and processed Reddit data as described in the Style Embedding paper.
- **Amazon & Fanfiction**
  - We follow the data processing protocol from the LUAR paper and include post-processing scripts for Amazon and Fanfiction in `data/`.
  - We share post-processed splits for Amazon and Fanfiction in `data/amazon` and `data/fanfiction`.
- All three datasets are expected in the `data/<dataset>` folders as:
  - `train.csv`, `dev.csv`, `test.csv`

### One-command LUAR-based Residualized Similarity (Reddit, Amazon, Fanfiction)

To train the **Residualized Similarity (RS)** model with **LUAR** as the neural base encoder on all three English datasets (Reddit, Amazon, Fanfiction), run **from the repo root**:

- **Train RS + LUAR on all three datasets**
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

### Outputs and experiments

- RS runs write results and artifacts under:
  - `experiments/<model>_<dataset>_<normalized|not_normalized>_<ln_setting>/<run_id>/`
- Within each run directory you will find:
  - `models/` – best model checkpoint and weights
  - `results/` – CSVs with predictions, metrics, and configuration
  - `graphs/` – AUC curves, loss curves, and residual histograms

### Notes

- The LUAR-based RS script (`scripts/train_rs_luar_all.sh`) is the easiest way to reproduce the main LUAR + RS experiments on Reddit, Amazon, and Fanfiction.
- Ensure you have a GPU with sufficient memory for LUAR and the chosen batch size (see `src/train_attention_residual.py` for per-model batch size logic).
- For detailed analysis, interpretability visualizations, and additional ablations, see the Jupyter notebooks and analysis scripts under `src/` and `data/`.


