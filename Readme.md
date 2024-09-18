## Residualized Similarity Prediction using Attention

### Setup
- `conda create -n rsp python=3.10`
- `conda activate rsp`
- `git clone https://github.com/eric-sclafani/gram2vec`
- `pip install gram2vec/`
- `pip install -r requirements.txt`

### Reproducing results
- To reproduce the results in the paper, as of now, one can run the three train files in the 'src' folder.
- As there are three types of systems that we produce, one must run each train file with each model-dataset pair.
- For example, to reproduce the results for RoBERTa and Reddit, one can run `train_residual.py -m roberta -d reddit`.

### Data:
- With respect to the Amazon and Fanfiction datasets, we've included post-processing scripts of the data after they have been downloaded according to the LUAR paper 
- We share the raw and processed files of the Reddit dataset as described in the Style Embedding paper.
- We share the post processed data for the Amazon and Fanfiction datasets.

### Notes:
- We plan on releasing a script so that one doesn't need to manually run these experiments.
- We plan on releasing the pre-trained models
