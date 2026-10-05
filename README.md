# AnOxFuse

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Offpass/AnOxFuse/blob/main/AnOxFuse_model.ipynb)

AnOxFuse predicts antioxidant activity from peptide sequences. It combines molecular fingerprints with a peptide language model to capture both chemical structure and sequence context.

![AnOxFuse architecture](AnOxFuse.png)

## Model

The molecular branch uses 2,048-dimensional ECFP4 count fingerprints and LightGBM. The sequence branch uses frozen embeddings from [peptide-adapted ESM-2](https://huggingface.co/jiahuizhang/esm-150m-peptide-fine-tune), mean and maximum pooling, and logistic regression. A second logistic regression combines the two branch logits, trained on out-of-fold predictions.

## Configuration and experimental settings

| Category | Implementation details |
|---|---|
| Input | Peptide sequences containing the 20 standard amino acids; sequence length 2–20 residues |
| Molecular representation | RDKit `MolFromFASTA`; Morgan count fingerprint (ECFP4), radius 2, 2,048 dimensions; bond and ring information enabled; chirality disabled |
| Molecular classifier | LightGBM binary GBDT; 500 estimators, learning rate 0.1, 31 leaves, unlimited depth, `min_child_samples=20`, `min_child_weight=0.001`, no class weighting, seed 70877 |
| Sequence encoder | Frozen `jiahuizhang/esm-150m-peptide-fine-tune`, revision `6d8cebf`; last hidden state; dynamic padding; no truncation |
| Sequence pooling | Mean and maximum residue pooling, excluding padding and special tokens; concatenated 1,280-dimensional vector |
| Sequence classifier | `StandardScaler` followed by L2 logistic regression; `C=1.0`, `liblinear`, maximum 4,000 iterations, tolerance `1e-4`, seed 70877 |
| Fusion | Branch probabilities clipped to `[1e-6, 1-1e-6]`, transformed to logits and combined by logistic regression; `C=1.0`, `lbfgs`, maximum 2,000 iterations |
| Development split | Five-fold shuffled `StratifiedGroupKFold`; sequence-similarity groups; seed 70877 |
| Nested training | Four grouped inner folds generate out-of-fold branch predictions for fitting the fusion model inside each outer fold |
| Independent test | 302 held-out peptides used only after model fitting; no exact sequence overlap with development data |
| Classification threshold | Fixed at 0.5; not selected using the independent test set |
| Evaluation metrics | ROC-AUC, AUPRC, accuracy, precision, recall, specificity, F1, MCC, Brier score and log loss |
| Confidence intervals | 2,500 class-stratified bootstrap resamples; percentile 95% confidence intervals |
| Embedding computation | Frozen encoder in evaluation mode; bfloat16 when supported, otherwise float16 autocast; pooled embeddings stored as `float32` |

## Data and evaluation

The [FASTA files](data/) come from the released RLAnOxPeptide dataset.

| Split | Antioxidant | Unannotated background | Total |
|---|---:|---:|---:|
| Development | 1,359 | 1,376 | 2,735 |
| Independent test | 150 | 152 | 302 |

The background peptides are not experimentally confirmed inactive controls. There is no exact sequence overlap between the development and test sets.

Development evaluation uses five-fold sequence-similarity-grouped cross-validation with nested fusion. The independent test is evaluated separately, with a classification threshold of 0.5. Additional sequence-similarity, length, shuffle and external-dataset analyses are in [reproducibility/](reproducibility/).

## Usage

Run [AnOxFuse_model.ipynb](AnOxFuse_model.ipynb) to train and evaluate the model. An NVIDIA GPU is recommended for extracting ESM-2 embeddings.

To predict from a FASTA file using the included trained classifiers, install a compatible PyTorch build, then:

```bash
python -m pip install -r requirements-gpu.txt
python predict.py --input examples/example_peptides.fasta --output predictions.csv --device auto
```

Sequences must use the 20 standard amino acids. The output contains both branch probabilities, the combined probability and the predicted class. Small batches can also run on CPU.

The [saved classifiers](models/) were refitted from the original feature matrices and reproduce the reported test metrics and class predictions. To check the release:

```bash
python -m pip install -r requirements.txt
python scripts/validate_release.py
```

## Results

| Evaluation | ROC-AUC | AUPRC | MCC |
|---|---:|---:|---:|
| Grouped development cross-validation | 0.9846 | 0.9854 | 0.8662 |
| Independent test | 0.9829 | 0.9838 | 0.8809 |
| Whole-pool cluster-disjoint sensitivity analysis | 0.9815 | 0.9820 | 0.8485 |

Independent-test accuracy and F1 are both **0.9404**, with precision **0.9342** and recall **0.9467**. The cluster-disjoint result is a separate sensitivity analysis, not a replacement test set.

![Independent-test ROC and precision-recall curves](figures/roc_pr_curves.png)
