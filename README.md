# AnOxFuse

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Offpass/AnOxFuse/blob/main/AnOxFuse_model.ipynb)

AnOxFuse predicts antioxidant-peptide probability by combining a molecular branch with a frozen peptide language-model branch. The repository contains the complete notebook, released data, trained release artifacts, a command-line predictor, and the analyses added during peer review.

![AnOxFuse architecture](AnOxFuse.png)

## Model

The molecular branch converts each peptide into a 2,048-dimensional ECFP4 count fingerprint and fits LightGBM. The context branch extracts frozen residue embeddings from `jiahuizhang/esm-150m-peptide-fine-tune` at revision `6d8cebf`, excludes padding and special tokens, concatenates residue-wise mean and maximum pooling, and fits a standardized logistic probe. A logistic meta-model combines the two branch logits. Development predictions used to fit the fusion layer are out of fold.

## Data

The released RLAnOxPeptide-derived files are in `data/`.

| Split | Antioxidant | Antioxidant-unannotated | Total |
|---|---:|---:|---:|
| Development | 1,359 | 1,376 | 2,735 |
| Independent test | 150 | 152 | 302 |

Class 1 is the released antioxidant class. Class 0 is an antioxidant-unannotated background class and should not be interpreted as a collection of experimentally verified inactive peptides. All released sequences contain only the 20 standard amino acids, and the development and test sets have no exact sequence overlap.

## Run

The notebook is the full training and analysis workflow. Open it in Colab with the badge above or run `AnOxFuse_model.ipynb` on an NVIDIA machine. The frozen ESM branch is the only GPU-heavy step.

For inference with the included release artifacts, install a PyTorch build compatible with the machine first, then run:

```bash
python -m pip install -r requirements-gpu.txt
python predict.py --input examples/example_peptides.fasta --output predictions.csv --device auto
```

Input is standard FASTA. Headers are arbitrary, lines may wrap, and sequences must be nonempty strings using the 20 standard amino acids. CUDA is recommended; CPU inference is supported for small inputs but is slower. The output reports both branch probabilities, the fused AnOxFuse probability, and the thresholded prediction.

The serialized classifiers in `models/` are a documented release refit from the frozen published feature matrices. Validate their integrity and released-test reproduction with:

```bash
python -m pip install -r requirements.txt
python scripts/validate_release.py
```

The validation reproduces ROC-AUC, AUPRC, MCC, and every thresholded test decision. The largest probability difference from the frozen historical vector is `3.824e-05`, which reflects the unavailable historical scikit-learn build and does not change any reported metric. `scripts/rebuild_release_artifacts.py` recreates the three small model files. The complete CPU/GPU revision workflows, locked inputs, manifests, result tables, and source spot-checks are in `reproducibility/`.

## Evaluation

The published development estimate uses five-fold similarity-grouped cross-validation and nested fusion. The 302-row released test remains separate until final evaluation. Reviewer analyses add stratified bootstrap intervals, calibration, exact length matching, strict Smith-Waterman cluster-disjoint folds, composition-preserving sequence shuffles, simple sequence baselines, and AOPP/AnOxPP external evaluations after exact-overlap and 60%-identity/80%-coverage filtering. These sensitivity analyses do not replace the published independent-test score.

## Results

| Evaluation | ROC-AUC | AUPRC | MCC |
|---|---:|---:|---:|
| Similarity-grouped development OOF | 0.9846 | 0.9854 | 0.8662 |
| Released independent test | 0.9829 | 0.9838 | 0.8809 |
| Strict whole-pool cluster-disjoint sensitivity | 0.9815 | 0.9820 | 0.8485 |

On the released independent test, accuracy is 0.9404, F1 is 0.9404, precision is 0.9342, recall is 0.9467, and Brier score is 0.0491. The strict whole-pool sensitivity has bootstrap 95% intervals of 0.9778-0.9848 for ROC-AUC and 0.9785-0.9853 for AUPRC. Unique composition-preserving shuffles reduce fusion ROC-AUC by 0.0397 on average, with a paired 95% interval of 0.0224-0.0600.

![Released-test benchmark](figures/benchmark_comparison.png)

![ROC and precision-recall curves](figures/roc_pr_curves.png)
