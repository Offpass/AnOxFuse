# AnOxFuse implementation and experimental settings

This document collects the configuration, implementation and evaluation details used for AnOxFuse and the reviewer-directed analyses.

## Data and labels

| Item | Setting |
|---|---|
| Source | Released RLAnOxPeptide-derived FASTA files in [`data/`](../data/) |
| Development set | 1,359 antioxidant peptides and 1,376 antioxidant-unannotated background peptides; 2,735 total |
| Independent test | 150 antioxidant peptides and 152 antioxidant-unannotated background peptides; 302 total |
| Label interpretation | Class 1 is the released antioxidant class. Class 0 is an unannotated background and is not treated as experimentally confirmed inactive |
| Accepted input | Uppercase sequences containing only `ACDEFGHIKLMNPQRSTVWY`; length 2–20 residues |
| Invalid input | Empty sequences or nonstandard residues stop the run; no unknown-residue substitution or row imputation |
| Development/test overlap | No exact sequence overlap between the released development and independent-test files |
| Label-orientation check | Seven positive sequences were traced by exact match to experimentally reported peptides from four source studies; this is a spot check rather than complete provenance for the anonymous benchmark |

## Final model implementation

| Component | Setting |
|---|---|
| Molecular construction | RDKit `Chem.MolFromFASTA`; standard peptide bonds, free neutral N terminus and neutral C-terminal carboxylic acid |
| Molecular conversion failure | Abort the run; no silent row removal or imputation |
| Molecular representation | Morgan count fingerprint (ECFP4), radius 2, 2,048 hashed `float32` dimensions |
| Morgan options | Bond types and ring membership enabled; chirality disabled |
| Representation scope | Sequence-derived two-dimensional connectivity; no three-dimensional conformation, pH-state ensemble or post-translational modifications |
| Molecular classifier | LightGBM binary GBDT |
| LightGBM configuration | 500 estimators; learning rate 0.1; 31 leaves; unlimited depth; `min_child_samples=20`; `min_child_weight=0.001`; full row and column sampling; no L1/L2 regularization; no class weighting; seed 70877 |
| Sequence encoder | Frozen `jiahuizhang/esm-150m-peptide-fine-tune`, revision `6d8cebf`; parent model `facebook/esm2_t30_150M_UR50D`; evaluation mode |
| Encoder input | Raw uppercase sequence; dynamic padding; no truncation |
| Hidden representation | Last hidden state; padding and all special tokens excluded |
| Pooling | Residue-wise mean concatenated with element-wise maximum; 1,280-dimensional `float32` output |
| Embedding precision | bfloat16 autocast when supported, otherwise float16; pooled arrays stored as `float32` |
| Sequence classifier | `StandardScaler` followed by L2 logistic regression; `C=1.0`; `liblinear`; maximum 4,000 iterations; tolerance `1e-4`; no class weighting; seed 70877 |
| Fusion input | Each branch probability clipped to `[1e-6, 1-1e-6]` and converted to a logit |
| Fusion model | Logistic regression; `C=1.0`; `lbfgs`; maximum 2,000 iterations |
| Final fusion parameters | Molecular-logit coefficient 0.12124174; contextual-logit coefficient 0.28677884; intercept 0.33672979 |
| Classification threshold | Fixed at 0.5; not selected using the independent test set |
| Hyperparameter policy | Settings are fixed in the public workflow; no grid or random search is run, and model choices are not reselected inside every outer fold |

## Experimental design and statistics

| Item | Setting |
|---|---|
| Global seed | 70877 |
| Development evaluation | Five-fold shuffled `StratifiedGroupKFold` |
| Headline similarity groups | Deterministic greedy grouping using RapidFuzz `fuzz.ratio`; sequences ordered by decreasing length, sequence text and original row index; grouping threshold 60 |
| Nested fusion | Four grouped inner folds generate out-of-fold branch predictions for fitting the fusion model within every outer fold |
| Final independent-test fit | Branches refitted on the complete development set; fusion fitted from complete-development out-of-fold branch predictions; independent test evaluated after all fitting |
| Probability metrics | ROC-AUC, AUPRC, Brier score and log loss |
| Threshold metrics | Accuracy, balanced accuracy, precision, recall or sensitivity, specificity, F1 and MCC at 0.5 |
| Confidence intervals | 2,500 class-stratified bootstrap resamples; 2.5th and 97.5th percentiles |
| Paired comparisons | The same bootstrap indices are used for both competing prediction vectors |
| Calibration | Brier score, log loss, 10-bin equal-width and equal-frequency ECE, calibration intercept and calibration slope |
| Repeated grouped analysis | Five grouped runs using seeds 42–46; variability reported as the standard deviation across seeds |
| Contextual t-SNE | 1,280-dimensional embeddings reduced to 50 PCA components, then t-SNE with perplexity 30, learning rate `auto`, PCA initialization, 1,500 iterations and seed 70877 |

## Reviewer-directed analyses

| Concern | Analysis and setting | Evidence |
|---|---|---|
| Sequence similarity | Smith-Waterman local alignment with BLOSUM62, gap-open 10 and gap-extension 1; edges require at least 60% identity and at least 80% coverage of both sequences; connected components define cluster-disjoint folds | [`strict_cluster_disjoint_fold_audit.csv`](../reproducibility/reviewer_revision/revision_outputs/cpu/tables/strict_cluster_disjoint_fold_audit.csv) |
| Simple confounding | Length-only, AAC plus length, AAC plus DPC plus length, character 1–3-mer and ECFP4 controls | [`final_results/tables/`](../reproducibility/final_results/tables/) |
| Length dependence | Exact length matching, matched-development retraining and overlap weighting | [`final_results/tables/`](../reproducibility/final_results/tables/) |
| Residue order | Ten unique non-original composition-preserving shuffles targeted per peptide; predictions averaged within the original peptide before paired bootstrapping | [`unique_shuffle_auc_drop.png`](../reproducibility/final_results/figures/unique_shuffle_auc_drop.png) |
| Fusion contribution | Nested logistic fusion compared with equal probability averaging, equal logit averaging, learned convex probability fusion and feature-level concatenation | [`final_results/tables/`](../reproducibility/final_results/tables/) |
| External transfer | AOPP and AnOxPP evaluated after exact-overlap removal and after excluding training peptides with at least 60% Smith-Waterman identity and at least 80% reciprocal coverage to an external peptide | [`external_protocol_metrics.csv`](../reproducibility/final_results/tables/external_protocol_metrics.csv) |
| External reference | AAC plus DPC plus length, 421 features, 500-tree random forest, square-root feature sampling, Gini splits, bootstrap sampling, unlimited depth, no class weighting and seed 70877 | [`external_dataset_audit.csv`](../reproducibility/final_results/tables/external_dataset_audit.csv) |
| Calibration and uncertainty | Bootstrap confidence intervals, calibration intercept and slope, ECE, Brier score and log loss | [`released_test_bootstrap_summary.csv`](../reproducibility/reviewer_revision/revision_outputs/cpu/tables/released_test_bootstrap_summary.csv) |

## Software and hardware

| Run | Environment |
|---|---|
| Headline run | Python 3.12.13; PyTorch 2.12.0+cu130; Transformers 5.14.1; CUDA 13.0; NVIDIA RTX 5080 |
| Reviewer GPU run | Python 3.12.14; PyTorch 2.11.0+cu128; CUDA 12.8; NVIDIA RTX 5090; runtime 1,759 seconds |
| Locked revision packages | NumPy 2.4.6; pandas 3.0.5; SciPy 1.18.0; scikit-learn 1.9.0; LightGBM 4.7.0; RDKit 2026.3.4; Biopython 1.87; parasail 1.3.4; matplotlib 3.11.1; seaborn 0.13.2; joblib 1.5.3; RapidFuzz 3.14.5; Transformers 5.14.1 |
| Historical limitation | Exact historical scikit-learn, LightGBM and RDKit versions used for the original headline run were not recorded; the locked revision environment documents the reproducibility rerun |

## Reproduction paths

| Purpose | Path |
|---|---|
| Main training and evaluation | [`AnOxFuse_model.ipynb`](../AnOxFuse_model.ipynb) |
| Released prediction interface | [`predict.py`](../predict.py) |
| Model artifact manifest | [`models/artifact_manifest.json`](../models/artifact_manifest.json) |
| Release validation | [`scripts/validate_release.py`](../scripts/validate_release.py) |
| Reviewer workflows and outputs | [`reproducibility/reviewer_revision/`](../reproducibility/reviewer_revision/) |
| Final reviewer result tables and figures | [`reproducibility/final_results/`](../reproducibility/final_results/) |
