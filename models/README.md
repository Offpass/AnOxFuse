# Trained classifiers

The three Joblib files contain the molecular, sequence and fusion classifiers used by `predict.py`. They were refitted from the original feature matrices and out-of-fold branch predictions; they are not the original in-memory estimators.

The refit reproduces the independent-test ROC-AUC, AUPRC, MCC and every class prediction. Individual probabilities differ by at most `3.824e-05` from the original run. Both sets of probabilities are in [released_test_refit_predictions.csv](released_test_refit_predictions.csv).

[artifact_manifest.json](artifact_manifest.json) records the feature definitions, encoder revision, package versions and file hashes. From the repository root:

```bash
python scripts/validate_release.py
```

To rebuild the classifiers from the cached features, run `python scripts/rebuild_release_artifacts.py`.
