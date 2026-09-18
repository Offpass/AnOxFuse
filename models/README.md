# Release artifacts

The three Joblib files are a release refit of the final classifiers using the frozen published ECFP4 features, peptide-adapted ESM-2 mean-plus-maximum features, development labels, and out-of-fold branch probabilities. They are not claimed to be byte-identical copies of the historical in-memory estimators.

`artifact_manifest.json` records the seed, feature definitions, encoder revision, dependency versions, file hashes, source hashes, and validation tolerance. `released_test_refit_predictions.csv` shows the frozen and refitted probabilities row by row. Run `python scripts/validate_release.py` from the repository root to verify integrity and reproduce the released-test metrics.

