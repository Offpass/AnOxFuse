from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 70877
EPS = 1e-6
MODEL_ID = "jiahuizhang/esm-150m-peptide-fine-tune"
MODEL_REVISION = "6d8cebf"
MAX_ALLOWED_PROBABILITY_DELTA = 5e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["X"], dtype=np.float32)


def local_factory() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def context_factory() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=4000,
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def meta_factory() -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)


def fusion_features(local_probability: np.ndarray, context_probability: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            logit(np.clip(local_probability, EPS, 1.0 - EPS)),
            logit(np.clip(context_probability, EPS, 1.0 - EPS)),
        ]
    )


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the lightweight AnOxFuse release artifacts from frozen published features."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reproducibility/reviewer_revision"),
        help="Self-contained reviewer-revision directory containing baseline/ and cache/.",
    )
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    baseline_path = source / "baseline" / "anoxfuse_main_run.npz"
    dev_local_path = source / "cache" / "dev_ecfp4_count.npz"
    test_local_path = source / "cache" / "test_ecfp4_count.npz"
    dev_context_path = source / "cache" / "dev_esm2_meanmax.npz"
    test_context_path = source / "cache" / "test_esm2_meanmax.npz"
    prediction_table_path = source / "baseline" / "tables" / "anoxfuse_test_predictions.csv"
    required = [
        baseline_path,
        dev_local_path,
        test_local_path,
        dev_context_path,
        test_context_path,
        prediction_table_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing locked source files:\n" + "\n".join(missing))

    with np.load(baseline_path) as payload:
        y_dev = np.asarray(payload["dev_y"], dtype=int)
        y_test = np.asarray(payload["test_y"], dtype=int)
        local_oof = np.asarray(payload["local_oof"], dtype=float)
        context_oof = np.asarray(payload["context_oof"], dtype=float)
        frozen_test = np.asarray(payload["anoxfuse_test"], dtype=float)

    dev_local = load_matrix(dev_local_path)
    test_local = load_matrix(test_local_path)
    dev_context = load_matrix(dev_context_path)
    test_context = load_matrix(test_context_path)

    local_model = local_factory().fit(dev_local, y_dev)
    context_model = context_factory().fit(dev_context, y_dev)
    meta_model = meta_factory().fit(fusion_features(local_oof, context_oof), y_dev)

    local_probability = local_model.predict_proba(test_local)[:, 1]
    context_probability = context_model.predict_proba(test_context)[:, 1]
    probability = meta_model.predict_proba(
        fusion_features(local_probability, context_probability)
    )[:, 1]
    probability_delta = np.abs(probability - frozen_test)
    maximum_delta = float(probability_delta.max())
    if maximum_delta > MAX_ALLOWED_PROBABILITY_DELTA:
        raise RuntimeError(
            f"Release refit differs from the frozen prediction vector by {maximum_delta:.8g}, "
            f"above the allowed {MAX_ALLOWED_PROBABILITY_DELTA:.8g}."
        )
    if not np.array_equal(probability >= 0.5, frozen_test >= 0.5):
        raise RuntimeError("Release refit changes at least one released-test class decision.")

    artifact_paths = {
        "local_model": output / "anoxfuse_local_model.joblib",
        "context_model": output / "anoxfuse_context_model.joblib",
        "meta_model": output / "anoxfuse_meta_model.joblib",
    }
    joblib.dump(local_model, artifact_paths["local_model"])
    joblib.dump(context_model, artifact_paths["context_model"])
    joblib.dump(meta_model, artifact_paths["meta_model"])

    release_table = pd.read_csv(prediction_table_path)
    release_table = release_table.rename(columns={"probability": "frozen_probability"})
    release_table["release_refit_probability"] = probability
    release_table["absolute_probability_delta"] = probability_delta
    release_table.to_csv(output / "released_test_refit_predictions.csv", index=False)

    observed_metrics = metrics(y_test, probability)
    frozen_metrics = metrics(y_test, frozen_test)
    manifest = {
        "schema_version": 1,
        "provenance": (
            "Release refit from frozen published ECFP4 and peptide-adapted ESM-2 features. "
            "These are not byte-identical copies of the historical in-memory estimators."
        ),
        "seed": SEED,
        "decision_threshold": 0.5,
        "encoder": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "features": {
            "molecular": "RDKit Morgan count fingerprint, radius 2, 2,048 dimensions",
            "context": "last-layer residue embeddings, special tokens excluded, mean and maximum pooled, 1,280 dimensions",
            "fusion": "logistic regression over clipped molecular and context logits",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
        },
        "source_files": {
            str(path.relative_to(source)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in required
        },
        "artifacts": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
        "validation": {
            "released_test_rows": int(len(y_test)),
            "maximum_absolute_probability_delta": maximum_delta,
            "allowed_probability_delta": MAX_ALLOWED_PROBABILITY_DELTA,
            "class_decisions_identical": True,
            "release_refit_metrics": observed_metrics,
            "frozen_published_metrics": frozen_metrics,
        },
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["validation"], indent=2))
    print(f"Saved release artifacts to {output}")


if __name__ == "__main__":
    main()

