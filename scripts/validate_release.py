from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logit
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
EPS = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence).upper()))
            header = line[1:]
            sequence = []
        else:
            if header is None:
                raise ValueError(f"Sequence appears before a FASTA header in {path}.")
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence).upper()))
    return records


def fusion_features(local_probability: np.ndarray, context_probability: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            logit(np.clip(local_probability, EPS, 1.0 - EPS)),
            logit(np.clip(context_probability, EPS, 1.0 - EPS)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the published AnOxFuse release package.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    model_dir = root / "models"
    source = root / "reproducibility" / "reviewer_revision"
    manifest = json.loads((model_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    for lock in manifest["artifacts"].values():
        path = model_dir / lock["path"]
        if path.stat().st_size != lock["bytes"] or sha256_file(path) != lock["sha256"]:
            raise RuntimeError(f"Artifact integrity check failed: {path}")

    positive = read_fasta(root / "data" / "remaining_positive.fasta")
    background = read_fasta(root / "data" / "remaining_negative.fasta")
    test = read_fasta(root / "data" / "independent_test_cleaned.fasta")
    all_records = positive + background + test
    invalid = [(header, sequence) for header, sequence in all_records if not sequence or not set(sequence) <= STANDARD_AA]
    if invalid:
        raise RuntimeError(f"Invalid released peptide records: {invalid[:3]}")
    if (len(positive), len(background), len(test)) != (1359, 1376, 302):
        raise RuntimeError("Released split counts changed.")
    development_sequences = {sequence for _, sequence in positive + background}
    if development_sequences.intersection(sequence for _, sequence in test):
        raise RuntimeError("Exact development/test sequence overlap detected.")

    def load_matrix(name: str) -> np.ndarray:
        with np.load(source / "cache" / name) as payload:
            return np.asarray(payload["X"], dtype=np.float32)

    test_local = load_matrix("test_ecfp4_count.npz")
    test_context = load_matrix("test_esm2_meanmax.npz")
    with np.load(source / "baseline" / "anoxfuse_main_run.npz") as payload:
        y_test = np.asarray(payload["test_y"], dtype=int)
        frozen_probability = np.asarray(payload["anoxfuse_test"], dtype=float)

    local_model = joblib.load(model_dir / manifest["artifacts"]["local_model"]["path"])
    context_model = joblib.load(model_dir / manifest["artifacts"]["context_model"]["path"])
    meta_model = joblib.load(model_dir / manifest["artifacts"]["meta_model"]["path"])
    local_probability = local_model.predict_proba(test_local)[:, 1]
    context_probability = context_model.predict_proba(test_context)[:, 1]
    probability = meta_model.predict_proba(fusion_features(local_probability, context_probability))[:, 1]

    maximum_delta = float(np.max(np.abs(probability - frozen_probability)))
    allowed = float(manifest["validation"]["allowed_probability_delta"])
    if maximum_delta > allowed:
        raise RuntimeError(f"Released-test probability delta {maximum_delta:.8g} exceeds {allowed:.8g}.")
    if not np.array_equal(probability >= 0.5, frozen_probability >= 0.5):
        raise RuntimeError("A released-test class decision changed.")

    second_local = joblib.load(model_dir / manifest["artifacts"]["local_model"]["path"])
    repeat_probability = second_local.predict_proba(test_local)[:, 1]
    if not np.array_equal(local_probability, repeat_probability):
        raise RuntimeError("Repeated artifact loading was not deterministic.")

    prediction = (probability >= 0.5).astype(int)
    report = {
        "development_rows": len(positive) + len(background),
        "released_test_rows": len(test),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "auprc": float(average_precision_score(y_test, probability)),
        "mcc": float(matthews_corrcoef(y_test, prediction)),
        "maximum_absolute_probability_delta": maximum_delta,
        "class_decisions_identical": True,
        "repeat_load_maximum_delta": float(np.max(np.abs(local_probability - repeat_probability))),
    }
    print(json.dumps(report, indent=2))
    print("PASS: AnOxFuse release artifacts, data contract, and released-test reproduction are valid.")


if __name__ == "__main__":
    main()

