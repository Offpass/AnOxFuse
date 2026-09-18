from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 70877
EPS = 1e-6
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def set_seed(seed: int = SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks).upper()))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def label_from_header(header: str) -> int:
    if header and header[0] in "01":
        return int(header[0])
    fields = header.split("|")
    for field in fields:
        if field in {"0", "1"}:
            return int(field)
    raise ValueError(f"Cannot infer class label from FASTA header: {header}")


def records_to_frame(
    records: Sequence[tuple[str, str]],
    labels: Sequence[int] | None = None,
    source: str = "",
) -> pd.DataFrame:
    if labels is None:
        labels = [label_from_header(header) for header, _ in records]
    frame = pd.DataFrame(
        {
            "header": [header for header, _ in records],
            "sequence": [sequence for _, sequence in records],
            "label": np.asarray(labels, dtype=int),
            "source": source,
        }
    )
    frame["length"] = frame["sequence"].str.len()
    frame["valid"] = frame["sequence"].map(
        lambda sequence: bool(sequence) and set(sequence) <= set(STANDARD_AA)
    )
    return frame


def load_released_frames(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    positive = read_fasta(data_dir / "remaining_positive.fasta")
    negative = read_fasta(data_dir / "remaining_negative.fasta")
    test = read_fasta(data_dir / "independent_test_cleaned.fasta")
    dev = pd.concat(
        [
            records_to_frame(positive, [1] * len(positive), "remaining_positive"),
            records_to_frame(negative, [0] * len(negative), "remaining_negative"),
        ],
        ignore_index=True,
    )
    test_frame = records_to_frame(test, source="published_independent_test")
    if not dev["valid"].all() or not test_frame["valid"].all():
        raise RuntimeError("Released FASTA files contain non-standard or empty sequences.")
    if len(dev) != 2735 or len(test_frame) != 302:
        raise RuntimeError("Released dataset row counts differ from the published code release.")
    return dev, test_frame


def binary_metrics(
    y_true: Sequence[int], probability: Sequence[float], threshold: float = 0.5
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    raw = np.asarray(probability, dtype=float)
    clipped = np.clip(raw, EPS, 1.0 - EPS)
    prediction = (raw >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "AUC": float(roc_auc_score(y, raw)) if len(np.unique(y)) == 2 else np.nan,
        "AUPRC": float(average_precision_score(y, raw)) if y.sum() else np.nan,
        "ACC": float(accuracy_score(y, prediction)),
        "Balanced_ACC": float(balanced_accuracy_score(y, prediction)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Recall": float(recall_score(y, prediction, zero_division=0)),
        "Sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(specificity),
        "MCC": float(matthews_corrcoef(y, prediction)),
        "Brier": float(brier_score_loss(y, clipped)),
        "LogLoss": float(log_loss(y, clipped, labels=[0, 1])),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def weighted_binary_metrics(
    y_true: Sequence[int], probability: Sequence[float], weights: Sequence[float], threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    w = np.asarray(weights, dtype=float)
    pred = (p >= threshold).astype(int)
    tn = float(w[(y == 0) & (pred == 0)].sum())
    fp = float(w[(y == 0) & (pred == 1)].sum())
    fn = float(w[(y == 1) & (pred == 0)].sum())
    tp = float(w[(y == 1) & (pred == 1)].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return {
        "n": float(w.sum()),
        "effective_n": float(w.sum() ** 2 / np.square(w).sum()),
        "AUC": float(roc_auc_score(y, p, sample_weight=w)),
        "AUPRC": float(average_precision_score(y, p, sample_weight=w)),
        "ACC": float(np.average(pred == y, weights=w)),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "Sensitivity": float(recall),
        "Specificity": float(specificity),
        "MCC": float(mcc),
        "Brier": float(np.average(np.square(clipped - y), weights=w)),
        "LogLoss": float(log_loss(y, clipped, sample_weight=w, labels=[0, 1])),
    }


def stratified_bootstrap_indices(y_true: Sequence[int], reps: int, seed: int) -> Iterable[np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == value) for value in (0, 1)]
    for _ in range(reps):
        sampled = [rng.choice(index, len(index), replace=True) for index in by_class]
        yield np.concatenate(sampled)


def bootstrap_metric_summary(
    y_true: Sequence[int],
    probability: Sequence[float],
    reps: int = 2500,
    seed: int = SEED,
    analysis: str = "",
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    point = binary_metrics(y, p)
    metric_names = [
        "AUC", "AUPRC", "ACC", "Balanced_ACC", "F1", "Precision",
        "Recall", "Sensitivity", "Specificity", "MCC", "Brier", "LogLoss",
    ]
    draws = {name: [] for name in metric_names}
    for index in stratified_bootstrap_indices(y, reps, seed):
        row = binary_metrics(y[index], p[index])
        for name in metric_names:
            draws[name].append(row[name])
    rows = []
    for name in metric_names:
        values = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "analysis": analysis,
                "metric": name,
                "estimate": point[name],
                "ci_low": float(np.nanquantile(values, 0.025)),
                "ci_high": float(np.nanquantile(values, 0.975)),
                "bootstrap_reps": reps,
                "bootstrap": "class-stratified percentile",
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_delta(
    y_true: Sequence[int],
    first_probability: Sequence[float],
    second_probability: Sequence[float],
    reps: int = 2500,
    seed: int = SEED,
    first_name: str = "first",
    second_name: str = "second",
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    first = np.asarray(first_probability, dtype=float)
    second = np.asarray(second_probability, dtype=float)
    names = ["AUC", "AUPRC", "ACC", "F1", "MCC", "Brier", "LogLoss"]
    first_point = binary_metrics(y, first)
    second_point = binary_metrics(y, second)
    draws = {name: [] for name in names}
    for index in stratified_bootstrap_indices(y, reps, seed):
        a = binary_metrics(y[index], first[index])
        b = binary_metrics(y[index], second[index])
        for name in names:
            draws[name].append(a[name] - b[name])
    rows = []
    for name in names:
        values = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "first": first_name,
                "second": second_name,
                "metric": name,
                "delta_first_minus_second": first_point[name] - second_point[name],
                "ci_low": float(np.nanquantile(values, 0.025)),
                "ci_high": float(np.nanquantile(values, 0.975)),
                "bootstrap_reps": reps,
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_manifest(root: str | Path) -> None:
    root = Path(root)
    manifest_path = root / "input_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing input manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("input_manifest.json does not contain a non-empty files mapping.")
    for relative_path, lock in files.items():
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing locked input: {relative_path}")
        expected_bytes = int(lock["bytes"])
        observed_bytes = path.stat().st_size
        if observed_bytes != expected_bytes:
            raise RuntimeError(
                f"Locked input size mismatch for {relative_path}: "
                f"expected {expected_bytes}, observed {observed_bytes}."
            )
        expected_sha256 = str(lock["sha256"]).lower()
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Locked input SHA-256 mismatch for {relative_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}."
            )
    print(f"Verified {len(files)} locked input files against input_manifest.json.")


def estimator_parameter_manifest(estimator) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, value in estimator.get_params(deep=True).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            output[name] = value
        else:
            output[name] = repr(value)
    return output


def save_json(payload: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def logit_features(*probabilities: Sequence[float]) -> np.ndarray:
    return np.column_stack([logit(np.clip(np.asarray(p), EPS, 1.0 - EPS)) for p in probabilities])


def equal_logit_mean(*probabilities: Sequence[float]) -> np.ndarray:
    return expit(np.mean(logit_features(*probabilities), axis=1))
