from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_SEED = 70877
DEFAULT_RESAMPLES = 2500
EPS = 1e-6


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def _write_csv(frame: pd.DataFrame, output_dir: Path | None, name: str) -> None:
    if output_dir is not None:
        frame.to_csv(output_dir / name, index=False)


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _validate_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"sequence", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    result = frame.copy().reset_index(drop=True)
    result["sequence"] = result["sequence"].astype(str).str.upper().str.strip()
    result["label"] = result["label"].astype(int)
    if not set(result["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name}.label must contain only 0 and 1.")
    if "header" not in result:
        result["header"] = [f"{name}_{i}" for i in range(len(result))]
    result["header"] = result["header"].astype(str)
    if "length" not in result:
        result["length"] = result["sequence"].str.len()
    result["length"] = result["length"].astype(int)
    observed_length = result["sequence"].str.len().to_numpy()
    if not np.array_equal(observed_length, result["length"].to_numpy()):
        raise ValueError(f"{name}.length is inconsistent with the sequence strings.")
    return result


def _normalise_aodb_sequences(
    aodb_sequences: Sequence[str] | pd.Series | pd.DataFrame | None,
) -> list[str]:
    if aodb_sequences is None:
        return []
    if isinstance(aodb_sequences, pd.DataFrame):
        if "Sequence" in aodb_sequences:
            values = aodb_sequences["Sequence"]
        elif "sequence" in aodb_sequences:
            values = aodb_sequences["sequence"]
        elif aodb_sequences.shape[1] == 1:
            values = aodb_sequences.iloc[:, 0]
        else:
            raise ValueError("AODB table must contain a Sequence column.")
    elif isinstance(aodb_sequences, pd.Series):
        values = aodb_sequences
    else:
        values = list(aodb_sequences)
    return [str(value).upper().strip() for value in values if str(value).strip()]


def infer_header_label(header: str) -> float:
    text = str(header).strip().lower()
    if re.search(r"(?:^|[|_\-])1(?:$|[|_\-])", text) or re.search(
        r"(?:^|[|_\-])pos(?:$|[|_\-])", text
    ):
        return 1.0
    if re.search(r"(?:^|[|_\-])0(?:$|[|_\-])", text) or re.search(
        r"(?:^|[|_\-])neg(?:$|[|_\-])", text
    ):
        return 0.0
    return np.nan


def run_label_integrity_audit(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    aodb_sequences: Sequence[str] | pd.Series | pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    paper_reported_positive: int = 1528,
    paper_reported_negative: int = 1509,
) -> dict[str, pd.DataFrame]:
    """Audit class orientation, counts, duplicates, and source consistency.

    AODB substring membership is an orientation/source-consistency check. It is
    not independent experimental validation of antioxidant activity.
    """

    out = _as_path(output_dir)
    dev = _validate_frame(development, "development")
    tst = _validate_frame(test, "test")
    dev["split"] = "development"
    tst["split"] = "released_test"
    combined = pd.concat([dev, tst], ignore_index=True, sort=False)
    combined["row_id"] = [f"row_{i:05d}" for i in range(len(combined))]
    combined["header_label"] = combined["header"].map(infer_header_label)
    combined["header_label_available"] = combined["header_label"].notna()
    combined["header_label_consistent"] = (
        ~combined["header_label_available"]
        | (combined["header_label"].astype("Int64") == combined["label"])
    )
    combined["valid_standard_amino_acids"] = combined["sequence"].map(
        lambda sequence: bool(sequence) and set(sequence).issubset(set(STANDARD_AA))
    )
    combined["duplicated_within_split"] = combined.groupby("split")[
        "sequence"
    ].transform(lambda values: values.duplicated(keep=False))

    dev_sequences = set(dev["sequence"])
    test_sequences = set(tst["sequence"])
    overlap = dev_sequences.intersection(test_sequences)
    combined["cross_split_exact_overlap"] = combined["sequence"].isin(overlap)
    label_nunique = combined.groupby("sequence")["label"].transform("nunique")
    combined["conflicting_label_anywhere"] = label_nunique > 1

    aodb = _normalise_aodb_sequences(aodb_sequences)
    if aodb:
        combined["aodb_protein_substring"] = combined["sequence"].map(
            lambda peptide: any(peptide in protein for protein in aodb)
        )
    else:
        combined["aodb_protein_substring"] = False

    released_positive = int(combined["label"].sum())
    released_negative = int(len(combined) - released_positive)
    apparent_transposition = (
        released_positive == paper_reported_negative
        and released_negative == paper_reported_positive
    )

    positive_aodb = int(
        ((combined["label"] == 1) & combined["aodb_protein_substring"]).sum()
    )
    negative_aodb = int(
        ((combined["label"] == 0) & combined["aodb_protein_substring"]).sum()
    )
    positive_not_aodb = released_positive - positive_aodb
    negative_not_aodb = released_negative - negative_aodb
    if aodb and positive_aodb + negative_aodb:
        orientation_fisher = fisher_exact(
            [
                [positive_aodb, positive_not_aodb],
                [negative_aodb, negative_not_aodb],
            ]
        )
        aodb_odds_ratio = float(orientation_fisher.statistic)
        aodb_p_value = float(orientation_fisher.pvalue)
    else:
        aodb_odds_ratio = np.nan
        aodb_p_value = np.nan

    manifest_columns = [
        "row_id",
        "split",
        "header",
        "sequence",
        "length",
        "label",
        "header_label",
        "header_label_available",
        "header_label_consistent",
        "valid_standard_amino_acids",
        "duplicated_within_split",
        "cross_split_exact_overlap",
        "conflicting_label_anywhere",
        "aodb_protein_substring",
    ]
    if "source" in combined:
        manifest_columns.insert(4, "source")
    manifest = combined[manifest_columns].copy()

    summary_rows: list[dict[str, Any]] = []

    def add(item: str, value: Any, notes: str = "") -> None:
        summary_rows.append({"item": item, "value": value, "notes": notes})

    for split_name, frame in (("development", dev), ("released_test", tst)):
        add(f"{split_name}_n", int(len(frame)))
        add(f"{split_name}_positive", int(frame["label"].sum()))
        add(f"{split_name}_negative", int(len(frame) - frame["label"].sum()))
    add("released_total_n", int(len(combined)))
    add("released_total_positive", released_positive)
    add("released_total_negative", released_negative)
    add("paper_reported_positive", int(paper_reported_positive))
    add("paper_reported_negative", int(paper_reported_negative))
    add(
        "paper_counts_are_apparent_transposition",
        bool(apparent_transposition),
        "Count-level discrepancy only; sequence-level orientation is audited separately.",
    )
    add("test_header_label_mismatches", int((~manifest.loc[manifest.split == "released_test", "header_label_consistent"]).sum()))
    add("nonstandard_sequence_rows", int((~manifest["valid_standard_amino_acids"]).sum()))
    add("within_split_duplicate_rows", int(manifest["duplicated_within_split"].sum()))
    add("development_test_exact_overlap_sequences", int(len(overlap)))
    add("conflicting_label_rows", int(manifest["conflicting_label_anywhere"].sum()))
    add("aodb_positive_substrings", positive_aodb)
    add("aodb_negative_substrings", negative_aodb)
    add(
        "aodb_orientation_odds_ratio",
        aodb_odds_ratio,
        "Source-consistency check only; not independent biological validation.",
    )
    add("aodb_orientation_fisher_p", aodb_p_value)
    summary = pd.DataFrame(summary_rows)

    class_counts = (
        combined.groupby(["split", "label"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    class_counts["class_name"] = class_counts["label"].map(
        {0: "negative", 1: "positive"}
    )

    _write_csv(manifest, out, "label_integrity_manifest.csv")
    _write_csv(summary, out, "label_integrity_summary.csv")
    _write_csv(class_counts, out, "label_integrity_class_counts.csv")
    return {
        "label_integrity_manifest": manifest,
        "label_integrity_summary": summary,
        "label_integrity_class_counts": class_counts,
    }


def _weighted_confusion(
    y: np.ndarray, prediction: np.ndarray, sample_weight: np.ndarray
) -> tuple[float, float, float, float]:
    tn = float(sample_weight[(y == 0) & (prediction == 0)].sum())
    fp = float(sample_weight[(y == 0) & (prediction == 1)].sum())
    fn = float(sample_weight[(y == 1) & (prediction == 0)].sum())
    tp = float(sample_weight[(y == 1) & (prediction == 1)].sum())
    return tn, fp, fn, tp


def binary_metrics(
    y: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    raw_probability = np.asarray(probability, dtype=float)
    if len(y_array) != len(raw_probability):
        raise ValueError("y and probability must have the same length.")
    if not np.isfinite(raw_probability).all():
        raise ValueError("probability contains non-finite values.")
    weights = (
        np.ones(len(y_array), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if len(weights) != len(y_array) or np.any(weights < 0):
        raise ValueError("sample_weight must be nonnegative and match y.")
    clipped = np.clip(raw_probability, EPS, 1 - EPS)
    prediction = (raw_probability >= threshold).astype(int)
    tn, fp, fn, tp = _weighted_confusion(y_array, prediction, weights)
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0
    auc = (
        float(roc_auc_score(y_array, raw_probability, sample_weight=weights))
        if len(np.unique(y_array)) == 2
        else np.nan
    )
    auprc = float(
        average_precision_score(y_array, raw_probability, sample_weight=weights)
    )
    brier = float(np.average((y_array - clipped) ** 2, weights=weights))
    return {
        "AUC": auc,
        "AUPRC": auprc,
        "ACC": (tp + tn) / total if total else np.nan,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "MCC": mcc,
        "Brier": brier,
        "LogLoss": float(
            log_loss(y_array, clipped, sample_weight=weights, labels=[0, 1])
        ),
    }


def _validate_prediction_map(
    prediction_map: Mapping[str, Sequence[float] | np.ndarray], expected_n: int
) -> dict[str, np.ndarray]:
    validated: dict[str, np.ndarray] = {}
    for name, values in prediction_map.items():
        probability = np.asarray(values, dtype=float)
        if probability.shape != (expected_n,):
            raise ValueError(
                f"Prediction vector {name!r} has shape {probability.shape}; "
                f"expected ({expected_n},)."
            )
        if not np.isfinite(probability).all():
            raise ValueError(f"Prediction vector {name!r} contains non-finite values.")
        validated[str(name)] = probability
    if not validated:
        raise ValueError("prediction_map cannot be empty.")
    return validated


def _stratified_bootstrap_indices(
    y: np.ndarray, reps: int, seed: int
) -> list[np.ndarray]:
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    if not len(positive) or not len(negative):
        raise ValueError("Both classes are required for a stratified bootstrap.")
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        for _ in range(reps)
    ]


def stratified_bootstrap_metrics(
    y: Sequence[int] | np.ndarray,
    prediction_map: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reps: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    threshold: float = 0.5,
    output_dir: str | Path | None = None,
    file_prefix: str = "prediction",
) -> dict[str, pd.DataFrame]:
    y_array = np.asarray(y, dtype=int)
    predictions = _validate_prediction_map(prediction_map, len(y_array))
    indices = _stratified_bootstrap_indices(y_array, reps, seed)
    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_name, probability in predictions.items():
        point = binary_metrics(y_array, probability, threshold=threshold)
        model_draws: dict[str, list[float]] = {key: [] for key in point}
        for replicate, index in enumerate(indices):
            values = binary_metrics(
                y_array[index], probability[index], threshold=threshold
            )
            draw_rows.extend(
                {
                    "model": model_name,
                    "replicate": replicate,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in values.items()
            )
            for metric, value in values.items():
                model_draws[metric].append(value)
        for metric, estimate in point.items():
            draws = np.asarray(model_draws[metric], dtype=float)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.nanquantile(draws, 0.025)),
                    "ci_high": float(np.nanquantile(draws, 0.975)),
                    "resamples": reps,
                    "resampling": "class-stratified bootstrap",
                }
            )
    draws_frame = pd.DataFrame(draw_rows)
    summary = pd.DataFrame(summary_rows)
    out = _as_path(output_dir)
    _write_csv(draws_frame, out, f"{file_prefix}_bootstrap_draws.csv")
    _write_csv(summary, out, f"{file_prefix}_bootstrap_summary.csv")
    return {
        f"{file_prefix}_bootstrap_draws": draws_frame,
        f"{file_prefix}_bootstrap_summary": summary,
    }


def run_exact_length_matching(
    test: pd.DataFrame,
    prediction_map: Mapping[str, Sequence[float] | np.ndarray],
    *,
    reps: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    threshold: float = 0.5,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Repeat exact within-length class balancing without replacement.

    The reported interval is matching-selection variability, not a conventional
    sampling confidence interval.
    """

    out = _as_path(output_dir)
    frame = _validate_frame(test, "test")
    y = frame["label"].to_numpy(dtype=int)
    lengths = frame["length"].to_numpy(dtype=int)
    predictions = _validate_prediction_map(prediction_map, len(frame))

    count_table = (
        frame.groupby(["length", "label"]).size().unstack(fill_value=0)
    )
    for label in (0, 1):
        if label not in count_table:
            count_table[label] = 0
    count_table = count_table[[0, 1]].rename(
        columns={0: "available_negative", 1: "available_positive"}
    )
    count_table["selected_per_class"] = count_table[
        ["available_negative", "available_positive"]
    ].min(axis=1)
    count_table["shared_length"] = count_table["selected_per_class"] > 0
    count_table = count_table.reset_index()
    common_lengths = count_table.loc[
        count_table["shared_length"], "length"
    ].astype(int).tolist()
    if not common_lengths:
        raise ValueError("No peptide length contains both classes.")

    rng = np.random.default_rng(seed)
    selection_indices: list[np.ndarray] = []
    selection_counts = np.zeros(len(frame), dtype=int)
    reference_index: np.ndarray | None = None
    for replicate in range(reps):
        chosen: list[int] = []
        for length in common_lengths:
            positive = np.where((lengths == length) & (y == 1))[0]
            negative = np.where((lengths == length) & (y == 0))[0]
            n = min(len(positive), len(negative))
            chosen.extend(rng.choice(positive, n, replace=False).tolist())
            chosen.extend(rng.choice(negative, n, replace=False).tolist())
        index = np.asarray(chosen, dtype=int)
        if reference_index is None:
            reference_index = index.copy()
        selection_indices.append(index)
        selection_counts[index] += 1

    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    n_per_replicate = len(selection_indices[0])
    for model_name, probability in predictions.items():
        ordinary = binary_metrics(y, probability, threshold=threshold)
        metric_draws: dict[str, list[float]] = {key: [] for key in ordinary}
        for replicate, index in enumerate(selection_indices):
            values = binary_metrics(
                y[index], probability[index], threshold=threshold
            )
            row: dict[str, Any] = {
                "model": model_name,
                "replicate": replicate,
                "n": len(index),
                "positive": int(y[index].sum()),
                "negative": int(len(index) - y[index].sum()),
            }
            row.update(values)
            draw_rows.append(row)
            for metric, value in values.items():
                metric_draws[metric].append(value)
        for metric, ordinary_estimate in ordinary.items():
            draws = np.asarray(metric_draws[metric], dtype=float)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "ordinary_estimate": ordinary_estimate,
                    "matched_mean": float(np.nanmean(draws)),
                    "interval_low": float(np.nanquantile(draws, 0.025)),
                    "interval_high": float(np.nanquantile(draws, 0.975)),
                    "interval_type": "matching-selection variability",
                    "matching_repetitions": reps,
                    "shared_lengths": len(common_lengths),
                    "n_per_replicate": n_per_replicate,
                }
            )

    assert reference_index is not None
    reference_columns = ["header", "sequence", "length", "label"]
    if "source" in frame:
        reference_columns.append("source")
    reference_membership = frame.loc[reference_index, reference_columns].copy()
    reference_membership.insert(0, "row_index", reference_index)
    reference_membership.insert(0, "replicate", 0)
    selection_frequency = frame[["header", "sequence", "length", "label"]].copy()
    selection_frequency.insert(0, "row_index", np.arange(len(frame)))
    selection_frequency["times_selected"] = selection_counts
    selection_frequency["selection_fraction"] = selection_counts / reps

    draws_frame = pd.DataFrame(draw_rows)
    summary = pd.DataFrame(summary_rows)
    _write_csv(count_table, out, "length_match_counts.csv")
    _write_csv(reference_membership, out, "length_match_membership_reference.csv")
    _write_csv(selection_frequency, out, "length_match_selection_frequency.csv")
    _write_csv(draws_frame, out, "length_match_draw_metrics.csv")
    _write_csv(summary, out, "length_match_summary.csv")
    return {
        "length_match_counts": count_table,
        "length_match_membership_reference": reference_membership,
        "length_match_selection_frequency": selection_frequency,
        "length_match_draw_metrics": draws_frame,
        "length_match_summary": summary,
    }


def peptide_descriptor_table(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Biopython is required for charge, GRAVY hydrophobicity, and "
            "molecular-weight descriptors. Install the pinned requirements "
            "before running the confound analysis."
        ) from exc

    data = _validate_frame(frame, "peptide_frame")
    rows: list[dict[str, Any]] = []
    for row_index, row in data.iterrows():
        sequence = row["sequence"]
        if not sequence or not set(sequence).issubset(set(STANDARD_AA)):
            raise ValueError(
                f"Cannot calculate descriptors for nonstandard sequence at row {row_index}."
            )
        analysis = ProteinAnalysis(sequence)
        counts = analysis.count_amino_acids()
        descriptor: dict[str, Any] = {
            "row_index": int(row_index),
            "header": row["header"],
            "sequence": sequence,
            "label": int(row["label"]),
            "length": len(sequence),
        }
        descriptor.update(
            {f"AAC_{aa}": counts.get(aa, 0) / len(sequence) for aa in STANDARD_AA}
        )
        descriptor["charge_pH7"] = float(analysis.charge_at_pH(7.0))
        descriptor["gravy_hydrophobicity"] = float(analysis.gravy())
        descriptor["molecular_weight"] = float(analysis.molecular_weight())
        rows.append(descriptor)
    return pd.DataFrame(rows)


def _weighted_mean_variance(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance


def _balance_table(
    descriptors: pd.DataFrame,
    feature_columns: Sequence[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    y = descriptors["label"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        values = descriptors[feature].to_numpy(dtype=float)
        for scheme, current_weights in (
            ("unweighted", np.ones(len(y), dtype=float)),
            ("overlap_weighted", weights),
        ):
            mean_positive, variance_positive = _weighted_mean_variance(
                values[y == 1], current_weights[y == 1]
            )
            mean_negative, variance_negative = _weighted_mean_variance(
                values[y == 0], current_weights[y == 0]
            )
            denominator = math.sqrt(
                max((variance_positive + variance_negative) / 2, 0.0)
            )
            smd = (
                (mean_positive - mean_negative) / denominator
                if denominator > 0
                else 0.0
            )
            rows.append(
                {
                    "scheme": scheme,
                    "feature": feature,
                    "positive_mean": mean_positive,
                    "negative_mean": mean_negative,
                    "standardized_mean_difference": smd,
                    "absolute_smd": abs(smd),
                }
            )
    return pd.DataFrame(rows)


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return float(weights.sum() ** 2 / denominator) if denominator else 0.0


def run_overlap_weighted_confound_analysis(
    frame: pd.DataFrame,
    prediction_map: Mapping[str, Sequence[float] | np.ndarray],
    *,
    propensity_c: float = 1.0,
    threshold: float = 0.5,
    bootstrap_reps: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run a descriptive overlap-weighted confound sensitivity analysis.

    When applied to the released test set, labels are used to estimate the
    weights. The result must therefore be reported as a descriptive sensitivity
    analysis, not as a new independent performance estimate.
    """

    out = _as_path(output_dir)
    data = _validate_frame(frame, "confound_frame")
    predictions = _validate_prediction_map(prediction_map, len(data))
    descriptors = peptide_descriptor_table(data)
    feature_columns = ["length"] + [f"AAC_{aa}" for aa in STANDARD_AA] + [
        "charge_pH7",
        "gravy_hydrophobicity",
        "molecular_weight",
    ]
    X = descriptors[feature_columns].to_numpy(dtype=float)
    y = descriptors["label"].to_numpy(dtype=int)
    propensity_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=propensity_c,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    propensity = propensity_model.fit(X, y).predict_proba(X)[:, 1]
    propensity = np.clip(propensity, EPS, 1 - EPS)
    overlap_weight = np.where(y == 1, 1 - propensity, propensity)
    descriptors["propensity_positive"] = propensity
    descriptors["overlap_weight"] = overlap_weight
    balance = _balance_table(descriptors, feature_columns, overlap_weight)

    overall_ess = _effective_sample_size(overlap_weight)
    positive_ess = _effective_sample_size(overlap_weight[y == 1])
    negative_ess = _effective_sample_size(overlap_weight[y == 0])
    weighted_max_smd = float(
        balance.loc[balance["scheme"] == "overlap_weighted", "absolute_smd"].max()
    )

    bootstrap_indices = _stratified_bootstrap_indices(y, bootstrap_reps, seed)
    metric_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    for model_name, probability in predictions.items():
        point = binary_metrics(
            y,
            probability,
            threshold=threshold,
            sample_weight=overlap_weight,
        )
        draws: dict[str, list[float]] = {metric: [] for metric in point}
        for replicate, index in enumerate(bootstrap_indices):
            values = binary_metrics(
                y[index],
                probability[index],
                threshold=threshold,
                sample_weight=overlap_weight[index],
            )
            draw_rows.extend(
                {
                    "model": model_name,
                    "replicate": replicate,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in values.items()
            )
            for metric, value in values.items():
                draws[metric].append(value)
        for metric, estimate in point.items():
            values = np.asarray(draws[metric], dtype=float)
            metric_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "overall_effective_sample_size": overall_ess,
                    "positive_effective_sample_size": positive_ess,
                    "negative_effective_sample_size": negative_ess,
                    "max_absolute_smd_after_weighting": weighted_max_smd,
                    "propensity_model": f"scaled L2 logistic regression, C={propensity_c}",
                    "interpretation": "descriptive confound-overlap sensitivity",
                }
            )

    metrics = pd.DataFrame(metric_rows)
    draws_frame = pd.DataFrame(draw_rows)
    _write_csv(descriptors, out, "confound_descriptors_and_weights.csv")
    _write_csv(balance, out, "confound_balance.csv")
    _write_csv(metrics, out, "overlap_weighted_metrics.csv")
    _write_csv(draws_frame, out, "overlap_weighted_bootstrap_draws.csv")
    return {
        "confound_descriptors_and_weights": descriptors,
        "confound_balance": balance,
        "overlap_weighted_metrics": metrics,
        "overlap_weighted_bootstrap_draws": draws_frame,
    }


def run_shuffle_geometry_audit(
    sequences: Sequence[str],
    *,
    headers: Sequence[str] | None = None,
    reps: int = 10,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Reproduce and characterize composition-preserving random permutations."""

    out = _as_path(output_dir)
    sequence_list = [str(sequence).upper().strip() for sequence in sequences]
    if headers is None:
        header_list = [f"sequence_{index}" for index in range(len(sequence_list))]
    else:
        header_list = list(map(str, headers))
        if len(header_list) != len(sequence_list):
            raise ValueError("headers and sequences must have the same length.")
    if reps < 1:
        raise ValueError("reps must be at least 1.")
    rng = np.random.default_rng(seed)
    ledger_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    all_shuffles: list[str] = []

    for owner_index, (header, original) in enumerate(
        zip(header_list, sequence_list, strict=True)
    ):
        characters = np.asarray(list(original))
        seen: set[str] = set()
        owner_hamming: list[int] = []
        unchanged_count = 0
        repeated_count = 0
        for shuffle_index in range(reps):
            shuffled = "".join(rng.permutation(characters))
            unchanged = shuffled == original
            repeated = shuffled in seen
            seen.add(shuffled)
            hamming = sum(left != right for left, right in zip(original, shuffled))
            normalized_hamming = hamming / len(original) if original else np.nan
            unchanged_count += int(unchanged)
            repeated_count += int(repeated)
            owner_hamming.append(hamming)
            all_shuffles.append(shuffled)
            ledger_rows.append(
                {
                    "owner_index": owner_index,
                    "header": header,
                    "shuffle_index": shuffle_index,
                    "original_sequence": original,
                    "shuffled_sequence": shuffled,
                    "length": len(original),
                    "unchanged_from_original": unchanged,
                    "repeated_within_owner": repeated,
                    "hamming_distance": hamming,
                    "normalized_hamming_distance": normalized_hamming,
                }
            )
        owner_rows.append(
            {
                "owner_index": owner_index,
                "header": header,
                "length": len(original),
                "draws": reps,
                "unique_draws": len(seen),
                "unchanged_draws": unchanged_count,
                "repeated_draws": repeated_count,
                "mean_hamming_distance": float(np.mean(owner_hamming)),
                "mean_normalized_hamming_distance": float(
                    np.mean(owner_hamming) / len(original)
                )
                if original
                else np.nan,
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    owner_summary = pd.DataFrame(owner_rows)
    generated = len(ledger)
    hamming = ledger["hamming_distance"].to_numpy(dtype=float)
    normalized = ledger["normalized_hamming_distance"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {"metric": "owners", "value": len(sequence_list)},
            {"metric": "shuffles_per_owner", "value": reps},
            {"metric": "generated_draws", "value": generated},
            {
                "metric": "unique_output_sequences_global",
                "value": len(set(all_shuffles)),
            },
            {
                "metric": "unchanged_draws",
                "value": int(ledger["unchanged_from_original"].sum()),
            },
            {
                "metric": "unchanged_fraction",
                "value": float(ledger["unchanged_from_original"].mean()),
            },
            {
                "metric": "within_owner_repeated_draws",
                "value": int(ledger["repeated_within_owner"].sum()),
            },
            {
                "metric": "within_owner_repeated_fraction",
                "value": float(ledger["repeated_within_owner"].mean()),
            },
            {
                "metric": "owners_with_at_least_one_unchanged_draw",
                "value": int((owner_summary["unchanged_draws"] > 0).sum()),
            },
            {"metric": "mean_hamming_distance", "value": float(np.mean(hamming))},
            {"metric": "median_hamming_distance", "value": float(np.median(hamming))},
            {
                "metric": "hamming_distance_q025",
                "value": float(np.quantile(hamming, 0.025)),
            },
            {
                "metric": "hamming_distance_q975",
                "value": float(np.quantile(hamming, 0.975)),
            },
            {
                "metric": "mean_normalized_hamming_distance",
                "value": float(np.nanmean(normalized)),
            },
            {
                "metric": "median_normalized_hamming_distance",
                "value": float(np.nanmedian(normalized)),
            },
            {
                "metric": "normalized_hamming_distance_q025",
                "value": float(np.nanquantile(normalized, 0.025)),
            },
            {
                "metric": "normalized_hamming_distance_q975",
                "value": float(np.nanquantile(normalized, 0.975)),
            },
        ]
    )
    _write_csv(ledger, out, "shuffle_draw_ledger.csv")
    _write_csv(owner_summary, out, "shuffle_owner_summary.csv")
    _write_csv(summary, out, "shuffle_geometry_summary.csv")
    return {
        "shuffle_draw_ledger": ledger,
        "shuffle_owner_summary": owner_summary,
        "shuffle_geometry_summary": summary,
    }


def expected_calibration_error_equal_width(
    y: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y_array = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if mask.any():
            value += float(mask.mean() * abs(y_array[mask].mean() - p[mask].mean()))
    return value


def expected_calibration_error_equal_frequency(
    y: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    y_array = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    groups = np.array_split(np.argsort(p, kind="mergesort"), min(bins, len(p)))
    return float(
        sum(
            len(group)
            / len(p)
            * abs(y_array[group].mean() - p[group].mean())
            for group in groups
            if len(group)
        )
    )


def calibration_intercept_slope(
    y: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    y_array = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    predictor = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    model = LogisticRegression(
        C=1e8,
        solver="lbfgs",
        max_iter=3000,
        fit_intercept=True,
    ).fit(predictor, y_array)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _calibration_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, EPS, 1 - EPS)
    intercept, slope = calibration_intercept_slope(y, clipped)
    return {
        "Brier": float(np.mean((y - clipped) ** 2)),
        "LogLoss": float(log_loss(y, clipped, labels=[0, 1])),
        "ECE_equal_width_10": expected_calibration_error_equal_width(
            y, clipped, bins=10
        ),
        "ECE_equal_frequency_10": expected_calibration_error_equal_frequency(
            y, clipped, bins=10
        ),
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
    }


def _calibration_bins(
    y: np.ndarray,
    probability: np.ndarray,
    evaluation: str,
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    equal_width_edges = np.linspace(0, 1, bins + 1)
    for bin_index, (left, right) in enumerate(
        zip(equal_width_edges[:-1], equal_width_edges[1:], strict=True), start=1
    ):
        mask = (probability >= left) & (
            (probability < right) if right < 1 else (probability <= right)
        )
        rows.append(
            {
                "evaluation": evaluation,
                "scheme": "equal_width",
                "bin": bin_index,
                "lower_bound": left,
                "upper_bound": right,
                "n": int(mask.sum()),
                "mean_probability": float(probability[mask].mean())
                if mask.any()
                else np.nan,
                "observed_rate": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    ordered_groups = np.array_split(
        np.argsort(probability, kind="mergesort"), min(bins, len(probability))
    )
    for bin_index, index in enumerate(ordered_groups, start=1):
        rows.append(
            {
                "evaluation": evaluation,
                "scheme": "equal_frequency",
                "bin": bin_index,
                "lower_bound": float(probability[index].min()),
                "upper_bound": float(probability[index].max()),
                "n": int(len(index)),
                "mean_probability": float(probability[index].mean()),
                "observed_rate": float(y[index].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_calibration_analysis(
    evaluation_map: Mapping[
        str, tuple[Sequence[int] | np.ndarray, Sequence[float] | np.ndarray]
    ],
    *,
    bootstrap_reps: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    out = _as_path(output_dir)
    summary_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    bin_frames: list[pd.DataFrame] = []
    for evaluation, (y_values, probability_values) in evaluation_map.items():
        y = np.asarray(y_values, dtype=int)
        probability = np.asarray(probability_values, dtype=float)
        if y.shape != probability.shape:
            raise ValueError(f"Calibration arrays for {evaluation!r} must match.")
        point = _calibration_metrics(y, probability)
        bin_frames.append(_calibration_bins(y, probability, str(evaluation)))
        indices = _stratified_bootstrap_indices(
            y, bootstrap_reps, _stable_seed(seed, "calibration", str(evaluation))
        )
        metric_draws: dict[str, list[float]] = {metric: [] for metric in point}
        for replicate, index in enumerate(indices):
            values = _calibration_metrics(y[index], probability[index])
            draw_rows.extend(
                {
                    "evaluation": evaluation,
                    "replicate": replicate,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in values.items()
            )
            for metric, value in values.items():
                metric_draws[metric].append(value)
        for metric, estimate in point.items():
            values = np.asarray(metric_draws[metric], dtype=float)
            summary_rows.append(
                {
                    "evaluation": evaluation,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.nanquantile(values, 0.025)),
                    "ci_high": float(np.nanquantile(values, 0.975)),
                    "bootstrap_reps": bootstrap_reps,
                    "resampling": "class-stratified bootstrap",
                }
            )
    summary = pd.DataFrame(summary_rows)
    draws = pd.DataFrame(draw_rows)
    bins = pd.concat(bin_frames, ignore_index=True)
    _write_csv(summary, out, "calibration_summary.csv")
    _write_csv(draws, out, "calibration_bootstrap_draws.csv")
    _write_csv(bins, out, "calibration_bins.csv")
    return {
        "calibration_summary": summary,
        "calibration_bootstrap_draws": draws,
        "calibration_bins": bins,
    }


def run_bias_statistics_suite(
    development: pd.DataFrame,
    test: pd.DataFrame,
    *,
    development_predictions: Mapping[str, Sequence[float] | np.ndarray],
    test_predictions: Mapping[str, Sequence[float] | np.ndarray],
    main_model_name: str,
    output_dir: str | Path,
    aodb_sequences: Sequence[str] | pd.Series | pd.DataFrame | None = None,
    seed: int = DEFAULT_SEED,
    matching_reps: int = DEFAULT_RESAMPLES,
    bootstrap_reps: int = DEFAULT_RESAMPLES,
    shuffle_reps: int = 10,
) -> dict[str, pd.DataFrame]:
    """Run all CPU-only bias and statistical reviewer analyses.

    This function consumes frozen predictions and never alters the published
    baseline. Training-time matched or overlap-weighted model fitting belongs in
    the orchestrating notebook because it requires the cached representation
    matrices and model factories.
    """

    root = _as_path(output_dir)
    assert root is not None
    dev = _validate_frame(development, "development")
    tst = _validate_frame(test, "test")
    validated_dev_predictions = _validate_prediction_map(
        development_predictions, len(dev)
    )
    validated_test_predictions = _validate_prediction_map(test_predictions, len(tst))
    if main_model_name not in validated_dev_predictions:
        raise KeyError(f"Missing development predictions for {main_model_name!r}.")
    if main_model_name not in validated_test_predictions:
        raise KeyError(f"Missing test predictions for {main_model_name!r}.")

    outputs: dict[str, pd.DataFrame] = {}
    outputs.update(
        run_label_integrity_audit(
            dev,
            tst,
            aodb_sequences=aodb_sequences,
            output_dir=root,
        )
    )
    outputs.update(
        run_exact_length_matching(
            tst,
            validated_test_predictions,
            reps=matching_reps,
            seed=seed,
            output_dir=root,
        )
    )
    outputs.update(
        run_overlap_weighted_confound_analysis(
            tst,
            validated_test_predictions,
            bootstrap_reps=bootstrap_reps,
            seed=seed,
            output_dir=root,
        )
    )
    outputs.update(
        run_shuffle_geometry_audit(
            tst["sequence"].tolist(),
            headers=tst["header"].tolist(),
            reps=shuffle_reps,
            seed=seed,
            output_dir=root,
        )
    )
    outputs.update(
        run_calibration_analysis(
            {
                "Similarity-grouped development OOF": (
                    dev["label"].to_numpy(dtype=int),
                    validated_dev_predictions[main_model_name],
                ),
                "Released independent test": (
                    tst["label"].to_numpy(dtype=int),
                    validated_test_predictions[main_model_name],
                ),
            },
            bootstrap_reps=bootstrap_reps,
            seed=seed,
            output_dir=root,
        )
    )
    outputs.update(
        stratified_bootstrap_metrics(
            tst["label"].to_numpy(dtype=int),
            validated_test_predictions,
            reps=bootstrap_reps,
            seed=seed,
            output_dir=root,
            file_prefix="released_test",
        )
    )

    manifest = {
        "module": "anox_revision_bias_stats",
        "seed": seed,
        "matching_repetitions": matching_reps,
        "bootstrap_repetitions": bootstrap_reps,
        "shuffle_repetitions": shuffle_reps,
        "main_model_name": main_model_name,
        "development_n": len(dev),
        "test_n": len(tst),
        "published_baseline_policy": (
            "Frozen predictions are authoritative; all stricter analyses are "
            "reported separately as sensitivity analyses."
        ),
        "gpu_required": False,
        "gpu_boundary": (
            "A GPU is needed only to embed new sequences, such as expanded "
            "shuffles or uncached external peptides."
        ),
    }
    (root / "bias_stats_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return outputs


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "STANDARD_AA",
    "binary_metrics",
    "calibration_intercept_slope",
    "expected_calibration_error_equal_frequency",
    "expected_calibration_error_equal_width",
    "infer_header_label",
    "peptide_descriptor_table",
    "run_bias_statistics_suite",
    "run_calibration_analysis",
    "run_exact_length_matching",
    "run_label_integrity_audit",
    "run_overlap_weighted_confound_analysis",
    "run_shuffle_geometry_audit",
    "stratified_bootstrap_metrics",
]
