"""Sequence-independence and uncertainty helpers for the AnOxFuse revision.

The sequence rule used throughout this module is Smith-Waterman local alignment
with BLOSUM62, gap-open 10, and gap-extension 1.  Pairwise identity is the
number of exact residue matches divided by all alignment columns (including
gap columns).  Coverage is the smaller of the aligned fractions of the two
full sequences.  The strict similarity graph uses identity >= 0.60 and
coverage >= 0.80 unless the caller explicitly overrides those values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import parasail
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
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_GAP_OPEN = 10
DEFAULT_GAP_EXTEND = 1
DEFAULT_IDENTITY_THRESHOLD = 0.60
DEFAULT_COVERAGE_THRESHOLD = 0.80
DEFAULT_BOOTSTRAP_REPLICATES = 2500


def read_fasta(
    path: str | Path,
    *,
    label: int | None = None,
    split: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Read a FASTA file without altering record order or sequence text."""

    path = Path(path)
    records: list[dict[str, Any]] = []
    header: str | None = None
    chunks: list[str] = []

    def append_record() -> None:
        if header is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"Empty FASTA sequence for record {header!r} in {path}.")
        records.append(
            {
                "record_id": header.split()[0],
                "header": header,
                "sequence": sequence,
                "length": len(sequence),
                "label": label,
                "split": split,
                "source": source if source is not None else path.name,
            }
        )

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                header = line[1:].strip()
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"Sequence text precedes the first FASTA header in {path}.")
                chunks.append(line)
    append_record()

    columns = ["record_id", "header", "sequence", "length", "label", "split", "source"]
    return pd.DataFrame.from_records(records, columns=columns)


def infer_binary_label(header: str) -> int:
    """Infer the released AnOxFuse binary label from a FASTA header."""

    text = header.strip().lower()
    positive_tokens = ("positive", "pos", "antioxidant", "anox")
    negative_tokens = ("negative", "neg", "non-antioxidant", "non_antioxidant", "nonanox")
    has_negative = any(token in text for token in negative_tokens)
    has_positive = any(token in text for token in positive_tokens)
    if has_negative:
        return 0
    if has_positive:
        return 1
    raise ValueError(f"Could not infer a binary label from FASTA header: {header!r}")


def load_anoxfuse_fastas(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the three released FASTA files into development and test tables."""

    data_dir = Path(data_dir)
    development = pd.concat(
        [
            read_fasta(
                data_dir / "remaining_positive.fasta",
                label=1,
                split="development",
                source="remaining_positive.fasta",
            ),
            read_fasta(
                data_dir / "remaining_negative.fasta",
                label=0,
                split="development",
                source="remaining_negative.fasta",
            ),
        ],
        ignore_index=True,
    )
    test = read_fasta(
        data_dir / "independent_test_cleaned.fasta",
        split="released_test",
        source="independent_test_cleaned.fasta",
    )
    test["label"] = [infer_binary_label(header) for header in test["header"]]

    if development["record_id"].duplicated().any():
        development["record_id"] = [f"development_{index:05d}" for index in range(len(development))]
    if test["record_id"].duplicated().any():
        test["record_id"] = [f"released_test_{index:05d}" for index in range(len(test))]
    return development, test


def smith_waterman_stats(
    query: str,
    reference: str,
    *,
    gap_open: int = DEFAULT_GAP_OPEN,
    gap_extend: int = DEFAULT_GAP_EXTEND,
) -> dict[str, float | int]:
    """Return explicit identity, coverage, and score statistics for one pair."""

    query = str(query).replace(" ", "").upper()
    reference = str(reference).replace(" ", "").upper()
    if not query or not reference:
        raise ValueError("Smith-Waterman alignment requires two non-empty sequences.")

    result = parasail.sw_trace_striped_16(
        query,
        reference,
        gap_open,
        gap_extend,
        parasail.blosum62,
    )
    query_alignment = result.traceback.query
    reference_alignment = result.traceback.ref
    alignment_columns = len(query_alignment)
    if alignment_columns == 0:
        return {
            "score": int(result.score),
            "matches": 0,
            "alignment_columns": 0,
            "identity": 0.0,
            "query_aligned_residues": 0,
            "reference_aligned_residues": 0,
            "query_coverage": 0.0,
            "reference_coverage": 0.0,
            "coverage": 0.0,
        }

    matches = sum(
        query_residue == reference_residue and query_residue != "-"
        for query_residue, reference_residue in zip(query_alignment, reference_alignment)
    )
    query_aligned = sum(residue != "-" for residue in query_alignment)
    reference_aligned = sum(residue != "-" for residue in reference_alignment)
    query_coverage = query_aligned / len(query)
    reference_coverage = reference_aligned / len(reference)
    return {
        "score": int(result.score),
        "matches": int(matches),
        "alignment_columns": int(alignment_columns),
        "identity": float(matches / alignment_columns),
        "query_aligned_residues": int(query_aligned),
        "reference_aligned_residues": int(reference_aligned),
        "query_coverage": float(query_coverage),
        "reference_coverage": float(reference_coverage),
        "coverage": float(min(query_coverage, reference_coverage)),
    }


def nearest_development_neighbors(
    development_sequences: Sequence[str],
    test_sequences: Sequence[str],
    *,
    development_ids: Sequence[Any] | None = None,
    test_ids: Sequence[Any] | None = None,
    development_labels: Sequence[int] | None = None,
    test_labels: Sequence[int] | None = None,
    coverage_floor: float = DEFAULT_COVERAGE_THRESHOLD,
    gap_open: int = DEFAULT_GAP_OPEN,
    gap_extend: int = DEFAULT_GAP_EXTEND,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Find each test peptide's nearest development peptide.

    The reported ``nearest_identity`` is the maximum identity among alignments
    meeting ``coverage_floor``.  If no development peptide reaches that
    coverage, it is set to 0.0 and ``eligible_neighbor_found`` is false.  The
    best unrestricted local alignment is retained in separate columns so the
    convention is fully auditable.
    """

    development_sequences = list(map(str, development_sequences))
    test_sequences = list(map(str, test_sequences))
    if not development_sequences:
        raise ValueError("At least one development sequence is required.")
    development_ids = list(range(len(development_sequences))) if development_ids is None else list(development_ids)
    test_ids = list(range(len(test_sequences))) if test_ids is None else list(test_ids)
    if len(development_ids) != len(development_sequences) or len(test_ids) != len(test_sequences):
        raise ValueError("Sequence and identifier arrays must have matching lengths.")
    if development_labels is not None and len(development_labels) != len(development_sequences):
        raise ValueError("development_labels has the wrong length.")
    if test_labels is not None and len(test_labels) != len(test_sequences):
        raise ValueError("test_labels has the wrong length.")

    development_labels_list = None if development_labels is None else list(development_labels)
    test_labels_list = None if test_labels is None else list(test_labels)
    total = len(test_sequences) * len(development_sequences)
    completed = 0
    rows: list[dict[str, Any]] = []

    for test_index, query in enumerate(test_sequences):
        best_eligible: tuple[tuple[float, float, int], int, dict[str, float | int]] | None = None
        best_unrestricted: tuple[tuple[float, float, int], int, dict[str, float | int]] | None = None
        for development_index, reference in enumerate(development_sequences):
            stats = smith_waterman_stats(
                query,
                reference,
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
            rank = (float(stats["identity"]), float(stats["coverage"]), int(stats["score"]))
            candidate = (rank, development_index, stats)
            if best_unrestricted is None or rank > best_unrestricted[0]:
                best_unrestricted = candidate
            if float(stats["coverage"]) >= coverage_floor - 1e-12 and (
                best_eligible is None or rank > best_eligible[0]
            ):
                best_eligible = candidate
            completed += 1
            if progress is not None:
                progress(completed, total)

        assert best_unrestricted is not None
        unrestricted_index = best_unrestricted[1]
        unrestricted_stats = best_unrestricted[2]
        row: dict[str, Any] = {
            "test_index": test_index,
            "test_id": test_ids[test_index],
            "test_sequence": query,
            "test_length": len(query),
            "coverage_floor": float(coverage_floor),
            "eligible_neighbor_found": best_eligible is not None,
            "best_local_dev_index": unrestricted_index,
            "best_local_dev_id": development_ids[unrestricted_index],
            "best_local_identity": float(unrestricted_stats["identity"]),
            "best_local_coverage": float(unrestricted_stats["coverage"]),
            "best_local_score": int(unrestricted_stats["score"]),
        }
        if test_labels_list is not None:
            row["test_label"] = int(test_labels_list[test_index])

        if best_eligible is None:
            row.update(
                {
                    "nearest_dev_index": -1,
                    "nearest_dev_id": None,
                    "nearest_dev_sequence": None,
                    "nearest_dev_label": np.nan,
                    "nearest_identity": 0.0,
                    "nearest_coverage": 0.0,
                    "nearest_score": 0,
                    "nearest_matches": 0,
                    "nearest_alignment_columns": 0,
                    "nearest_query_coverage": 0.0,
                    "nearest_reference_coverage": 0.0,
                }
            )
        else:
            development_index = best_eligible[1]
            stats = best_eligible[2]
            row.update(
                {
                    "nearest_dev_index": development_index,
                    "nearest_dev_id": development_ids[development_index],
                    "nearest_dev_sequence": development_sequences[development_index],
                    "nearest_dev_label": (
                        np.nan
                        if development_labels_list is None
                        else int(development_labels_list[development_index])
                    ),
                    "nearest_identity": float(stats["identity"]),
                    "nearest_coverage": float(stats["coverage"]),
                    "nearest_score": int(stats["score"]),
                    "nearest_matches": int(stats["matches"]),
                    "nearest_alignment_columns": int(stats["alignment_columns"]),
                    "nearest_query_coverage": float(stats["query_coverage"]),
                    "nearest_reference_coverage": float(stats["reference_coverage"]),
                }
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.component_size = np.ones(size, dtype=np.int64)

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = int(self.parent[index])
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]


def build_similarity_components(
    sequences: Sequence[str],
    *,
    record_ids: Sequence[Any] | None = None,
    labels: Sequence[int] | None = None,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    gap_open: int = DEFAULT_GAP_OPEN,
    gap_extend: int = DEFAULT_GAP_EXTEND,
    return_edges: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Create whole-pool similarity components using the declared SW rule."""

    sequences = list(map(str, sequences))
    count = len(sequences)
    if count == 0:
        raise ValueError("At least one sequence is required.")
    record_ids = list(range(count)) if record_ids is None else list(record_ids)
    if len(record_ids) != count:
        raise ValueError("record_ids has the wrong length.")
    labels_list = None if labels is None else list(labels)
    if labels_list is not None and len(labels_list) != count:
        raise ValueError("labels has the wrong length.")

    disjoint_set = _DisjointSet(count)
    edge_rows: list[dict[str, Any]] = []
    edge_count = 0
    total = count * (count - 1) // 2
    completed = 0
    for right in range(count):
        for left in range(right):
            stats = smith_waterman_stats(
                sequences[right],
                sequences[left],
                gap_open=gap_open,
                gap_extend=gap_extend,
            )
            if (
                float(stats["identity"]) >= identity_threshold - 1e-12
                and float(stats["coverage"]) >= coverage_threshold - 1e-12
            ):
                disjoint_set.union(left, right)
                edge_count += 1
                if return_edges:
                    edge_rows.append(
                        {
                            "left_index": left,
                            "right_index": right,
                            "left_id": record_ids[left],
                            "right_id": record_ids[right],
                            "identity": float(stats["identity"]),
                            "coverage": float(stats["coverage"]),
                            "score": int(stats["score"]),
                            "matches": int(stats["matches"]),
                            "alignment_columns": int(stats["alignment_columns"]),
                        }
                    )
            completed += 1
            if progress is not None:
                progress(completed, total)

    roots = np.asarray([disjoint_set.find(index) for index in range(count)], dtype=np.int64)
    root_to_first_index: dict[int, int] = {}
    for index, root in enumerate(roots):
        root_to_first_index.setdefault(int(root), index)
    ordered_roots = sorted(root_to_first_index, key=root_to_first_index.get)
    root_to_group = {root: group for group, root in enumerate(ordered_roots)}
    groups = np.asarray([root_to_group[int(root)] for root in roots], dtype=np.int64)
    identity_tag = int(round(100 * identity_threshold))
    coverage_tag = int(round(100 * coverage_threshold))
    group_ids = np.asarray(
        [f"SW{identity_tag:02d}C{coverage_tag:02d}_{group:05d}" for group in groups],
        dtype=object,
    )

    membership = pd.DataFrame(
        {
            "index": np.arange(count, dtype=np.int64),
            "record_id": record_ids,
            "sequence": sequences,
            "length": [len(sequence) for sequence in sequences],
            "group_index": groups,
            "group_id": group_ids,
        }
    )
    if labels_list is not None:
        membership["label"] = np.asarray(labels_list, dtype=np.int64)

    aggregation: dict[str, tuple[str, str]] = {
        "size": ("index", "size"),
        "first_index": ("index", "min"),
        "minimum_length": ("length", "min"),
        "maximum_length": ("length", "max"),
    }
    component_table = membership.groupby(["group_index", "group_id"], as_index=False).agg(**aggregation)
    if labels_list is not None:
        class_counts = (
            membership.groupby(["group_index", "label"]).size().unstack(fill_value=0).rename_axis(None, axis=1)
        )
        class_counts = class_counts.rename(columns={0: "negative_n", 1: "positive_n"})
        for column in ("negative_n", "positive_n"):
            if column not in class_counts:
                class_counts[column] = 0
        component_table = component_table.merge(
            class_counts[["negative_n", "positive_n"]].reset_index(),
            on="group_index",
            how="left",
        )

    return {
        "groups": groups,
        "group_ids": group_ids,
        "membership": membership,
        "components": component_table,
        "edges": pd.DataFrame.from_records(edge_rows),
        "edge_count": edge_count,
        "pair_count": total,
        "identity_threshold": float(identity_threshold),
        "coverage_threshold": float(coverage_threshold),
        "gap_open": int(gap_open),
        "gap_extend": int(gap_extend),
    }


def assign_strict_group_folds(
    labels: Sequence[int],
    groups: Sequence[Any],
    *,
    n_splits: int = 5,
    random_state: int = 70877,
    shuffle: bool = False,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign every whole-pool similarity component to exactly one fold."""

    labels_array = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray(groups)
    if labels_array.ndim != 1 or groups_array.ndim != 1 or len(labels_array) != len(groups_array):
        raise ValueError("labels and groups must be one-dimensional arrays of equal length.")
    if not set(np.unique(labels_array)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1.")
    if len(np.unique(groups_array)) < n_splits:
        raise ValueError("There are fewer unique sequence groups than requested folds.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    fold_ids = np.full(len(labels_array), -1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros((len(labels_array), 1)), labels_array, groups_array)
    ):
        fold_ids[test_index] = fold
        train_groups = set(groups_array[train_index].tolist())
        test_groups = set(groups_array[test_index].tolist())
        overlap = train_groups.intersection(test_groups)
        test_positive = int(labels_array[test_index].sum())
        test_negative = int(len(test_index) - test_positive)
        train_positive = int(labels_array[train_index].sum())
        train_negative = int(len(train_index) - train_positive)
        if overlap:
            raise RuntimeError(f"Fold {fold} has {len(overlap)} groups shared between train and test.")
        if require_both_classes and min(test_positive, test_negative, train_positive, train_negative) == 0:
            raise RuntimeError(f"Fold {fold} does not contain both classes in train and test.")
        rows.append(
            {
                "fold": fold,
                "train_n": len(train_index),
                "test_n": len(test_index),
                "train_positive_n": train_positive,
                "train_negative_n": train_negative,
                "test_positive_n": test_positive,
                "test_negative_n": test_negative,
                "train_group_n": len(train_groups),
                "test_group_n": len(test_groups),
                "shared_group_n": 0,
            }
        )
    if np.any(fold_ids < 0):
        raise RuntimeError("At least one row was not assigned to a strict fold.")
    return fold_ids, pd.DataFrame.from_records(rows)


def binary_metrics(
    y_true: Sequence[int],
    y_probability: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discrimination, decision, and calibration metrics."""

    y_true_array = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(y_probability, dtype=np.float64)
    if y_true_array.ndim != 1 or probability.ndim != 1 or len(y_true_array) != len(probability):
        raise ValueError("y_true and y_probability must be one-dimensional arrays of equal length.")
    if not set(np.unique(y_true_array)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and lie between 0 and 1.")

    prediction = (probability >= threshold).astype(np.int64)
    negative_true, positive_true = np.bincount(y_true_array, minlength=2)
    if negative_true and positive_true:
        roc_auc = float(roc_auc_score(y_true_array, probability))
        average_precision = float(average_precision_score(y_true_array, probability))
    else:
        roc_auc = np.nan
        average_precision = np.nan
    matrix = confusion_matrix(y_true_array, prediction, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    specificity = (
        float(true_negative / specificity_denominator) if specificity_denominator else np.nan
    )
    clipped = np.clip(probability, 1e-15, 1 - 1e-15)
    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(y_true_array, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, prediction)),
        "precision": float(precision_score(y_true_array, prediction, zero_division=0)),
        "sensitivity": float(recall_score(y_true_array, prediction, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true_array, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_array, prediction)),
        "brier": float(brier_score_loss(y_true_array, probability)),
        "log_loss": float(log_loss(y_true_array, clipped, labels=[0, 1])),
    }


def stratified_bootstrap_summary(
    y_true: Sequence[int],
    y_probability: Sequence[float],
    *,
    threshold: float = 0.5,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = 70877,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize binary metrics with class-stratified percentile intervals."""

    y_true_array = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(y_probability, dtype=np.float64)
    if len(y_true_array) != len(probability):
        raise ValueError("y_true and y_probability have different lengths.")
    negative_index = np.flatnonzero(y_true_array == 0)
    positive_index = np.flatnonzero(y_true_array == 1)
    if not len(negative_index) or not len(positive_index):
        raise ValueError("Stratified bootstrap requires both outcome classes.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between 0 and 1.")

    estimate = binary_metrics(y_true_array, probability, threshold=threshold)
    samples = {metric: np.empty(n_bootstrap, dtype=np.float64) for metric in estimate}
    generator = np.random.default_rng(random_state)
    for replicate in range(n_bootstrap):
        resampled_index = np.concatenate(
            [
                generator.choice(negative_index, size=len(negative_index), replace=True),
                generator.choice(positive_index, size=len(positive_index), replace=True),
            ]
        )
        replicate_metrics = binary_metrics(
            y_true_array[resampled_index],
            probability[resampled_index],
            threshold=threshold,
        )
        for metric, value in replicate_metrics.items():
            samples[metric][replicate] = value

    alpha = (1 - confidence_level) / 2
    rows = []
    for metric, value in estimate.items():
        distribution = samples[metric]
        finite = distribution[np.isfinite(distribution)]
        rows.append(
            {
                "metric": metric,
                "estimate": value,
                "ci_low": float(np.quantile(finite, alpha)) if len(finite) else np.nan,
                "ci_high": float(np.quantile(finite, 1 - alpha)) if len(finite) else np.nan,
                "confidence_level": confidence_level,
                "bootstrap_replicates": n_bootstrap,
                "valid_replicates": len(finite),
                "n": len(y_true_array),
                "positive_n": len(positive_index),
                "negative_n": len(negative_index),
                "threshold": threshold,
                "random_state": random_state,
            }
        )
    return pd.DataFrame.from_records(rows)


def identity_sensitivity_summary(
    neighbor_table: pd.DataFrame,
    y_true: Sequence[int],
    y_probability: Sequence[float],
    *,
    identity_cutoffs: Iterable[float] = (0.40, 0.50, 0.60),
    threshold: float = 0.5,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = 70877,
) -> pd.DataFrame:
    """Evaluate predictions after excluding increasingly similar test peptides."""

    y_true_array = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(y_probability, dtype=np.float64)
    if len(neighbor_table) != len(y_true_array) or len(y_true_array) != len(probability):
        raise ValueError("neighbor_table, y_true, and y_probability must have equal lengths.")
    if "nearest_identity" not in neighbor_table:
        raise ValueError("neighbor_table must contain a nearest_identity column.")

    tables: list[pd.DataFrame] = []
    nearest_identity = neighbor_table["nearest_identity"].to_numpy(dtype=float)
    for cutoff_index, cutoff in enumerate(identity_cutoffs):
        retained = nearest_identity < float(cutoff)
        retained_labels = y_true_array[retained]
        if retained.sum() == 0 or len(np.unique(retained_labels)) < 2:
            tables.append(
                pd.DataFrame(
                    [
                        {
                            "identity_cutoff": float(cutoff),
                            "metric": "not_estimable",
                            "estimate": np.nan,
                            "ci_low": np.nan,
                            "ci_high": np.nan,
                            "n": int(retained.sum()),
                            "positive_n": int(retained_labels.sum()) if len(retained_labels) else 0,
                            "negative_n": int(len(retained_labels) - retained_labels.sum()) if len(retained_labels) else 0,
                        }
                    ]
                )
            )
            continue
        summary = stratified_bootstrap_summary(
            retained_labels,
            probability[retained],
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            random_state=random_state + cutoff_index,
        )
        summary.insert(0, "identity_cutoff", float(cutoff))
        summary.insert(1, "retained_fraction", float(retained.mean()))
        tables.append(summary)
    return pd.concat(tables, ignore_index=True, sort=False)


def paired_stratified_bootstrap_difference(
    y_true: Sequence[int],
    first_probability: Sequence[float],
    second_probability: Sequence[float],
    *,
    threshold: float = 0.5,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_state: int = 70877,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Estimate paired metric differences as first model minus second model."""

    y_true_array = np.asarray(y_true, dtype=np.int64)
    first = np.asarray(first_probability, dtype=np.float64)
    second = np.asarray(second_probability, dtype=np.float64)
    if not (len(y_true_array) == len(first) == len(second)):
        raise ValueError("All paired arrays must have equal lengths.")
    negative_index = np.flatnonzero(y_true_array == 0)
    positive_index = np.flatnonzero(y_true_array == 1)
    if not len(negative_index) or not len(positive_index):
        raise ValueError("Paired stratified bootstrap requires both outcome classes.")

    first_estimate = binary_metrics(y_true_array, first, threshold=threshold)
    second_estimate = binary_metrics(y_true_array, second, threshold=threshold)
    metrics = list(first_estimate)
    distributions = {metric: np.empty(n_bootstrap, dtype=float) for metric in metrics}
    generator = np.random.default_rng(random_state)
    for replicate in range(n_bootstrap):
        indices = np.concatenate(
            [
                generator.choice(negative_index, size=len(negative_index), replace=True),
                generator.choice(positive_index, size=len(positive_index), replace=True),
            ]
        )
        first_metrics = binary_metrics(y_true_array[indices], first[indices], threshold=threshold)
        second_metrics = binary_metrics(y_true_array[indices], second[indices], threshold=threshold)
        for metric in metrics:
            distributions[metric][replicate] = first_metrics[metric] - second_metrics[metric]

    alpha = (1 - confidence_level) / 2
    rows = []
    for metric in metrics:
        distribution = distributions[metric]
        finite = distribution[np.isfinite(distribution)]
        rows.append(
            {
                "metric": metric,
                "first_estimate": first_estimate[metric],
                "second_estimate": second_estimate[metric],
                "difference": first_estimate[metric] - second_estimate[metric],
                "difference_ci_low": float(np.quantile(finite, alpha)) if len(finite) else np.nan,
                "difference_ci_high": float(np.quantile(finite, 1 - alpha)) if len(finite) else np.nan,
                "confidence_level": confidence_level,
                "bootstrap_replicates": n_bootstrap,
                "valid_replicates": len(finite),
                "n": len(y_true_array),
                "threshold": threshold,
                "random_state": random_state,
            }
        )
    return pd.DataFrame.from_records(rows)


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_COVERAGE_THRESHOLD",
    "DEFAULT_GAP_EXTEND",
    "DEFAULT_GAP_OPEN",
    "DEFAULT_IDENTITY_THRESHOLD",
    "assign_strict_group_folds",
    "binary_metrics",
    "build_similarity_components",
    "identity_sensitivity_summary",
    "infer_binary_label",
    "load_anoxfuse_fastas",
    "nearest_development_neighbors",
    "paired_stratified_bootstrap_difference",
    "read_fasta",
    "smith_waterman_stats",
    "stratified_bootstrap_summary",
]
