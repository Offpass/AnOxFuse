from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anox_revision_common import STANDARD_AA, binary_metrics
from anox_revision_models import lgbm_factory

AA_INDEX = {residue: index for index, residue in enumerate(STANDARD_AA)}


def aac_vector(sequence: str) -> np.ndarray:
    output = np.zeros(20, dtype=np.float32)
    for residue in sequence:
        output[AA_INDEX[residue]] += 1.0
    return output / max(len(sequence), 1)


def dpc_vector(sequence: str) -> np.ndarray:
    output = np.zeros(400, dtype=np.float32)
    if len(sequence) < 2:
        return output
    for first, second in zip(sequence[:-1], sequence[1:]):
        output[AA_INDEX[first] * 20 + AA_INDEX[second]] += 1.0
    return output / (len(sequence) - 1)


def handcrafted_matrix(sequences: Sequence[str], include_dpc: bool) -> np.ndarray:
    rows = []
    for sequence in sequences:
        parts = [np.asarray([len(sequence)], dtype=np.float32), aac_vector(sequence)]
        if include_dpc:
            parts.append(dpc_vector(sequence))
        rows.append(np.concatenate(parts))
    return np.vstack(rows)


def logistic_model(seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=4000,
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )


def rf_model(seed: int):
    return RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight=None,
        random_state=seed,
        n_jobs=-1,
    )


def char_model(seed: int):
    return Pipeline(
        [
            (
                "ngrams",
                CountVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    lowercase=False,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    max_iter=4000,
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )


def _crossfit_tabular(name, X, y, folds, model_builder, seed):
    probability = np.full(len(y), np.nan)
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds):
        model = model_builder(seed + fold)
        model.fit(X[train_index], y[train_index])
        probability[valid_index] = model.predict_proba(X[valid_index])[:, 1]
        rows.append(
            {
                "model": name,
                "fold": fold + 1,
                **binary_metrics(y[valid_index], probability[valid_index]),
            }
        )
    return probability, rows


def _crossfit_sequences(name, sequences, y, folds, seed):
    sequences = np.asarray(sequences)
    probability = np.full(len(y), np.nan)
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds):
        model = char_model(seed + fold)
        model.fit(sequences[train_index], y[train_index])
        probability[valid_index] = model.predict_proba(sequences[valid_index])[:, 1]
        rows.append(
            {
                "model": name,
                "fold": fold + 1,
                **binary_metrics(y[valid_index], probability[valid_index]),
            }
        )
    return probability, rows


def run_simple_representation_controls(
    dev_sequences: Sequence[str],
    test_sequences: Sequence[str],
    y_dev: np.ndarray,
    y_test: np.ndarray,
    dev_ecfp4: np.ndarray,
    test_ecfp4: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 70877,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev_sequences = np.asarray(dev_sequences)
    test_sequences = np.asarray(test_sequences)
    X_length_dev = np.asarray([len(sequence) for sequence in dev_sequences], dtype=np.float32).reshape(-1, 1)
    X_length_test = np.asarray([len(sequence) for sequence in test_sequences], dtype=np.float32).reshape(-1, 1)
    X_aac_dev = handcrafted_matrix(dev_sequences, include_dpc=False)
    X_aac_test = handcrafted_matrix(test_sequences, include_dpc=False)
    X_dpc_dev = handcrafted_matrix(dev_sequences, include_dpc=True)
    X_dpc_test = handcrafted_matrix(test_sequences, include_dpc=True)
    specifications = [
        ("Length LR", X_length_dev, X_length_test, logistic_model),
        ("AAC plus length LR", X_aac_dev, X_aac_test, logistic_model),
        ("AAC plus DPC plus length LR", X_dpc_dev, X_dpc_test, logistic_model),
        ("AAC plus DPC plus length RF", X_dpc_dev, X_dpc_test, rf_model),
        ("ECFP4 count LightGBM", dev_ecfp4, test_ecfp4, lgbm_factory),
    ]
    metric_rows = []
    fold_rows = []
    prediction_columns: dict[str, np.ndarray] = {"label": y_test}
    for name, X_dev, X_test, builder in specifications:
        oof, rows = _crossfit_tabular(name, X_dev, y_dev, folds, builder, seed)
        model = builder(seed)
        model.fit(X_dev, y_dev)
        test_probability = model.predict_proba(X_test)[:, 1]
        metric_rows.extend(
            [
                {"evaluation": "development grouped OOF", "model": name, **binary_metrics(y_dev, oof)},
                {"evaluation": "released test", "model": name, **binary_metrics(y_test, test_probability)},
            ]
        )
        fold_rows.extend(rows)
        prediction_columns[name] = test_probability

    char_name = "Character 1-3-mer LR"
    char_oof, rows = _crossfit_sequences(char_name, dev_sequences, y_dev, folds, seed)
    model = char_model(seed)
    model.fit(dev_sequences, y_dev)
    char_test = model.predict_proba(test_sequences)[:, 1]
    metric_rows.extend(
        [
            {"evaluation": "development grouped OOF", "model": char_name, **binary_metrics(y_dev, char_oof)},
            {"evaluation": "released test", "model": char_name, **binary_metrics(y_test, char_test)},
        ]
    )
    fold_rows.extend(rows)
    prediction_columns[char_name] = char_test
    return pd.DataFrame(metric_rows), pd.DataFrame(fold_rows), pd.DataFrame(prediction_columns)
