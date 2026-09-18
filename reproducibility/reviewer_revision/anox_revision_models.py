from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anox_revision_common import EPS, binary_metrics, logit_features


def lgbm_factory(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def context_factory(seed: int) -> Pipeline:
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


def concat_factory(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=0.1,
                    solver="liblinear",
                    max_iter=5000,
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit(model, X, y, sample_weight=None):
    if sample_weight is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, model__sample_weight=sample_weight)
    return model.fit(X, y, sample_weight=sample_weight)


def _probability(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def learned_convex_weight(y: Sequence[int], local: Sequence[float], context: Sequence[float]) -> float:
    y = np.asarray(y, dtype=int)
    local = np.asarray(local, dtype=float)
    context = np.asarray(context, dtype=float)

    def objective(weight: float) -> float:
        probability = np.clip(weight * local + (1.0 - weight) * context, EPS, 1.0 - EPS)
        return float(log_loss(y, probability, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


@dataclass
class FoldResult:
    predictions: dict[str, np.ndarray]
    fold_rows: list[dict]
    fusion_rows: list[dict]


def nested_fold_predictions(
    X_local: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    include_concat: bool = True,
    sample_weight: np.ndarray | None = None,
) -> FoldResult:
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    outputs = {
        "Molecular branch": np.full(len(y), np.nan),
        "Contextual branch": np.full(len(y), np.nan),
        "Equal probability mean": np.full(len(y), np.nan),
        "Equal logit mean": np.full(len(y), np.nan),
        "Learned convex probability mean": np.full(len(y), np.nan),
        "Nested logistic fusion": np.full(len(y), np.nan),
    }
    if include_concat:
        outputs["Feature concatenation LR"] = np.full(len(y), np.nan)
        X_concat = np.column_stack([X_local, X_context])
    fold_rows: list[dict] = []
    fusion_rows: list[dict] = []

    for outer_fold, (outer_train, outer_valid) in enumerate(folds):
        inner_splitter = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=seed + 1000 + outer_fold,
        )
        inner_folds = list(
            inner_splitter.split(
                np.zeros(len(outer_train)),
                y[outer_train],
                groups=groups[outer_train],
            )
        )
        inner_scores: list[np.ndarray] = []
        outer_scores: list[np.ndarray] = []
        branch_specs: list[tuple[np.ndarray, Callable[[int], object], str]] = [
            (X_local, lgbm_factory, "Molecular branch"),
            (X_context, context_factory, "Contextual branch"),
        ]
        for X, factory, branch_name in branch_specs:
            inner_oof = np.full(len(outer_train), np.nan)
            for inner_fold, (inner_train_rel, inner_valid_rel) in enumerate(inner_folds):
                train_index = outer_train[inner_train_rel]
                valid_index = outer_train[inner_valid_rel]
                model = factory(seed + 10_000 * outer_fold + inner_fold)
                weights = None if sample_weight is None else sample_weight[train_index]
                _fit(model, X[train_index], y[train_index], weights)
                inner_oof[inner_valid_rel] = _probability(model, X[valid_index])
            model = factory(seed + 20_000 + outer_fold)
            weights = None if sample_weight is None else sample_weight[outer_train]
            _fit(model, X[outer_train], y[outer_train], weights)
            outer_probability = _probability(model, X[outer_valid])
            outputs[branch_name][outer_valid] = outer_probability
            inner_scores.append(inner_oof)
            outer_scores.append(outer_probability)

        inner_local, inner_context = inner_scores
        outer_local, outer_context = outer_scores
        equal_probability = 0.5 * (outer_local + outer_context)
        equal_logit = expit(
            0.5
            * (
                logit(np.clip(outer_local, EPS, 1.0 - EPS))
                + logit(np.clip(outer_context, EPS, 1.0 - EPS))
            )
        )
        convex_weight = learned_convex_weight(
            y[outer_train], inner_local, inner_context
        )
        convex_probability = convex_weight * outer_local + (1.0 - convex_weight) * outer_context

        meta = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight=None,
        )
        inner_Z = logit_features(inner_local, inner_context)
        outer_Z = logit_features(outer_local, outer_context)
        weights = None if sample_weight is None else sample_weight[outer_train]
        meta.fit(inner_Z, y[outer_train], sample_weight=weights)
        nested_probability = _probability(meta, outer_Z)

        outputs["Equal probability mean"][outer_valid] = equal_probability
        outputs["Equal logit mean"][outer_valid] = equal_logit
        outputs["Learned convex probability mean"][outer_valid] = convex_probability
        outputs["Nested logistic fusion"][outer_valid] = nested_probability

        fusion_rows.append(
            {
                "seed": seed,
                "fold": outer_fold + 1,
                "learned_local_weight": convex_weight,
                "nested_local_coefficient": float(meta.coef_[0, 0]),
                "nested_context_coefficient": float(meta.coef_[0, 1]),
                "nested_intercept": float(meta.intercept_[0]),
            }
        )

        if include_concat:
            model = concat_factory(seed + 30_000 + outer_fold)
            weights = None if sample_weight is None else sample_weight[outer_train]
            _fit(model, X_concat[outer_train], y[outer_train], weights)
            outputs["Feature concatenation LR"][outer_valid] = _probability(
                model, X_concat[outer_valid]
            )

        for model_name, probability in outputs.items():
            fold_probability = probability[outer_valid]
            if np.isfinite(fold_probability).all():
                fold_rows.append(
                    {
                        "seed": seed,
                        "fold": outer_fold + 1,
                        "model": model_name,
                        **binary_metrics(y[outer_valid], fold_probability),
                    }
                )

    for model_name, probability in outputs.items():
        if not np.isfinite(probability).all():
            raise RuntimeError(f"Missing OOF predictions for {model_name}.")
    return FoldResult(outputs, fold_rows, fusion_rows)


def fit_full_and_predict(
    X_local: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    X_local_test: np.ndarray,
    X_context_test: np.ndarray,
    branch_oof: dict[str, np.ndarray],
    seed: int,
    include_concat: bool = True,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    y = np.asarray(y, dtype=int)
    local_model = lgbm_factory(seed)
    context_model = context_factory(seed)
    _fit(local_model, X_local, y, sample_weight)
    _fit(context_model, X_context, y, sample_weight)
    local_test = _probability(local_model, X_local_test)
    context_test = _probability(context_model, X_context_test)
    predictions = {
        "Molecular branch": local_test,
        "Contextual branch": context_test,
        "Equal probability mean": 0.5 * (local_test + context_test),
        "Equal logit mean": expit(
            0.5
            * (
                logit(np.clip(local_test, EPS, 1.0 - EPS))
                + logit(np.clip(context_test, EPS, 1.0 - EPS))
            )
        ),
    }
    local_oof = branch_oof["Molecular branch"]
    context_oof = branch_oof["Contextual branch"]
    local_weight = learned_convex_weight(y, local_oof, context_oof)
    predictions["Learned convex probability mean"] = (
        local_weight * local_test + (1.0 - local_weight) * context_test
    )
    meta = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, class_weight=None
    )
    meta.fit(logit_features(local_oof, context_oof), y, sample_weight=sample_weight)
    predictions["Nested logistic fusion"] = _probability(
        meta, logit_features(local_test, context_test)
    )
    if include_concat:
        concat_model = concat_factory(seed)
        X_concat = np.column_stack([X_local, X_context])
        X_concat_test = np.column_stack([X_local_test, X_context_test])
        _fit(concat_model, X_concat, y, sample_weight)
        predictions["Feature concatenation LR"] = _probability(concat_model, X_concat_test)
    parameters = {
        "learned_local_weight": local_weight,
        "nested_local_coefficient": float(meta.coef_[0, 0]),
        "nested_context_coefficient": float(meta.coef_[0, 1]),
        "nested_intercept": float(meta.intercept_[0]),
    }
    return predictions, parameters


def repeated_grouped_comparison(
    X_local: np.ndarray,
    X_context: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    X_local_test: np.ndarray,
    X_context_test: np.ndarray,
    y_test: np.ndarray,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    fold_rows: list[dict] = []
    parameter_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups=groups))
        result = nested_fold_predictions(
            X_local, X_context, y, groups, folds, seed, include_concat=True
        )
        test_predictions, parameters = fit_full_and_predict(
            X_local,
            X_context,
            y,
            X_local_test,
            X_context_test,
            result.predictions,
            seed,
            include_concat=True,
        )
        for model_name, probability in result.predictions.items():
            metric_rows.append(
                {
                    "evaluation": "development grouped OOF",
                    "seed": seed,
                    "model": model_name,
                    **binary_metrics(y, probability),
                }
            )
        for model_name, probability in test_predictions.items():
            metric_rows.append(
                {
                    "evaluation": "released test sensitivity",
                    "seed": seed,
                    "model": model_name,
                    **binary_metrics(y_test, probability),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "row_index": np.arange(len(y_test)),
                        "label": y_test,
                        "model": model_name,
                        "probability": probability,
                    }
                )
            )
        fold_rows.extend(result.fold_rows)
        parameter_rows.append({"seed": seed, **parameters})
        parameter_rows.extend(result.fusion_rows)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(parameter_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def summarize_repeated_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "AUC", "AUPRC", "ACC", "Balanced_ACC", "F1", "Precision", "Recall",
        "Specificity", "MCC", "Brier", "LogLoss",
    ]
    grouped = metrics.groupby(["evaluation", "model"], sort=False)[value_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_sd")
    return mean.join(std).reset_index()
