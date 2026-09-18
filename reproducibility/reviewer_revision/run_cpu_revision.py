from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anox_revision_bias_stats import (
    peptide_descriptor_table,
    run_bias_statistics_suite,
)
from anox_revision_common import (
    SEED,
    binary_metrics,
    estimator_parameter_manifest,
    load_released_frames,
    paired_bootstrap_delta,
    save_json,
    set_seed,
    verify_input_manifest,
)
from anox_revision_controls import run_simple_representation_controls
from anox_revision_models import (
    context_factory,
    fit_full_and_predict,
    lgbm_factory,
    nested_fold_predictions,
    repeated_grouped_comparison,
    summarize_repeated_metrics,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
from anox_revision_sequence import (
    assign_strict_group_folds,
    build_similarity_components,
    identity_sensitivity_summary,
    nearest_development_neighbors,
)

OUTPUT_ROOT = ROOT / "revision_outputs" / "cpu"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
CACHE_DIR = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_npz_X(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["X"], dtype=np.float32)


def original_group_folds(y: np.ndarray, groups: np.ndarray, seed: int = SEED):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups=groups))


def write_metric_table(prediction_map, y, evaluation, path):
    rows = [
        {"evaluation": evaluation, "model": name, **binary_metrics(y, probability)}
        for name, probability in prediction_map.items()
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def run_sequence_audit(dev, test, baseline, X_local_dev, X_context_dev, X_local_test, X_context_test):
    nearest_path = TABLE_DIR / "test_to_development_nearest_alignment.csv"
    if nearest_path.exists():
        nearest = pd.read_csv(nearest_path)
    else:
        started = time.time()
        nearest = nearest_development_neighbors(
            dev.sequence.tolist(),
            test.sequence.tolist(),
            development_ids=dev.header.tolist(),
            test_ids=test.header.tolist(),
            development_labels=dev.label.tolist(),
            test_labels=test.label.tolist(),
            coverage_floor=0.80,
        )
        nearest.to_csv(nearest_path, index=False)
        print(f"Test-to-development alignment audit completed in {time.time() - started:.1f} seconds.")

    identity_summary = identity_sensitivity_summary(
        nearest,
        baseline["test_y"],
        baseline["anoxfuse_test"],
        identity_cutoffs=(0.40, 0.50, 0.60),
        n_bootstrap=2500,
        random_state=SEED,
    )
    identity_summary.to_csv(TABLE_DIR / "identity_threshold_bootstrap_summary.csv", index=False)

    combined = pd.concat(
        [
            dev.assign(original_split="development"),
            test.assign(original_split="released_test"),
        ],
        ignore_index=True,
    )
    membership_path = TABLE_DIR / "whole_pool_similarity_clusters.csv"
    edges_path = TABLE_DIR / "whole_pool_similarity_edges.csv"
    groups_path = CACHE_DIR / "whole_pool_similarity_groups.npy"
    if membership_path.exists() and groups_path.exists() and edges_path.exists():
        membership = pd.read_csv(membership_path)
        edges = pd.read_csv(edges_path)
        groups = np.load(groups_path)
    else:
        started = time.time()
        graph = build_similarity_components(
            combined.sequence.tolist(),
            record_ids=combined.header.tolist(),
            labels=combined.label.tolist(),
            identity_threshold=0.60,
            coverage_threshold=0.80,
            return_edges=True,
        )
        groups = graph["groups"]
        membership = graph["membership"].copy()
        membership["original_split"] = combined.original_split.to_numpy()
        membership["source"] = combined.source.to_numpy()
        edges = graph["edges"]
        membership.to_csv(membership_path, index=False)
        graph["components"].to_csv(TABLE_DIR / "whole_pool_similarity_component_summary.csv", index=False)
        edges.to_csv(edges_path, index=False)
        np.save(groups_path, groups)
        print(f"Whole-pool clustering completed in {time.time() - started:.1f} seconds.")

    y_all = combined.label.to_numpy(dtype=int)
    fold_id, fold_audit = assign_strict_group_folds(
        y_all, groups, n_splits=5, shuffle=False
    )
    fold_audit["fold"] = fold_audit["fold"] + 1
    fold_audit.to_csv(TABLE_DIR / "strict_cluster_disjoint_fold_audit.csv", index=False)
    assignments = membership.copy()
    assignments["strict_fold"] = fold_id + 1
    assignments.to_csv(TABLE_DIR / "strict_cluster_disjoint_assignments.csv", index=False)
    strict_folds = [
        (np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold))
        for fold in range(5)
    ]
    X_local_all = np.vstack([X_local_dev, X_local_test])
    X_context_all = np.vstack([X_context_dev, X_context_test])
    print("Running whole-pool cluster-disjoint nested comparison...")
    strict = nested_fold_predictions(
        X_local_all,
        X_context_all,
        y_all,
        groups,
        strict_folds,
        SEED,
        include_concat=True,
    )
    strict_metrics = write_metric_table(
        strict.predictions,
        y_all,
        "whole-pool 60% identity / 80% coverage cluster-disjoint OOF",
        TABLE_DIR / "strict_cluster_disjoint_metrics.csv",
    )
    pd.DataFrame(strict.fold_rows).to_csv(
        TABLE_DIR / "strict_cluster_disjoint_fold_metrics.csv", index=False
    )
    pd.DataFrame(strict.fusion_rows).to_csv(
        TABLE_DIR / "strict_cluster_disjoint_fusion_parameters.csv", index=False
    )
    prediction_frame = combined[["header", "sequence", "length", "label", "original_split"]].copy()
    prediction_frame["strict_fold"] = fold_id + 1
    prediction_frame["similarity_cluster"] = groups
    for name, probability in strict.predictions.items():
        prediction_frame[name] = probability
    prediction_frame.to_csv(TABLE_DIR / "strict_cluster_disjoint_predictions.csv", index=False)
    return nearest, identity_summary, membership, fold_audit, strict_metrics


def exact_length_matched_indices(frame: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    retained = []
    for length, group in frame.groupby("length", sort=True):
        positive = group.index[group.label == 1].to_numpy()
        negative = group.index[group.label == 0].to_numpy()
        size = min(len(positive), len(negative))
        if size:
            retained.extend(rng.choice(positive, size, replace=False).tolist())
            retained.extend(rng.choice(negative, size, replace=False).tolist())
    return np.asarray(sorted(retained), dtype=int)


def run_training_confound_sensitivities(
    dev,
    test,
    groups,
    X_local_dev,
    X_context_dev,
    X_local_test,
    X_context_test,
    baseline,
):
    matched_rows = []
    matched_predictions = []
    for offset, seed in enumerate((42, 43, 44, 45, 46)):
        retained = exact_length_matched_indices(dev, seed)
        if len(retained) != 912:
            raise RuntimeError(f"Expected 912 matched development rows, observed {len(retained)}.")
        y = dev.label.to_numpy(dtype=int)[retained]
        retained_groups = groups[retained]
        folds = original_group_folds(y, retained_groups, seed)
        result = nested_fold_predictions(
            X_local_dev[retained],
            X_context_dev[retained],
            y,
            retained_groups,
            folds,
            seed,
            include_concat=False,
        )
        test_predictions, parameters = fit_full_and_predict(
            X_local_dev[retained],
            X_context_dev[retained],
            y,
            X_local_test,
            X_context_test,
            result.predictions,
            seed,
            include_concat=False,
        )
        for model_name, probability in result.predictions.items():
            matched_rows.append(
                {
                    "seed": seed,
                    "evaluation": "matched-development grouped OOF",
                    "model": model_name,
                    "training_n": len(retained),
                    **binary_metrics(y, probability),
                }
            )
        for model_name, probability in test_predictions.items():
            matched_rows.append(
                {
                    "seed": seed,
                    "evaluation": "untouched released test after matched training",
                    "model": model_name,
                    "training_n": len(retained),
                    **binary_metrics(test.label.to_numpy(dtype=int), probability),
                    **parameters,
                }
            )
            matched_predictions.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "test_index": np.arange(len(test)),
                        "label": test.label.to_numpy(dtype=int),
                        "model": model_name,
                        "probability": probability,
                    }
                )
            )
    matched_metrics = pd.DataFrame(matched_rows)
    matched_metrics.to_csv(TABLE_DIR / "matched_training_seed_metrics.csv", index=False)
    pd.concat(matched_predictions, ignore_index=True).to_csv(
        TABLE_DIR / "matched_training_test_predictions.csv", index=False
    )
    summarize_repeated_metrics(matched_metrics).to_csv(
        TABLE_DIR / "matched_training_summary.csv", index=False
    )

    descriptors = peptide_descriptor_table(dev)
    feature_columns = [
        column for column in descriptors.columns
        if column == "length" or column.startswith("AAC_")
        or column in {"charge_pH7", "gravy_hydrophobicity", "molecular_weight"}
    ]
    propensity = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    ).fit(descriptors[feature_columns], dev.label)
    positive_probability = propensity.predict_proba(descriptors[feature_columns])[:, 1]
    weights = np.where(dev.label.to_numpy(dtype=int) == 1, 1 - positive_probability, positive_probability)
    descriptors["propensity_positive"] = positive_probability
    descriptors["overlap_weight"] = weights
    descriptors.to_csv(TABLE_DIR / "development_confound_descriptors_and_weights.csv", index=False)
    folds = original_group_folds(dev.label.to_numpy(dtype=int), groups, SEED)
    unweighted_oof = nested_fold_predictions(
        X_local_dev,
        X_context_dev,
        dev.label.to_numpy(dtype=int),
        groups,
        folds,
        SEED,
        include_concat=False,
    )
    weighted_test, parameters = fit_full_and_predict(
        X_local_dev,
        X_context_dev,
        dev.label.to_numpy(dtype=int),
        X_local_test,
        X_context_test,
        unweighted_oof.predictions,
        SEED,
        include_concat=False,
        sample_weight=weights,
    )
    weighted_rows = []
    for model_name, probability in weighted_test.items():
        weighted_rows.append(
            {
                "evaluation": "untouched released test after overlap-weighted full-development training",
                "model": model_name,
                **binary_metrics(test.label, probability),
                **parameters,
            }
        )
    weighted_metrics = pd.DataFrame(weighted_rows)
    weighted_metrics.to_csv(TABLE_DIR / "overlap_weighted_training_metrics.csv", index=False)
    weighted_prediction_frame = pd.DataFrame(
        {"test_index": np.arange(len(test)), "label": test.label.to_numpy(dtype=int)}
    )
    for name, values in weighted_test.items():
        weighted_prediction_frame[name] = values
    weighted_prediction_frame.to_csv(
        TABLE_DIR / "overlap_weighted_training_test_predictions.csv", index=False
    )
    return matched_metrics, weighted_metrics


def create_figures(identity_summary, repeated_summary, calibration_bins, context_matrix, dev):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

    selected_metrics = identity_summary[
        identity_summary["metric"].isin(["roc_auc", "average_precision", "mcc"])
    ].copy()
    label_map = {"roc_auc": "ROC-AUC", "average_precision": "AUPRC", "mcc": "MCC"}
    selected_metrics["metric"] = selected_metrics["metric"].map(label_map)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for metric, subset in selected_metrics.groupby("metric"):
        ax.errorbar(
            subset["identity_cutoff"] * 100,
            subset["estimate"],
            yerr=[subset["estimate"] - subset["ci_low"], subset["ci_high"] - subset["estimate"]],
            marker="o",
            capsize=3,
            label=metric,
        )
    ax.set_xlabel("Maximum qualifying identity to development set (%)")
    ax.set_ylabel("Score")
    ax.set_title("AnOxFuse on increasingly sequence-independent test subsets")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sequence_identity_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    test_summary = repeated_summary[repeated_summary.evaluation == "released test sensitivity"].copy()
    test_summary = test_summary.sort_values("AUC_mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(test_summary.model, test_summary.AUC_mean, xerr=test_summary.AUC_sd, color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Released-test ROC-AUC across five development split seeds")
    ax.set_xlim(max(0.90, float(test_summary.AUC_mean.min() - 0.02)), 1.0)
    ax.set_title("Fusion sensitivity comparison")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fusion_method_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    for ax, evaluation in zip(axes, ["Similarity-grouped development OOF", "Released independent test"]):
        subset = calibration_bins[
            (calibration_bins.evaluation == evaluation)
            & (calibration_bins.scheme == "equal_frequency")
        ]
        ax.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
        ax.plot(subset.mean_probability, subset.observed_rate, marker="o", color="#6f4aa8")
        ax.set_title(evaluation)
        ax.set_xlabel("Mean predicted probability")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Observed positive fraction")
    fig.suptitle("AnOxFuse calibration (10 equal-frequency bins)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_equal_frequency.png", bbox_inches="tight")
    plt.close(fig)

    tsne_cache = CACHE_DIR / "development_context_tsne_length.npz"
    if tsne_cache.exists():
        with np.load(tsne_cache) as payload:
            coordinates = payload["coordinates"]
    else:
        reduced = PCA(n_components=50, random_state=SEED).fit_transform(context_matrix)
        coordinates = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            max_iter=1500,
            random_state=SEED,
        ).fit_transform(reduced)
        np.savez_compressed(tsne_cache, coordinates=coordinates)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    first = axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=dev.label, cmap="coolwarm", s=9, alpha=0.7)
    axes[0].set_title("Class label")
    second = axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=dev.length, cmap="viridis", s=9, alpha=0.7)
    axes[1].set_title("Peptide length")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(second, ax=axes[1], label="Residues")
    fig.suptitle("Frozen peptide-adapted ESM-2 development representation")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "context_tsne_class_and_length.png", bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    verify_input_manifest(ROOT)
    started = time.time()
    print("Loading the immutable published baseline and cached representations...")
    dev, test = load_released_frames(ROOT / "data")
    with np.load(ROOT / "baseline" / "anoxfuse_main_run.npz") as payload:
        baseline = {key: payload[key].copy() for key in payload.files}
    X_local_dev = load_npz_X(ROOT / "cache" / "dev_ecfp4_count.npz")
    X_local_test = load_npz_X(ROOT / "cache" / "test_ecfp4_count.npz")
    X_context_dev = load_npz_X(ROOT / "cache" / "dev_esm2_meanmax.npz")
    X_context_test = load_npz_X(ROOT / "cache" / "test_esm2_meanmax.npz")
    original_groups = np.load(ROOT / "cache" / "groups_published_rapidfuzz60.npy")

    np.testing.assert_array_equal(dev.label.to_numpy(dtype=int), baseline["dev_y"])
    np.testing.assert_array_equal(test.label.to_numpy(dtype=int), baseline["test_y"])
    np.testing.assert_allclose(X_context_dev, baseline["dev_context"], rtol=0, atol=0)
    np.testing.assert_allclose(X_context_test, baseline["test_context"], rtol=0, atol=0)
    headline = binary_metrics(baseline["test_y"], baseline["anoxfuse_test"])
    if abs(headline["AUC"] - 0.9828508771929825) > 1e-12:
        raise RuntimeError("Published baseline guard failed.")
    published_map_dev = {
        "AnOxFuse": baseline["anoxfuse_oof"],
        "Molecular ECFP4 branch": baseline["local_oof"],
        "Contextual peptide-adapted ESM-2 branch": baseline["context_oof"],
    }
    published_map_test = {
        "AnOxFuse": baseline["anoxfuse_test"],
        "Molecular ECFP4 branch": baseline["local_test"],
        "Contextual peptide-adapted ESM-2 branch": baseline["context_test"],
    }
    write_metric_table(
        published_map_test,
        baseline["test_y"],
        "immutable released-test benchmark",
        TABLE_DIR / "published_baseline_metrics.csv",
    )

    aodb = pd.read_csv(ROOT / "external_source" / "AODB_PROTEIN.csv")
    print("Running label, length, confound, shuffle-geometry, calibration, and bootstrap audits...")
    bias_outputs = run_bias_statistics_suite(
        dev,
        test,
        development_predictions=published_map_dev,
        test_predictions=published_map_test,
        main_model_name="AnOxFuse",
        output_dir=TABLE_DIR,
        aodb_sequences=aodb,
        seed=SEED,
        matching_reps=2500,
        bootstrap_reps=2500,
        shuffle_reps=10,
    )

    print("Running strict sequence-independence analyses...")
    nearest, identity_summary, membership, strict_fold_audit, strict_metrics = run_sequence_audit(
        dev,
        test,
        baseline,
        X_local_dev,
        X_context_dev,
        X_local_test,
        X_context_test,
    )

    print("Running repeated grouped fusion comparisons across five split seeds...")
    repeated_metrics, repeated_fold_metrics, repeated_parameters, repeated_predictions = repeated_grouped_comparison(
        X_local_dev,
        X_context_dev,
        baseline["dev_y"],
        original_groups,
        X_local_test,
        X_context_test,
        baseline["test_y"],
        seeds=(42, 43, 44, 45, 46),
    )
    repeated_summary = summarize_repeated_metrics(repeated_metrics)
    repeated_metrics.to_csv(TABLE_DIR / "repeated_grouped_fusion_metrics.csv", index=False)
    repeated_fold_metrics.to_csv(TABLE_DIR / "repeated_grouped_fusion_fold_metrics.csv", index=False)
    repeated_parameters.to_csv(TABLE_DIR / "repeated_grouped_fusion_parameters.csv", index=False)
    repeated_predictions.to_csv(TABLE_DIR / "repeated_grouped_fusion_test_predictions.csv", index=False)
    repeated_summary.to_csv(TABLE_DIR / "repeated_grouped_fusion_summary.csv", index=False)

    frozen_equal_probability = 0.5 * (baseline["local_test"] + baseline["context_test"])
    branch_delta = paired_bootstrap_delta(
        baseline["test_y"],
        baseline["anoxfuse_test"],
        baseline["context_test"],
        reps=2500,
        seed=SEED + 401,
        first_name="AnOxFuse",
        second_name="Contextual peptide-adapted ESM-2 branch",
    )
    average_delta = paired_bootstrap_delta(
        baseline["test_y"],
        baseline["anoxfuse_test"],
        frozen_equal_probability,
        reps=2500,
        seed=SEED + 402,
        first_name="AnOxFuse nested logistic fusion",
        second_name="Equal probability mean",
    )
    pd.concat([branch_delta, average_delta], ignore_index=True).to_csv(
        TABLE_DIR / "paired_fusion_differences.csv", index=False
    )

    print("Running simple sequence and chemical representation controls...")
    control_folds = original_group_folds(baseline["dev_y"], original_groups, SEED)
    control_metrics, control_fold_metrics, control_predictions = run_simple_representation_controls(
        dev.sequence,
        test.sequence,
        baseline["dev_y"],
        baseline["test_y"],
        X_local_dev,
        X_local_test,
        control_folds,
        seed=SEED,
    )
    control_metrics.to_csv(TABLE_DIR / "simple_representation_control_metrics.csv", index=False)
    control_fold_metrics.to_csv(TABLE_DIR / "simple_representation_control_fold_metrics.csv", index=False)
    control_predictions.to_csv(TABLE_DIR / "simple_representation_control_test_predictions.csv", index=False)

    print("Running matched-development and overlap-weighted training sensitivities...")
    matched_metrics, weighted_metrics = run_training_confound_sensitivities(
        dev,
        test,
        original_groups,
        X_local_dev,
        X_context_dev,
        X_local_test,
        X_context_test,
        baseline,
    )

    print("Creating reviewer figures...")
    create_figures(
        identity_summary,
        repeated_summary,
        bias_outputs["calibration_bins"],
        X_context_dev,
        dev,
    )

    method_manifest = {
        "analysis": "AnOxFuse reviewer revision CPU suite",
        "published_baseline_policy": "The frozen released-test prediction vector remains the headline benchmark. Stricter analyses are additional sensitivity analyses.",
        "published_released_test_auc": headline["AUC"],
        "seed": SEED,
        "bootstrap_repetitions": 2500,
        "repeated_grouped_seeds": [42, 43, 44, 45, 46],
        "alignment": {
            "algorithm": "Smith-Waterman local alignment",
            "library": "parasail 1.3.4",
            "matrix": "BLOSUM62",
            "gap_open": 10,
            "gap_extension": 1,
            "identity": "exact residue matches divided by alignment columns, including gaps",
            "coverage": "aligned non-gap residues divided by each full sequence length; both must be at least 0.80",
            "cluster_edge": "identity at least 0.60 and both coverages at least 0.80",
        },
        "model": {
            "molecular": "RDKit MolFromFASTA ECFP4 count, radius 2, 2048 dimensions, LightGBM 500 estimators",
            "context": "jiahuizhang/esm-150m-peptide-fine-tune revision 6d8cebf, frozen last hidden state, residue-only mean plus maximum pooling, StandardScaler plus L2 logistic regression",
            "fusion": "logistic regression on clipped branch logits, epsilon 1e-6",
            "threshold": 0.5,
        },
        "effective_model_parameters": {
            "molecular_lightgbm": estimator_parameter_manifest(lgbm_factory(SEED)),
            "context_logistic_pipeline": estimator_parameter_manifest(context_factory(SEED)),
        },
        "hardware": "CPU analyses use frozen GPU-derived embeddings. No new GPU computation is performed in this suite.",
        "runtime_seconds": time.time() - started,
        "python": platform.python_version(),
    }
    save_json(method_manifest, OUTPUT_ROOT / "method_manifest.json")
    summary = {
        "published_test_auc": headline["AUC"],
        "published_test_mcc": headline["MCC"],
        "identity_sensitivity_summary_rows": len(identity_summary),
        "whole_pool_clusters": int(membership.group_index.nunique()),
        "repeated_comparison_rows": len(repeated_metrics),
        "matched_training_metric_rows": len(matched_metrics),
        "overlap_weighted_metric_rows": len(weighted_metrics),
        "matched_development_training_n_per_seed": 912,
        "runtime_seconds": time.time() - started,
    }
    save_json(summary, OUTPUT_ROOT / "completion_summary.json")
    print(json.dumps(summary, indent=2))
    print("CPU reviewer revision completed:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
