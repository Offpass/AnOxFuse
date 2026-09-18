from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import time
import warnings
from collections import Counter
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import parasail
from lightgbm import LGBMClassifier
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anox_revision_common import (
    EPS,
    SEED,
    binary_metrics,
    bootstrap_metric_summary,
    estimator_parameter_manifest,
    load_released_frames,
    paired_bootstrap_delta,
    read_fasta,
    records_to_frame,
    save_json,
    set_seed,
    verify_input_manifest,
)
from anox_revision_controls import handcrafted_matrix
from anox_revision_models import context_factory, lgbm_factory
from anox_revision_sequence import build_similarity_components, smith_waterman_stats

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

MODEL_ID = "jiahuizhang/esm-150m-peptide-fine-tune"
MODEL_REVISION = "6d8cebf"
OUTPUT_ROOT = ROOT / "revision_outputs" / "gpu"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
CACHE_DIR = OUTPUT_ROOT / "cache"
for directory in (OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def sequence_hash(sequences) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(str(sequence).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def load_npz_X(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["X"], dtype=np.float32)


def save_npz_atomic(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def possible_nonoriginal_permutations(sequence: str) -> int:
    total = factorial(len(sequence))
    for count in Counter(sequence).values():
        total //= factorial(count)
    return max(0, total - 1)


def multiset_permutations(sequence: str):
    counts = Counter(sequence)
    alphabet = sorted(counts)
    target_length = len(sequence)

    def recurse(prefix):
        if len(prefix) == target_length:
            yield "".join(prefix)
            return
        for residue in alphabet:
            if counts[residue]:
                counts[residue] -= 1
                prefix.append(residue)
                yield from recurse(prefix)
                prefix.pop()
                counts[residue] += 1

    yield from recurse([])


def unique_nonoriginal_shuffles(sequence: str, peptide_index: int, wanted: int = 10):
    possible = possible_nonoriginal_permutations(sequence)
    target = min(wanted, possible)
    if target == 0:
        return [], possible
    rng = np.random.default_rng(SEED + 1009 * (peptide_index + 1))
    seen = {sequence}
    output = []
    for _ in range(max(200, target * 100)):
        candidate = "".join(rng.permutation(list(sequence)))
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
            if len(output) == target:
                break
    if len(output) < target:
        for candidate in multiset_permutations(sequence):
            if candidate not in seen:
                seen.add(candidate)
                output.append(candidate)
                if len(output) == target:
                    break
    if len(output) != target or sequence in output or len(output) != len(set(output)):
        raise RuntimeError("Unique-shuffle construction failed.")
    if any(Counter(candidate) != Counter(sequence) for candidate in output):
        raise RuntimeError("A shuffle changed peptide composition.")
    return output, possible


def build_shuffle_table(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in test.reset_index(drop=True).iterrows():
        variants, possible = unique_nonoriginal_shuffles(row.sequence, index)
        for shuffle_index, variant in enumerate(variants):
            rows.append(
                {
                    "test_index": index,
                    "header": row.header,
                    "label": int(row.label),
                    "shuffle_index": shuffle_index,
                    "original_sequence": row.sequence,
                    "shuffled_sequence": variant,
                    "possible_nonoriginal_permutations": possible,
                    "hamming_distance": sum(a != b for a, b in zip(row.sequence, variant)),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 2719 or frame.test_index.nunique() != 301:
        raise RuntimeError(
            f"Expected 2,719 unique shuffles for 301 peptides, observed {len(frame)} for {frame.test_index.nunique()}."
        )
    return frame


def fingerprint_matrix(sequences, cache_name: str) -> np.ndarray:
    sequences = list(map(str, sequences))
    path = CACHE_DIR / f"{cache_name}_{sequence_hash(sequences)}.npz"
    if path.exists():
        return load_npz_X(path)
    rows = []
    for sequence in sequences:
        molecule = Chem.MolFromFASTA(sequence)
        if molecule is None:
            raise ValueError(f"RDKit could not parse peptide sequence: {sequence}")
        rows.append(FINGERPRINT_GENERATOR.GetCountFingerprintAsNumPy(molecule).astype(np.float32))
    matrix = np.vstack(rows)
    save_npz_atomic(path, X=matrix, sequences=np.asarray(sequences, dtype="U"))
    return matrix


def extract_esm_embeddings(sequences, cache_name: str, batch_size: int = 256) -> np.ndarray:
    sequences = list(map(str, sequences))
    path = CACHE_DIR / f"{cache_name}_{MODEL_REVISION}_{sequence_hash(sequences)}.npz"
    if path.exists():
        with np.load(path) as payload:
            cached_sequences = payload["sequences"].astype(str).tolist()
            if cached_sequences != sequences:
                raise RuntimeError(f"Sequence order changed for cache {path.name}.")
            return np.asarray(payload["X"], dtype=np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "New peptide-adapted ESM-2 embeddings require an NVIDIA CUDA GPU. "
            "The published baseline remains available in the CPU notebook."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda").eval()
    special_ids = torch.tensor(tokenizer.all_special_ids, dtype=torch.long, device="cuda")
    output = []
    start = 0
    current_batch = batch_size
    while start < len(sequences):
        batch = sequences[start : start + current_batch]
        try:
            tokens = tokenizer(batch, return_tensors="pt", padding=True)
            tokens = {key: value.to("cuda") for key, value in tokens.items()}
            input_ids = tokens["input_ids"]
            attention = tokens["attention_mask"].bool()
            special = (input_ids.unsqueeze(-1) == special_ids).any(-1)
            residue_mask = attention & ~special
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype):
                hidden = model(**tokens).last_hidden_state
            mean = (hidden * residue_mask.unsqueeze(-1)).sum(1) / residue_mask.sum(1, keepdim=True)
            maximum = hidden.masked_fill(~residue_mask.unsqueeze(-1), -torch.inf).max(1).values
            output.append(torch.cat([mean, maximum], dim=1).float().cpu().numpy())
            start += len(batch)
            print(f"Embedded {start:,}/{len(sequences):,} sequences")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            current_batch //= 2
            if current_batch < 1:
                raise
            print("Reduced embedding batch size to", current_batch)
    matrix = np.vstack(output).astype(np.float32)
    save_npz_atomic(
        path,
        X=matrix,
        sequences=np.asarray(sequences, dtype="U"),
        model_id=np.asarray([MODEL_ID]),
        model_revision=np.asarray([MODEL_REVISION]),
    )
    del model, tokenizer
    torch.cuda.empty_cache()
    return matrix


def load_external_frames():
    workbook = pd.read_excel(
        ROOT / "external_source" / "Antiox_dataset.xls",
        sheet_name="Antiox_train",
        engine="xlrd",
    ).dropna(subset=["Sequence", "Label"])
    train = pd.DataFrame(
        {
            "header": [f"xls_{index}" for index in range(len(workbook))],
            "sequence": workbook.Sequence.astype(str).str.upper().str.strip(),
            "label": workbook.Label.astype(int),
            "source": "Antiox_train_xls",
        }
    )
    train["length"] = train.sequence.str.len()
    train = train[
        train.sequence.map(lambda value: bool(value) and set(value) <= set("ACDEFGHIKLMNPQRSTVWY"))
    ].drop_duplicates("sequence").reset_index(drop=True)

    aopp = records_to_frame(
        read_fasta(ROOT / "external_source" / "Total-test.fasta"),
        source="AOPP_test",
    ).drop_duplicates("sequence").reset_index(drop=True)
    positive = read_fasta(ROOT / "external_source" / "test_AnOxPs.txt")
    negative = read_fasta(ROOT / "external_source" / "test_non-AnOxPs.txt")
    anoxpp = pd.concat(
        [
            records_to_frame(positive, [1] * len(positive), "AnOxPP_test"),
            records_to_frame(negative, [0] * len(negative), "AnOxPP_test"),
        ],
        ignore_index=True,
    ).drop_duplicates("sequence").reset_index(drop=True)
    expected = {
        "train": (2077, 1020, 1057),
        "AOPP": (606, 414, 192),
        "AnOxPP": (424, 212, 212),
    }
    observed = {
        "train": (len(train), int(train.label.sum()), int((1 - train.label).sum())),
        "AOPP": (len(aopp), int(aopp.label.sum()), int((1 - aopp.label).sum())),
        "AnOxPP": (len(anoxpp), int(anoxpp.label.sum()), int((1 - anoxpp.label).sum())),
    }
    if observed != expected:
        raise RuntimeError(f"External dataset counts changed: {observed}")
    return train, {"AOPP": aopp, "AnOxPP": anoxpp}


def filter_external_training(train: pd.DataFrame, test: pd.DataFrame, dataset_name: str):
    cache = TABLE_DIR / f"{dataset_name.lower()}_similarity_filter_audit.csv"
    if cache.exists():
        audit = pd.read_csv(cache)
        expected_index = np.arange(len(train), dtype=int)
        cache_is_valid = (
            len(audit) == len(train)
            and "train_index" in audit
            and "train_sequence" in audit
            and np.array_equal(audit["train_index"].to_numpy(dtype=int), expected_index)
            and audit["train_sequence"].astype(str).tolist() == train.sequence.astype(str).tolist()
        )
        if not cache_is_valid:
            cache.unlink()
            return filter_external_training(train, test, dataset_name)
        removed = audit["removed"].map(
            lambda value: value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"true", "1", "yes"}
        )
        kept = train.loc[~removed.to_numpy(dtype=bool)].reset_index(drop=True)
        return kept, audit
    rows = []
    test_sequences = test.sequence.tolist()
    for train_index, row in train.reset_index(drop=True).iterrows():
        best_any = None
        best_hit = None
        for test_index, test_sequence in enumerate(test_sequences):
            stats = smith_waterman_stats(row.sequence, test_sequence)
            candidate = {
                "nearest_test_index": test_index,
                "nearest_test_sequence": test_sequence,
                "identity": stats["identity"],
                "coverage": stats["coverage"],
                "alignment_score": stats["score"],
            }
            any_key = (
                float(stats["identity"]) * float(stats["coverage"]),
                float(stats["identity"]),
                float(stats["coverage"]),
                int(stats["score"]),
            )
            if best_any is None or any_key > best_any[0]:
                best_any = (any_key, candidate)
            if stats["identity"] >= 0.60 - 1e-12 and stats["coverage"] >= 0.80 - 1e-12:
                hit_key = (float(stats["identity"]), float(stats["coverage"]), int(stats["score"]))
                if best_hit is None or hit_key > best_hit[0]:
                    best_hit = (hit_key, candidate)
        chosen = (best_hit or best_any)[1]
        rows.append(
            {
                "dataset": dataset_name,
                "train_index": train_index,
                "train_sequence": row.sequence,
                "label": int(row.label),
                "removed": best_hit is not None,
                **chosen,
            }
        )
        if (train_index + 1) % 250 == 0:
            print(f"{dataset_name} similarity audit: {train_index + 1:,}/{len(train):,}")
    audit = pd.DataFrame(rows)
    audit.to_csv(cache, index=False)
    kept = train.loc[~audit.removed.to_numpy()].reset_index(drop=True)
    return kept, audit


def exact_overlap_filter(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    test_sequences = set(test.sequence)
    return train.loc[~train.sequence.isin(test_sequences)].reset_index(drop=True)


def group_folds(frame: pd.DataFrame, dataset_name: str, protocol: str):
    group_path = CACHE_DIR / f"{dataset_name.lower()}_{protocol}_groups.npy"
    audit_path = TABLE_DIR / f"{dataset_name.lower()}_{protocol}_group_audit.csv"
    if group_path.exists():
        groups = np.load(group_path)
    else:
        graph = build_similarity_components(
            frame.sequence.tolist(),
            record_ids=frame.header.tolist(),
            labels=frame.label.tolist(),
            identity_threshold=0.60,
            coverage_threshold=0.80,
            return_edges=False,
        )
        groups = graph["groups"]
        np.save(group_path, groups)
        graph["components"].to_csv(
            TABLE_DIR / f"{dataset_name.lower()}_{protocol}_component_summary.csv", index=False
        )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(frame)), frame.label, groups=groups))
    rows = []
    for fold, (train_index, valid_index) in enumerate(folds):
        rows.append(
            {
                "dataset": dataset_name,
                "protocol": protocol,
                "fold": fold + 1,
                "train_n": len(train_index),
                "valid_n": len(valid_index),
                "valid_positive": int(frame.label.iloc[valid_index].sum()),
                "valid_negative": int(len(valid_index) - frame.label.iloc[valid_index].sum()),
                "valid_groups": int(len(np.unique(groups[valid_index]))),
            }
        )
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    return groups, folds


def crossfit_tabular(X, y, folds, factory):
    probability = np.full(len(y), np.nan)
    for fold, (train_index, valid_index) in enumerate(folds):
        model = factory(SEED + fold)
        model.fit(X[train_index], y[train_index])
        probability[valid_index] = model.predict_proba(X[valid_index])[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("Cross-fitting left missing predictions.")
    return probability


def logistic_factory(seed: int):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2", C=1.0, solver="liblinear", max_iter=4000, random_state=seed
                ),
            ),
        ]
    )


def rf_factory(seed: int):
    return RandomForestClassifier(
        n_estimators=500, max_features="sqrt", random_state=seed, n_jobs=-1
    )


def evaluate_external_protocol(
    dataset_name,
    protocol,
    train,
    test,
    embedding_map,
):
    groups, folds = group_folds(train, dataset_name, protocol)
    y_train = train.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)
    train_sequences = train.sequence.tolist()
    test_sequences = test.sequence.tolist()
    X_train_length = train.length.to_numpy(dtype=np.float32).reshape(-1, 1)
    X_test_length = test.length.to_numpy(dtype=np.float32).reshape(-1, 1)
    X_train_hand = handcrafted_matrix(train_sequences, include_dpc=True)
    X_test_hand = handcrafted_matrix(test_sequences, include_dpc=True)
    X_train_local = fingerprint_matrix(train_sequences, f"{dataset_name}_{protocol}_train_ecfp4")
    X_test_local = fingerprint_matrix(test_sequences, f"{dataset_name}_test_ecfp4")
    X_train_context = np.vstack([embedding_map[sequence] for sequence in train_sequences])
    X_test_context = np.vstack([embedding_map[sequence] for sequence in test_sequences])

    specifications = {
        "Length LR": (X_train_length, X_test_length, logistic_factory),
        "AAC plus DPC plus length RF": (X_train_hand, X_test_hand, rf_factory),
        "Molecular ECFP4 branch": (X_train_local, X_test_local, lgbm_factory),
        "Contextual peptide-adapted ESM-2 branch": (X_train_context, X_test_context, context_factory),
    }
    oof = {}
    test_predictions = {}
    for model_name, (X_train, X_test, factory) in specifications.items():
        oof[model_name] = crossfit_tabular(X_train, y_train, folds, factory)
        fitted = factory(SEED).fit(X_train, y_train)
        test_predictions[model_name] = fitted.predict_proba(X_test)[:, 1]

    local_oof = oof["Molecular ECFP4 branch"]
    context_oof = oof["Contextual peptide-adapted ESM-2 branch"]
    local_test = test_predictions["Molecular ECFP4 branch"]
    context_test = test_predictions["Contextual peptide-adapted ESM-2 branch"]
    oof["Equal probability mean"] = 0.5 * (local_oof + context_oof)
    test_predictions["Equal probability mean"] = 0.5 * (local_test + context_test)
    oof["Equal logit mean"] = expit(
        0.5 * (logit(np.clip(local_oof, EPS, 1 - EPS)) + logit(np.clip(context_oof, EPS, 1 - EPS)))
    )
    test_predictions["Equal logit mean"] = expit(
        0.5 * (logit(np.clip(local_test, EPS, 1 - EPS)) + logit(np.clip(context_test, EPS, 1 - EPS)))
    )
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(
        np.column_stack(
            [
                logit(np.clip(local_oof, EPS, 1 - EPS)),
                logit(np.clip(context_oof, EPS, 1 - EPS)),
            ]
        ),
        y_train,
    )
    test_predictions["Cross-fitted logistic fusion"] = meta.predict_proba(
        np.column_stack(
            [
                logit(np.clip(local_test, EPS, 1 - EPS)),
                logit(np.clip(context_test, EPS, 1 - EPS)),
            ]
        )
    )[:, 1]

    metric_rows = []
    oof_frames = []
    test_frames = []
    bootstrap_frames = []
    for model_name, probability in oof.items():
        metric_rows.append(
            {
                "dataset": dataset_name,
                "protocol": protocol,
                "evaluation": "training grouped OOF",
                "model": model_name,
                **binary_metrics(y_train, probability),
            }
        )
        oof_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "protocol": protocol,
                    "row_index": np.arange(len(train)),
                    "sequence": train.sequence,
                    "label": y_train,
                    "model": model_name,
                    "probability": probability,
                }
            )
        )
    for model_name, probability in test_predictions.items():
        metric_rows.append(
            {
                "dataset": dataset_name,
                "protocol": protocol,
                "evaluation": "external test",
                "model": model_name,
                **binary_metrics(y_test, probability),
            }
        )
        test_frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset_name,
                    "protocol": protocol,
                    "row_index": np.arange(len(test)),
                    "sequence": test.sequence,
                    "label": y_test,
                    "model": model_name,
                    "probability": probability,
                }
            )
        )
        bootstrap = bootstrap_metric_summary(
            y_test,
            probability,
            reps=2500,
            seed=SEED,
            analysis=f"{dataset_name} | {protocol} | {model_name}",
        )
        bootstrap["dataset"] = dataset_name
        bootstrap["protocol"] = protocol
        bootstrap["model"] = model_name
        bootstrap_frames.append(bootstrap)
    for comparison_name in ("Contextual peptide-adapted ESM-2 branch", "AAC plus DPC plus length RF"):
        paired = paired_bootstrap_delta(
            y_test,
            test_predictions["Cross-fitted logistic fusion"],
            test_predictions[comparison_name],
            reps=2500,
            seed=SEED + 100,
            first_name="Cross-fitted logistic fusion",
            second_name=comparison_name,
        )
        paired["dataset"] = dataset_name
        paired["protocol"] = protocol
        paired.to_csv(
            TABLE_DIR / f"{dataset_name.lower()}_{protocol}_{comparison_name.lower().replace(' ', '_')}_paired_delta.csv",
            index=False,
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(oof_frames, ignore_index=True),
        pd.concat(test_frames, ignore_index=True),
        pd.concat(bootstrap_frames, ignore_index=True),
    )


def score_unique_shuffles(dev, test, shuffles, shuffle_context):
    X_local_dev = load_npz_X(ROOT / "cache" / "dev_ecfp4_count.npz")
    X_context_dev = load_npz_X(ROOT / "cache" / "dev_esm2_meanmax.npz")
    X_local_test = load_npz_X(ROOT / "cache" / "test_ecfp4_count.npz")
    X_context_test = load_npz_X(ROOT / "cache" / "test_esm2_meanmax.npz")
    X_local_shuffle = fingerprint_matrix(
        shuffles.shuffled_sequence.tolist(), "main_test_unique_shuffles_ecfp4"
    )
    with np.load(ROOT / "baseline" / "anoxfuse_main_run.npz") as payload:
        baseline = {key: payload[key].copy() for key in payload.files}
    local_model = lgbm_factory(SEED).fit(X_local_dev, baseline["dev_y"])
    context_model = context_factory(SEED).fit(X_context_dev, baseline["dev_y"])
    local_original = local_model.predict_proba(X_local_test)[:, 1]
    context_original = context_model.predict_proba(X_context_test)[:, 1]
    local_shuffle = local_model.predict_proba(X_local_shuffle)[:, 1]
    context_shuffle = context_model.predict_proba(shuffle_context)[:, 1]
    coefficients = baseline["final_meta_coef"]
    intercept = float(baseline["final_meta_intercept"][0])

    def frozen_fusion(local_probability, context_probability):
        matrix = np.column_stack(
            [
                logit(np.clip(local_probability, EPS, 1 - EPS)),
                logit(np.clip(context_probability, EPS, 1 - EPS)),
            ]
        )
        return expit(intercept + matrix @ coefficients)

    original_refit = {
        "Molecular ECFP4 branch": local_original,
        "Contextual peptide-adapted ESM-2 branch": context_original,
        "Frozen-coefficient logistic fusion": frozen_fusion(local_original, context_original),
        "Equal probability mean": 0.5 * (local_original + context_original),
        "Equal logit mean": expit(
            0.5
            * (
                logit(np.clip(local_original, EPS, 1 - EPS))
                + logit(np.clip(context_original, EPS, 1 - EPS))
            )
        ),
    }
    shuffled = {
        "Molecular ECFP4 branch": local_shuffle,
        "Contextual peptide-adapted ESM-2 branch": context_shuffle,
        "Frozen-coefficient logistic fusion": frozen_fusion(local_shuffle, context_shuffle),
        "Equal probability mean": 0.5 * (local_shuffle + context_shuffle),
        "Equal logit mean": expit(
            0.5
            * (
                logit(np.clip(local_shuffle, EPS, 1 - EPS))
                + logit(np.clip(context_shuffle, EPS, 1 - EPS))
            )
        ),
    }
    drift = pd.DataFrame(
        [
            {
                "model": "Molecular ECFP4 branch",
                "max_absolute_probability_difference": float(
                    np.max(np.abs(local_original - baseline["local_test"]))
                ),
            },
            {
                "model": "Contextual peptide-adapted ESM-2 branch",
                "max_absolute_probability_difference": float(
                    np.max(np.abs(context_original - baseline["context_test"]))
                ),
            },
            {
                "model": "Frozen-coefficient logistic fusion",
                "max_absolute_probability_difference": float(
                    np.max(
                        np.abs(
                            original_refit["Frozen-coefficient logistic fusion"]
                            - baseline["anoxfuse_test"]
                        )
                    )
                ),
            },
        ]
    )
    drift.to_csv(TABLE_DIR / "shuffle_refit_drift_guard.csv", index=False)
    drift_tolerance = 1e-3
    maximum_drift = float(drift["max_absolute_probability_difference"].max())
    if maximum_drift > drift_tolerance:
        raise RuntimeError(
            f"Frozen-model refit drift {maximum_drift:.6g} exceeds the declared "
            f"tolerance {drift_tolerance:.6g}. Do not score shuffled sequences in this environment."
        )

    individual = shuffles.copy()
    for model_name, probability in shuffled.items():
        individual[model_name] = probability
    individual.to_csv(TABLE_DIR / "shuffle_predictions_individual.csv", index=False)

    eligible_indices = sorted(individual.test_index.unique())
    summary_rows = []
    averaged_frames = []
    for model_name in shuffled:
        averaged = (
            individual.groupby("test_index", as_index=False)
            .agg(
                label=("label", "first"),
                shuffle_count=("shuffle_index", "size"),
                shuffled_mean_probability=(model_name, "mean"),
                shuffled_probability_sd=(model_name, "std"),
            )
        )
        averaged["model"] = model_name
        averaged["original_probability"] = original_refit[model_name][averaged.test_index]
        averaged["probability_change"] = (
            averaged.shuffled_mean_probability - averaged.original_probability
        )
        y = averaged.label.to_numpy(dtype=int)
        original_probability = averaged.original_probability.to_numpy(dtype=float)
        shuffled_probability = averaged.shuffled_mean_probability.to_numpy(dtype=float)
        original_metrics = binary_metrics(y, original_probability)
        shuffled_metrics = binary_metrics(y, shuffled_probability)
        summary_rows.append(
            {
                "model": model_name,
                "eligible_peptides": len(averaged),
                "total_unique_shuffles": len(individual),
                "original_auc_same_eligible_peptides": original_metrics["AUC"],
                "shuffled_mean_auc": shuffled_metrics["AUC"],
                "auc_drop": original_metrics["AUC"] - shuffled_metrics["AUC"],
                "original_mcc": original_metrics["MCC"],
                "shuffled_mean_mcc": shuffled_metrics["MCC"],
                "pearson_probability_correlation": float(
                    np.corrcoef(original_probability, shuffled_probability)[0, 1]
                ),
                "mean_absolute_probability_change": float(
                    np.mean(np.abs(shuffled_probability - original_probability))
                ),
                "mean_change_positive": float(averaged.loc[averaged.label == 1, "probability_change"].mean()),
                "mean_change_negative": float(averaged.loc[averaged.label == 0, "probability_change"].mean()),
            }
        )
        averaged_frames.append(averaged)
    averaged_all = pd.concat(averaged_frames, ignore_index=True)
    averaged_all.to_csv(TABLE_DIR / "shuffle_predictions_per_original.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "shuffle_summary.csv", index=False)
    return summary


def create_gpu_figures(shuffle_summary, external_metrics):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ordered = shuffle_summary.sort_values("auc_drop")
    ax.barh(ordered.model, ordered.auc_drop, color="#d77a28")
    ax.set_xlabel("ROC-AUC reduction after averaging unique sequence shuffles")
    ax.set_title("Sensitivity to residue-order perturbation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "unique_shuffle_auc_drop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    subset = external_metrics[
        (external_metrics.evaluation == "external test")
        & external_metrics.model.isin(
            [
                "AAC plus DPC plus length RF",
                "Molecular ECFP4 branch",
                "Contextual peptide-adapted ESM-2 branch",
                "Cross-fitted logistic fusion",
            ]
        )
    ].copy()
    labels = [f"{row.dataset}\n{row.protocol}\n{row.model}" for row in subset.itertuples()]
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.32 * len(subset))))
    order = np.argsort(subset.AUC.to_numpy())
    ax.barh(np.asarray(labels)[order], subset.AUC.to_numpy()[order], color="#4c78a8")
    ax.set_xlim(max(0.5, float(subset.AUC.min() - 0.08)), 1.0)
    ax.set_xlabel("External-test ROC-AUC")
    ax.set_title("External evaluation before and after sequence-similarity filtering")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "external_similarity_protocol_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    verify_input_manifest(ROOT)
    started = time.time()
    dev, test = load_released_frames(ROOT / "data")
    shuffles = build_shuffle_table(test)
    shuffles.to_csv(TABLE_DIR / "shuffle_sequences.csv", index=False)
    external_train, external_tests = load_external_frames()

    audit_rows = []
    protocol_frames = {}
    for dataset_name, external_test in external_tests.items():
        exact = exact_overlap_filter(external_train, external_test)
        strict, audit = filter_external_training(external_train, external_test, dataset_name)
        expected = {
            "AOPP": {"exact_overlap_only": 1678, "identity60_coverage80": 1345},
            "AnOxPP": {"exact_overlap_only": 1794, "identity60_coverage80": 1487},
        }
        if len(exact) != expected[dataset_name]["exact_overlap_only"]:
            raise RuntimeError(f"{dataset_name} exact-overlap count changed.")
        if len(strict) != expected[dataset_name]["identity60_coverage80"]:
            raise RuntimeError(f"{dataset_name} strict-filter count changed.")
        protocol_frames[(dataset_name, "exact_overlap_only")] = exact
        protocol_frames[(dataset_name, "identity60_coverage80")] = strict
        for protocol, frame in (("exact_overlap_only", exact), ("identity60_coverage80", strict)):
            audit_rows.append(
                {
                    "dataset": dataset_name,
                    "protocol": protocol,
                    "original_training_n": len(external_train),
                    "removed_training_n": len(external_train) - len(frame),
                    "retained_training_n": len(frame),
                    "training_positive": int(frame.label.sum()),
                    "training_negative": int((1 - frame.label).sum()),
                    "test_n": len(external_test),
                    "test_positive": int(external_test.label.sum()),
                    "test_negative": int((1 - external_test.label).sum()),
                }
            )
    pd.DataFrame(audit_rows).to_csv(TABLE_DIR / "external_dataset_audit.csv", index=False)

    new_sequences = shuffles.shuffled_sequence.tolist()
    for frame in [external_train, *external_tests.values()]:
        new_sequences.extend(frame.sequence.tolist())
    unique_new_sequences = list(dict.fromkeys(new_sequences))
    print(f"Embedding {len(unique_new_sequences):,} unique new sequences with {MODEL_ID}@{MODEL_REVISION}.")
    embedding_matrix = extract_esm_embeddings(unique_new_sequences, "reviewer_new_sequence_union")
    embedding_map = dict(zip(unique_new_sequences, embedding_matrix, strict=True))
    shuffle_context = np.vstack([embedding_map[sequence] for sequence in shuffles.shuffled_sequence])
    shuffle_summary = score_unique_shuffles(dev, test, shuffles, shuffle_context)

    metric_frames = []
    oof_frames = []
    prediction_frames = []
    bootstrap_frames = []
    for (dataset_name, protocol), train_frame in protocol_frames.items():
        print(f"Running {dataset_name}: {protocol} with {len(train_frame):,} training peptides")
        outputs = evaluate_external_protocol(
            dataset_name,
            protocol,
            train_frame,
            external_tests[dataset_name],
            embedding_map,
        )
        metric_frames.append(outputs[0])
        oof_frames.append(outputs[1])
        prediction_frames.append(outputs[2])
        bootstrap_frames.append(outputs[3])
    external_metrics = pd.concat(metric_frames, ignore_index=True)
    external_metrics.to_csv(TABLE_DIR / "external_protocol_metrics.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(TABLE_DIR / "external_oof_predictions.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(TABLE_DIR / "external_test_predictions.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(
        TABLE_DIR / "external_bootstrap_summary.csv", index=False
    )
    create_gpu_figures(shuffle_summary, external_metrics)

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        gpu_name = torch_version = cuda_version = None
    manifest = {
        "analysis": "AnOxFuse reviewer GPU completion",
        "published_baseline_policy": "Frozen published predictions are unchanged; all new results are sensitivity analyses.",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_description": "Frozen peptide-adapted ESM-2 150M, last hidden state, residue-only mean plus maximum pooling",
        "not_esm_c": True,
        "not_esm_plusplus": True,
        "seed": SEED,
        "unique_shuffle_target": 10,
        "unique_shuffles_created": len(shuffles),
        "external_protocols": ["exact_overlap_only", "identity60_coverage80"],
        "effective_model_parameters": {
            "molecular_lightgbm": estimator_parameter_manifest(lgbm_factory(SEED)),
            "context_logistic_pipeline": estimator_parameter_manifest(context_factory(SEED)),
            "external_meta_logistic": estimator_parameter_manifest(
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            ),
        },
        "runtime_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda_version,
        "gpu": gpu_name,
    }
    save_json(manifest, OUTPUT_ROOT / "gpu_environment_and_method.json")
    archive_base = OUTPUT_ROOT.parent / "AnOxFuse_GPU_Reviewer_Completion_Outputs"
    archive_file = archive_base.with_suffix(".zip")
    if archive_file.exists():
        archive_file.unlink()
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUT_ROOT)
    print("GPU reviewer work complete. Download before deleting the instance:")
    print(archive_path)


if __name__ == "__main__":
    main()
