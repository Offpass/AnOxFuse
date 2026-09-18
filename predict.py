from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import logit


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
EPS = 1e-6
MODEL_ID = "jiahuizhang/esm-150m-peptide-fine-tune"
MODEL_REVISION = "6d8cebf"
ECFP4_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


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
                raise ValueError("A FASTA sequence appeared before its header.")
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence).upper()))
    if not records:
        raise ValueError(f"No FASTA records found in {path}.")
    invalid = [(header, sequence) for header, sequence in records if not sequence or not set(sequence) <= STANDARD_AA]
    if invalid:
        raise ValueError(f"Only nonempty sequences using the 20 standard amino acids are accepted: {invalid[:3]}")
    return records


def fingerprint_matrix(sequences: list[str]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sequence in sequences:
        molecule = Chem.MolFromFASTA(sequence)
        if molecule is None:
            raise ValueError(f"RDKit could not construct the peptide molecule: {sequence}")
        rows.append(ECFP4_GENERATOR.GetCountFingerprintAsNumPy(molecule).astype(np.float32))
    return np.vstack(rows)


def contextual_matrix(sequences: list[str], device_name: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or a CUDA-enabled runtime.")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    special_ids = torch.tensor(tokenizer.all_special_ids, dtype=torch.long, device=device)
    rows: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True)
        tokens = {name: value.to(device) for name, value in tokens.items()}
        input_ids = tokens["input_ids"]
        attention = tokens["attention_mask"].bool()
        special = (input_ids.unsqueeze(-1) == special_ids).any(-1)
        residue_mask = attention & ~special
        if not residue_mask.any(dim=1).all():
            raise RuntimeError("The tokenizer produced a sequence with no residue tokens.")
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            autocast = torch.amp.autocast("cuda", dtype=dtype)
        else:
            autocast = nullcontext()
        with torch.inference_mode(), autocast:
            hidden = model(**tokens).last_hidden_state
        mean = (hidden * residue_mask.unsqueeze(-1)).sum(1) / residue_mask.sum(1, keepdim=True)
        maximum = hidden.masked_fill(~residue_mask.unsqueeze(-1), -torch.inf).max(1).values
        rows.append(torch.cat([mean, maximum], dim=1).float().cpu().numpy())
    return np.vstack(rows).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict antioxidant-peptide probability with AnOxFuse.")
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file.")
    parser.add_argument("--models", type=Path, default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    records = read_fasta(args.input)
    identifiers = [identifier for identifier, _ in records]
    sequences = [sequence for _, sequence in records]
    local_model = joblib.load(args.models / "anoxfuse_local_model.joblib")
    context_model = joblib.load(args.models / "anoxfuse_context_model.joblib")
    meta_model = joblib.load(args.models / "anoxfuse_meta_model.joblib")

    local_probability = local_model.predict_proba(fingerprint_matrix(sequences))[:, 1]
    context_probability = context_model.predict_proba(
        contextual_matrix(sequences, args.device, args.batch_size)
    )[:, 1]
    fusion_features = np.column_stack(
        [
            logit(np.clip(local_probability, EPS, 1.0 - EPS)),
            logit(np.clip(context_probability, EPS, 1.0 - EPS)),
        ]
    )
    probability = meta_model.predict_proba(fusion_features)[:, 1]
    output = pd.DataFrame(
        {
            "id": identifiers,
            "sequence": sequences,
            "local_probability": local_probability,
            "context_probability": context_probability,
            "anoxfuse_probability": probability,
            "prediction": (probability >= args.threshold).astype(int),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()

