from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def markdown(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


LOCATE = r'''
from pathlib import Path
import json, os, sys, zipfile

def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)

def complete_bundle(path):
    manifest_path = path / "input_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return all((path / relative_path).is_file() for relative_path in manifest["files"])
    except Exception:
        return False

candidates = [
    Path.cwd() / "AnOxFuse_Reviewer_Revision_Bundle",
    Path("/data/AnOxFuse_Reviewer_Revision_Bundle"),
    Path.cwd(),
]
ROOT = next((path.resolve() for path in candidates if complete_bundle(path)), None)
if ROOT is None:
    archives = [Path.cwd() / "AnOxFuse_Reviewer_Revision_Bundle.zip", Path("/data/AnOxFuse_Reviewer_Revision_Bundle.zip")]
    archive = next((path for path in archives if path.is_file()), None)
    if archive is None:
        raise RuntimeError("Upload the complete AnOxFuse_Reviewer_Revision_Bundle folder or ZIP.")
    destination = archive.parent
    safe_extract(archive, destination)
    ROOT = destination / "AnOxFuse_Reviewer_Revision_Bundle"
    if not complete_bundle(ROOT):
        if complete_bundle(destination):
            ROOT = destination
        else:
            matches = [path.parent for path in destination.glob("*/input_manifest.json") if complete_bundle(path.parent)]
            if len(matches) != 1:
                raise RuntimeError("Could not locate a complete extracted revision bundle.")
            ROOT = matches[0]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Revision bundle:", ROOT)
'''


cpu_cells = [
    markdown(
        """# AnOxFuse reviewer revision: CPU analyses

This notebook preserves the published AnOxFuse released-test predictions and runs the reviewer analyses that use cached ECFP4 and frozen peptide-adapted ESM-2 representations. It does not require a GPU. Stricter results are reported as sensitivity analyses and do not replace the released benchmark."""
    ),
    code(LOCATE),
    code(
        r'''
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements_cpu.txt")])
print("Dependencies installed. If this cell changed NumPy or scikit-learn in an already active kernel, restart the kernel once before continuing.")
'''
    ),
    code(
        r'''
import importlib, platform
required = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "rdkit", "Bio", "parasail", "matplotlib"]
versions = {}
for module_name in required:
    module = importlib.import_module(module_name)
    versions[module_name] = getattr(module, "__version__", "not reported")
print("Python", platform.python_version())
print(versions)
'''
    ),
    code(
        r'''
import runpy
runpy.run_path(str(ROOT / "run_cpu_revision.py"), run_name="__main__")
'''
    ),
    code(
        r'''
import pandas as pd
display(pd.read_csv(ROOT / "revision_outputs" / "cpu" / "tables" / "published_baseline_metrics.csv").round(4))
display(pd.read_csv(ROOT / "revision_outputs" / "cpu" / "tables" / "identity_threshold_bootstrap_summary.csv").query("metric in ['roc_auc','average_precision','mcc']").round(4))
display(pd.read_csv(ROOT / "revision_outputs" / "cpu" / "tables" / "strict_cluster_disjoint_metrics.csv").round(4))
display(pd.read_csv(ROOT / "revision_outputs" / "cpu" / "tables" / "repeated_grouped_fusion_summary.csv").round(4))
'''
    ),
]

gpu_cells = [
    markdown(
        """# AnOxFuse reviewer revision: GPU completion

This fresh-device notebook performs only the reviewer analyses that require new contextual embeddings: unique non-original residue shuffles and similarity-filtered AOPP and AnOxPP external evaluation. The published AnOxFuse result remains frozen. The final contextual encoder is peptide-adapted ESM-2 150M, not ESM-C or ESM++."""
    ),
    code(LOCATE),
    code(
        r'''
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements_gpu.txt")])
print("Non-torch dependencies installed. This notebook intentionally does not install or replace PyTorch.")
'''
    ),
    code(
        r'''
import importlib, platform
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Rent an NVIDIA machine with a driver-compatible PyTorch template.")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
'''
    ),
    code(
        r'''
import runpy
runpy.run_path(str(ROOT / "run_gpu_revision.py"), run_name="__main__")
'''
    ),
    code(
        r'''
import pandas as pd
display(pd.read_csv(ROOT / "revision_outputs" / "gpu" / "tables" / "shuffle_summary.csv").round(4))
display(pd.read_csv(ROOT / "revision_outputs" / "gpu" / "tables" / "external_dataset_audit.csv"))
display(pd.read_csv(ROOT / "revision_outputs" / "gpu" / "tables" / "external_protocol_metrics.csv").query("evaluation == 'external test'").round(4))
print("Download this file before deleting the rented instance:")
print(ROOT / "revision_outputs" / "AnOxFuse_GPU_Reviewer_Completion_Outputs.zip")
'''
    ),
]

(ROOT / "AnOxFuse_Reviewer_Revision_CPU.ipynb").write_text(
    json.dumps(notebook(cpu_cells), indent=1), encoding="utf-8"
)
(ROOT / "AnOxFuse_Reviewer_Revision_GPU.ipynb").write_text(
    json.dumps(notebook(gpu_cells), indent=1), encoding="utf-8"
)
print("Created both revision notebooks.")
