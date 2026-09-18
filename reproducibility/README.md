# Reproducibility package

This directory contains the complete computational material prepared during peer review. The published independent-test prediction vector and headline result remain unchanged. New tables are sensitivity analyses or external evaluations and are identified as such.

`reviewer_revision` is a self-contained copy of the executable revision workflow. It includes locked development and test inputs, cached ECFP4 and peptide-adapted ESM-2 representations, the CPU and GPU notebooks, their Python modules, dependency locks, and completed CPU/GPU outputs. The CPU notebook performs integrity checks, strict cluster-disjoint evaluation, repeated grouped comparisons, calibration, length controls, matched-training analyses, simple sequence controls, bootstrap intervals, and figures. The GPU notebook evaluates unique composition-preserving shuffles and the AOPP and AnOxPP external sets after exact-overlap and 60%-identity/80%-coverage filtering.

`final_results` collects the tables and figures used in the reviewer response. `positive_label_source_spotcheck.csv` links seven released positive entries, by exact full sequence, to original experimental reports. This supports the positive-label orientation but is not a complete provenance reconstruction of the anonymous FASTA collection.

Run the local workflow from this directory with:

```bash
python -m pip install -r reviewer_revision/requirements_cpu.txt
python reviewer_revision/run_cpu_revision.py
```

For the GPU workflow, keep the CUDA-compatible PyTorch build supplied by the machine, install `reviewer_revision/requirements_gpu.txt`, and run `reviewer_revision/AnOxFuse_Reviewer_Revision_GPU.ipynb`. The completed GPU outputs in this repository were produced with an RTX 5090, PyTorch 2.11.0+cu128, and the frozen `jiahuizhang/esm-150m-peptide-fine-tune` checkpoint at revision `6d8cebf`.

Class 1 denotes a peptide released as antioxidant. Class 0 is an antioxidant-unannotated background class, not a set of experimentally proven inactive peptides. That limitation applies to the reported discrimination metrics and is addressed with explicit controls rather than hidden by the label wording.
