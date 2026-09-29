# Additional analyses

Code, data and results for the analyses added during peer review. The original independent-test predictions are preserved in [reviewer_revision/baseline/](reviewer_revision/baseline/).

- [Final tables and figures](final_results/) contain the results used in the reviewer response.
- [CPU notebook](reviewer_revision/AnOxFuse_Reviewer_Revision_CPU.ipynb): sequence-similarity splits, calibration, length controls, simple baselines and confidence intervals.
- [GPU notebook](reviewer_revision/AnOxFuse_Reviewer_Revision_GPU.ipynb): composition-preserving sequence shuffles and external evaluation on AOPP and AnOxPP, with overlap and similarity filtering.

The [reviewer_revision/](reviewer_revision/) folder is self-contained, including cached features and completed outputs. Keep its internal paths when copying it to another machine.

From this directory, run the CPU analysis with:

```bash
python -m pip install -r reviewer_revision/requirements_cpu.txt
python reviewer_revision/run_cpu_revision.py
```

For the GPU analysis, keep the machine's CUDA-compatible PyTorch installation, install `reviewer_revision/requirements_gpu.txt`, and run the GPU notebook. The saved run used an RTX 5090, PyTorch 2.11.0+cu128 and peptide-adapted ESM-2 revision `6d8cebf`.

The [positive-label source checks](final_results/tables/positive_label_source_spotcheck.csv) link seven sequences to experimental reports; they do not establish the provenance of every dataset entry. Class 0 remains an antioxidant-unannotated background class.
