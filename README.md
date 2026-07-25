# AnOxFuse

![AnOxFuse architecture](AnOxFuse.png)

**AnOxFuse** is a dual-branch antioxidant peptide classifier that combines a local biochemical pathway using ECFP4 count fingerprints and LightGBM with a contextual sequence pathway built from frozen, peptide-adapted ESM-2 embeddings, mean and max pooling, and a logistic probe. The two branch logits are combined through leakage-safe nested logistic fusion trained from similarity-grouped out-of-fold predictions, with final performance evaluated on the released independent test set. The repository provides the clean training and inference workflow in [`AnOxFuse_model.ipynb`](AnOxFuse_model.ipynb) and expects `remaining_positive.fasta`, `remaining_negative.fasta`, and `independent_test_cleaned.fasta` inside the `data/` directory.
