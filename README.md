# Painting Style Classification — BostonGene IML-2

A multi-task machine learning project that classifies paintings into eight style
categories (*ArtDeco, Cubism, Impressionism, Japonism, Naturalism, Rococo,
cartoon, photo*) and explores their latent structure through self-supervised
learning, variational autoencoders, unsupervised clustering, and gradient boosting.
The dataset contains 1 422 images, is naturally imbalanced, and is split
70/15/15 into train (995) / val (213) / test (214) using stratified sampling.
Three tasks are addressed: (1) supervised classification baselines and SimCLR
self-supervision, (2) VAE-based latent-space learning and unsupervised clustering,
and (3) gradient boosting on the learned embeddings with XAI interpretation.

---

## Results at a Glance

### Task 1 — Supervised Classification

| Model | Accuracy | Macro-F1 | Balanced Acc |
|---|---|---|---|
| SimpleCNN (baseline) | ~0.50 | ~0.45 | - |
| **ResNet18 transfer** | **0.757** | **0.732** | — |
| EfficientNet-B0 transfer | ~0.74 | ~0.72 | — |
| SimCLR linear probe (Task 1b) | 0.388 | 0.361 | 0.381 |

### Task 2 — VAE Latent Space + Clustering

| Model / Metric | Value |
|---|---|
| Pixel VAE (unsupervised, 256-d) | Blurry reconstructions; weak cluster separation |
| DINOv2 feature-space VAE (128-d) — NN head | Acc 0.832 / F1 0.818 / Bal 0.813 |
| DINOv2 feature-space VAE — linear probe ceiling | 0.848 |
| Clustering (GMM on pixel-VAE, PCA-denoised) | ARI 0.036 / NMI 0.084 / Silhouette 0.073 |

### Task 3 — Gradient Boosting on Embeddings

| Model | Accuracy | Macro-F1 | Balanced Acc |
|---|---|---|---|
| **XGBoost** | **0.874** | **0.865** | **0.858** |
| HistGradientBoosting | 0.860 | 0.848 | 0.835 |
| Neural net (VAE head, same embeddings) | 0.832 | 0.818 | 0.813 |
| Boosting on unsupervised pixel-VAE | ~0.44 | ~0.37 | — |

---

## Project Structure

```
painting-style-classification/
├── notebooks/              # Run in order 01 → 05
│   ├── 01_eda_and_split.ipynb
│   ├── 02_train_classifier.ipynb       # Task 1 — supervised + Grad-CAM XAI
│   ├── 02b_self_supervised_simclr.ipynb # Task 1b — SimCLR
│   ├── 03_train_vae_embeddings.ipynb   # Task 2a — pixel VAE
│   ├── 03b_class_regularized_vae_embeddings.ipynb  # Task 2b — DINOv2 VAE
│   ├── 04_clustering_analysis.ipynb    # Task 2c — KMeans + GMM clustering
│   └── 05_boosting_on_embeddings.ipynb # Task 3 — boosting + XAI
├── experiments/            # Superseded draft notebooks (archived)
├── src/
│   ├── config.py           # All paths and constants (TRAIN_DIR, MODELS_DIR, …)
│   ├── data.py             # DataLoaders, transforms, PadToSquare
│   ├── models.py           # SimpleCNN definition
│   └── metrics.py          # Evaluation helpers
├── data/
│   ├── processed/          # Train/val/test ImageFolder splits (gitignored)
│   └── splits/             # CSV manifests + JSON class mappings (tracked)
├── models/                 # Saved .pt checkpoints (gitignored; backed up to Drive)
├── outputs/                # Logs, CSVs, .npz embeddings (gitignored)
├── reports/
│   ├── figures/            # Generated plots (gitignored; regenerate by running notebooks)
│   └── report.md           # Full project report (tracked)
├── requirements.txt
└── README.md
```

---

## Notebook Run Order

```
01_eda_and_split                 # EDA + stratified split → data/processed/
    ↓
02_train_classifier              # SimpleCNN + ResNet18 + EfficientNet + TTA + Grad-CAM
02b_self_supervised_simclr       # SimCLR SSL + linear probe + fine-tune (parallel)
    ↓
03_train_vae_embeddings          # Pixel VAE → selected_vae_embeddings.npz
    ↓
03b_class_regularized_vae_embeddings  # DINOv2 VAE → class_regularized_vae_embeddings.npz
    ↓                                   (depends on 03 for comparison)
04_clustering_analysis           # KMeans + GMM on selected_vae_embeddings.npz (depends on 03)
    ↓
05_boosting_on_embeddings        # XGBoost + HistGBM on class_regularized embeddings (depends on 03b)
```

**Key dependency note**: notebooks 04 and 05 consume `.npz` embedding files produced
by 03 and 03b respectively. Each notebook includes a session-safety cell that
restores the file from a Google Drive backup folder if it is not present locally.

---

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **DINOv2**: loaded at runtime via `torch.hub.load("facebookresearch/dinov2", …)`.
> No extra package needed; outbound internet access required on first run of `03b`.

> **Windows note**: if `xgboost` or `shap` fail to install, both are optional.
> `HistGradientBoostingClassifier` from scikit-learn is the dependency-free fallback
> for gradient boosting; permutation importance from scikit-learn replaces SHAP.

### 3. Place raw image data

Raw images must be in `../images/` relative to this project directory
(i.e., at `Task_IML_2_data/images/<class_name>/<image>.jpg`).
Run `01_eda_and_split.ipynb` to generate the processed split under `data/processed/`.

### 4. Launch Jupyter

```bash
jupyter notebook
```

---

## Running in Google Colab (recommended for GPU training)

1. Upload `bostongene_classifier_bundle.zip` to
   `My Drive/bostongene_project/` on Google Drive.
2. Open any notebook in Colab.
3. Run the first cell — it mounts Drive, extracts the project, and sets `PROJECT_ROOT`.
4. The final cell of each notebook backs up results back to Drive under
   `My Drive/bostongene_project/<stage>_results/`.

---

## Where Data and Large Artifacts Live

| Artifact | Location |
|---|---|
| Raw images | `../images/` (not tracked) |
| Train/val/test splits | `data/processed/` (gitignored; regenerate via NB 01) |
| Split manifests (CSV/JSON) | `data/splits/` (tracked) |
| Model checkpoints (`.pt`) | `models/` (gitignored; backed up to Drive `classifier_results/models/`) |
| Pixel-VAE embeddings | `outputs/selected_vae_embeddings.npz` (gitignored; Drive `vae_results/outputs/`) |
| DINOv2 VAE embeddings | `outputs/class_regularized_vae_embeddings.npz` (gitignored; Drive `class_regularized_vae_results/outputs/`) |
| Figures | `reports/figures/` (gitignored; regenerated by running notebooks in Colab) |

---

## Metrics

Because the dataset is imbalanced, all evaluations report:

- **Macro-F1** — primary metric (equal weight per class)
- **Balanced Accuracy** — macro recall
- **Accuracy** — overall, for reference
- **Confusion Matrix** (normalised)
