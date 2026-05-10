# Brain Connectivity Subtypes and Disorder Prediction
### A Multi-Model Framework on Resting-State fMRI Data
**David Sen, Rutgers University**

---

## Overview

We propose an integrated pipeline that combines unsupervised brain connectivity clustering with supervised disorder prediction on the ABIDE I dataset. The pipeline:

1. Extracts functional connectivity (FC) matrices from preprocessed BOLD signals
2. Applies PCA + K-Means clustering to discover latent connectivity subtypes
3. Trains and compares three models — SVM, MLP, and GCN — on two tasks:
   - Binary ASD vs. neurotypical classification
   - Brain age regression

---

## Results Summary

| Model | Accuracy | F1 | ROC-AUC | Age MAE |
|-------|----------|----|---------|---------|
| SVM   | 0.665    | 0.646 | **0.761** | 3.40 |
| MLP   | 0.685    | 0.730 | 0.721 | 3.88 |
| GCN   | 0.595    | 0.548 | 0.631 | **3.36** |

---

## Repository Structure

```
brain-connectivity-abide/
│
├── data/                        # Not included — see Data Download below
│   ├── roi_timeseries/          # .1D files from ABIDE I (CC200, C-PAC)
│   └── Phenotypic_V1_0b_preprocessed1.csv
│
├── figures/                     # Generated automatically by main.py
│   ├── fig1_cluster_selection.png
│   ├── fig2_tsne.png
│   ├── fig3_roc_curves.png
│   ├── fig4_confusion_matrix.png
│   └── fig5_age_scatter.png
│
├── results/                     # Generated automatically by main.py
│   ├── classification_results.csv
│   └── regression_results.csv
│
├── synthetic_data.py            # Synthetic ABIDE-like data for pipeline testing
├── preprocess.py                # Preprocessing: standardization, PCA, train/test split
├── cluster.py                   # K-Means clustering, elbow/silhouette, t-SNE
├── svm_model.py                 # SVM classifier and regressor
├── mlp_model.py                 # MLP classifier and regressor (PyTorch)
├── gcn_model.py                 # GCN classifier and regressor (PyTorch Geometric)
├── main.py                      # Master pipeline script
├── download_abide.py            # Script to download ABIDE I data from S3
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/DavidSen05/brain-connectivity-abide.git
cd brain-connectivity-abide
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install PyTorch Geometric** (for GCN support)

First check your PyTorch version:
```bash
python -c "import torch; print(torch.__version__)"
```
Then follow the installation instructions at https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Data Download

This project uses the **ABIDE I** dataset, preprocessed by the Preprocessed Connectomes Project using the C-PAC pipeline, CC200 atlas, and filt_global strategy.

**To download the data, run:**
```bash
python download_abide.py
```

This will:
1. Download the phenotypic CSV with subject demographics and QC metrics
2. Filter subjects by motion (mean framewise displacement ≤ 0.5 mm)
3. Download `.1D` ROI time series files into `data/roi_timeseries/`

The ABIDE data is publicly available but requires downloading from the Amazon S3 bucket hosted by the Preprocessed Connectomes Project. See http://preprocessed-connectomes-project.org/abide/ for more details.

---

## Running the Pipeline

**Option 1 — Quick test with synthetic data (no download needed)**
```bash
python main.py
```
This generates synthetic data mirroring the ABIDE I structure and runs the full pipeline. Was used to verify the code works before downloading the real data. results will not match the paper.

**Option 2 — Full pipeline on real ABIDE data**
```bash
python main.py --real
```

All figures are saved to `figures/` and result tables to `results/`.

---

## Pipeline Details

### Preprocessing (`preprocess.py`)
- Stratified 80/20 train/test split by diagnosis label and acquisition site
- StandardScaler fit on training set only (no data leakage)
- PCA retaining 95% of explained variance, further reduced to 100 components for clustering stability

### Clustering (`cluster.py`)
- K-Means on PCA-reduced FC features
- Optimal k selected by Silhouette Score (k=3 on real data)
- t-SNE visualization colored by cluster and diagnosis

### Models
| Model | Input | Key details |
|-------|-------|-------------|
| SVM | PCA-reduced FC vector | RBF kernel, 5-fold CV hyperparameter tuning |
| MLP | PCA-reduced FC vector | [256,128,64], BCEWithLogitsLoss, weighted sampler |
| GCN | Subject-level brain graph | Top-k (k=10) edges, 204-dim node features, mean+max pooling |

---

## Reproducing Paper Results

```bash
# Download data
python download_abide.py

# Run full pipeline with fixed seed
python main.py --real --seed 42
```

Results will appear in `results/classification_results.csv` and `results/regression_results.csv`. Figures will be saved to `figures/`.

---
