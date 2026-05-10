"""
preprocess.py
-------------
Shared preprocessing pipeline for SVM, MLP, and GCN models.

All sklearn fit calls happen ONLY on training data, then transform
is applied to test data — zero data leakage.

To swap in real ABIDE data, implement load_real_data() and call it
instead of generate_dataset() in get_splits().
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Tuple, List

from synthetic_data import generate_dataset, dataset_to_arrays, SubjectData


# ── data container returned to all downstream models ─────────────────────────

@dataclass
class ProcessedSplit:
    # PCA-reduced features (for SVM and MLP)
    X_train_pca:  np.ndarray   # (N_train, n_components)
    X_test_pca:   np.ndarray   # (N_test,  n_components)

    # raw FC vectors after standardization but before PCA (for ablation)
    X_train_raw:  np.ndarray   # (N_train, 19900)
    X_test_raw:   np.ndarray   # (N_test,  19900)

    # labels
    y_cls_train:  np.ndarray   # (N_train,) int  — ASD classification
    y_cls_test:   np.ndarray   # (N_test,)  int
    y_age_train:  np.ndarray   # (N_train,) float — brain age regression
    y_age_test:   np.ndarray   # (N_test,)  float

    # subject objects (GCN needs fc_matrix; also useful for inspection)
    subjects_train: List[SubjectData]
    subjects_test:  List[SubjectData]

    # fitted transformers (needed to inverse-transform predictions)
    scaler: StandardScaler
    pca:    PCA

    # metadata
    n_components:   int
    variance_ratio: float   # cumulative explained variance kept


def preprocess(
    dataset: list,
    pca_variance: float = 0.95,
    test_size: float    = 0.20,
    random_seed: int    = 42,
) -> ProcessedSplit:
    """
    Full preprocessing pipeline:
      1. Stratified 80/20 train/test split (stratified by label + site)
      2. StandardScaler fit on train, applied to train and test
      3. PCA fit on train (retaining `pca_variance` of variance), applied to both

    Parameters
    ----------
    dataset      : list of SubjectData from synthetic_data or real loader
    pca_variance : cumulative explained variance to retain (default 0.95)
    test_size    : fraction of data held out for testing
    random_seed  : for reproducibility
    """
    X, y_cls, y_age, sites = dataset_to_arrays(dataset)

    # ── 1. stratified split ────────────────────────────────────────────────
    # Combine label + site into a single stratification key so both dimensions
    # are balanced across train and test.
    strat_key = np.array([f"{lbl}_{site}" for lbl, site in zip(y_cls, sites)])

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    train_idx, test_idx = next(sss.split(X, strat_key))

    X_train, X_test       = X[train_idx],     X[test_idx]
    y_cls_train, y_cls_test = y_cls[train_idx], y_cls[test_idx]
    y_age_train, y_age_test = y_age[train_idx], y_age[test_idx]
    subjects_train = [dataset[i] for i in train_idx]
    subjects_test  = [dataset[i] for i in test_idx]

    # ── 2. standardize (fit on train only) ────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 3. PCA (fit on train only) ────────────────────────────────────────
    pca = PCA(n_components=pca_variance, random_state=random_seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    pca_cluster = PCA(n_components=min(100, X_train_pca.shape[1]), random_state=random_seed)
    X_train_pca = pca_cluster.fit_transform(X_train_pca)
    X_test_pca  = pca_cluster.transform(X_test_pca)

    n_components   = pca.n_components_
    variance_ratio = float(pca.explained_variance_ratio_.sum())

    print(f"  Train / Test  : {len(train_idx)} / {len(test_idx)} subjects")
    print(f"  PCA kept      : {n_components} components "
          f"({variance_ratio*100:.1f}% variance)")
    print(f"  ASD in train  : {y_cls_train.sum()} "
          f"({y_cls_train.mean()*100:.1f}%)")
    print(f"  ASD in test   : {y_cls_test.sum()} "
          f"({y_cls_test.mean()*100:.1f}%)")

    return ProcessedSplit(
        X_train_pca    = X_train_pca.astype(np.float32),
        X_test_pca     = X_test_pca.astype(np.float32),
        X_train_raw    = X_train.astype(np.float32),
        X_test_raw     = X_test.astype(np.float32),
        y_cls_train    = y_cls_train,
        y_cls_test     = y_cls_test,
        y_age_train    = y_age_train,
        y_age_test     = y_age_test,
        subjects_train = subjects_train,
        subjects_test  = subjects_test,
        scaler         = scaler,
        pca            = pca,
        n_components   = n_components,
        variance_ratio = variance_ratio,
    )


def load_real_data(data_dir: str = "data/roi_timeseries",
                   pheno_path: str = "data/Phenotypic_V1_0b_preprocessed1.csv",
                   max_mean_fd: float = 0.5,
                   min_timepoints: int = 50) -> list:
    """
    Load real ABIDE I preprocessed data (C-PAC, CC200, filt_global).

    Parameters
    ----------
    data_dir       : folder containing *_rois_cc200.1D files
    pheno_path     : path to Phenotypic_V1_0b_preprocessed1.csv
    max_mean_fd    : motion exclusion threshold (framewise displacement)
    min_timepoints : minimum usable time points after scrubbing

    Returns
    -------
    List of SubjectData objects (same format as synthetic_data.py)
    """
    import os
    import pandas as pd
    import numpy as np
    from synthetic_data import SubjectData

    N_ROIS = 200

    # ── 1. load phenotypic CSV ────────────────────────────────────────────
    print(f"  Loading phenotypic data from {pheno_path}...")
    pheno = pd.read_csv(pheno_path)

    # normalise column names (strip whitespace)
    pheno.columns = pheno.columns.str.strip()

    # ── 2. apply QC filters ───────────────────────────────────────────────
    before = len(pheno)

    # motion threshold
    if "func_mean_fd" in pheno.columns:
        pheno = pheno[pheno["func_mean_fd"] <= max_mean_fd]

    # minimum time points: func_num_fd = number of frames ABOVE fd threshold,
    # so usable frames ≈ total_frames - func_num_fd. We keep subjects where
    # enough frames remain. Use a simple proxy: drop top-fd subjects.
    # (Real frame count isn't directly in pheno; we'll verify after loading.)
    after = len(pheno)
    print(f"  Subjects after motion QC: {after} (excluded {before - after})")

    # ── 3. load .1D files ─────────────────────────────────────────────────
    dataset   = []
    skipped   = 0
    loaded    = 0

    for _, row in pheno.iterrows():
        file_id = str(row["FILE_ID"]).strip()
        fname   = f"{file_id}_rois_cc200.1D"
        fpath   = os.path.join(data_dir, fname)

        if not os.path.exists(fpath):
            skipped += 1
            continue

        # load time series — .1D files are whitespace-separated, possibly
        # with comment lines starting with # at the top
        try:
            ts = np.loadtxt(fpath, comments="#")   # shape (T, 200)
        except Exception as e:
            print(f"    Warning: could not load {fname}: {e}")
            skipped += 1
            continue

        # handle edge cases
        if ts.ndim == 1:
            skipped += 1
            continue
        if ts.shape[1] != N_ROIS:
            # some files are transposed (200, T) — fix it
            if ts.shape[0] == N_ROIS:
                ts = ts.T
            else:
                skipped += 1
                continue

        # minimum time points check
        if ts.shape[0] < min_timepoints:
            skipped += 1
            continue

        # ── compute FC matrix ─────────────────────────────────────────
        # corrcoef expects (N_ROIS, T) — transpose ts
        fc_matrix = np.corrcoef(ts.T)              # (200, 200)
        fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
        np.fill_diagonal(fc_matrix, 1.0)
        fc_matrix = np.clip(fc_matrix, -1.0, 1.0)

        # upper-triangle feature vector
        idx       = np.triu_indices(N_ROIS, k=1)
        fc_vector = fc_matrix[idx]                 # (19900,)

        # ── labels & metadata ─────────────────────────────────────────
        # DX_GROUP: 1 = ASD, 2 = neurotypical control → remap to 1/0
        dx     = int(row["DX_GROUP"])
        label  = 1 if dx == 1 else 0

        age    = float(row["AGE_AT_SCAN"]) if "AGE_AT_SCAN" in row else 0.0
        site   = str(row["SITE_ID"]).strip() if "SITE_ID" in row else "UNKNOWN"

        dataset.append(SubjectData(
            subject_id = file_id,
            site       = site,
            age        = age,
            label      = label,
            fc_vector  = fc_vector.astype(np.float32),
            timeseries = ts.astype(np.float32),
            fc_matrix  = fc_matrix.astype(np.float32),
        ))
        loaded += 1

    print(f"  Successfully loaded : {loaded} subjects")
    print(f"  Skipped (no file / bad data): {skipped}")
    asd_count  = sum(s.label == 1 for s in dataset)
    ctrl_count = sum(s.label == 0 for s in dataset)
    print(f"  ASD: {asd_count}  |  Control: {ctrl_count}")

    if len(dataset) == 0:
        raise RuntimeError(
            "No subjects loaded. Check that data_dir points to your .1D files "
            "and that FILE_ID values in the CSV match the filenames."
        )

    return dataset



if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    print("Generating synthetic dataset...")
    dataset = generate_dataset()
    print("Running preprocessing pipeline...")
    split = preprocess(dataset)
    print(f"\nX_train_pca shape : {split.X_train_pca.shape}")
    print(f"X_test_pca  shape : {split.X_test_pca.shape}")
    print("Preprocessing OK.")
