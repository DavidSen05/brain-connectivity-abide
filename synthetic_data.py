"""
synthetic_data.py
-----------------
Generates synthetic data that mirrors the structure of ABIDE I
preprocessed with the CC200 atlas.

When you have the real data, replace load_data() in preprocess.py
with your actual loader — everything downstream stays the same.

Real data download:
  http://preprocessed-connectomes-project.org/abide/
  Pipeline: C-PAC | Atlas: CC200 | Derivative: rois_cc200
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ── constants that match real ABIDE I CC200 ───────────────────────────────────
N_ROIS        = 200    # number of brain parcels (CC200 atlas)
N_TIMEPOINTS  = 78     # minimum scan length after scrubbing
N_FC_FEATURES = N_ROIS * (N_ROIS - 1) // 2   # 19 900 upper-triangle entries
N_SITES       = 17     # acquisition sites in ABIDE I


@dataclass
class SubjectData:
    subject_id:   str
    site:         str
    age:          float          # chronological age (years)
    label:        int            # 1 = ASD, 0 = neurotypical control
    fc_vector:    np.ndarray     # shape (19 900,)  — upper-triangle of FC matrix
    timeseries:   np.ndarray     # shape (78, 200)  — BOLD signal (kept for reference)
    fc_matrix:    np.ndarray     # shape (200, 200) — full symmetric FC matrix


def _make_fc_matrix(label: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a plausible synthetic FC matrix.
    ASD subjects get slightly lower long-range correlations (underconnectivity)
    and slightly higher local correlations (overconnectivity) — a pattern
    reported in the literature (Just et al., 2004).
    """
    # base correlation structure via a random low-rank signal
    signal = rng.standard_normal((N_ROIS, 10))
    base   = signal @ signal.T
    base   = base / (np.max(np.abs(base)) + 1e-8)   # normalise to [-1, 1]

    # add site-specific noise
    noise  = rng.standard_normal((N_ROIS, N_ROIS)) * 0.1
    noise  = (noise + noise.T) / 2

    fc = base + noise

    if label == 1:          # ASD: attenuate long-range, amplify local
        local_mask   = np.abs(np.arange(N_ROIS)[:, None] - np.arange(N_ROIS)[None, :]) < 10
        distant_mask = ~local_mask
        fc[local_mask]   *= 1.05
        fc[distant_mask] *= 0.90

    np.fill_diagonal(fc, 1.0)   # self-correlations are always 1
    fc = np.clip(fc, -1.0, 1.0)
    return fc


def generate_dataset(
    n_subjects: int = 1112,
    asd_fraction: float = 0.485,
    random_seed: int = 42,
) -> list:
    """
    Generate a synthetic ABIDE-like dataset.

    Parameters
    ----------
    n_subjects    : total number of subjects (default matches ABIDE I)
    asd_fraction  : fraction with ASD label  (default matches ABIDE I)
    random_seed   : for reproducibility

    Returns
    -------
    List of SubjectData objects.
    """
    rng      = np.random.default_rng(random_seed)
    n_asd    = int(n_subjects * asd_fraction)
    n_ctrl   = n_subjects - n_asd
    labels   = [1] * n_asd + [0] * n_ctrl
    sites    = [f"SITE_{i+1:02d}" for i in range(N_SITES)]

    dataset = []
    for i, label in enumerate(labels):
        site = sites[i % N_SITES]

        # age: ASD group skews slightly younger on average (reflects ABIDE I)
        age = float(rng.normal(17.0 if label == 1 else 20.0, 8.0))
        age = float(np.clip(age, 7.0, 64.0))

        fc_matrix  = _make_fc_matrix(label, rng)
        # upper-triangle vector (the actual feature used by SVM and MLP)
        idx        = np.triu_indices(N_ROIS, k=1)
        fc_vector  = fc_matrix[idx]

        # synthetic BOLD timeseries (used only to verify GCN pipeline)
        timeseries = rng.standard_normal((N_TIMEPOINTS, N_ROIS)) * 0.5
        # inject FC structure weakly into the timeseries
        timeseries += (fc_matrix @ timeseries.T).T * 0.1

        dataset.append(SubjectData(
            subject_id = f"SUB_{i+1:04d}",
            site       = site,
            age        = age,
            label      = label,
            fc_vector  = fc_vector.astype(np.float32),
            timeseries = timeseries.astype(np.float32),
            fc_matrix  = fc_matrix.astype(np.float32),
        ))

    rng.shuffle(dataset)
    return dataset


def dataset_to_arrays(
    dataset: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Unpack a list of SubjectData into numpy arrays.

    Returns
    -------
    X      : (N, 19900)  FC feature vectors
    y_cls  : (N,)        binary classification labels (ASD=1, ctrl=0)
    y_age  : (N,)        continuous age regression targets
    sites  : list of site strings, length N
    """
    X     = np.stack([s.fc_vector  for s in dataset], axis=0)
    y_cls = np.array([s.label      for s in dataset], dtype=np.int64)
    y_age = np.array([s.age        for s in dataset], dtype=np.float32)
    sites = [s.site                for s in dataset]
    return X, y_cls, y_age, sites


if __name__ == "__main__":
    dataset = generate_dataset()
    X, y_cls, y_age, sites = dataset_to_arrays(dataset)
    print(f"Dataset size  : {len(dataset)} subjects")
    print(f"FC vector dim : {X.shape[1]:,}")
    print(f"ASD subjects  : {y_cls.sum()} ({y_cls.mean()*100:.1f}%)")
    print(f"Age range     : {y_age.min():.1f} – {y_age.max():.1f} years")
    print(f"Sites         : {len(set(sites))}")
    print("Synthetic data generation OK.")
