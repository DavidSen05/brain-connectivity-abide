"""
Download ABIDE I preprocessed CC200 ROI time series data.
Pipeline: C-PAC | Strategy: filt_global | Derivative: rois_cc200

Steps:
  1. Downloads the phenotypic/QC spreadsheet
  2. Filters subjects by motion QC (mean_fd <= 0.5)
  3. Downloads .1D time series files for passing subjects
"""

import os
import urllib.request
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR       = "data/roi_timeseries"   # where .1D files are saved
PHENO_PATH    = "data/Phenotypic_V1_0b_preprocessed1.csv"
MAX_MEAN_FD   = 0.5                     # motion exclusion threshold (mm)
MIN_TIMEPOINTS = 50                     # minimum usable time points after scrubbing

PHENO_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Phenotypic_V1_0b_preprocessed1.csv"
)
S3_TEMPLATE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
    "Outputs/cpac/filt_global/rois_cc200/{file_id}_rois_cc200.1D"
)
# ─────────────────────────────────────────────────────────────────────────────


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return  # skip already downloaded
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"  ✗ {os.path.basename(dest_path)} — {e}")


def main():
    # 1. Download phenotypic spreadsheet
    if not os.path.exists(PHENO_PATH):
        print("Downloading phenotypic spreadsheet...")
        os.makedirs(os.path.dirname(PHENO_PATH), exist_ok=True)
        urllib.request.urlretrieve(PHENO_URL, PHENO_PATH)
        print(f"  Saved to {PHENO_PATH}\n")

    # 2. Load and inspect
    df = pd.read_csv(PHENO_PATH)
    print(f"Total subjects in spreadsheet: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # 3. Apply QC filters
    #    func_mean_fd  = mean framewise displacement
    #    func_num_fd   = number of frames exceeding FD threshold (proxy for usable TPs)
    before = len(df)
    df = df[df["func_mean_fd"] <= MAX_MEAN_FD]
    df = df[df["func_num_fd"] <= (df["func_num_fd"].max() - MIN_TIMEPOINTS)]  # rough proxy
    after = len(df)
    print(f"Subjects after motion QC: {after} (excluded {before - after})")

    # Diagnostic breakdown
    n_asd  = (df["DX_GROUP"] == 1).sum()
    n_ctrl = (df["DX_GROUP"] == 2).sum()
    print(f"  ASD: {n_asd}  |  Control: {n_ctrl}\n")

    # 4. Download .1D files
    print(f"Downloading CC200 ROI time series to {OUT_DIR}/...")
    os.makedirs(OUT_DIR, exist_ok=True)

    for _, row in df.iterrows():
        file_id   = row["FILE_ID"]
        url       = S3_TEMPLATE.format(file_id=file_id)
        dest_path = os.path.join(OUT_DIR, f"{file_id}_rois_cc200.1D")
        download_file(url, dest_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
