from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import wfdb

from pmecg import ECGStats

CACHE_DIR = "tests/.ptbxl-cache"
PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
FEATURES_URL = "https://physionet.org/files/ptb-xl-plus/1.0.1/features/unig_features.csv"


def _ensure_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def _get_cached_csv_row(url: str, filename: str, ecg_id: int) -> dict[str, Any]:
    _ensure_cache()
    local_path = os.path.join(CACHE_DIR, filename)

    # Download the full CSV if not cached
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    # Read the CSV and find the row
    df = pd.read_csv(local_path)
    # Ensure ecg_id is treated as int for matching
    row = df[df["ecg_id"] == ecg_id]

    if row.empty:
        return {}

    return row.iloc[0].to_dict()


def _filter_ptbxlplus_features(features_dict: dict[str, Any]) -> ECGStats:
    return ECGStats(
        bpm=features_dict.get("HR__Global"),
        rr_interval_ms=features_dict.get("RR_Mean_Global"),
        hrv_ms=features_dict.get("RR_StdDev_Global"),
        pr_interval_ms=features_dict.get("PR_Int_Global"),
        qrs_duration_ms=features_dict.get("QRS_Dur_Global"),
        qt_interval_ms=features_dict.get("QT_Int_Global"),
        qtc_interval_ms=features_dict.get("QT_IntCorr_Global"),
        p_axis_deg=features_dict.get("P_AxisFront_Global"),
        qrs_axis_deg=features_dict.get("QRS_AxisFront_Global"),
        t_axis_deg=features_dict.get("T_AxisFront_Global"),
    )


def get_ptbxl_data(ecg_id: int, fs: int = 500) -> tuple[wfdb.Record, dict[str, Any], ECGStats]:
    _ensure_cache()

    folder = (ecg_id // 1000) * 1000
    suffix = "hr" if fs == 500 else "lr"
    base = f"{ecg_id:05d}_{suffix}"
    subdir = f"records{fs}/{folder:05d}"

    # 1. Get/Cache Record
    local_record_path = os.path.join(CACHE_DIR, base)
    if not os.path.exists(local_record_path + ".hea"):
        print(f"Downloading record {base}...")
        for ext in [".hea", ".dat"]:
            r = requests.get(f"{PTBXL_BASE_URL}{subdir}/{base}{ext}")
            r.raise_for_status()
            with open(os.path.join(CACHE_DIR, base + ext), "wb") as f:
                f.write(r.content)

    record = wfdb.rdrecord(local_record_path)

    # 2. Get Metadata from ptbxl_database.csv
    db_url = f"{PTBXL_BASE_URL}ptbxl_database.csv"
    meta_row = _get_cached_csv_row(db_url, "ptbxl_database.csv", ecg_id)

    metadata = {
        "date": str(meta_row.get("recording_date", "")),
        "sex": "Female" if meta_row.get("sex") == 1 else "Male",
        "age": int(meta_row.get("age", 0)),
    }

    # 3. Get UNIG Features
    features_row = _get_cached_csv_row(FEATURES_URL, "unig_features.csv", ecg_id)

    return record, metadata, _filter_ptbxlplus_features(features_row)


def get_ptbxl_record(ecg_id: int, fs: int = 500) -> wfdb.Record:
    record, _, _ = get_ptbxl_data(ecg_id, fs)
    return record
