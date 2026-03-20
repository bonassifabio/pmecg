"""Integration test that fetches PTB-XL records and saves deterministic PNG and PDF plot outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import pandas as pd
import pytest
from output_helpers import save_if_changed
from ptbxl_helper import get_ptbxl_data

from pmecg import LeadsMap, RhythmStripsConfig, template_factory

# PTB-XL uses uppercase "AVR"/"AVL"/"AVF"; map them to canonical "aVR"/"aVL"/"aVF"
_PTBXL_LEADS_MAP = LeadsMap(aVR="AVR", aVL="AVL", aVF="AVF")
from pmecg.plot import ECGInformation, ECGPlotter

pytestmark = pytest.mark.integration

ECG_IDS = [1, 2, 3, 4, 5]
SPEED = 25  # mm/s
CONFIGURATIONS = ["1x3", "2x6", "4x3", "4x3+1", "4x3+3"]
OUTPUT_ROOT = "example/artifacts/no-attention"


@pytest.mark.parametrize("configuration", CONFIGURATIONS)
@pytest.mark.parametrize("ecg_id", ECG_IDS)
# Checks that each sampled PTB-XL record can be plotted and saved as non-empty PNG and PDF outputs.
def test_ptbxl_plot_saved(ecg_id, configuration):
    """Plot a PTB-XL record and save it as PNG and PDF."""
    record, metadata, stats = get_ptbxl_data(ecg_id)
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)

    info = ECGInformation(
        age=metadata["age"],
        sex=metadata["sex"],
        date=metadata["date"],
    )

    out_dir = os.path.join(OUTPUT_ROOT, str(ecg_id))
    os.makedirs(out_dir, exist_ok=True)

    plotter = ECGPlotter(grid_mode="cm", speed=SPEED, print_information=True)
    plot_configuration = template_factory(configuration, df, leads_map=_PTBXL_LEADS_MAP)
    fig = plotter.plot(
        df,
        plot_configuration,
        sampling_frequency=record.fs,
        show=False,
        information=info,
        stats=stats,
    )

    try:
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"{configuration}.{ext}")
            save_if_changed(fig, path, ext)
            assert os.path.exists(path), f"Expected output file not found: {path}"
            assert os.path.getsize(path) > 0, f"Output file is empty: {path}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


RHYTHM_STRIP_LEAD = "II"
RHYTHM_STRIP_SPEED = SPEED / 2  # mm/s — half speed; doubled rhythm strip fills the same width


@pytest.mark.parametrize("ecg_id", ECG_IDS)
def test_ptbxl_rhythm_strip_plot_saved(ecg_id):
    """Plot a PTB-XL record with a 4x3 + full-width rhythm strip layout.

    Lead II is concatenated with itself so the rhythm strip spans twice the normal
    recording duration. Half speed is used so the rhythm strip fits the same physical
    width as the regular leads at full speed.
    """
    record, metadata, stats = get_ptbxl_data(ecg_id)
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)

    ii_signal = record.p_signal[:, list(record.sig_name).index(RHYTHM_STRIP_LEAD)]
    rhythm_strip_df = pd.DataFrame({RHYTHM_STRIP_LEAD: np.concatenate([ii_signal, ii_signal])})

    plot_configuration = template_factory("4x3", df, leads_map=_PTBXL_LEADS_MAP)

    info = ECGInformation(
        age=metadata["age"],
        sex=metadata["sex"],
        date=metadata["date"],
    )

    out_dir = os.path.join(OUTPUT_ROOT, str(ecg_id))
    os.makedirs(out_dir, exist_ok=True)

    plotter = ECGPlotter(grid_mode="cm", speed=SPEED, print_information=True)
    fig = plotter.plot(
        df,
        plot_configuration,
        sampling_frequency=record.fs,
        show=False,
        information=info,
        stats=stats,
        rhythm_strips=RhythmStripsConfig(ecg_data=rhythm_strip_df, speed=RHYTHM_STRIP_SPEED),
    )

    try:
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"4x3-strip.{ext}")
            save_if_changed(fig, path, ext)
            assert os.path.exists(path), f"Expected output file not found: {path}"
            assert os.path.getsize(path) > 0, f"Output file is empty: {path}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
