"""Integration test that fetches PTB-XL records and saves deterministic PNG and PDF plot outputs."""

import matplotlib

matplotlib.use("Agg")

import os

import pandas as pd
import pytest
from output_helpers import save_if_changed
from ptbxl_helper import get_ptbxl_data

from pmecg import template_factory
from pmecg.plot import ECGInformation, ECGPlotter

pytestmark = pytest.mark.integration

ECG_IDS = [1, 2, 3, 4, 5]
CONFIGURATIONS = ["1x3", "2x6", "4x3"]
OUTPUT_ROOT = "example/outputs/no-attention"


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

    plotter = ECGPlotter(grid_mode="cm", print_information=True)
    plot_configuration = template_factory(configuration, df, leads_map=None)
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
