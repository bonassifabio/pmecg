"""Integration test that fetches PTB-XL records and saves deterministic PNG and PDF plot outputs."""

import matplotlib

matplotlib.use("Agg")

import hashlib
import io
import os
from datetime import datetime

import pandas as pd
import pytest
from ptbxl_helper import get_ptbxl_data

from pmecg import template_factory
from pmecg.plot import ECGInformation, ECGPlotter

pytestmark = pytest.mark.integration

ECG_IDS = [1, 2, 3, 4, 5]
CONFIGURATIONS = ["1x3", "2x6", "4x3"]
OUTPUT_ROOT = "example/outputs"

# Fixed metadata removes time-varying fields so identical figures produce identical bytes.
_EPOCH = datetime(1970, 1, 1, 0, 0, 0)
_STABLE_METADATA: dict[str, dict] = {
    "png": {"Software": "pmecg", "Creation Time": _EPOCH.isoformat()},
    "pdf": {"Creator": "pmecg", "Producer": "pmecg", "CreationDate": _EPOCH},
}


def _figure_bytes(fig, fmt: str) -> bytes:
    """Render *fig* to bytes with fixed metadata so the result is deterministic."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight", metadata=_STABLE_METADATA[fmt])
    return buf.getvalue()


def _save_if_changed(fig, path: str, fmt: str) -> bool:
    """Write a figure only when the newly rendered bytes differ from the file already on disk."""
    new_bytes = _figure_bytes(fig, fmt)
    new_hash = hashlib.md5(new_bytes).hexdigest()

    if os.path.exists(path):
        with open(path, "rb") as fh:
            if hashlib.md5(fh.read()).hexdigest() == new_hash:
                return False

    with open(path, "wb") as fh:
        fh.write(new_bytes)
    return True


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
            _save_if_changed(fig, path, ext)
            assert os.path.exists(path), f"Expected output file not found: {path}"
            assert os.path.getsize(path) > 0, f"Output file is empty: {path}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
