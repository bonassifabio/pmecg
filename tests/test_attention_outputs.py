"""Integration test that generates deterministic PTB-XL attention-map outputs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from output_helpers import save_if_changed
from ptbxl_helper import get_ptbxl_data

import pmecg
from pmecg import BackgroundAttentionMap, IntervalAttentionMap, LineColorAttentionMap, template_factory
from pmecg.plot import ECGInformation, ECGPlotter

pytestmark = pytest.mark.integration

ECG_ID = 1
SAMPLING_FREQUENCY = 500
CONFIGURATION_NAME = "4x3"
OUTPUT_ROOT = Path("example/outputs/attention")


def _make_signed_attention(n_samples: int, lead_names: list[str]) -> pd.DataFrame:
    """Create a smooth signed attention map with per-lead phase shifts."""
    t = np.linspace(0.0, 4.0 * np.pi, n_samples)
    attention = {}
    for index, lead_name in enumerate(lead_names):
        phase = index * np.pi / 10.0
        attention[lead_name] = np.sin(t + phase) + 0.35 * np.cos(2.0 * t - phase)
    return pd.DataFrame(attention, columns=lead_names)


def _make_unipolar_attention(n_samples: int, lead_names: list[str]) -> pd.DataFrame:
    """Create a smooth uni-polar attention map in [0, 1] with per-lead phase shifts."""
    t = np.linspace(0.0, 4.0 * np.pi, n_samples)
    attention = {}
    for index, lead_name in enumerate(lead_names):
        phase = index * np.pi / 10.0
        attention[lead_name] = np.clip(np.sin(t + phase), a_min=0.0, a_max=None)
    return pd.DataFrame(attention, columns=lead_names)


def _build_attention_map(kind: str, attention_variant: str, attention_df: pd.DataFrame) -> pmecg.AbstractAttentionMap:
    if attention_variant == "signed":
        common_kwargs = {"data": attention_df, "polarity": "signed", "color": ("blue", "red")}
    elif attention_variant == "positive":
        common_kwargs = {"data": attention_df, "polarity": "positive", "color": "red"}
    else:
        raise ValueError(f"Unsupported attention variant: {attention_variant}")

    if kind == "interval":
        return IntervalAttentionMap(max_attention_mV=0.5, alpha=0.4, **common_kwargs)
    if kind == "line-color":
        return LineColorAttentionMap(**common_kwargs)
    if kind == "background":
        return BackgroundAttentionMap(**common_kwargs)
    raise ValueError(f"Unsupported attention-map kind: {kind}")


@pytest.mark.parametrize(
    ("attention_variant", "attention_factory"),
    [
        ("signed", _make_signed_attention),
        ("positive", _make_unipolar_attention),
    ],
)
@pytest.mark.parametrize("attention_kind", ["interval", "line-color", "background"])
def test_attention_map_plot_saved(attention_kind: str, attention_variant: str, attention_factory) -> None:
    """Plot a PTB-XL record with each public attention-map configuration and save PNG/PDF outputs."""
    record, metadata, stats = get_ptbxl_data(ecg_id=ECG_ID, fs=SAMPLING_FREQUENCY)
    ecg_df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    attention_df = attention_factory(len(ecg_df), list(ecg_df.columns))

    plotter = ECGPlotter(grid_mode="cm", print_information=True)
    plot_configuration = template_factory(CONFIGURATION_NAME, ecg_df, leads_map=None)
    information = ECGInformation(
        age=metadata["age"],
        sex=metadata["sex"],
        date=metadata["date"],
        machine_model=f"PTB-XL attention example ({attention_kind}, {attention_variant})",
    )

    fig = plotter.plot(
        ecg_df,
        plot_configuration,
        sampling_frequency=record.fs,
        show=False,
        information=information,
        stats=stats,
        attention_map=_build_attention_map(attention_kind, attention_variant, attention_df),
    )

    try:
        for ext in ("png", "pdf"):
            path = OUTPUT_ROOT / f"{CONFIGURATION_NAME}-{attention_kind}-{attention_variant}.{ext}"
            save_if_changed(fig, path, ext)
            assert path.exists(), f"Expected output file not found: {path}"
            assert path.stat().st_size > 0, f"Output file is empty: {path}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
