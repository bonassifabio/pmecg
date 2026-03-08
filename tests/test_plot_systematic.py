"""Automated structural tests for ECGPlotter.

Instead of manually inspecting images, these tests introspect the returned
``matplotlib.Figure`` to verify that the plotter's output has the correct
structure for each combination of parameters and configuration.

Strategy
--------
``ECGPlotter.plot()`` returns a ``Figure``.  We can assert:

* **Figure size** — ``fig.get_size_inches()`` must match ``_compute_figure_size()``.
* **Line count** — each row draws 1 signal line + 1 calibration line (if enabled);
  ``grid_mode='cm'`` adds many more lines via ``axvline``/``axhline``.
* **Text content** — lead labels (``show_leads_labels``), calibration "1mV" tags
  (``show_calibration``), and diagnostic / patient / stats strings
  (``print_information``) are all reflected in ``ax.texts``.
* **Time axis visibility** — ``ax.xaxis.get_visible()`` tracks ``show_time_axis``.

All tests use synthetic ECG data so no network access is required.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pmecg.plot import ECGInformation, ECGPlotter, ECGStats
from pmecg.utils.data import TEMPLATE_CONFIGURATIONS
from pmecg.utils.plot import (
    LEFT_MARGIN_MM,
    MM_PER_INCH,
    _adjust_row_distance,
    _compute_figure_size,
    _compute_row_offsets,
)

# ── Shared test data ───────────────────────────────────────────────────────

LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500  # Hz
N_SAMPLES = 1000  # 2 s — short enough to keep tests fast


@pytest.fixture(scope="module")
def ecg_df() -> pd.DataFrame:
    """12-lead synthetic ECG as a DataFrame (no network access)."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_SAMPLES, len(LEAD_NAMES))) * 0.5
    return pd.DataFrame(data, columns=LEAD_NAMES)


def _ax(fig):
    return fig.axes[0]


def _texts(fig) -> list[str]:
    return [t.get_text() for t in _ax(fig).texts]


def _grid_lines(ax):
    """Return (vline_x_positions, hline_y_positions) for all 2-point grid lines on ax.

    ``axvline``/``axhline`` each create a 2-point ``Line2D``:
    * axvline(x): xdata=[x, x]  ydata=[0, 1] (axes coords)
    * axhline(y): xdata=[0, 1]  ydata=[y, y] (data coords)
    """
    two_pt = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
    vx = sorted(ln.get_xdata()[0] for ln in two_pt if ln.get_xdata()[0] == ln.get_xdata()[1])
    hy = sorted(ln.get_ydata()[0] for ln in two_pt if ln.get_ydata()[0] == ln.get_ydata()[1])
    return vx, hy


def _resolve_rows(configuration) -> list[list[str]]:
    """Normalize any configuration format to list[list[str]] (one sublist per row)."""
    if configuration is None:
        return [[lead] for lead in LEAD_NAMES]
    if isinstance(configuration, str):
        if configuration in TEMPLATE_CONFIGURATIONS:
            config = TEMPLATE_CONFIGURATIONS[configuration]
        else:
            # Single lead name string
            config = [configuration]
    else:
        config = configuration
    return [[e] if isinstance(e, str) else e for e in config]


def _should_warn_divisible(configuration, n_samples):
    """Return True if any row in the configuration does not evenly divide n_samples."""
    rows = _resolve_rows(configuration)
    for row in rows:
        if n_samples % len(row) != 0:
            return True
    return False


def maybe_warns_divisible(configuration, n_samples):
    """Context manager to handle the divisibility warning if it's expected."""
    if _should_warn_divisible(configuration, n_samples):
        return pytest.warns(UserWarning, match="is not evenly divisible")
    return nullcontext()


# ── Figure size ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "configuration,n_rows",
    [
        (["I"], 1),
        (["I", "II", "III"], 3),
        ([["I", "II", "III", "AVR", "AVL", "AVF"], ["V1", "V2", "V3", "V4", "V5", "V6"]], 2),
        (None, 12),
    ],
)
def test_figure_size_matches_layout(ecg_df, configuration, n_rows):
    """Figure dimensions must equal _compute_figure_size() for the given layout."""
    plotter = ECGPlotter(grid_mode=None, print_information=False)
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        adjusted_row_distance = _adjust_row_distance(plotter.row_distance, plotter.voltage)

        exp_w, exp_h = _compute_figure_size(
            n_rows,
            N_SAMPLES,
            FS,
            plotter.speed,
            plotter.voltage,
            adjusted_row_distance,
            print_information=False,
        )
        w, h = fig.get_size_inches()
        assert abs(w - exp_w) < 1e-6
        assert abs(h - exp_h) < 1e-6
    finally:
        plt.close(fig)


@pytest.mark.parametrize("print_information", [True, False])
def test_figure_height_grows_with_print_information(ecg_df, print_information):
    """Extra top/bottom margins are added when print_information=True."""
    plotter = ECGPlotter(grid_mode=None, print_information=print_information)
    configuration = ["I", "II"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        adjusted_row_distance = _adjust_row_distance(plotter.row_distance, plotter.voltage)

        exp_w, exp_h = _compute_figure_size(
            2,
            N_SAMPLES,
            FS,
            plotter.speed,
            plotter.voltage,
            adjusted_row_distance,
            print_information=print_information,
        )
        _, h = fig.get_size_inches()
        assert abs(h - exp_h) < 1e-6
    finally:
        plt.close(fig)


# ── Signal / calibration line count ───────────────────────────────────────


@pytest.mark.parametrize(
    "n_rows,show_calibration",
    [
        (1, True),
        (1, False),
        (3, True),
        (3, False),
        (6, False),
    ],
)
def test_line_count(ecg_df, n_rows, show_calibration):
    """Without grid, each row contributes exactly 1 signal line + 1 calibration line (if on)."""
    configuration = LEAD_NAMES[:n_rows]
    plotter = ECGPlotter(grid_mode=None, print_information=False, show_calibration=show_calibration)
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        expected = n_rows * (2 if show_calibration else 1)
        assert len(_ax(fig).lines) == expected
    finally:
        plt.close(fig)


# ── Grid ───────────────────────────────────────────────────────────────────


def test_grid_minor_spacing(ecg_df):
    """Adjacent grid lines (minor ticks) are spaced exactly 1 mm apart in both axes."""
    plotter = ECGPlotter(grid_mode="cm", show_calibration=False)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        vx, hy = _grid_lines(_ax(fig))
        minor_step = 1.0 / MM_PER_INCH  # 1 mm in inches
        assert np.allclose(np.diff(vx), minor_step, atol=1e-9), "Vertical minor grid spacing ≠ 1 mm"
        assert np.allclose(np.diff(hy), minor_step, atol=1e-9), "Horizontal minor grid spacing ≠ 1 mm"
    finally:
        plt.close(fig)


def test_grid_major_spacing(ecg_df):
    """Major (thick) grid lines are spaced exactly 5 mm apart in both axes."""
    major_lw = 0.6
    plotter = ECGPlotter(grid_mode="cm", show_calibration=False)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        ax = _ax(fig)
        two_pt = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
        major_vx = sorted(
            ln.get_xdata()[0]
            for ln in two_pt
            if ln.get_xdata()[0] == ln.get_xdata()[1] and abs(ln.get_linewidth() - major_lw) < 0.01
        )
        major_hy = sorted(
            ln.get_ydata()[0]
            for ln in two_pt
            if ln.get_ydata()[0] == ln.get_ydata()[1] and abs(ln.get_linewidth() - major_lw) < 0.01
        )
        major_step = 5.0 / MM_PER_INCH  # 5 mm in inches
        assert np.allclose(np.diff(major_vx), major_step, atol=1e-9), "Major vertical grid spacing ≠ 5 mm"
        assert np.allclose(np.diff(major_hy), major_step, atol=1e-9), "Major horizontal grid spacing ≠ 5 mm"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "grid_mode,expect_many_lines",
    [
        ("cm", True),
        (None, False),
    ],
)
def test_grid_line_count(ecg_df, grid_mode, expect_many_lines):
    """grid_mode='cm' adds many axvline/axhline lines; None adds none."""
    plotter = ECGPlotter(grid_mode=grid_mode, print_information=False, show_calibration=False)
    configuration = ["I", "II"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        n = len(_ax(fig).lines)
        if expect_many_lines:
            assert n > 50, f"Expected many grid lines, got {n}"
        else:
            assert n == 2, f"Expected 2 signal lines, got {n}"
    finally:
        plt.close(fig)


# ── Time axis ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("show_time_axis", [True, False])
def test_time_axis_visibility(ecg_df, show_time_axis):
    plotter = ECGPlotter(grid_mode=None, print_information=False, show_time_axis=show_time_axis)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        assert _ax(fig).xaxis.get_visible() == show_time_axis
    finally:
        plt.close(fig)


# ── Lead labels ────────────────────────────────────────────────────────────

# 1xL keys: flat-list templates (no split rows) whose leads form the "source" DataFrame.
_ONE_X_L_KEYS = [k for k, v in TEMPLATE_CONFIGURATIONS.items() if all(isinstance(e, str) for e in v)]


def _template_unique_leads(template_key: str) -> list[str]:
    """Return deduplicated leads referenced by a template (order-preserving)."""
    seen: dict[str, None] = {}
    for entry in TEMPLATE_CONFIGURATIONS[template_key]:
        for lead in [entry] if isinstance(entry, str) else entry:
            seen[lead] = None
    return list(seen)


@pytest.mark.parametrize("show_leads_labels", [True, False])
@pytest.mark.parametrize("plot_key", list(TEMPLATE_CONFIGURATIONS.keys()))
@pytest.mark.parametrize("source_key", _ONE_X_L_KEYS)
def test_lead_labels(source_key, plot_key, show_leads_labels):
    """Lead labels are rendered iff show_leads_labels=True; missing leads raise an error.

    For every 1xL source configuration (DataFrame restricted to that template's leads)
    and every plot template:
    * If the template only references leads present in the source DataFrame → plot()
      must succeed; labels appear or are absent according to show_leads_labels.
    * If the template references any lead absent from the source DataFrame → plot()
      must raise a KeyError (pandas column access) regardless of show_leads_labels.
    """
    source_leads = list(TEMPLATE_CONFIGURATIONS[source_key])
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_SAMPLES, len(source_leads))) * 0.5
    df = pd.DataFrame(data, columns=source_leads)

    plot_leads = _template_unique_leads(plot_key)
    template_fits = set(plot_leads).issubset(set(source_leads))

    plotter = ECGPlotter(grid_mode=None, print_information=False, show_leads_labels=show_leads_labels)

    if not template_fits:
        with maybe_warns_divisible(plot_key, N_SAMPLES):
            with pytest.raises(KeyError):
                plotter.plot(df, plot_key, sampling_frequency=FS, show=False)
        return

    with maybe_warns_divisible(plot_key, N_SAMPLES):
        fig = plotter.plot(df, plot_key, sampling_frequency=FS, show=False)
    try:
        texts = _texts(fig)
        if show_leads_labels:
            for lead in plot_leads:
                assert lead in texts, f"[{source_key} → {plot_key}] Expected label '{lead}' in texts; got {texts}"
        else:
            assert not any(t in LEAD_NAMES for t in texts), (
                f"[{source_key} → {plot_key}] Unexpected lead labels with show_leads_labels=False: {texts}"
            )
    finally:
        plt.close(fig)


# ── Calibration pulse annotation ───────────────────────────────────────────


@pytest.mark.parametrize(
    "show_calibration,n_rows",
    [
        (True, 1),
        (True, 3),
        (False, 1),
        (False, 3),
    ],
)
def test_calibration_1mv_text(ecg_df, show_calibration, n_rows):
    """Each calibration pulse adds a '1mV' text annotation; count must match n_rows."""
    configuration = LEAD_NAMES[:n_rows]
    plotter = ECGPlotter(grid_mode=None, print_information=False, show_calibration=show_calibration)
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        n_1mv = sum(1 for t in _texts(fig) if t == "1mV")
        assert n_1mv == (n_rows if show_calibration else 0)
    finally:
        plt.close(fig)


# ── print_information: diagnostics ────────────────────────────────────────


@pytest.mark.parametrize("print_information", [True, False])
def test_diagnostics_text_presence(ecg_df, print_information):
    """Speed/Voltage/Freq diagnostics appear iff print_information=True."""
    plotter = ECGPlotter(grid_mode=None, print_information=print_information)
    configuration = ["I", "II"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        has_diag = any("Speed" in t and "Voltage" in t for t in _texts(fig))
        assert has_diag == print_information
    finally:
        plt.close(fig)


# ── print_information: patient metadata ───────────────────────────────────


@pytest.mark.parametrize(
    "information,needle",
    [
        (ECGInformation(patient_name="Jane Doe"), "Jane Doe"),
        (ECGInformation(age=65, sex="Female"), "65"),
        (ECGInformation(hospital="City Hospital"), "City Hospital"),
        (ECGInformation(date="2024-06-01"), "2024-06-01"),
        (ECGInformation(machine_model="ECG-9000"), "ECG-9000"),
        (ECGInformation(filter="0.05-150 Hz"), "0.05-150 Hz"),
    ],
)
def test_patient_info_content(ecg_df, information, needle):
    """Each ECGInformation field is rendered as text when print_information=True."""
    plotter = ECGPlotter(grid_mode=None, print_information=True)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False, information=information)
    try:
        combined = " ".join(_texts(fig))
        assert needle in combined, f"Expected '{needle}' in figure texts"
    finally:
        plt.close(fig)


# ── print_information: ECGStats ───────────────────────────────────────────


@pytest.mark.parametrize(
    "stats,label,value_substr",
    [
        (ECGStats(bpm=72.0), "BPM", "72"),
        (ECGStats(rr_interval_ms=833.0), "RR", "833"),
        (ECGStats(hrv_ms=45.0), "HRV", "45"),
        (ECGStats(pr_interval_ms=160.0), "PR", "160"),
        (ECGStats(qrs_duration_ms=90.0), "QRS", "90"),
        (ECGStats(qt_interval_ms=400.0), "QT", "400"),
        (ECGStats(qtc_interval_ms=420.0), "QTc", "420"),
        (ECGStats(p_axis_deg=60.0), "P ax.", "60"),
        (ECGStats(qrs_axis_deg=-30.0), "QRS ax.", "-30"),
        (ECGStats(t_axis_deg=45.0), "T ax.", "45"),
    ],
)
def test_stats_content(ecg_df, stats, label, value_substr):
    """Each ECGStats field is rendered as 'LABEL: value' text when print_information=True."""
    plotter = ECGPlotter(grid_mode=None, print_information=True)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False, stats=stats)
    try:
        combined = " ".join(_texts(fig))
        assert label in combined, f"Expected label '{label}' in figure texts"
        assert value_substr in combined, f"Expected value '{value_substr}' in figure texts"
    finally:
        plt.close(fig)


def test_stats_absent_when_no_print_information(ecg_df):
    """Stats are not rendered when print_information=False."""
    plotter = ECGPlotter(grid_mode=None, print_information=False)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(
            ecg_df,
            configuration,
            sampling_frequency=FS,
            show=False,
            stats=ECGStats(bpm=72.0, qrs_duration_ms=90.0),
        )
    try:
        assert not any("BPM" in t for t in _texts(fig))
    finally:
        plt.close(fig)


# ── Full ECGInformation and ECGStats ───────────────────────────────────────

ALL_STATS_LABELS = ["BPM", "S/N", "RR", "HRV", "PR", "QRS", "QT", "QTc", "P ax.", "QRS ax.", "T ax."]


def test_full_ecginformation_all_fields_printed(ecg_df):
    """When all ECGInformation fields are set every field appears in the figure."""
    info = ECGInformation(
        hospital="City Hospital",
        patient_name="John Doe",
        age=65,
        sex="Male",
        date="2024-01-15",
        machine_model="ECG-9000",
        filter="0.05-150 Hz",
    )
    plotter = ECGPlotter(grid_mode=None, print_information=True)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False, information=info)
    try:
        combined = " ".join(_texts(fig))
        for needle in ["City Hospital", "John Doe", "65", "Male", "2024-01-15", "ECG-9000", "0.05-150 Hz"]:
            assert needle in combined, f"'{needle}' not found in figure text"
    finally:
        plt.close(fig)


def test_full_ecgstats_all_fields_printed(ecg_df):
    """When all ECGStats fields are set every label appears in the figure."""
    stats = ECGStats(
        bpm=72.0,
        snr=25.5,
        rr_interval_ms=833.0,
        hrv_ms=45.0,
        pr_interval_ms=160.0,
        qrs_duration_ms=90.0,
        qt_interval_ms=400.0,
        qtc_interval_ms=420.0,
        p_axis_deg=60.0,
        qrs_axis_deg=-30.0,
        t_axis_deg=45.0,
    )
    plotter = ECGPlotter(grid_mode=None, print_information=True)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False, stats=stats)
    try:
        combined = " ".join(_texts(fig))
        for label in ALL_STATS_LABELS:
            assert label in combined, f"Stats label '{label}' not found in figure text"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "stats,present_labels,absent_labels",
    [
        # Only BPM and QRS set; every other field is None
        pytest.param(
            ECGStats(bpm=72.0, qrs_duration_ms=90.0),
            ["BPM", "QRS"],
            ["S/N", "RR", "HRV", "PR", "QT", "QTc", "P ax.", "QRS ax.", "T ax."],
            id="bpm+qrs-only",
        ),
        # Only axis fields set
        pytest.param(
            ECGStats(p_axis_deg=60.0, qrs_axis_deg=-30.0, t_axis_deg=45.0),
            ["P ax.", "QRS ax.", "T ax."],
            ["BPM", "S/N", "RR", "HRV", "PR", "QRS", "QT", "QTc"],
            id="axes-only",
        ),
        # All fields None → no stats labels at all
        pytest.param(
            ECGStats(),
            [],
            ALL_STATS_LABELS,
            id="all-none",
        ),
        # float('nan') is not None: the field is treated as set and renders as "nan"
        pytest.param(
            ECGStats(bpm=float("nan"), rr_interval_ms=float("nan")),
            ["BPM", "RR"],
            ["HRV", "PR", "QRS", "QT", "QTc", "P ax.", "QRS ax.", "T ax."],
            id="nan-floats",
        ),
    ],
)
def test_partial_ecgstats(ecg_df, stats, present_labels, absent_labels):
    """Only non-None ECGStats fields produce a label; None fields are silently omitted.

    We search for ``"LABEL:"`` (with colon) rather than bare label strings to avoid
    substring false-positives (e.g. ``"QRS"`` inside ``"QRS ax.: …"``).
    """
    plotter = ECGPlotter(grid_mode=None, print_information=True)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False, stats=stats)
    try:
        combined = " ".join(_texts(fig))
        for label in present_labels:
            assert f"{label}:" in combined, f"Expected present label '{label}:' not found"
        for label in absent_labels:
            assert f"{label}:" not in combined, f"Absent label '{label}:' found unexpectedly"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "information,present_strings,absent_strings",
    [
        # All fields populated → all appear
        pytest.param(
            ECGInformation(hospital="CH", patient_name="Jane", age=42, sex="Female", date="2024-01-01"),
            ["CH", "Jane", "42", "Female", "2024-01-01"],
            [],
            id="all-fields",
        ),
        # All defaults (None) → no info header lines
        pytest.param(
            ECGInformation(),
            [],
            ["Hospital:", "Patient:", "Date:"],
            id="all-none",
        ),
        # information=None → same as all-None ECGInformation
        pytest.param(
            None,
            [],
            ["Hospital:", "Patient:", "Date:"],
            id="none-object",
        ),
        # Only hospital set → Patient: and Date: absent
        pytest.param(
            ECGInformation(hospital="City Hospital"),
            ["City Hospital"],
            ["Patient:", "Date:"],
            id="hospital-only",
        ),
        # Only date set → Hospital: and Patient: absent
        pytest.param(
            ECGInformation(date="2024-06-15"),
            ["2024-06-15"],
            ["Hospital:", "Patient:"],
            id="date-only",
        ),
        # Only age set → "Patient:  42 yrs" appears; Hospital: and Date: absent
        pytest.param(
            ECGInformation(age=42),
            ["42"],
            ["Hospital:", "Date:"],
            id="age-only",
        ),
        # Only sex + age → patient line present; hospital and date absent
        pytest.param(
            ECGInformation(sex="Male", age=55),
            ["Male", "55"],
            ["Hospital:", "Date:"],
            id="sex+age",
        ),
    ],
)
def test_partial_ecginformation(ecg_df, information, present_strings, absent_strings):
    """Only set ECGInformation fields appear; unset fields produce no text."""
    plotter = ECGPlotter(grid_mode=None, print_information=True)
    configuration = ["I"]
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False, information=information)
    try:
        combined = " ".join(_texts(fig))
        for s in present_strings:
            assert s in combined, f"Expected '{s}' in figure text"
        for s in absent_strings:
            assert s not in combined, f"'{s}' should not appear when field is unset"
    finally:
        plt.close(fig)


# ── Signal horizontal extent ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "configuration,n_rows",
    [
        (["I"], 1),
        (["I", "II", "III"], 3),
        ([["I", "II"], ["III", "AVR"]], 2),
    ],
)
def test_signal_horizontal_extent(ecg_df, configuration, n_rows):
    """Signal lines span exactly (N_SAMPLES - 1) / FS * speed mm horizontally.

    This verifies the time-to-inches conversion: the physical width of the plotted
    signal must equal the recording duration times the paper speed.
    """
    speed = 50.0  # ECGPlotter default
    expected_span_mm = (N_SAMPLES - 1) * speed / FS
    left_margin_in = LEFT_MARGIN_MM / MM_PER_INCH

    plotter = ECGPlotter(grid_mode=None, show_calibration=False, print_information=False)
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        # Signal lines: len(xdata) == N_SAMPLES and start exactly at the left margin
        signal_lines = [
            ln for ln in _ax(fig).lines if len(ln.get_xdata()) == N_SAMPLES and abs(ln.get_xdata()[0] - left_margin_in) < 1e-9
        ]
        assert len(signal_lines) == n_rows, f"Expected {n_rows} signal lines, got {len(signal_lines)}"
        for line in signal_lines:
            x = line.get_xdata()
            span_mm = (x[-1] - x[0]) * MM_PER_INCH
            assert abs(span_mm - expected_span_mm) < speed / FS, (
                f"Signal span {span_mm:.4f} mm ≠ expected {expected_span_mm:.4f} mm"
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "template_key,expected_n_rows",
    [
        ("1x1", 1),
        ("1x2", 2),
        ("1x3", 3),
        ("1x4", 4),
        ("1x6", 6),
        ("1x8", 8),
        ("1x12", 12),
        ("2x4", 5),  # 4 split rows + 1 rhythm strip
        ("2x6", 7),  # 6 split rows + 1 rhythm strip
        ("4x3", 4),  # 3 split rows + 1 rhythm strip
    ],
)
def test_template_row_count(ecg_df, template_key, expected_n_rows):
    """Each named template produces the correct number of ECG rows."""
    plotter = ECGPlotter(grid_mode=None, print_information=False, show_calibration=False)
    with maybe_warns_divisible(template_key, N_SAMPLES):
        fig = plotter.plot(ecg_df, template_key, sampling_frequency=FS, show=False)
    try:
        # 1 line per row (show_calibration=False, no grid)
        assert len(_ax(fig).lines) == expected_n_rows
    finally:
        plt.close(fig)


@pytest.mark.parametrize("template_key", list(TEMPLATE_CONFIGURATIONS.keys()))
def test_template_all_lead_labels(ecg_df, template_key):
    """Every lead in a template appears as a text label on the figure."""
    config = TEMPLATE_CONFIGURATIONS[template_key]
    expected_leads: list[str] = []
    for entry in config:
        if isinstance(entry, list):
            expected_leads.extend(entry)
        else:
            expected_leads.append(entry)
    expected_leads = list(dict.fromkeys(expected_leads))  # deduplicate, preserve order

    plotter = ECGPlotter(grid_mode=None, print_information=False, show_leads_labels=True)
    with maybe_warns_divisible(template_key, N_SAMPLES):
        fig = plotter.plot(ecg_df, template_key, sampling_frequency=FS, show=False)
    try:
        texts = _texts(fig)
        for lead in expected_leads:
            assert lead in texts, f"Lead '{lead}' missing from figure texts for template '{template_key}'"
    finally:
        plt.close(fig)


# ── Lead label positions ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "configuration",
    [
        ["I", "II", "III"],
        [["I", "V1"], ["II", "V2"]],
        # User's example: split rows + repeated lead "II" as a rhythm strip
        [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"], "II"],
        "4x3",
    ],
)
def test_lead_label_positions(ecg_df, configuration):
    """Each lead label appears at the correct row (y) and left-to-right segment order (x).

    Strategy
    --------
    We recompute the expected (x, y) for every label using the same formulas as
    ``_plot_row``, then verify:

    * Each label is present at its expected (x, y) position (within 1e-9 in).
    * Within every row, label x-coordinates are strictly increasing left to right.
    * Repeated leads (e.g. "II" in two different rows) are distinguished by y,
      so each occurrence is checked independently.
    """
    speed, voltage, row_distance = 50.0, 20.0, 2.0
    rows = _resolve_rows(configuration)
    n_rows = len(rows)

    plotter = ECGPlotter(
        grid_mode=None,
        print_information=False,
        show_calibration=False,
        show_leads_labels=True,
        speed=speed,
        voltage=voltage,
        row_distance=row_distance,
    )
    with maybe_warns_divisible(configuration, N_SAMPLES):
        fig = plotter.plot(ecg_df, configuration, sampling_frequency=FS, show=False)
    try:
        ax = _ax(fig)

        adjusted_row_distance = _adjust_row_distance(row_distance, voltage)

        time_to_inches = speed / (FS * MM_PER_INCH)
        row_distance_in = adjusted_row_distance * voltage / MM_PER_INCH
        left_margin = LEFT_MARGIN_MM / MM_PER_INCH

        _, height_in = _compute_figure_size(n_rows, N_SAMPLES, FS, speed, voltage, adjusted_row_distance)
        y_offsets = _compute_row_offsets(n_rows, height_in, row_distance_in)
        row_half = row_distance_in / 2.0

        # All text artists: (x, y, label)
        all_texts = [(t.get_position()[0], t.get_position()[1], t.get_text()) for t in ax.texts]

        for i, leads in enumerate(rows):
            exp_y = y_offsets[i] + row_half
            segment_len = N_SAMPLES // len(leads)

            # Group texts belonging to this row by y proximity
            row_texts = sorted(
                [(x, label) for x, y, label in all_texts if abs(y - exp_y) < 1e-9],
                key=lambda t: t[0],
            )

            assert len(row_texts) == len(leads), f"Row {i} ({leads}): expected {len(leads)} labels, found {len(row_texts)}"

            for j, (x, label) in enumerate(row_texts):
                exp_x = left_margin + j * segment_len * time_to_inches

                # Correct label in the correct left-to-right slot
                assert label == leads[j], f"Row {i}, slot {j}: expected '{leads[j]}', got '{label}'"
                # Correct horizontal position
                assert abs(x - exp_x) < 1e-9, f"Row {i}, lead '{label}': expected x={exp_x:.6f} in, got {x:.6f} in"
                # Strictly to the right of the previous label
                if j > 0:
                    assert x > row_texts[j - 1][0], (
                        f"Row {i}: '{label}' (x={x:.4f}) not to the right of "
                        f"'{row_texts[j - 1][1]}' (x={row_texts[j - 1][0]:.4f})"
                    )
    finally:
        plt.close(fig)
