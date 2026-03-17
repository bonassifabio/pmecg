from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import matplotlib.axes
import numpy as np
from matplotlib.collections import Collection
from matplotlib.patches import Rectangle

from pmecg.types import LeadSegment

from .attention import AbstractAttentionMap

MM_PER_INCH = 25.4
MARGIN_MM = 5.0  # margin above the first row, below the last row, and between rows
INFO_TOP_EXTRA_MARGIN_MM = 14.0  # extra top margin added when print_information=True
INFO_BOT_EXTRA_MARGIN_MM = 8.0  # extra bottom margin added when print_information=True
LEFT_MARGIN_MM = 15.0  # 1.5 cm left margin (accommodates calibration pulse)
RIGHT_MARGIN_MM = 10.0  # 1 cm right margin
COLORBAR_WIDTH_MM = 4.0
COLORBAR_INNER_PAD_MM = 2.0
COLORBAR_LABEL_PAD_MM = 1.5
COLORBAR_TICK_LENGTH_MM = 1.0

# Calibration pulse dimensions
CAL_PULSE_WIDTH_MM = 5.0  # 1 large square wide
CAL_PULSE_AMP_MV = 1.0  # standard 1 mV amplitude
CAL_PULSE_OFFSET_MM = 3.0  # gap from left figure edge to the rising edge
_STAT_FORMATTERS: tuple[tuple[str, str, str], ...] = (
    ("bpm", "BPM", "{value:.0f}"),
    ("snr", "S/N", "{value:.1f} dB"),
    ("rr_interval_ms", "RR", "{value:.0f} ms"),
    ("hrv_ms", "HRV", "{value:.0f} ms"),
    ("pr_interval_ms", "PR", "{value:.0f} ms"),
    ("qrs_duration_ms", "QRS", "{value:.0f} ms"),
    ("qt_interval_ms", "QT", "{value:.0f} ms"),
    ("qtc_interval_ms", "QTc", "{value:.0f} ms"),
    ("p_axis_deg", "P ax.", "{value:.0f}°"),
    ("qrs_axis_deg", "QRS ax.", "{value:.0f}°"),
    ("t_axis_deg", "T ax.", "{value:.0f}°"),
)


@dataclass
class _RenderContext:
    """Derived rendering values computed once per :meth:`ECGPlotter.plot` call.

    Bundles the per-configuration constants that would otherwise be forwarded
    individually to every low-level drawing helper.

    Attributes
    ----------
    mv_to_inches : float
        Conversion factor: 1 mV → inches  (= ``voltage / MM_PER_INCH``).
    time_to_inches : float
        Conversion factor: 1 sample → inches
        (= ``speed / (sampling_frequency * MM_PER_INCH)``).
    row_distance_inches : float
        Distance between consecutive row zero-lines in inches.
    line_width : float
        Thickness of ECG signal lines in points.
    grid_color : str
        Matplotlib color string for the grid lines.
    speed : float
        Paper speed in mm/s (used in the diagnostics label).
    voltage : float
        Vertical scale in mm/mV (used in the diagnostics label).
    show_calibration : bool
        Whether to draw the 1 mV calibration pulse in the left margin of each row.
    show_leads_labels : bool
        Whether to print lead names onto the plot.
    """

    mv_to_inches: float
    time_to_inches: float
    row_distance_inches: float
    line_width: float
    grid_color: str
    speed: float
    voltage: float
    show_calibration: bool
    show_leads_labels: bool


def _nice_tick_step(total_time_s: float) -> float:
    """Choose a human-friendly tick spacing (in seconds) for a given recording duration.

    Targets roughly 10 ticks across the recording.

    Returns
    -------
    float
        Tick spacing in seconds.
    """
    raw_step = total_time_s / 10.0
    # Round to the nearest value in [0.1, 0.2, 0.5, 1, 2, 5, 10, ...]
    magnitude = 10 ** np.floor(np.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized < 1.5:
        nice = 1.0
    elif normalized < 3.5:
        nice = 2.0
    else:
        nice = 5.0
    return nice * magnitude


def _adjust_row_distance(row_distance: float, voltage: float) -> float:
    """Adjust row_distance so that it is a multiple of 5 mm in vertical space.

    Rounding is applied to avoid float precision issues before the ceiling
    operation.

    Parameters
    ----------
    row_distance : float
        The distance between rows in mV.
    voltage : float
        Vertical scale: mm per mV.

    Returns
    -------
    float
        Adjusted row distance in mV.
    """
    row_distance_mm = np.round(row_distance * voltage, decimals=5)
    row_distance_mm = np.ceil(row_distance_mm / 5.0) * 5.0
    return row_distance_mm / voltage


def _compute_figure_size(
    n_rows: int,
    seq_len: int,
    sampling_frequency: float,
    speed: float,
    voltage: float,
    row_distance_mv: float,
    print_information: bool = False,
    right_margin_mm: float = RIGHT_MARGIN_MM,
    strip_seq_len: int | None = None,
    strip_speed: float | None = None,
) -> tuple[float, float]:
    """Compute the figure width and height in inches based on ECG parameters.

    Parameters
    ----------
    n_rows : int
        Number of rows in the layout.
    seq_len : int
        Number of samples in the ECG (length of each row's signal).
    sampling_frequency : float
        Sampling frequency in Hz.
    speed : float
        Paper speed in mm/s.
    voltage : float
        Vertical scale: mm per mV.
    row_distance_mv : float
        Distance between consecutive row zero-lines, expressed in mV.
    print_information : bool, optional
        When True, extra top and bottom margins are added to accommodate
        the patient information and statistics text, by default False.
    right_margin_mm : float, optional
        Width of the right margin in millimetres. This is expanded when an
        attention color scale needs to be drawn, by default ``RIGHT_MARGIN_MM``.
    strip_seq_len : int | None, optional
        Number of samples in the strip rows (typically the full recording
        length). When provided, the figure width is expanded if the strip
        rows are wider than the main config rows. By default ``None``.
    strip_speed : float | None, optional
        Paper speed in mm/s for strip rows. When ``None``, ``speed`` is used.
        Only meaningful when ``strip_seq_len`` is provided. By default ``None``.

    Returns
    -------
    tuple[float, float]
        (width_inches, height_inches)
    """
    # --- Width ---
    # Total recording duration in seconds
    total_time_s = seq_len / sampling_frequency
    # Convert to mm: duration * speed, then add left and right margins
    width_mm = total_time_s * speed + LEFT_MARGIN_MM + right_margin_mm
    if strip_seq_len is not None:
        effective_strip_speed = strip_speed if strip_speed is not None else speed
        strip_width_mm = strip_seq_len / sampling_frequency * effective_strip_speed + LEFT_MARGIN_MM + right_margin_mm
        width_mm = max(width_mm, strip_width_mm)
    width_inches = width_mm / MM_PER_INCH

    # --- Height ---
    # Each row is allocated row_distance_mv * voltage mm of vertical space (centred on its zero line).
    # Add a top and bottom margin (MARGIN_MM each) to avoid clipping.
    # When print_information is enabled, add extra space at top and bottom for the annotation text.
    row_distance_mm = row_distance_mv * voltage
    top_extra_mm = INFO_TOP_EXTRA_MARGIN_MM if print_information else 0.0
    bot_extra_mm = INFO_BOT_EXTRA_MARGIN_MM if print_information else 0.0

    # Calculate spaces from the zero-lines to the edges of the figure
    top_space_mm = MARGIN_MM + top_extra_mm + row_distance_mm / 2.0
    ideal_bottom_space_mm = MARGIN_MM + bot_extra_mm + row_distance_mm / 2.0

    # Ensure the bottom space is a multiple of 5mm so that zero-lines align with major grid lines
    actual_bottom_space_mm = np.ceil(ideal_bottom_space_mm / 5.0) * 5.0

    total_height_mm = top_space_mm + (n_rows - 1) * row_distance_mm + actual_bottom_space_mm
    height_inches = total_height_mm / MM_PER_INCH

    return width_inches, height_inches


def _compute_row_offsets(
    n_rows: int,
    height_inches: float,
    row_distance_inches: float,
    print_information: bool = False,
) -> list[float]:
    """Pre-compute the vertical centre (zero-line position, in inches) for each ECG row.

    Rows are laid out top-to-bottom with a fixed spacing between zero-lines and a
    MARGIN_MM margin above the first row and below the last row. When
    ``print_information`` is True the extra top margin is also accounted for so that
    the patient-info text can sit in the space above the first row.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    height_inches : float
        Total figure height in inches (as returned by `_compute_figure_size`).
    row_distance_inches : float
        Distance between consecutive zero-lines in inches.
    print_information : bool, optional
        When True, include the extra top margin when computing the first row
        position, by default False.

    Returns
    -------
    list[float]
        y-coordinate (in inches from figure bottom) of the zero-line of each row.
    """
    top_extra_mm = INFO_TOP_EXTRA_MARGIN_MM if print_information else 0.0
    top_margin_inches = (MARGIN_MM + top_extra_mm) / MM_PER_INCH
    # First row zero-line sits half a spacing below the top margin.
    # (Rows are evenly spaced; the half-spacing gives equal room above row 0 and below last row.)
    first_zero = height_inches - top_margin_inches - row_distance_inches / 2.0
    return [first_zero - i * row_distance_inches for i in range(n_rows)]


def _plot_calibration_pulse(
    ax: matplotlib.axes.Axes,
    ctx: _RenderContext,
    y_offset: float,
) -> None:
    """Draw a 1 mV square calibration pulse (_|-|_) in the left margin for a row.

    The pulse is 1 large square wide (CAL_PULSE_WIDTH_MM = 5 mm) and 1 mV tall,
    positioned within the left margin starting at CAL_PULSE_OFFSET_MM from the
    left edge of the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on. Coordinates are in inches.
    ctx : _RenderContext
        Rendering context; ``ctx.mv_to_inches`` and ``ctx.line_width`` are used.
    y_offset : float
        Vertical zero-line of the row in figure inches.
    """
    x0 = CAL_PULSE_OFFSET_MM / MM_PER_INCH
    x1 = (CAL_PULSE_OFFSET_MM + CAL_PULSE_WIDTH_MM) / MM_PER_INCH
    x_signal = LEFT_MARGIN_MM / MM_PER_INCH
    amp = CAL_PULSE_AMP_MV * ctx.mv_to_inches

    xs = [0.0, x0, x0, x1, x1, x_signal]
    ys = [y_offset, y_offset, y_offset + amp, y_offset + amp, y_offset, y_offset]
    ax.plot(xs, ys, color="black", linewidth=ctx.line_width)

    # Label centred above the top of the pulse
    x_mid = (x0 + x1) / 2
    ax.text(x_mid, y_offset + amp, "1mV", ha="center", va="bottom", fontsize=6, fontfamily="monospace")


def _plot_row(
    ax: matplotlib.axes.Axes,
    row: tuple[np.ndarray, list[str]],
    ctx: _RenderContext,
    y_offset: float,
    attention_values: np.ndarray | None = None,
    attention_map: AbstractAttentionMap | None = None,
    time_to_inches: float | None = None,
    segment_offsets: list[int] | None = None,
) -> None:
    """Plot a single ECG row onto `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on. Coordinates are in inches.
    row : tuple[np.ndarray, list[str]]
        A (signal, leads) pair as produced by `_apply_configuration`.
    ctx : _RenderContext
        Rendering context with conversion factors and style settings.
    y_offset : float
        Vertical zero-line of this row in figure inches (pre-computed by
        `_compute_row_offsets`).
    attention_values : np.ndarray | None, optional
        1-D attention array of length ``len(signal)`` for this row.
        Must be provided together with ``attention_map``. By default ``None``.
    attention_map : AbstractAttentionMap | None, optional
        Prepared attention map used to build the overlay artists.
        Must be provided together with ``attention_values``. By default ``None``.
    time_to_inches : float | None, optional
        Override for the time-to-inches conversion factor. When provided,
        this is used instead of ``ctx.time_to_inches`` (e.g. for strip
        leads running at a different paper speed). By default ``None``.
    segment_offsets : list[int] | None, optional
        Sample index in the concatenated signal at which each lead starts.
        Length must equal ``len(leads)``. When provided, lead labels are
        placed at the true segment boundary rather than at equal divisions.
        By default ``None`` (even-split assumed, matching classic string-based rows).
    """
    tti = time_to_inches if time_to_inches is not None else ctx.time_to_inches
    signal, leads = row
    n_samples = len(signal)
    n_leads = len(leads)

    # --- x axis ---
    # x[i] = sample_index * (speed_mm_s / (fs_hz * mm_per_inch)) + left_margin  [samples → inches]
    left_margin_inches = LEFT_MARGIN_MM / MM_PER_INCH
    x = np.arange(n_samples) * tti + left_margin_inches

    # --- y axis ---
    y = signal * ctx.mv_to_inches + y_offset
    row_half_height_inches = ctx.row_distance_inches / 2.0

    if attention_map is not None and attention_values is not None:
        for artist in attention_map.build_artists(
            ax,
            x,
            y,
            attention_values,
            y_offset,
            row_half_height_inches,
            ctx.mv_to_inches,
            ctx.line_width,
        ):
            if isinstance(artist, Collection):
                ax.add_collection(artist)
            else:
                ax.add_artist(artist)

    ax.plot(x, y, color="black", linewidth=ctx.line_width, zorder=3)

    if ctx.show_calibration:
        _plot_calibration_pulse(ax, ctx, y_offset)

    # --- Labels ---
    if ctx.show_leads_labels:
        for i, lead_name in enumerate(leads):
            sample_start = segment_offsets[i] if segment_offsets is not None else i * (n_samples // n_leads)
            x_label = left_margin_inches + sample_start * tti
            y_label = y_offset + row_half_height_inches
            ax.text(x_label, y_label, lead_name, va="top", ha="left", fontsize=9, fontfamily="monospace")


def _plot_grid(
    ax: matplotlib.axes.Axes,
    grid_mode: Literal["cm"],
    width_inches: float,
    height_inches: float,
    ctx: _RenderContext,
) -> None:
    """Draw a background grid on `ax` with lines spaced according to `grid_mode`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the grid on. Coordinates are in inches.
    grid_mode : {'cm'}
        Grid spacing unit. ``'cm'`` draws lines every 0.1 cm (= 1 mm); every 5th line
        is slightly thicker to form major squares. ``'inch'`` is not implemented and
        will raise :exc:`NotImplementedError`.
    width_inches : float
        Width of the plot area in inches (used to bound vertical grid lines).
    height_inches : float
        Height of the plot area in inches (used to bound horizontal grid lines).
    ctx : _RenderContext
        Rendering context; ``ctx.grid_color`` sets the line colour.
    """
    if grid_mode == "inch":
        raise NotImplementedError("'inch' grid mode is not supported. Use 'cm' or None.")
    step = 1.0 / MM_PER_INCH  # 0.1 cm = 1 mm expressed in inches

    minor_lw = 0.2
    major_lw = 0.6

    xs = np.arange(0, width_inches + step * 0.5, step)
    for i, x in enumerate(xs):
        lw = major_lw if i % 5 == 0 else minor_lw
        ax.axvline(x, color=ctx.grid_color, linewidth=lw, zorder=0)

    ys = np.arange(0, height_inches + step * 0.5, step)
    for i, y in enumerate(ys):
        lw = major_lw if i % 5 == 0 else minor_lw
        ax.axhline(y, color=ctx.grid_color, linewidth=lw, zorder=0)


def _plot_attention_color_scale(
    ax: matplotlib.axes.Axes,
    attention_map: AbstractAttentionMap,
    width_inches: float,
    right_margin_mm: float,
    top_inches: float,
    bottom_inches: float,
) -> None:
    """Draw a vertical attention color scale inside the expanded right margin."""
    rgba_image = attention_map.colormap_rgba()

    colorbar_left = width_inches - (right_margin_mm / MM_PER_INCH) + (COLORBAR_INNER_PAD_MM / MM_PER_INCH)
    colorbar_width = COLORBAR_WIDTH_MM / MM_PER_INCH
    colorbar_right = colorbar_left + colorbar_width

    ax.imshow(
        rgba_image,
        extent=(colorbar_left, colorbar_right, bottom_inches, top_inches),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        zorder=4,
    )
    ax.add_patch(
        Rectangle(
            (colorbar_left, bottom_inches),
            colorbar_width,
            top_inches - bottom_inches,
            fill=False,
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
        )
    )

    tick_end = colorbar_right + (COLORBAR_TICK_LENGTH_MM / MM_PER_INCH)
    label_x = tick_end + (COLORBAR_LABEL_PAD_MM / MM_PER_INCH)
    lower, upper = attention_map.range
    tick_values = [upper, lower]
    if lower < 0 < upper:
        tick_values.insert(1, 0.0)

    for value in tick_values:
        y_pos = bottom_inches + ((value - lower) / (upper - lower)) * (top_inches - bottom_inches)
        ax.plot([colorbar_right, tick_end], [y_pos, y_pos], color="black", linewidth=0.4, zorder=5)
        ax.text(
            label_x,
            y_pos,
            f"{value:.2g}",
            va="center",
            ha="left",
            fontsize=6,
            fontfamily="monospace",
            zorder=5,
        )


def _print_information(
    ax: matplotlib.axes.Axes,
    ctx: _RenderContext,
    width_inches: float,
    sampling_frequency: float,
    leads: list[str],
    first_row_top_inches: float,
    last_row_zero_inches: float,
    information=None,
    stats=None,
    strip_speed: float | None = None,
) -> None:
    """Annotate the figure with diagnostic parameters and optional patient information.

    Diagnostics (speed, voltage, sampling frequency, leads) are placed in the
    bottom-left corner.  The machine model (from ``information.machine_model``) is
    placed in the bottom-right corner.  Patient/recording metadata (hospital,
    patient name, date) are placed just above the first ECG row, in the top margin.
    ECG statistics (from ``stats``) are placed in the top-right corner, arranged
    in columns of up to three rows.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    ctx : _RenderContext
        Rendering context; ``ctx.speed`` and ``ctx.voltage`` are used in the
        diagnostics label.
    width_inches : float
        Total figure width in inches, used to position the bottom-right text.
    sampling_frequency : float
        Sampling frequency in Hz.
    leads : list of str
        Lead names in original input order (no deduplication needed).
    first_row_top_inches : float
        Y coordinate (inches) of the top edge of the first ECG row. Patient info
        is anchored just above this line.
    last_row_zero_inches : float
        Y coordinate (inches) of the zero-line of the last ECG row. Used to position
        the bottom text so it doesn't overlap with the ECG signal.
    information : ECGInformation, optional
        Patient/recording metadata. ``information.machine_model`` is printed
        bottom-right; hospital, patient_name and date are printed top-left.
    stats : ECGStats, optional
        Computed ECG statistics. Non-None fields are printed top-right in a
        column-major grid (up to 3 rows per column).
    strip_speed : float | None, optional
        Paper speed in mm/s for strip rows, printed in the diagnostics line
        when it differs from the main speed. By default ``None``.
    """
    font = {"fontsize": 7, "fontfamily": "monospace"}
    x_left = LEFT_MARGIN_MM / MM_PER_INCH
    x_right = width_inches - (RIGHT_MARGIN_MM / MM_PER_INCH)

    row_half_mm = (ctx.row_distance_inches * MM_PER_INCH) / 2.0
    dist_mm = np.ceil(row_half_mm / 5.0) * 5.0 + MARGIN_MM + INFO_BOT_EXTRA_MARGIN_MM / 2.0
    bottom_info_top_inches = last_row_zero_inches - dist_mm / MM_PER_INCH

    line_height = 0.13  # inches between lines

    # --- Bottom-left: diagnostics (single line) ---
    leads_str = " ".join(leads)
    strip_speed_part = f"   Strip speed: {strip_speed:g} mm/s" if strip_speed is not None else ""
    diag_line = (
        f"Speed: {ctx.speed:g} mm/s{strip_speed_part}   "
        f"Voltage: {ctx.voltage:g} mm/mV   "
        f"Freq: {sampling_frequency:g} Hz   "
        f"Leads: {leads_str}"
    )
    if information is not None and getattr(information, "filter", None):
        diag_line += f"   Filter: {information.filter}"

    ax.text(x_left, bottom_info_top_inches, diag_line, va="top", ha="left", zorder=5, **font)

    # --- Bottom-right: machine model ---
    if information is not None and getattr(information, "machine_model", None):
        ax.text(x_right, bottom_info_top_inches, information.machine_model, va="top", ha="right", zorder=5, **font)

    # --- Top-left: patient / recording info, anchored just above the first ECG row ---
    if information is not None:
        info_lines: list[str] = []
        if getattr(information, "hospital", None):
            info_lines.append(f"Hospital: {information.hospital}")

        # Patient name, sex, age
        patient_line = ""
        if getattr(information, "patient_name", None):
            patient_line = f"Patient:  {information.patient_name}"
        if getattr(information, "sex", None):
            prefix = ", " if patient_line else "Patient:  "
            patient_line += f"{prefix}{information.sex}"
        if getattr(information, "age", None):
            prefix = ", " if patient_line else "Patient:  "
            patient_line += f"{prefix}{information.age} yrs"
        if patient_line:
            info_lines.append(patient_line)

        if getattr(information, "date", None):
            info_lines.append(f"Date:     {information.date}")
        # y_base sits 5 mm above the first row's top edge; lines stack upward
        y_base = first_row_top_inches + 5.0 / MM_PER_INCH
        for idx, line in enumerate(info_lines):
            y_pos = y_base + idx * line_height
            ax.text(x_left, y_pos, line, va="bottom", ha="left", zorder=5, **font)

    # --- Top-right: ECG statistics grid ---
    if stats is not None:
        stat_items = [
            (label, fmt.format(value=getattr(stats, attr_name)))
            for attr_name, label, fmt in _STAT_FORMATTERS
            if getattr(stats, attr_name) is not None
        ]

        if stat_items:
            n_cols = ceil(len(stat_items) / 3)

            # Uniform cell widths for clean column alignment
            label_w = max(len(lbl) for lbl, _ in stat_items)
            value_w = max(len(val) for _, val in stat_items)
            cells = [f"{lbl:>{label_w}}: {val:<{value_w}}" for lbl, val in stat_items]

            # Column width: cell chars + 14 padding chars, converted to inches
            # (~0.075 in per char at 7 pt monospace)
            char_width_in = 0.075
            col_width = (label_w + 2 + value_w + 14) * char_width_in

            # Anchor: bottom of the stats block aligns with the patient-info baseline
            y_base = first_row_top_inches + 5.0 / MM_PER_INCH

            for i, cell in enumerate(cells):
                col = i // 3  # column index (0 = leftmost)
                row = i % 3  # row index   (0 = top)
                x = x_right - (n_cols - col) * col_width
                y = y_base + (2 - row) * line_height
                ax.text(x, y, cell, va="bottom", ha="left", zorder=5, **font)


def _validate_time_axis_config(
    all_segments: list[list[LeadSegment]],
    strip_speed: float | None,
    main_speed: float,
) -> None:
    """Raise ValueError if the layout is incompatible with ``show_time_axis=True``.

    The time axis is only meaningful when every row shares the same time origin,
    segments within each row are strictly contiguous, and strip leads (if any)
    run at the same paper speed as the main layout.
    """
    if not all_segments:
        return

    # All rows must begin at the same sample index.
    first_starts = [segs[0].start for segs in all_segments]
    if len(set(first_starts)) > 1:
        rows_info = ", ".join(f"row {i}: sample {s}" for i, s in enumerate(first_starts))
        raise ValueError(
            "show_time_axis=True requires all rows to begin at the same sample index, "
            f"but rows start at different positions ({rows_info}). "
            "Use a consistent start index across all rows, or set show_time_axis=False."
        )

    # Within each row, segments must be strictly contiguous (no gaps or overlaps).
    for row_idx, segs in enumerate(all_segments):
        for seg_idx in range(1, len(segs)):
            prev, curr = segs[seg_idx - 1], segs[seg_idx]
            if curr.start != prev.end:
                raise ValueError(
                    f"show_time_axis=True requires segments to be strictly contiguous within "
                    f"each row (each segment must start exactly where the previous one ends), "
                    f"but row {row_idx} has a gap or overlap between "
                    f"lead '{prev.lead}' (end={prev.end}) and "
                    f"lead '{curr.lead}' (start={curr.start}). "
                    "Adjust the segment boundaries or set show_time_axis=False."
                )

    # Strip leads must run at the same speed as the main layout.
    if strip_speed is not None:
        raise ValueError(
            f"show_time_axis=True is not supported when strip_leads uses a different speed "
            f"({strip_speed:g} mm/s) than the main layout ({main_speed:g} mm/s). "
            "Set strip_leads.speed to match the plotter speed, or set show_time_axis=False."
        )
