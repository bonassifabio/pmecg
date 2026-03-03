import numpy as np
import matplotlib.axes
from typing import List, Tuple

MM_PER_INCH = 25.4
MARGIN_MM = 5.0        # margin above the first row, below the last row, and between rows
LEFT_MARGIN_MM = 10.0  # 1 cm left margin
RIGHT_MARGIN_MM = 10.0 # 1 cm right margin


def _nice_tick_step(total_time_s: float) -> float:
    """Choose a human-friendly tick spacing (in seconds) for a given recording duration.

    Targets roughly 10 ticks across the recording.
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


def _compute_figure_size(
    n_rows: int,
    seq_len: int,
    sampling_frequency: float,
    speed: float,
    voltage: float,
    row_spacing_mv: float,
) -> Tuple[float, float]:
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
    row_spacing_mv : float
        Distance between consecutive row zero-lines, expressed in mV.

    Returns
    -------
    Tuple[float, float]
        (width_inches, height_inches)
    """
    # --- Width ---
    # Total recording duration in seconds
    total_time_s = seq_len / sampling_frequency
    # Convert to mm: duration * speed, then add 1 cm on left and right
    width_mm = total_time_s * speed + LEFT_MARGIN_MM + RIGHT_MARGIN_MM
    width_inches = width_mm / MM_PER_INCH

    # --- Height ---
    # Each row is allocated row_spacing_mv * voltage mm of vertical space (centred on its zero line).
    # Add a top and bottom margin (MARGIN_MM each) to avoid clipping.
    row_spacing_mm = row_spacing_mv * voltage
    total_height_mm = n_rows * row_spacing_mm + 2 * MARGIN_MM
    height_inches = total_height_mm / MM_PER_INCH

    return width_inches, height_inches


def _compute_row_offsets(
    n_rows: int,
    height_inches: float,
    row_spacing_inches: float,
) -> List[float]:
    """Pre-compute the vertical centre (zero-line position, in inches) for each ECG row.

    Rows are laid out top-to-bottom with a fixed spacing between zero-lines and a
    MARGIN_MM margin above the first row and below the last row.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    height_inches : float
        Total figure height in inches (as returned by `_compute_figure_size`).
    row_spacing_inches : float
        Distance between consecutive zero-lines in inches.

    Returns
    -------
    List[float]
        y-coordinate (in inches from figure bottom) of the zero-line of each row.
    """
    top_margin_inches = MARGIN_MM / MM_PER_INCH
    # First row zero-line sits half a spacing below the top margin.
    # (Rows are evenly spaced; the half-spacing gives equal room above row 0 and below last row.)
    first_zero = height_inches - top_margin_inches - row_spacing_inches / 2.0
    return [first_zero - i * row_spacing_inches for i in range(n_rows)]


def _plot_row(
    ax: matplotlib.axes.Axes,
    row: Tuple[np.ndarray, List[str]],
    mv_to_inches: float,
    time_to_inches: float,
    row_idx: int,
    y_offset: float,
    row_half_height_inches: float,
) -> None:
    """Plot a single ECG row onto `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on. Coordinates are in inches.
    row : Tuple[np.ndarray, List[str]]
        A (signal, leads) pair as produced by `_apply_configuration`.
    mv_to_inches : float
        Conversion factor: 1 mV → inches  (= voltage / MM_PER_INCH).
    time_to_inches : float
        Conversion factor: 1 sample → inches  (= speed / (fs * MM_PER_INCH)).
    row_idx : int
        Zero-based index of this row (used for labelling).
    y_offset : float
        Vertical zero-line of this row in figure inches (pre-computed by
        `_compute_row_offsets`).
    row_half_height_inches : float
        Half the allocated row height in inches (= row_spacing_inches / 2).
        Used to position labels at the top of each segment's bounding box.
    """
    signal, leads = row
    n_samples = len(signal)
    n_leads = len(leads)

    # --- x axis ---
    # Convert sample indices to time (seconds) then to inches via the paper speed.
    # time_to_inches already encodes both sampling frequency and speed, so a simple
    # multiplication suffices. Shift right by the left margin (1 cm).
    left_margin_inches = LEFT_MARGIN_MM / MM_PER_INCH
    x = np.arange(n_samples) * time_to_inches + left_margin_inches

    # --- y axis ---
    # Scale the signal from mV to inches, then translate to the row's zero-line.
    y = signal * mv_to_inches + y_offset

    ax.plot(x, y, color="black", linewidth=0.5)

    # --- Labels ---
    # Each lead occupies an equal segment of the total sample length.
    # Place each label at the top-left corner of its segment's bounding box.
    segment_len = n_samples // n_leads
    for i, lead_name in enumerate(leads):
        # x: left edge of this segment (sample i*segment_len), shifted by the left margin
        x_label = left_margin_inches + i * segment_len * time_to_inches
        # y: top edge of the row's allocated vertical space
        y_label = y_offset + row_half_height_inches
        ax.text(x_label, y_label, lead_name, va="top", ha="left", fontsize=6,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
