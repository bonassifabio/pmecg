import numpy as np
import matplotlib.axes
from typing import List, Literal, Optional, Tuple

MM_PER_INCH = 25.4
MARGIN_MM = 5.0        # margin above the first row, below the last row, and between rows
LEFT_MARGIN_MM = 15.0  # 1.5 cm left margin (accommodates calibration pulse)
RIGHT_MARGIN_MM = 10.0 # 1 cm right margin

# Calibration pulse dimensions
CAL_PULSE_WIDTH_MM = 5.0   # 1 large square wide
CAL_PULSE_AMP_MV = 1.0     # standard 1 mV amplitude
CAL_PULSE_OFFSET_MM = 3.0  # gap from left figure edge to the rising edge


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


def _plot_calibration_pulse(
    ax: matplotlib.axes.Axes,
    mv_to_inches: float,
    y_offset: float,
    line_width: float,
) -> None:
    """Draw a 1 mV square calibration pulse (_|-|_) in the left margin for a row.

    The pulse is 1 large square wide (CAL_PULSE_WIDTH_MM = 5 mm) and 1 mV tall,
    positioned within the left margin starting at CAL_PULSE_OFFSET_MM from the
    left edge of the figure.
    """
    x0 = CAL_PULSE_OFFSET_MM / MM_PER_INCH
    x1 = (CAL_PULSE_OFFSET_MM + CAL_PULSE_WIDTH_MM) / MM_PER_INCH
    x_signal = LEFT_MARGIN_MM / MM_PER_INCH
    amp = CAL_PULSE_AMP_MV * mv_to_inches

    xs = [0.0,      x0,             x0,             x1,             x1,      x_signal]
    ys = [y_offset, y_offset,       y_offset + amp, y_offset + amp, y_offset, y_offset]
    ax.plot(xs, ys, color="black", linewidth=line_width)

    # Label centred above the top of the pulse
    x_mid = (x0 + x1) / 2
    ax.text(x_mid, y_offset + amp, "1mV", ha="center", va="bottom", fontsize=6, fontfamily="serif")


def _plot_row(
    ax: matplotlib.axes.Axes,
    row: Tuple[np.ndarray, List[str]],
    mv_to_inches: float,
    time_to_inches: float,
    row_idx: int,
    y_offset: float,
    row_half_height_inches: float,
    line_width: float,
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
    line_width : float
        Thickness of the plotted lines in points.
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

    ax.plot(x, y, color="black", linewidth=line_width)

    _plot_calibration_pulse(ax, mv_to_inches, y_offset, line_width)

    # --- Labels ---
    # Each lead occupies an equal segment of the total sample length.
    # Place each label at the top-left corner of its segment's bounding box.
    segment_len = n_samples // n_leads
    for i, lead_name in enumerate(leads):
        # x: left edge of this segment (sample i*segment_len), shifted by the left margin
        x_label = left_margin_inches + i * segment_len * time_to_inches
        # y: top edge of the row's allocated vertical space
        y_label = y_offset + row_half_height_inches
        ax.text(x_label, y_label, lead_name, va="top", ha="left", fontsize=9, fontfamily="serif")


def _plot_grid(
    ax: matplotlib.axes.Axes,
    grid_mode: Literal['cm'],
    width_inches: float,
    height_inches: float,
    grid_color: str = '#f4aaaa',
) -> None:
    """Draw a background grid on `ax` with lines spaced according to `grid_mode`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the grid on. Coordinates are in inches.
    grid_mode : {'cm'}
        Grid spacing unit. 'cm' draws lines every 0.1 cm (= 1 mm). Every 5th line is
        slightly thicker to form major squares.
    width_inches : float
        Width of the plot area in inches (used to bound vertical grid lines).
    height_inches : float
        Height of the plot area in inches (used to bound horizontal grid lines).
    grid_color : str, optional
        Color of the grid lines, by default '#f4aaaa' (light ECG-paper red).
    """
    if grid_mode == 'inch':
        raise NotImplementedError("'inch' grid mode is not supported. Use 'cm' or None.")
    step = 1.0 / MM_PER_INCH  # 0.1 cm = 1 mm expressed in inches

    minor_lw = 0.2
    major_lw = 0.6
    color = grid_color

    xs = np.arange(0, width_inches + step * 0.5, step)
    for i, x in enumerate(xs):
        lw = major_lw if i % 5 == 0 else minor_lw
        ax.axvline(x, color=color, linewidth=lw, zorder=0)

    ys = np.arange(0, height_inches + step * 0.5, step)
    for i, y in enumerate(ys):
        lw = major_lw if i % 5 == 0 else minor_lw
        ax.axhline(y, color=color, linewidth=lw, zorder=0)
