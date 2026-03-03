from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .utils.data import (
    _apply_configuration,
    _numpy_to_dataframe,
    _validate_lead_names,
)
from .utils.plot import (
    LEFT_MARGIN_MM,
    MM_PER_INCH,
    _compute_figure_size,
    _compute_row_offsets,
    _nice_tick_step,
    _plot_grid,
    _plot_row,
    _print_information,
    _RenderContext,
)


@dataclass
class ECGStats:
    """Computed ECG diagnostic statistics to be printed on the plot.

    All fields default to ``None`` (not shown). Any field that is set will be
    displayed in the top-right corner when ``print_diagnostics=True``, arranged
    in columns of three rows.

    Parameters
    ----------
    bpm : float, optional
        Heart rate in beats per minute.
    snr : float, optional
        Signal-to-noise ratio in dB.
    rr_interval_ms : float, optional
        Mean RR interval (beat-to-beat) in milliseconds.
    hrv_ms : float, optional
        Heart-rate variability — statistical spread of RR intervals (ms).
    pr_interval_ms : float, optional
        PR interval in milliseconds.
    qrs_duration_ms : float, optional
        QRS complex duration in milliseconds.
    qt_interval_ms : float, optional
        QT interval in milliseconds.
    qtc_interval_ms : float, optional
        Corrected QT interval (QTc) in milliseconds.
    p_axis_deg : float, optional
        P-wave axis in degrees.
    qrs_axis_deg : float, optional
        QRS axis in degrees.
    t_axis_deg : float, optional
        T-wave axis in degrees.
    """
    bpm: float | None = None
    snr: float | None = None
    rr_interval_ms: float | None = None
    hrv_ms: float | None = None
    pr_interval_ms: float | None = None
    qrs_duration_ms: float | None = None
    qt_interval_ms: float | None = None
    qtc_interval_ms: float | None = None
    p_axis_deg: float | None = None
    qrs_axis_deg: float | None = None
    t_axis_deg: float | None = None


@dataclass
class ECGInformation:
    """Patient and recording metadata to be printed on the ECG plot.

    Parameters
    ----------
    hospital : str, optional
        Name of the hospital or clinic where the ECG was recorded.
    patient_name : str, optional
        Name of the patient.
    date : str, optional
        Date of the recording (any human-readable format, e.g. "2024-01-15").
    machine_model : str, optional
        ECG machine model, printed in the bottom-right corner.
    """
    hospital: str | None = None
    patient_name: str | None = None
    date: str | None = None
    machine_model: str | None = None

ECGDataType = Union[
    tuple[Union[list[np.ndarray], np.ndarray], list[str]],
    pd.DataFrame,
]
ConfigurationDataType = Union[list[Union[list[str], str]], str]


class ECGPlotter:

    def __init__(
        self,
        grid_mode: Literal['inch', 'cm'] | None = 'cm',
        speed: float = 50.0,
        voltage: float = 20.0,
        row_spacing: float = 2.0,
        line_width: float = 0.5,
        grid_color: str = '#f4aaaa',
        print_diagnostics: bool = False,
        show_time_axis: bool = False,
        disconnect_segments: bool = True,
    ):
        """The ECGPlotter class can be used to generate plots for multiple ECGs using the same plotting configuration.

        Parameters
        ----------
        grid_mode : {'cm'} or None, optional
            Grid style to overlay on the plot. 'cm' draws lines every 0.1 cm (= 1 mm),
            with every 5th line slightly thicker. Pass None to disable the grid.
            By default 'cm'.
        speed : float, optional
            The speed of the plot in mm/s, by default 50.0
        voltage : float, optional
            The space (in mm) corresponding to 1 mV, by default 20.0
        row_spacing : float, optional
            Distance between the zero-lines of consecutive rows, expressed in mV, by default 2.0
        line_width : float, optional
            Thickness of the ECG signal lines (and calibration pulse) in points, by default 0.5
        grid_color : str, optional
            Color of the grid lines. Any matplotlib color string is accepted (e.g. '#f4aaaa',
            'lightgray', 'gray'). By default '#f4aaaa' (light ECG-paper red).
        print_diagnostics : bool, optional
            Whether to print diagnostic parameters (speed, voltage, sampling frequency,
            leads) and any extra metadata in the corners of the figure, by default False.
        show_time_axis : bool, optional
            Whether to show the time axis (x-axis ticks and spine) at the bottom of the
            figure, by default False.
        disconnect_segments : bool, optional
            If True, the last sample of each segment is set to NaN so that adjacent
            segments are not visually connected in the plot. By default True.
        """
        assert grid_mode in (None, 'cm'), "grid_mode must be None or 'cm'"
        assert isinstance(speed, (int, float)) and speed > 0, "speed must be a positive number"
        assert isinstance(voltage, (int, float)) and voltage > 0, "voltage must be a positive number"
        assert isinstance(row_spacing, (int, float)) and row_spacing > 0, "row_spacing must be a positive number"
        assert isinstance(line_width, (int, float)) and line_width > 0, "line_width must be a positive number"
        assert isinstance(grid_color, str) and len(grid_color) > 0, "grid_color must be a non-empty string"

        self.grid_mode = grid_mode
        self.speed = speed
        self.voltage = voltage
        self.row_spacing = row_spacing
        self.line_width = line_width
        self.grid_color = grid_color
        self.print_diagnostics = print_diagnostics
        self.show_time_axis = show_time_axis
        self.disconnect_segments = disconnect_segments

    def plot(self,
             ecg_data: ECGDataType,
             configuration: ConfigurationDataType,
             sampling_frequency: float = 500.0,
             show: bool = True,
             information: ECGInformation | None = None,
             stats: ECGStats | None = None) -> Figure:
        """Plot the ECG in `ecg_data` using the plotting configuration specified in `configuration`.

        Parameters
        ----------
        ecg_data : ECGDataType
            The ECG data to be plotted. The following formats are supported
                - tuple[list[np.ndarray], list[str]], where each array corresponds to a lead
                  and has shape (n_samples,), and the list of strings contains the lead names
                - tuple[np.ndarray, list[str]], where the array has shape (n_samples, n_leads)
                  and the list of strings contains the names of the leads
                - pd.DataFrame, where each column corresponds to a lead and the column names are the names of the leads
        configuration : ConfigurationDataType
            The plotting configuration to be used. The following formats are supported:
                - list[list[str] | str], where sub-lists indicate what leads are plotted in
                  each row, while strings are used to indicate that the lead should be plotted
                  for its entire duration.
                - str, to indicate notable templates.
        sampling_frequency : float, optional
            The sampling frequency of the ECG data in Hz, by default 500.0
        show : bool, optional
            Whether to show the plot, by default True
        information : ECGInformation, optional
            Patient and recording metadata. When ``self.print_diagnostics`` is True, the
            hospital, patient name and date are printed above the first ECG row, and
            the machine model is printed in the bottom-right corner.
        stats : ECGStats, optional
            Computed ECG statistics. When ``self.print_diagnostics`` is True, any
            non-None field is printed in the top-right corner, arranged in columns
            of up to three rows.

        Returns
        -------
        Figure
            The matplotlib figure object containing the plot
        """
        if isinstance(ecg_data, tuple):
            _validate_lead_names(ecg_data[1])
            df_data = _numpy_to_dataframe(ecg_data[0], ecg_data[1])
        elif (
            isinstance(ecg_data, np.ndarray)
            or (
                isinstance(ecg_data, list)
                and len(ecg_data) > 0
                and all(isinstance(row, np.ndarray) for row in ecg_data)
            )
        ):
            df_data = _numpy_to_dataframe(ecg_data)
        elif isinstance(ecg_data, pd.DataFrame):
            df_data = ecg_data
        else:
            raise ValueError(
                "ecg_data must be a tuple of (list of numpy arrays, list of lead names), "
                "a numpy array, a list of numpy arrays, or a pandas DataFrame"
            )

        # Apply the layout configuration → one (signal, leads) pair per row
        rows = _apply_configuration(df_data, configuration, self.disconnect_segments)
        n_rows = len(rows)

        # Number of samples is the same for every row (the full recording length)
        seq_len = df_data.shape[0]

        # Conversion factors and per-call render context
        ctx = _RenderContext(
            mv_to_inches=self.voltage / MM_PER_INCH,
            time_to_inches=self.speed / (sampling_frequency * MM_PER_INCH),
            row_spacing_inches=self.row_spacing * self.voltage / MM_PER_INCH,
            line_width=self.line_width,
            grid_color=self.grid_color,
            speed=self.speed,
            voltage=self.voltage,
        )

        # Figure dimensions
        width_inches, height_inches = _compute_figure_size(
            n_rows, seq_len, sampling_frequency, self.speed, self.voltage, self.row_spacing,
            print_diagnostics=self.print_diagnostics,
        )

        # Pre-compute the zero-line y position (in inches) for every row
        y_offsets = _compute_row_offsets(
            n_rows, height_inches, ctx.row_spacing_inches, self.print_diagnostics,
        )

        # Create figure with exact physical dimensions
        fig, ax = plt.subplots(1, 1, figsize=(width_inches, height_inches))
        ax.set_xlim(0, width_inches)
        ax.set_ylim(0, height_inches)
        ax.set_aspect("equal")

        if self.grid_mode is not None:
            _plot_grid(ax, self.grid_mode, width_inches, height_inches, ctx)

        for i, row in enumerate(rows):
            _plot_row(ax, row, ctx, y_offsets[i])

        # --- Time axis ---
        left_margin_inches = LEFT_MARGIN_MM / MM_PER_INCH

        if self.show_time_axis:
            # Choose a sensible tick spacing (0.2 s, rounded to a nice step)
            total_time_s = seq_len / sampling_frequency
            tick_step_s = _nice_tick_step(total_time_s)
            tick_times_s = np.arange(0, total_time_s + tick_step_s / 2, tick_step_s)

            # Convert time values (seconds) → x position in inches
            tick_positions_inches = tick_times_s * (self.speed / MM_PER_INCH) + left_margin_inches

            ax.set_xticks(tick_positions_inches)
            ax.set_xticklabels([f"{t:.2g} s" for t in tick_times_s], fontsize=7, fontfamily="monospace")
            ax.spines["bottom"].set_position(("axes", 0))
        else:
            ax.xaxis.set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Remove the box: keep only the bottom spine (when visible)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_visible(False)

        if self.print_diagnostics:
            original_leads = list(df_data.columns)
            first_row_top_inches = y_offsets[0] + ctx.row_spacing_inches / 2.0
            _print_information(
                ax,
                ctx,
                width_inches,
                sampling_frequency,
                original_leads,
                first_row_top_inches,
                information=information,
                stats=stats,
            )

        if show:
            plt.show()

        return fig
