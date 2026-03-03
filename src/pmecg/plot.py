from typing import List, Literal, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .utils.data import _numpy_to_dataframe, _validate_lead_names, _apply_configuration
from .utils.plot import (
    MM_PER_INCH,
    LEFT_MARGIN_MM,
    _compute_figure_size,
    _compute_row_offsets,
    _nice_tick_step,
    _plot_grid,
    _plot_row,
)

ECGDataType = Tuple[List[np.ndarray] | np.ndarray, List[str]] | pd.DataFrame
ConfigurationDataType = List[List[str] | str] | str


class ECGPlotter:

    def __init__(self, grid_mode: Optional[Literal['inch', 'cm']] = 'cm', speed: float = 50.0, voltage: float = 20.0, row_spacing: float = 2.0, line_width: float = 0.5, grid_color: str = '#f4aaaa'):
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

    def plot(self,
             ecg_data: ECGDataType,
             configuration: ConfigurationDataType,
             sampling_frequency: float = 500.0,
             show: bool = True) -> Figure:
        """Plot the ECG in `ecg_data` using the plotting configuration specified in `configuration`.

        Parameters
        ----------
        ecg_data : ECGDataType
            The ECG data to be plotted. The following formats are supported
                - Tuple[list[np.ndarray], list[str]], where each array corresponds to a lead and has shape (n_samples,), and the list of strings contains the names of the leads
                - Tuple[np.ndarray, list[str]], where the array has shape (n_leads, n_samples) and the list of strings contains the names of the leads
                - pd.DataFrame, where each column corresponds to a lead and the column names are the names of the leads
        configuration : ConfigurationDataType
            The plotting configuration to be used. The following formats are supported:
                - List[List[str], str], where sub-lists indicate what leads are plotted in each row, while strings are used to indicate that the lead should be plotted for its entire duration.
                - str, to indicate notable templates.
        sampling_frequency : float, optional
            The sampling frequency of the ECG data in Hz, by default 500.0
        show : bool, optional
            Whether to show the plot, by default True

        Returns
        -------
        Figure
            The matplotlib figure object containing the plot
        """
        if isinstance(ecg_data, tuple):
            _validate_lead_names(ecg_data[1])
            df_data = _numpy_to_dataframe(ecg_data[0], ecg_data[1])
        elif isinstance(ecg_data, np.ndarray) or isinstance(ecg_data, list) and len(ecg_data) > 0 and all(isinstance(row, np.ndarray) for row in ecg_data):
            df_data = _numpy_to_dataframe(ecg_data)
        elif isinstance(ecg_data, pd.DataFrame):
            df_data = ecg_data
        else:
            raise ValueError("ecg_data must be a tuple of (list of numpy arrays, list of lead names), a numpy array, a list of numpy arrays, or a pandas DataFrame")

        # Apply the layout configuration → one (signal, leads) pair per row
        rows = _apply_configuration(df_data, configuration)
        n_rows = len(rows)

        # Number of samples is the same for every row (the full recording length)
        seq_len = df_data.shape[0]

        # Conversion factors
        mv_to_inches = self.voltage / MM_PER_INCH          # 1 mV  → inches
        time_to_inches = self.speed / (sampling_frequency * MM_PER_INCH)  # 1 sample → inches

        # Fixed spacing between row zero-lines
        row_spacing_inches = self.row_spacing * self.voltage / MM_PER_INCH  # row_spacing mV → inches

        # Figure dimensions
        width_inches, height_inches = _compute_figure_size(
            n_rows, seq_len, sampling_frequency, self.speed, self.voltage, self.row_spacing
        )

        # Pre-compute the zero-line y position (in inches) for every row
        y_offsets = _compute_row_offsets(n_rows, height_inches, row_spacing_inches)

        # Create figure with exact physical dimensions
        fig, ax = plt.subplots(1, 1, figsize=(width_inches, height_inches))
        ax.set_xlim(0, width_inches)
        ax.set_ylim(0, height_inches)
        ax.set_aspect("equal")

        if self.grid_mode is not None:
            _plot_grid(ax, self.grid_mode, width_inches, height_inches, self.grid_color)

        # Draw each row; half the allocated height is used to position labels
        row_half_height_inches = row_spacing_inches / 2.0
        for i, row in enumerate(rows):
            _plot_row(ax, row, mv_to_inches, time_to_inches, i, y_offsets[i], row_half_height_inches, self.line_width)

        # --- Time axis ---
        left_margin_inches = LEFT_MARGIN_MM / MM_PER_INCH

        # Choose a sensible tick spacing (0.2 s, rounded to a nice step)
        total_time_s = seq_len / sampling_frequency
        tick_step_s = _nice_tick_step(total_time_s)
        tick_times_s = np.arange(0, total_time_s + tick_step_s / 2, tick_step_s)

        # Convert time values (seconds) → x position in inches
        tick_positions_inches = tick_times_s * (self.speed / MM_PER_INCH) + left_margin_inches

        ax.set_xticks(tick_positions_inches)
        ax.set_xticklabels([f"{t:.2g} s" for t in tick_times_s], fontsize=7)

        # Remove the box: keep only the bottom spine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_visible(False)

        # Position the bottom spine exactly at the bottom of the axis
        ax.spines["bottom"].set_position(("axes", 0))

        if show:
            plt.show()

        return fig