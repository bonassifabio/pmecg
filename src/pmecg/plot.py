from typing import List, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .utils.data import _numpy_to_dataframe, _validate_lead_names

ECGDataType = Tuple[List[np.ndarray] | np.ndarray, List[str]] | pd.DataFrame
ConfigurationDataType = List[List[str], str] | str


class ECGPlotter:

    def __init__(self, show_grid: bool = True, speed: float = 55.0, voltage: float = 20.0):
        """The ECGPlotter class can be used to generate plots for multiple ECGs using the same plotting configuration.

        Parameters
        ----------
        show_grid : bool, optional
            Whether to show the grid in the plot, by default True
        speed : float, optional
            The speed of the plot in mm/s, by default 55.0
        voltage : float, optional
            The space (in mm) corresponding to 1 mV, by default 20.0
        """
        assert isinstance(show_grid, bool), "show_grid must be a boolean"
        assert isinstance(speed, (int, float)) and speed > 0, "speed must be a positive number"
        assert isinstance(voltage, (int, float)) and voltage > 0, "voltage must be a positive number"

        self.show_grid = show_grid
        self.speed = speed
        self.voltage = voltage

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
        
        pass