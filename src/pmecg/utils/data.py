from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SUPPORTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
TEMPLATE_CONFIGURATIONS = {
    "1x1": ["I"],
    "1x2": ["I", "II"],
    "1x3": ["I", "II", "V2"],
    "1x4": ["I", "II", "III", "V2"],
    "1x6": ["I", "II", "III", "AVR", "AVL", "AVF"],
    "1x8": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    "1x12": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "2x4": [["I", "V3"], ["II", "V4"], ["III", "V5"], ["AVR", "V6"], "II"],
    "2x6": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"], "II"],
    "4x3": [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"], "II"],
}


def _numpy_to_dataframe(ecg_data: np.ndarray, lead_names: list[str] | None = None) -> pd.DataFrame:
    """Convert ECG data in numpy array format to a pandas DataFrame.

    Parameters
    ----------
    ecg_data : np.ndarray | list[np.ndarray]
        The ECG data to be converted. It should either be a numpy array with shape
        (n_samples, n_leads) or a list of numpy arrays, each with shape (n_samples,).
    lead_names : list[str] | None, defaults to None
        The names of the leads corresponding to the number of leads. If None, the
        function will use the standard 12 leads as default.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the ECG data, where each column corresponds to
        a lead and the column names are the names of the leads.
    """
    if isinstance(ecg_data, np.ndarray):
        assert ecg_data.ndim == 2, "ecg_data must be a 2D numpy array with shape (n_samples, n_leads)"
        assert ecg_data.shape[1] <= len(SUPPORTED_LEADS), (
            f"lead_names should be specified when the number of leads is not {len(SUPPORTED_LEADS)}"
        )
        if lead_names is None:
            assert ecg_data.shape[1] == len(SUPPORTED_LEADS), (
                "If lead_names is not provided, ecg_data must have the same number of leads as the standard 12 leads"
            )
            # Deep copy of leads names
            lead_names = [str(lead) for lead in SUPPORTED_LEADS]
        else:
            assert ecg_data.shape[1] == len(lead_names), (
                "The number of leads in ecg_data must match the number of lead names provided in lead_names"
            )

    elif isinstance(ecg_data, list) and len(ecg_data) > 0 and all(isinstance(row, np.ndarray) for row in ecg_data):
        if lead_names is None:
            assert len(ecg_data) == len(SUPPORTED_LEADS), (
                "If lead_names is not provided, ecg_data must have the same number of leads as the standard 12 leads"
            )
            # Deep copy of leads names
            lead_names = [str(lead) for lead in SUPPORTED_LEADS]
        else:
            assert len(lead_names) == len(ecg_data), (
                "The number of leads in ecg_data must match the number of lead names provided in lead_names"
            )

        assert all(arr.shape[0] == ecg_data[0].shape[0] for arr in ecg_data), "All arrays in ecg_data must have the same length"
        ecg_data = np.stack([ecg.flatten() for ecg in ecg_data], axis=1)

    else:
        raise ValueError("ecg_data must be a numpy array or list of numpy arrays")

    # At this stage, ecg_data is a 2D numpy array with shape (n_samples, n_leads)
    # and lead_names is a list of strings with length equal to the number of leads.
    return pd.DataFrame(ecg_data, columns=[name.upper() for name in lead_names])


def _validate_lead_names(lead_name: str | list[str]) -> None:
    """Validate that the provided lead name is supported.

    Parameters
    ----------
    lead_name : str | list[str]
        The name, or list of names, of the lead(s) to be validated.

    Raises
    ------
    ValueError
        If the provided lead name is not supported.
    """
    if isinstance(lead_name, list):
        for name in lead_name:
            if name.upper() not in SUPPORTED_LEADS:
                raise ValueError(
                    f"Lead name '{name}' in configuration is not supported. Supported leads are: {SUPPORTED_LEADS}"
                )
    else:
        if lead_name.upper() not in SUPPORTED_LEADS:
            raise ValueError(
                f"Lead name '{lead_name}' in configuration is not supported. Supported leads are: {SUPPORTED_LEADS}"
            )


# segment_leads
def _segment_leads(
    df: pd.DataFrame, selected_leads: list[str], disconnect_segments: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Segment the ECG data so that segments of the leads are concatenated in a single vector.

       Let $n$ denote the number of leads in `selected_leads`, and let $N$ be the sequence length (in number of data-points).
       The output of this function will be a numpy array of shape (N,).

       The segment i*N/n to (i+1)*N/n of the output will contain the data of the lead `selected_leads[i]` for i=0,...,n-1.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ECG data, where each column corresponds to a
        lead and the column names are the names of the leads.
    selected_leads : list[str]
        The names of the leads to be included in the segmented DataFrame.
    disconnect_segments : bool, optional
        If True, the last sample of each segment is set to NaN so that adjacent
        segments are not visually connected in the plot. By default True.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A tuple containing the segmented ECG data as a numpy array and the list of selected lead names.
    """
    if isinstance(selected_leads, str):
        # Ensure we can handle the case where selected_leads is a single string instead of a list of strings
        selected_leads = [selected_leads]

    signal = np.full((df.shape[0],), np.nan)
    segment_len = df.shape[0] // len(selected_leads)

    if df.shape[0] != len(selected_leads) * segment_len:
        warnings.warn(
            f"df.shape[0] ({df.shape[0]}) is not evenly divisible by the "
            f"number of selected leads ({len(selected_leads)}). "
            "The resulting sequence might contain NaNs.",
            stacklevel=2,
        )

    for i, lead in enumerate(selected_leads):
        start_idx = i * segment_len
        end_idx = start_idx + segment_len
        signal[start_idx:end_idx] = df[lead].values[:segment_len]
        if disconnect_segments:
            signal[end_idx - 1] = np.nan

    return signal, selected_leads


def _apply_configuration(
    df: pd.DataFrame,
    configuration: list[list[str] | str] | str | None = None,
    disconnect_segments: bool = True,
) -> tuple[tuple[np.ndarray, list[str]], ...]:
    """Apply the plotting configuration to the ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ECG data, where each column corresponds to a
        lead and the column names are the names of the leads.
    configuration : list[list[str] | str] | str | None, optional
        The plotting configuration to be applied.
        - If a list is provided, each element represents a row.
          - If the element is a string, it is a lead plotted for its entire duration.
          - If the element is a list of strings, those leads are concatenated in that row.
        - If the configuration is a string that matches one of the keys in
          ``TEMPLATE_CONFIGURATIONS``, the corresponding template is applied.
        - If None, all leads in the DataFrame are plotted for their entire duration.
        By default None.
    disconnect_segments : bool, optional
        Passed through to :func:`_segment_leads`. By default True.

    Returns
    -------
    tuple[tuple[np.ndarray, list[str]], ...]
        A tuple of (signal, selected_leads) pairs — one per row in the
        configuration — where signal is the segmented ECG data for that row.
    """
    if configuration is None:
        configuration = [[lead] for lead in df.columns]

    result = []
    if isinstance(configuration, str):
        if configuration in SUPPORTED_LEADS:
            result.append(_segment_leads(df, configuration, disconnect_segments))
        elif configuration in TEMPLATE_CONFIGURATIONS:
            result.extend(_apply_configuration(df, TEMPLATE_CONFIGURATIONS[configuration], disconnect_segments))
        else:
            raise ValueError(
                f"configuration string '{configuration}' is not supported. "
                "It should either be a lead name or one of the following "
                f"template configurations: {list(TEMPLATE_CONFIGURATIONS.keys())}"
            )

    elif isinstance(configuration, list):
        for entry in configuration:
            if isinstance(entry, (str, list)):
                result.append(_segment_leads(df, entry, disconnect_segments))
            else:
                raise ValueError(
                    "configuration list must contain either strings (lead names) "
                    "or sub-lists of strings (lead names for each row)"
                )
    else:
        raise ValueError("configuration must be either a string or a list of lists of strings")

    return tuple(result)
