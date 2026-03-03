import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Optional

SUPPORTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
TEMPLATE_CONFIGURATIONS = {
    '1x1': ['I'],
    '1x2': ['I', 'II'],
    '1x3': ['I', 'II', 'V2'],
    '1x4': ['I', 'II', 'III', 'V2'],
    '1x6': ['I', 'II', 'III', 'AVR', 'AVL', 'AVF'],
    '1x8': ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
    '1x12': ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
    '2x4': [['I', 'II', 'V1', 'V2',], ['V3', 'V4', 'V5', 'V6'], 'II'],
    '2x6': [['I', 'II', 'III', 'AVR', 'AVL', 'AVF'], ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 'II'],
    '4x3': [['I', 'II', 'III'], ['AVR', 'AVL', 'AVF'], ['V1', 'V2', 'V3'], ['V4', 'V5', 'V6'], 'II']
}

def _numpy_to_dataframe(ecg_data: np.ndarray, lead_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert ECG data in numpy array format to a pandas DataFrame.

    Parameters
    ----------
    ecg_data : np.ndarray | List[np.ndarray]
        The ECG data to be converted. It should either be a numpy array with shape (n_samples, n_leads) or a list of numpy arrays, each with shape (n_samples,).
    lead_names : Optional[List[str]], defaults to None
        The names of the leads corresponding to the number of leads. If None, the function will use the standard 12 leads as default.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the ECG data, where each column corresponds to a lead and the column names are the names of the leads.
    """
    if isinstance(ecg_data, np.ndarray):
        assert ecg_data.ndim == 2, "ecg_data must be a 2D numpy array with shape (n_samples, n_leads)"
        assert ecg_data.shape[1] <= len(SUPPORTED_LEADS), f"lead_names should be specified when the number of leads is not {len(SUPPORTED_LEADS)}"
        if lead_names is None:
            assert ecg_data.shape[1] == len(SUPPORTED_LEADS), "If lead_names is not provided, ecg_data must have the same number of leads as the standard 12 leads"
            # Deep copy of leads names
            lead_names = [str(lead) for lead in SUPPORTED_LEADS]
        else:
            assert ecg_data.shape[1] == len(lead_names), "The number of leads in ecg_data must match the number of lead names provided in lead_names"
    
    elif isinstance(ecg_data, list) and len(ecg_data) > 0 and all(isinstance(row, np.ndarray) for row in ecg_data):
        if lead_names is None:
            assert len(ecg_data) == len(SUPPORTED_LEADS), "If lead_names is not provided, ecg_data must have the same number of leads as the standard 12 leads"
            # Deep copy of leads names
            lead_names = [str(lead) for lead in SUPPORTED_LEADS]
        else:
            assert len(lead_names) == len(ecg_data), "The number of leads in ecg_data must match the number of lead names provided in lead_names"

        assert all(arr.shape[0] == ecg_data[0].shape[0] for arr in ecg_data), "All arrays in ecg_data must have the same length"
        ecg_data = np.stack([ecg.flatten() for ecg in ecg_data], axis=1)

    else:
        raise ValueError("ecg_data must be a numpy array or list of numpy arrays")
    
    # At this stage, we know ecg_data is a 2D numpy array with shape (n_samples, n_leads) and lead_names is a list of strings with length equal to the number of leads
    return pd.DataFrame(ecg_data, columns=[l.upper() for l in lead_names])


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
                raise ValueError(f"Lead name '{name}' in configuration is not supported. Supported leads are: {SUPPORTED_LEADS}")
    else:
        if lead_name.upper() not in SUPPORTED_LEADS:
            raise ValueError(f"Lead name '{lead_name}' in configuration is not supported. Supported leads are: {SUPPORTED_LEADS}")
    

# segment_leads 
def _segment_leads(df: pd.DataFrame, selected_leads: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Segment the ECG data so that segments of the leads are concatenated in a single vector.
       
       Let $n$ denote the number of leads in `selected_leads`, and let $N$ be the sequence length (in number of data-points).
       The output of this function will be a numpy array of shape (N,).

       The segment i*N/n to (i+1)*N/n of the output will contain the data of the lead `selected_leads[i]` for i=0,...,n-1.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ECG data, where each column corresponds to a lead and the column names are the names of the leads.
    selected_leads : List[str]
        The names of the leads to be included in the segmented DataFrame.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple containing the segmented ECG data as a numpy array and the list of selected lead names.
    """
    if isinstance(selected_leads, str):
        # Ensure we can handle the case where selected_leads is a single string instead of a list of strings
        selected_leads = [selected_leads]

    signal = np.full((df.shape[0], ), np.nan)
    segment_len = df.shape[0] // len(selected_leads)

    if df.shape[0] != len(selected_leads) * segment_len:
        warnings.warn(f"df.shape[0] ({df.shape[0]}) is not evenly divisible by the number of selected leads ({len(selected_leads)}). The resulting sequence might contain NaNs.")

    for i, lead in enumerate(selected_leads):
        start_idx = i * segment_len
        end_idx = start_idx + segment_len
        signal[start_idx:end_idx] = df[lead].values[:segment_len]

    return signal, selected_leads


def _apply_configuration(df: pd.DataFrame, configuration: List[List[str]] | str) -> Tuple[Tuple[np.ndarray, List[str]]]:
    """Apply the plotting configuration to the ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ECG data, where each column corresponds to a lead and the column names are the names of the leads.
    configuration : List[List[str]] | str
        The plotting configuration to be applied. If a list of lists of strings is provided, it indicates what leads are plotted in each row. 
        If a single lead string is provided, it indicates that only that lead should be plotted for its entire duration.
        If the configuration is a string that matches one of the keys in `TEMPLATE_CONFIGURATIONS`, the corresponding template configuration will be applied.

    Returns
    -------
    Tuple[Tuple[np.ndarray, List[str]]]
        A N-uple containing the pairs (signal: np.ndarray, selected_leads: List[str]) for each row in the configuration, where signal is the segmented ECG data corresponding to the selected leads for that row.
    """
    result = []
    if isinstance(configuration, str):
        if configuration in SUPPORTED_LEADS:
            result.append(_segment_leads(df, configuration))
        elif configuration in TEMPLATE_CONFIGURATIONS:
            result.append(_segment_leads(df, TEMPLATE_CONFIGURATIONS[configuration]))
        else:
            raise ValueError(f"configuration string '{configuration}' is not supported. It should either be a lead name or one of the following template configurations: {list(TEMPLATE_CONFIGURATIONS.keys())}")

    elif isinstance(configuration, list):
        if all(isinstance(entry, str) for entry in configuration):
            result.append(_segment_leads(df, configuration))
        elif any(isinstance(entry, list) for entry in configuration):
            for entry in configuration:
                result.append(_apply_configuration(df, entry))
        else:
            raise ValueError("configuration list must contain either strings (lead names) or sub-lists of strings (lead names for each row)")
    else:
        raise ValueError("configuration must be either a string or a list of lists of strings")
    
    return tuple(result)