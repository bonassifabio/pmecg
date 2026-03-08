from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

SUPPORTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
SUPPORTED_TEMPLATES = ("1x1", "1x2", "1x3", "1x4", "1x6", "1x8", "1x12", "2x4", "2x6", "4x3")


class LeadsMap(NamedTuple):
    """Optional mapping from canonical leads to input lead names."""

    I: str | None = None  # noqa: E741
    II: str | None = None
    III: str | None = None
    AVR: str | None = None
    AVL: str | None = None
    AVF: str | None = None
    V1: str | None = None
    V2: str | None = None
    V3: str | None = None
    V4: str | None = None
    V5: str | None = None
    V6: str | None = None


ECGDataType = tuple[list[np.ndarray] | np.ndarray, list[str]] | pd.DataFrame
ConfigurationDataType = list[list[str] | str]


def _normalize_canonical_lead_name(lead_name: str) -> str:
    """Return the conventional spelling for a supported standard lead."""
    normalized = lead_name.upper()
    if normalized not in SUPPORTED_LEADS:
        raise ValueError(f"Lead name '{lead_name}' is not supported. Supported leads are: {SUPPORTED_LEADS}")
    return normalized


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
    return pd.DataFrame(ecg_data, columns=[str(name) for name in lead_names])


def _validate_input_lead_names(lead_names: Sequence[str]) -> None:
    """Validate that user-provided input lead names are unique and non-empty."""
    normalized_names = [str(name) for name in lead_names]
    if any(len(name) == 0 for name in normalized_names):
        raise ValueError("Lead names must be non-empty strings")

    duplicates = sorted(name for name, count in Counter(normalized_names).items() if count > 1)
    if duplicates:
        duplicate_names = ", ".join(repr(name) for name in duplicates)
        raise ValueError(f"Duplicate lead names are not allowed: {duplicate_names}")


def _normalize_leads_map(
    leads_map: LeadsMap | None,
    input_leads: Sequence[str],
) -> dict[str, str]:
    """Validate and normalize a conventional-to-custom lead-name mapping."""
    _validate_input_lead_names(input_leads)

    if leads_map is None:
        return {}

    canonical_to_custom: dict[str, str] = {}
    available_input_leads = set(input_leads)

    for canonical_name, custom_name in leads_map._asdict().items():
        if custom_name is None:
            continue
        canonical_key = _normalize_canonical_lead_name(str(canonical_name))
        custom_value = str(custom_name)
        if custom_value not in available_input_leads:
            raise ValueError(
                f"Leads map value '{custom_value}' for conventional lead '{canonical_key}' is not present in the input data"
            )
        if canonical_key in canonical_to_custom:
            raise ValueError(f"Duplicate conventional lead mapping for '{canonical_key}' is not allowed")
        if custom_value in canonical_to_custom.values():
            raise ValueError(f"Duplicate custom lead name '{custom_value}' in leads_map is not allowed")

        canonical_to_custom[canonical_key] = custom_value

    return canonical_to_custom


def _template_configuration(template: str) -> ConfigurationDataType:
    """Return the conventional configuration for a supported built-in template."""
    if template == "1x1":
        return ["I"]
    if template == "1x2":
        return ["I", "II"]
    if template == "1x3":
        return ["I", "II", "V2"]
    if template == "1x4":
        return ["I", "II", "III", "V2"]
    if template == "1x6":
        return ["I", "II", "III", "AVR", "AVL", "AVF"]
    if template == "1x8":
        return ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    if template == "1x12":
        return ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if template == "2x4":
        return [["I", "V3"], ["II", "V4"], ["III", "V5"], ["AVR", "V6"], "II"]
    if template == "2x6":
        return [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"], "II"]
    if template == "4x3":
        return [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"], "II"]
    raise ValueError(f"Template '{template}' is not supported. Supported templates are: {SUPPORTED_TEMPLATES}")


def _extract_input_leads(ecg_data: ECGDataType) -> list[str]:
    """Return input lead names from a DataFrame or a tuple ECG representation."""
    if isinstance(ecg_data, pd.DataFrame):
        return [str(name) for name in ecg_data.columns]
    return [str(name) for name in ecg_data[1]]


def template_factory(template: str, ecg_data: ECGDataType, leads_map: LeadsMap | None) -> ConfigurationDataType:
    """Resolve a built-in template to an explicit configuration for the provided ECG input."""
    input_leads = _extract_input_leads(ecg_data)
    canonical_to_custom = _normalize_leads_map(leads_map, input_leads)
    available_input_leads = set(input_leads)
    resolved_configuration: ConfigurationDataType = []

    for entry in _template_configuration(template):
        if isinstance(entry, list):
            resolved_entry = []
            for canonical_name in entry:
                canonical_key = _normalize_canonical_lead_name(canonical_name)
                if canonical_key in canonical_to_custom:
                    resolved_entry.append(canonical_to_custom[canonical_key])
                elif canonical_key in available_input_leads:
                    resolved_entry.append(canonical_key)
                else:
                    raise ValueError(
                        f"Template '{template}' requires conventional lead '{canonical_key}', "
                        "but it is missing from leads_map and the input data"
                    )
            resolved_configuration.append(resolved_entry)
        else:
            canonical_key = _normalize_canonical_lead_name(entry)
            if canonical_key in canonical_to_custom:
                resolved_configuration.append(canonical_to_custom[canonical_key])
            elif canonical_key in available_input_leads:
                resolved_configuration.append(canonical_key)
            else:
                raise ValueError(
                    f"Template '{template}' requires conventional lead '{canonical_key}', "
                    "but it is missing from leads_map and the input data"
                )

    return resolved_configuration


def _resolve_configuration(
    configuration: ConfigurationDataType | None,
    input_leads: Sequence[str],
) -> ConfigurationDataType | None:
    """Validate that a user configuration only references lead names present in the input data."""
    _validate_input_lead_names(input_leads)
    available_input_leads = list(input_leads)

    if configuration is None:
        return None

    if isinstance(configuration, list):
        resolved_configuration: ConfigurationDataType = []
        for entry in configuration:
            if isinstance(entry, str):
                if entry not in available_input_leads:
                    raise ValueError(f"Lead name '{entry}' in configuration is not present in the input data")
                resolved_configuration.append(entry)
            elif isinstance(entry, list):
                resolved_entry = []
                for lead_name in entry:
                    if lead_name not in available_input_leads:
                        raise ValueError(f"Lead name '{lead_name}' in configuration is not present in the input data")
                    resolved_entry.append(lead_name)
                resolved_configuration.append(resolved_entry)
            else:
                raise ValueError(
                    "configuration list must contain either strings (lead names) "
                    "or sub-lists of strings (lead names for each row)"
                )
        return resolved_configuration

    raise ValueError("configuration must be a list containing lead names or lists of lead names")


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
    configuration: ConfigurationDataType | None = None,
    disconnect_segments: bool = True,
) -> tuple[tuple[np.ndarray, list[str]], ...]:
    """Apply the plotting configuration to the ECG data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the ECG data, where each column corresponds to a
        lead and the column names are the names of the leads.
    configuration : ConfigurationDataType | None, optional
        The plotting configuration to be applied.
        - If a list is provided, each element represents a row.
          - If the element is a string, it is a lead plotted for its entire duration.
          - If the element is a list of strings, those leads are concatenated in that row.
        - If None, all leads in the DataFrame are plotted on separate full-width rows.
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
    if isinstance(configuration, list):
        for entry in configuration:
            if isinstance(entry, (str, list)):
                result.append(_segment_leads(df, entry, disconnect_segments))
            else:
                raise ValueError(
                    "configuration list must contain either strings (lead names) "
                    "or sub-lists of strings (lead names for each row)"
                )
    else:
        raise ValueError("configuration must be a list containing lead names or lists of lead names")

    return tuple(result)
