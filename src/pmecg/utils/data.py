from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from pmecg.types import ConfigurationDataType, ECGDataType, LeadSegment

SUPPORTED_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
SUPPORTED_TEMPLATES = (
    "1x1",
    "1x2",
    "1x3",
    "1x4",
    "1x6",
    "1x8",
    "1x12",
    "2x4",
    "2x6",
    "4x3",
    "2x4+1",
    "2x6+1",
    "4x3+1",
    "2x4+3",
    "2x6+3",
    "4x3+3",
)

# Cabrera limb leads: the order in which the 6 limb leads appear in Cabrera format.
_CABRERA_LIMB_ORDER = ("AVL", "I", "-AVR", "II", "AVF", "III")

# Maps each standard limb lead name (as it appears in a built-in template)
# to the Cabrera-format lead name that should replace it.
_CABRERA_SUBSTITUTION: dict[str, str] = {
    "I": "AVL",
    "II": "I",
    "III": "-AVR",
    "AVR": "II",
    "AVL": "AVF",
    "AVF": "III",
}

_CABRERA_LIMB_LEADS = frozenset(("I", "II", "III", "AVR", "AVL", "AVF"))
_CONFIGURATION_ENTRY_ERROR = (
    "configuration list must contain either strings (lead names), sub-lists of strings (lead names for each row), "
    "LeadSegment objects, or sub-lists of LeadSegment objects; "
    "all rows must use the same type (all string-based or all LeadSegment-based)"
)


class LeadsMap(NamedTuple):
    """Optional mapping from canonical leads to input lead names.

    All 12 fields default to ``None``; only the leads that differ from their
    canonical names need to be specified.  For example, if the input ECG data
    uses ``"LI"`` for lead I and ``"-aVR"`` for lead AVR::

        LeadsMap(I="LI", AVR="-aVR")

    This allows the built-in templates to be resolved correctly even when the
    input data uses non-canonical lead names.
    """

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


_TEMPLATE_CONFIGURATIONS: dict[str, ConfigurationDataType] = {
    "1x1": ["I"],
    "1x2": ["I", "II"],
    "1x3": ["I", "II", "V2"],
    "1x4": ["I", "II", "III", "V2"],
    "1x6": ["I", "II", "III", "AVR", "AVL", "AVF"],
    "1x8": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    "1x12": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "2x4": [["I", "V3"], ["II", "V4"], ["III", "V5"], ["AVR", "V6"]],
    "2x4+1": [["I", "V3"], ["II", "V4"], ["III", "V5"], ["AVR", "V6"], "II"],
    "2x6": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"]],
    "2x6+1": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"], "II"],
    "4x3": [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"]],
    "4x3+1": [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"], "II"],
    "2x4+3": [["I", "V3"], ["II", "V4"], ["III", "V5"], ["AVR", "V6"], "II", "V1", "V5"],
    "2x6+3": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"], "II", "V1", "V5"],
    "4x3+3": [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"], "II", "V1", "V5"],
}


def _normalize_canonical_lead_name(lead_name: str) -> str:
    """Validate that a supported standard lead name matches exactly."""
    if lead_name not in SUPPORTED_LEADS:
        raise ValueError(f"Lead name '{lead_name}' is not supported. Supported leads are: {SUPPORTED_LEADS}")
    return lead_name


def _numpy_to_dataframe(ecg_data: np.ndarray | list[np.ndarray], lead_names: list[str] | None = None) -> pd.DataFrame:
    """Convert ECG data in numpy array format to a pandas DataFrame.

    Parameters
    ----------
    ecg_data : numpy.ndarray | list[numpy.ndarray]
        The ECG data to be converted. It should either be a numpy array with shape
        (n_samples, n_leads) or a list of numpy arrays, each with shape (n_samples,).
    lead_names : list[str] | None, defaults to None
        The names of the leads corresponding to the number of leads. If None, the
        function will use the standard 12 leads as default.

    Returns
    -------
    pandas.DataFrame
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


def _validate_and_resolve_leads_map(
    leads_map: LeadsMap | None,
    input_leads: Sequence[str],
) -> dict[str, str]:
    """Validate ``leads_map`` and convert it into the internal canonical-to-input lookup.

    Parameters
    ----------
    leads_map : LeadsMap | None
        Optional mapping whose keys are conventional ECG lead names (``I``, ``II``,
        ``AVR``, ``V1``, ...) and whose values are the corresponding column names
        present in the caller's input data.
    input_leads : Sequence[str]
        Lead names available in the provided ECG input.

    Returns
    -------
    dict[str, str]
        Mapping from canonical lead names to the matching input lead names.
        For example, ``{"I": "LI", "V1": "Chest-1"}``.

    Raises
    ------
    ValueError
        If input lead names are invalid, if a mapped input lead is missing from the
        ECG data, or if the mapping contains duplicates.
    """
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


def _resolve_template_lead(
    template: str,
    canonical_name: str,
    canonical_to_custom: dict[str, str],
    available_input_leads: set[str],
) -> str:
    """Resolve one canonical lead from a built-in template into an actual input lead name.

    Resolution order: (1) explicit entry in ``canonical_to_custom``,
    (2) canonical name present as-is in the input, (3) raise ``ValueError``.

    Returns
    -------
    str
        The resolved input column name for this lead.
    """
    canonical_key = _normalize_canonical_lead_name(canonical_name)
    if canonical_key in canonical_to_custom:
        return canonical_to_custom[canonical_key]
    if canonical_key in available_input_leads:
        return canonical_key
    raise ValueError(
        f"Template '{template}' requires conventional lead '{canonical_key}', "
        "but it is missing from leads_map and the input data"
    )


def _validate_configuration_row_definition(
    entry: list[str] | str | list[LeadSegment] | LeadSegment,
    available_input_leads: Sequence[str] | None = None,
) -> list[str] | str | list[LeadSegment] | LeadSegment:
    """Validate one row definition from a user configuration.

    A valid row definition is either:
    - a single lead name as ``str`` for a full-width row
    - a ``list[str]`` of lead names to be concatenated within the same row
    - a single :class:`~pmecg.types.LeadSegment` for a full-width row with explicit range
    - a ``list[LeadSegment]`` for leads with explicit ranges in one row

    Within a list, all entries must be the same type (all strings or all LeadSegments).

    When ``available_input_leads`` is provided, every referenced lead name must be
    present in that sequence.

    Returns
    -------
    list[str] | str | list[LeadSegment] | LeadSegment
        The validated entry, mirroring the input type (e.g. ``str`` in → ``str`` out).
    """
    if isinstance(entry, str):
        leads: list[str] | str | list[LeadSegment] | LeadSegment = entry
        leads_to_check = [entry]
    elif isinstance(entry, LeadSegment):
        if available_input_leads is not None and entry.lead not in available_input_leads:
            raise ValueError(f"Lead name '{entry.lead}' in configuration is not present in the input data")
        return entry
    elif isinstance(entry, list) and len(entry) == 0:
        raise ValueError("configuration row must not be an empty list")
    elif isinstance(entry, list):
        if all(isinstance(e, str) for e in entry):
            leads = list(entry)
            leads_to_check = leads
        elif all(isinstance(e, LeadSegment) for e in entry):
            if available_input_leads is not None:
                for e in entry:
                    if e.lead not in available_input_leads:
                        raise ValueError(f"Lead name '{e.lead}' in configuration is not present in the input data")
            return list(entry)
        else:
            raise ValueError(
                "Within a configuration row, all entries must be the same type: all strings or all LeadSegment objects"
            )
    else:
        raise ValueError(_CONFIGURATION_ENTRY_ERROR)

    if available_input_leads is not None:
        for lead_name in leads_to_check:
            if lead_name not in available_input_leads:
                raise ValueError(f"Lead name '{lead_name}' in configuration is not present in the input data")

    return leads


def _template_configuration(template: str) -> ConfigurationDataType:
    """Return a copy of the built-in row layout for ``template``.

    The returned value uses the public configuration format:
    strings for full-width rows and ``list[str]`` for concatenated rows.
    """
    try:
        template_configuration = _TEMPLATE_CONFIGURATIONS[template]
    except KeyError as exc:
        raise ValueError(f"Template '{template}' is not supported. Supported templates are: {SUPPORTED_TEMPLATES}") from exc
    return [list(entry) if isinstance(entry, list) else entry for entry in template_configuration]


def _extract_input_leads(ecg_data: ECGDataType) -> list[str]:
    """Return input lead names from a DataFrame or a tuple ECG representation."""
    if isinstance(ecg_data, pd.DataFrame):
        return [str(name) for name in ecg_data.columns]
    return [str(name) for name in ecg_data[1]]


def template_factory(template: str, ecg_data: ECGDataType, leads_map: LeadsMap | None) -> ConfigurationDataType:
    """Resolve a built-in template to an explicit configuration for the provided ECG input.

    Parameters
    ----------
    template : str
        Name of the built-in template to expand. Supported values:
        ``'1x1'``, ``'1x2'``, ``'1x3'``, ``'1x4'``, ``'1x6'``, ``'1x8'``,
        ``'1x12'``, ``'2x4'``, ``'2x6'``, ``'4x3'``, ``'2x4+1'``, ``'2x6+1'``,
        ``'4x3+1'``, ``'2x4+3'``, ``'2x6+3'``, ``'4x3+3'``.
    ecg_data : ECGDataType
        The ECG input used to resolve the final lead names. Must be the same
        object (or an object of the same type and with the same columns/lead
        names) that will later be passed to :meth:`~pmecg.ECGPlotter.plot`.
    leads_map : LeadsMap | None
        Optional mapping from conventional template lead names (``I``, ``II``,
        ``AVR``, ``V1``, …) to the corresponding column names in ``ecg_data``.
        Pass ``None`` when the input already uses the canonical names.

    Returns
    -------
    ConfigurationDataType
        Explicit plotting configuration: a list where each element is either a
        string (full-width rhythm strip) or a list of strings (leads concatenated
        within the same row).

    Raises
    ------
    ValueError
        If ``template`` is not one of the supported template names, if a
        required canonical lead is missing from both ``leads_map`` and
        ``ecg_data``, or if ``leads_map`` contains invalid or duplicate
        mappings.
    """
    input_leads = _extract_input_leads(ecg_data)
    canonical_to_custom = _validate_and_resolve_leads_map(leads_map, input_leads)
    available_input_leads = set(input_leads)
    resolved_configuration: ConfigurationDataType = []

    for entry in _template_configuration(template):
        if isinstance(entry, list):
            resolved_entry = [
                _resolve_template_lead(template, canonical_name, canonical_to_custom, available_input_leads)
                for canonical_name in entry
            ]
            resolved_configuration.append(resolved_entry)
        else:
            resolved_configuration.append(_resolve_template_lead(template, entry, canonical_to_custom, available_input_leads))

    return resolved_configuration


def cabrera_factory(
    template: str,
    ecg_data: ECGDataType,
    leads_map: LeadsMap | None = None,
) -> tuple[ECGDataType, ConfigurationDataType]:
    """Build Cabrera-ordered ECG data and plotting configuration from a template.

    Cabrera format reorders the six limb leads as AVL, I, -AVR, II, AVF, III
    (instead of the standard I, II, III, AVR, AVL, AVF) and creates a new
    ``-AVR`` lead (negated AVR).

    Parameters
    ----------
    template : str
        Name of a built-in template to expand. The template must reference
        all six limb leads (supported: ``'1x6'``, ``'1x12'``, ``'2x6'``,
        ``'4x3'``, ``'2x6+1'``, ``'4x3+1'``, ``'2x6+3'``, ``'4x3+3'``).
    ecg_data : ECGDataType
        ECG input. Must include all six limb leads. When the input uses
        non-canonical column names, provide ``leads_map`` to map them.
    leads_map : LeadsMap | None, optional
        Optional mapping from canonical lead names (``I``, ``II``, ``AVR``,
        …) to the corresponding column names in ``ecg_data``. Pass ``None``
        when the input already uses canonical names. By default ``None``.

        If the custom column name mapped to ``AVR`` starts with ``'-'``
        (e.g. ``LeadsMap(AVR='-aVR')`` or ``LeadsMap(AVR='-AVR')``), the
        data are assumed to be already negated and the sign flip is skipped;
        only the rename to ``'-AVR'`` is performed.

    Returns
    -------
    tuple[ECGDataType, ConfigurationDataType]
        A pair of ``(modified_ecg_data, cabrera_configuration)`` where:

        - ``modified_ecg_data`` is a copy of the input where the ``'AVR'``
          column (or lead) has been renamed to ``'-AVR'``. The sign is
          flipped unless the source column name already starts with ``'-'``,
          in which case the data are treated as pre-negated.
        - ``cabrera_configuration`` is the layout configuration with limb
          leads reordered into Cabrera sequence, using the same column names
          as the returned ``modified_ecg_data``. Rhythm-strip rows (string
          entries in multi-row templates) are also resolved through
          ``leads_map``.

    Raises
    ------
    ValueError
        If the template does not reference all six limb leads, or if the
        ``'AVR'`` lead is missing from the input data.
    """
    # Validate template and check it uses all 6 limb leads
    config = _template_configuration(template)
    template_leads: set[str] = set()
    for entry in config:
        if isinstance(entry, list):
            template_leads.update(entry)
        else:
            template_leads.add(entry)

    missing_limb = _CABRERA_LIMB_LEADS - template_leads
    if missing_limb:
        raise ValueError(
            f"Cabrera format requires all six limb leads in the template. "
            f"Template '{template}' is missing: {', '.join(sorted(missing_limb))}"
        )

    # Resolve leads_map and find the actual column name for AVR
    input_leads = _extract_input_leads(ecg_data)
    canonical_to_custom = _validate_and_resolve_leads_map(leads_map, input_leads)
    avr_col = canonical_to_custom.get("AVR", "AVR")
    if avr_col not in input_leads:
        raise ValueError("Cabrera format requires 'AVR' lead in the input data")

    # Replace AVR with -AVR (rename + flip sign, unless already negated)
    already_negated = avr_col.startswith("-")
    if isinstance(ecg_data, pd.DataFrame):
        new_data: ECGDataType = ecg_data.copy()
        new_data = new_data.rename(columns={avr_col: "-AVR"})
        if not already_negated:
            new_data["-AVR"] = -new_data["-AVR"]
    else:
        array, names = ecg_data
        avr_idx = [str(n) for n in names].index(avr_col)
        new_names = list(names)
        new_names[avr_idx] = "-AVR"
        if isinstance(array, np.ndarray):
            new_array: np.ndarray | list[np.ndarray] = array.copy()
            if not already_negated:
                new_array[:, avr_idx] = -new_array[:, avr_idx]
        else:
            new_array = list(array)
            if not already_negated:
                new_array[avr_idx] = -new_array[avr_idx]
        new_data = (new_array, new_names)

    # Determine if the template has mixed row types (lists + strings = has rhythm strips)
    has_list_entries = any(isinstance(e, list) for e in config)

    def _resolve(canonical: str) -> str:
        """Apply Cabrera substitution then resolve to the custom column name."""
        cabrera = _CABRERA_SUBSTITUTION.get(canonical, canonical)
        if cabrera == "-AVR":
            return "-AVR"
        return canonical_to_custom.get(cabrera, cabrera)

    # Apply Cabrera substitution and resolve to actual column names
    cabrera_config: ConfigurationDataType = []
    for entry in config:
        if isinstance(entry, list):
            cabrera_config.append([_resolve(lead) for lead in entry])
        elif has_list_entries:
            # String entry in a mixed template = rhythm strip; resolve through leads_map
            cabrera_config.append(canonical_to_custom.get(entry, entry))
        else:
            # All-string template (1xN), remap
            cabrera_config.append(_resolve(entry))

    return new_data, cabrera_config


def expand_to_12_leads(
    ecg_data: ECGDataType,
    leads_map: LeadsMap | None = None,
) -> pd.DataFrame:
    """Derive the four missing limb leads and return a full 12-lead ECG DataFrame.

    Given an 8-lead ECG containing leads I, II, V1–V6, this function computes
    the four remaining limb leads using Einthoven's law:

    .. math::

        \\text{III} &= \\text{II} - \\text{I} \\\\
        \\text{AVR} &= -\\tfrac{\\text{I} + \\text{II}}{2} \\\\
        \\text{AVL} &= \\text{I} - \\tfrac{\\text{II}}{2} \\\\
        \\text{AVF} &= \\text{II} - \\tfrac{\\text{I}}{2}

    Parameters
    ----------
    ecg_data : ECGDataType
        8-lead ECG input. Must contain leads I, II, V1–V6 (eight leads total).
        When the input uses non-canonical column names, supply ``leads_map`` to
        map them to canonical names.
    leads_map : LeadsMap | None, optional
        Mapping from canonical lead names (``I``, ``II``, ``V1``, …) to the
        column names present in ``ecg_data``. Pass ``None`` when the input
        already uses canonical names. By default ``None``.

    Returns
    -------
    pandas.DataFrame
        12-lead ECG DataFrame with columns in standard order:
        I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6.

    Raises
    ------
    ValueError
        If any of the eight required leads (I, II, V1–V6) is missing from the
        input data or the leads map.
    """
    _8_LEAD_CANONICAL = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")

    input_leads = _extract_input_leads(ecg_data)
    canonical_to_custom = _validate_and_resolve_leads_map(leads_map, input_leads)

    # Resolve each of the 8 required leads to its actual column name
    available = set(input_leads)

    def _resolve(canonical: str) -> str:
        if canonical in canonical_to_custom:
            return canonical_to_custom[canonical]
        if canonical in available:
            return canonical
        raise ValueError(f"expand_to_12_leads requires lead '{canonical}', but it is missing from leads_map and the input data")

    col = {c: _resolve(c) for c in _8_LEAD_CANONICAL}

    # Load into a DataFrame for easy arithmetic
    if isinstance(ecg_data, pd.DataFrame):
        src = ecg_data
    else:
        src = _numpy_to_dataframe(ecg_data[0], ecg_data[1])

    lead_I = src[col["I"]].values
    lead_II = src[col["II"]].values

    derived = {
        "III": lead_II - lead_I,
        "AVR": -(lead_I + lead_II) / 2.0,
        "AVL": lead_I - lead_II / 2.0,
        "AVF": lead_II - lead_I / 2.0,
    }

    data: dict[str, np.ndarray] = {}
    for canonical in SUPPORTED_LEADS:
        if canonical in derived:
            data[canonical] = derived[canonical]
        else:
            data[canonical] = src[col[canonical]].values

    return pd.DataFrame(data)


def _resolve_configuration(
    configuration: ConfigurationDataType | None,
    input_leads: Sequence[str],
) -> ConfigurationDataType | None:
    """Validate and normalise a user configuration against the input lead names.

    Each row entry is passed through :func:`_validate_configuration_row_definition`,
    which validates referenced lead names and normalises the entry type.
    """
    _validate_input_lead_names(input_leads)
    available_input_leads = list(input_leads)

    if configuration is None:
        return None

    if isinstance(configuration, list):
        resolved_configuration: ConfigurationDataType = []
        for entry in configuration:
            resolved_configuration.append(_validate_configuration_row_definition(entry, available_input_leads))
        return resolved_configuration

    raise ValueError("configuration must be a list containing lead names or lists of lead names")


def _even_leads_split(entry: list[str] | str, total_samples: int) -> list[LeadSegment]:
    """Convert a classic string-based row entry to equal-length :class:`LeadSegment` slices.

    Parameters
    ----------
    entry : list[str] | str
        A single lead name or a list of lead names.
    total_samples : int
        Total number of samples in the recording. Each lead receives
        ``total_samples // len(leads)`` samples.

    Returns
    -------
    list[LeadSegment]
        One :class:`LeadSegment` per lead with ``start``/``end`` set to equal slices.

    Warns
    -----
    UserWarning
        When ``total_samples`` is not evenly divisible by the number of leads;
        the trailing samples are silently dropped.
    """
    selected = [entry] if isinstance(entry, str) else list(entry)
    n = len(selected)
    segment_len = total_samples // n
    if total_samples != n * segment_len:
        warnings.warn(
            f"total_samples ({total_samples}) is not evenly divisible by the "
            f"number of selected leads ({n}). "
            "The last few samples will not be plotted.",
            stacklevel=3,
        )
    return [LeadSegment(lead=lead, start=i * segment_len, end=(i + 1) * segment_len) for i, lead in enumerate(selected)]


def _build_row_signal(
    df: pd.DataFrame,
    lead_configs: list[LeadSegment],
    disconnect_segments: bool = True,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build a 1-D signal by concatenating per-lead sample ranges.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the ECG data.
    lead_configs : list[LeadSegment]
        One :class:`~pmecg.types.LeadSegment` per segment.
    disconnect_segments : bool, optional
        If True, the last sample of each segment is set to NaN so that adjacent
        segments are not visually connected. By default True.

    Returns
    -------
    tuple[numpy.ndarray, list[str], list[int]]
        Concatenated signal array, the corresponding lead names, and the
        sample offset (in the concatenated signal) at which each lead starts.
    """
    total_samples = sum(lc.end - lc.start for lc in lead_configs)
    signal = np.full((total_samples,), np.nan)
    lead_names: list[str] = []
    offsets: list[int] = []
    offset = 0

    for lc in lead_configs:
        lead = lc.lead
        start = lc.start
        end = lc.end
        seg_len = end - start
        n_samples = len(df)
        if start >= n_samples or end > n_samples:
            raise ValueError(
                f"LeadSegment for lead '{lead}' requests samples [{start}:{end}], "
                f"but the DataFrame only has {n_samples} samples."
            )
        offsets.append(offset)
        signal[offset : offset + seg_len] = df[lead].values[start:end]
        if disconnect_segments and seg_len > 0:
            signal[offset + seg_len - 1] = np.nan
        lead_names.append(lead)
        offset += seg_len

    return signal, lead_names, offsets


def _apply_configuration(
    df: pd.DataFrame,
    configuration: ConfigurationDataType | None = None,
    disconnect_segments: bool = True,
) -> tuple[tuple[np.ndarray, list[str], list[int], list[LeadSegment]], ...]:
    """Apply the plotting configuration to the ECG data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the ECG data, where each column corresponds to a
        lead and the column names are the names of the leads.
    configuration : ConfigurationDataType | None, optional
        The plotting configuration to be applied.
        - If a list is provided, each element represents a row. All rows must be the
          same kind — either all string-based or all :class:`~pmecg.types.LeadSegment`-based;
          mixing is not allowed.
          - If the element is a string, it is a lead plotted for its entire duration.
          - If the element is a list of strings, those leads are concatenated in that row.
          - If the element is a LeadSegment, it is a lead with an explicit sample range.
          - If the element is a list of LeadSegments, those leads are concatenated.
        - If None, all leads in the DataFrame are plotted on separate full-width rows.
        By default None.
    disconnect_segments : bool, optional
        Passed through to :func:`_build_row_signal`. By default True.

    Returns
    -------
    tuple[tuple[numpy.ndarray, list[str], list[int], list[LeadSegment]], ...]
        A tuple of (signal, selected_leads, offsets, segments) 4-tuples — one per row
        in the configuration — where signal is the segmented ECG data for that row,
        offsets[i] is the sample index in the signal where lead i starts, and
        segments[i] is the :class:`~pmecg.types.LeadSegment` describing lead i's slice.

    Warns
    -----
    UserWarning
        When any rows that use :class:`~pmecg.types.LeadSegment` entries have
        unequal total sample counts across rows.
    """
    if configuration is None:
        configuration = [[lead] for lead in df.columns]

    result: list[tuple[np.ndarray, list[str], list[int], list[LeadSegment]]] = []
    has_lead_segment_rows = False
    if isinstance(configuration, list):
        # Validate cross-row homogeneity: all rows must be string-based or all LeadSegment-based.
        def _is_segment_entry(e: object) -> bool:
            return isinstance(e, LeadSegment) or (isinstance(e, list) and len(e) > 0 and isinstance(e[0], LeadSegment))

        has_segment = any(_is_segment_entry(e) for e in configuration)
        has_string = any(isinstance(e, (str, list)) and not _is_segment_entry(e) for e in configuration)
        if has_segment and has_string:
            raise ValueError(
                "configuration mixes string-based and LeadSegment-based rows; "
                "all rows must use the same type (all string-based or all LeadSegment-based)"
            )
        for entry in configuration:
            validated = _validate_configuration_row_definition(entry)
            is_str_list = isinstance(validated, list) and len(validated) > 0 and isinstance(validated[0], str)
            if isinstance(validated, str) or is_str_list:
                lead_configs = _even_leads_split(validated, df.shape[0])
            elif isinstance(validated, LeadSegment):
                lead_configs = [validated]
                has_lead_segment_rows = True
            else:
                lead_configs = validated  # list[LeadSegment]
                has_lead_segment_rows = True
            signal, leads, offsets = _build_row_signal(df, lead_configs, disconnect_segments)
            result.append((signal, leads, offsets, lead_configs))
    else:
        raise ValueError("configuration must be a list containing lead names or lists of lead names")

    if has_lead_segment_rows:
        row_lengths = [len(row[0]) for row in result]
        if len(set(row_lengths)) > 1:
            warnings.warn(
                f"Rows have unequal total sample counts: {row_lengths}. "
                "For consistent visual output, each row should span the same number of samples.",
                stacklevel=2,
            )

    return tuple(result)
