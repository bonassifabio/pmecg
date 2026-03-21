"""Type aliases used throughout pmecg.

Import these in annotations to describe input/output shapes clearly::

    from pmecg.types import ECGDataType, ConfigurationDataType, LeadSegment, RhythmStripsConfig, AttentionDataType
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd

ECGDataType = Union[Tuple[Union[List[np.ndarray], np.ndarray], List[str]], pd.DataFrame]
"""ECG signal input accepted by :class:`~pmecg.ECGPlotter`.

The following formats are accepted:

- a tuple of ``(signal, lead_names)`` where ``signal`` is either a 2D NumPy array
  of shape ``(n_samples, n_leads)`` or a list of 1D arrays of shape ``(n_samples,)``
  for each lead, and ``lead_names`` is a list of strings naming the leads.
- a :class:`pandas.DataFrame` whose columns are named after the leads.
"""


@dataclass(frozen=True)
class LeadSegment:
    """Advanced per-lead configuration entry for a row.

    Parameters
    ----------
    lead : str
        Lead name as it appears in the input data.
    start : int
        First sample index (inclusive).
    end : int
        Last sample index (exclusive).

    Example::

        LeadSegment(lead='I', start=0, end=500)
    """

    lead: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.lead:
            raise ValueError("LeadSegment 'lead' must be a non-empty string")
        if self.start < 0:
            raise ValueError("LeadSegment 'start' must be non-negative")
        if self.end <= self.start:
            raise ValueError(f"LeadSegment 'end' ({self.end}) must be greater than 'start' ({self.start})")


_StringConfig = List[Union[List[str], str]]
_SegmentConfig = List[Union[List[LeadSegment], LeadSegment]]

ConfigurationDataType = Union[_StringConfig, _SegmentConfig]
"""Layout configuration accepted by :meth:`~pmecg.ECGPlotter.plot`.

Either a purely string-based layout or a purely :class:`LeadSegment`-based layout —
the two kinds **cannot be mixed** within the same configuration.

**String-based** (each row element is a ``str`` or ``list[str]``):

- a **list of lead name strings** — those leads are concatenated side-by-side in one row, or
- a **single lead name string** — that lead occupies the full row width.

**Segment-based** (each row element is a :class:`LeadSegment` or ``list[LeadSegment]``):

- a **list of LeadSegment objects** — leads with explicit start/end sample indices in one row, or
- a **single LeadSegment object** — a lead with explicit range occupying the full row width.

Example (string-based)::

    config = [
        ["I", "II", "III"],  # Leads I, II, III in the first row
        "aVR",                # Lead aVR in the second row
        ["aVL", "aVF"],       # Leads aVL and aVF in the third row
    ]

Example (segment-based)::

    config = [
        [LeadSegment(lead='I', start=0, end=500), LeadSegment(lead='II', start=0, end=500)],
        [LeadSegment(lead='III', start=0, end=1000)],
    ]
"""


@dataclass(frozen=True)
class RhythmStripsConfig:
    """Configuration for rhythm strip rows appended after the main layout rows.

    Parameters
    ----------
    ecg_data : ECGDataType
        ECG signal source for the rhythm strip rows. All leads present in this
        dataset are plotted as full-width rhythm strip rows. Lead names are derived
        from the data itself (DataFrame column names, or the lead-name list
        in the tuple form), so no separate ``leads`` argument is needed.
    speed : float | None, optional
        Paper speed in mm/s for the rhythm strips. When ``None``, the
        plotter's main speed is used. By default ``None``.
    """

    ecg_data: ECGDataType
    speed: float | None = None

    def __post_init__(self) -> None:
        if self.speed is not None and self.speed <= 0:
            raise ValueError("RhythmStripsConfig 'speed' must be a positive number")


AttentionArrayType = Union[np.ndarray, List[np.ndarray]]
"""Raw attention scores. The following data types are supported:

- a 2-D NumPy array of shape ``(n_samples, n_leads)``
- a list of 1-D arrays of shape ``(n_samples,)`` for each lead
"""

AttentionDataType = Union[Tuple[AttentionArrayType, List[str]], pd.DataFrame]
"""Attention scores input accepted by the ``*AttentionMap`` classes.

The following formats are accepted:

- a tuple of ``(scores, lead_names)`` where ``scores`` is :class:`AttentionArrayType`
  and ``lead_names`` is a list of lead name strings.
- a :class:`pandas.DataFrame` whose columns are named after the leads.
"""

AttentionPolarity = Literal["positive", "signed"]
"""Whether attention values are positive-only or span negative and positive."""

AttentionColorType = Union[str, Tuple[str, str]]
"""A single color string (positive-only) or a ``(negative, positive)`` color pair (signed).

Colors should be specified in a format accepted by Matplotlib, e.g. hex strings like ``#FF0000``
or named colors like ``"red"``.
"""

__all__ = [
    "AttentionArrayType",
    "AttentionColorType",
    "AttentionDataType",
    "AttentionPolarity",
    "ConfigurationDataType",
    "ECGDataType",
    "LeadSegment",
    "RhythmStripsConfig",
]
