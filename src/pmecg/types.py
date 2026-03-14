"""Type aliases used throughout pmecg.

Import these in annotations to describe input/output shapes clearly::

    from pmecg.types import ECGDataType, ConfigurationDataType, AttentionDataType
"""

from __future__ import annotations

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

ConfigurationDataType = List[Union[List[str], str]]
"""Layout configuration accepted by :meth:`~pmecg.ECGPlotter.plot`.

A list where each element is either:

- a **list of lead name strings** — those leads are concatenated side-by-side in one row, or
- a **single lead name string** — that lead occupies the full row width.

Example::

    config = [
        ["I", "II", "III"],  # Leads I, II, III in the first row
        "AVR",                # Lead AVR in the second row
        ["AVL", "AVF"],       # Leads AVL and AVF in the third row
        ["V1", "V2", "V3"],   # Leads V1, V2, V3 in the fourth row
        ["V4", "V5", "V6"],   # Leads V4, V5, V6 in the fifth row
    ]
"""

AttentionArrayType = Union[np.ndarray, List[np.ndarray]]
"""Raw attention scores. The following data types are supported:

- a 2-D NumPy array of shape ``(n_leads, n_samples)``
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
]
