# pmecg — Plot My ECG
# Copyright (C) 2026 Fabio Bonassi <fabio.bonassi@polimi.it>
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pmecg")
except PackageNotFoundError:
    __version__ = "unknown"
from .plot import (
    AbstractAttentionMap,
    BackgroundAttentionMap,
    ECGInformation,
    ECGPlotter,
    ECGStats,
    IntervalAttentionMap,
    LineColorAttentionMap,
    attention_map_from_indices_annotations,
    attention_map_from_time_annotations,
)
from .types import LeadSegment, RhythmStripsConfig
from .utils.data import SUPPORTED_LEADS, LeadsMap, cabrera_factory, expand_to_12_leads, template_factory

__all__ = [
    "AbstractAttentionMap",
    "BackgroundAttentionMap",
    "ECGInformation",
    "ECGPlotter",
    "ECGStats",
    "IntervalAttentionMap",
    "LeadSegment",
    "LeadsMap",
    "LineColorAttentionMap",
    "SUPPORTED_LEADS",
    "RhythmStripsConfig",
    "attention_map_from_indices_annotations",
    "attention_map_from_time_annotations",
    "cabrera_factory",
    "expand_to_12_leads",
    "template_factory",
]
