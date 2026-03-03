# pmecg — Plot My ECG
# Copyright (C) 2026 Fabio Bonassi <fabio.bonassi@polimi.it>
# SPDX-License-Identifier: GPL-3.0-or-later

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pmecg")
except PackageNotFoundError:
    __version__ = "unknown"
__author__ = "Fabio Bonassi"
