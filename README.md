# pmecg — Plot My ECG

[![PyPI](https://img.shields.io/pypi/v/pmecg)](https://pypi.org/project/pmecg/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/pypi/pyversions/pmecg)](https://pypi.org/project/pmecg/)

**pmecg** is a Python library for plotting ECG (electrocardiogram) signals on a paper-like support, with flexible support for variable lead configurations, time scales, and visual styles.

## Features

- Render ECG signals on a classic grid paper background
- Support for any lead configuration (1-lead, 6-lead, 12-lead, …)
- Configurable time/amplitude scales
- Matplotlib-based — export to PNG, PDF, SVG, and more

## Installation

```bash
pip install pmecg
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pmecg
```

## Quick start

```python
import numpy as np
import pmecg

# 10-second, 12-lead ECG sampled at 500 Hz (synthetic example)
fs = 500
t = np.linspace(0, 10, fs * 10)
leads = {name: np.random.randn(len(t)) * 0.5 for name in pmecg.STANDARD_12_LEADS}

fig = pmecg.plot(leads, fs=fs)
fig.savefig("ecg.png", dpi=150, bbox_inches="tight")
```

## Development

```bash
git clone https://github.com/bonassifabio/pmecg.git
cd pmecg
uv sync --all-groups
uv run pytest
```

## License

Copyright (C) 2026 Fabio Bonassi

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3** as published by the Free Software Foundation.  
See [LICENSE](LICENSE) for the full text.
