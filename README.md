<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bonassifabio/pmecg/main/assets/logo-dark.svg"/>
    <img src="https://raw.githubusercontent.com/bonassifabio/pmecg/main/assets/logo.svg" alt="pmecg logo" width="333"/>
  </picture>
</p>

# pmecg — Plot My ECG

[![PyPI](https://img.shields.io/pypi/v/pmecg)](https://pypi.org/project/pmecg/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/pypi/pyversions/pmecg)](https://pypi.org/project/pmecg/)

**pmecg** is a Python library for plotting ECG signals on a paper-like support, with flexible lead configurations, time scales, and attention map overlays.

| 1×3 layout | 4×3 layout |
|:---:|:---:|
| ![1x3 ECG](https://raw.githubusercontent.com/bonassifabio/pmecg/main/example/artifacts/no-attention/1/1x3.png) | ![4x3 ECG](https://raw.githubusercontent.com/bonassifabio/pmecg/main/example/artifacts/no-attention/1/4x3.png) |

| Interval attention | Line-color attention | Background attention |
|:---:|:---:|:---:|
| ![Interval attention ECG](https://raw.githubusercontent.com/bonassifabio/pmecg/main/example/artifacts/attention/4x3+1-interval-signed.png) | ![Line-color attention ECG](https://raw.githubusercontent.com/bonassifabio/pmecg/main/example/artifacts/attention/4x3+1-line-color-signed.png) | ![Background attention ECG](https://raw.githubusercontent.com/bonassifabio/pmecg/main/example/artifacts/attention/4x3+1-background-signed.png) |

## Installation

```bash
pip install pmecg
```

## Quick start

```python
import numpy as np
import pandas as pd
import pmecg

fs = 500
t = np.linspace(0, 10, int(fs * 10))
df = pd.DataFrame({name: np.random.randn(len(t)) * 0.1 for name in pmecg.SUPPORTED_LEADS})

plotter = pmecg.ECGPlotter()
configuration = pmecg.template_factory("4x3", df, leads_map=None)
fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs)
fig.savefig("ecg.png", dpi=300, bbox_inches="tight")
```

For full documentation, visit [pmecg.readthedocs.io](https://pmecg.readthedocs.io/en/stable/).

## License

Copyright (C) 2026 Fabio Bonassi — [GNU General Public License v3](LICENSE)
