# Plot My ECG

**pmecg** is a Python library for plotting high-quality, paper-like ECG signals using Matplotlib.
It is designed with multiple users in mind:
- researchers and clinicians who need publication-ready, customizable ECG visualisations
- Machine Learning scientists training image models for ECG classification

## Features

- Paper-like ECG rendering with correct physical dimensions (mm/s, mm/mV)
- Flexible lead layouts: built-in templates (`"4x3"`, `"2x6"`, `"1x12"`, …) or fully custom configurations
- Attention map overlays — highlight intervals, colour individual leads, or shade background regions
- Clean public API: a single `ECGPlotter` class with sensible defaults

## Quick Start

```python
import numpy as np
import pmecg

# 12-lead ECG at 500 Hz, 10 seconds
fs = 500
signal = np.random.randn(fs * 10, 12) * 0.5   # shape: (n_samples, n_leads)
lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

ecg_data = (signal, lead_names)
config = pmecg.template_factory("4x3", ecg_data, leads_map=None)

plotter = pmecg.ECGPlotter(print_information=True)
fig = plotter.plot(ecg_data, sampling_frequency=fs, configuration=config)
fig.savefig("ecg.pdf")
```

## Installation

```bash
pip install pmecg
```

## Contents

```{tableofcontents}
```

---

## Acknowledgements

Special thanks to Antônio H. Ribeiro (Uppsala University), Stefan Gustaffson (Uppsala University Hospital), and Johan Sundström (Uppsala University Hospital) for their insights, feedbacks, and contributions!

## License

Copyright (C) 2026 Fabio Bonassi

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3** as published by the Free Software Foundation.  
See [LICENSE](https://github.com/bonassifabio/pmecg/LICENSE) for the full text.
