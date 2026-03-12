<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg"/>
    <img src="assets/logo.svg" alt="pmecg logo" width="333"/>
  </picture>
</p>

# pmecg — Plot My ECG

[![PyPI](https://img.shields.io/pypi/v/pmecg)](https://pypi.org/project/pmecg/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/pypi/pyversions/pmecg)](https://pypi.org/project/pmecg/)

**pmecg** is a Python library for plotting ECG (electrocardiogram) signals on a paper-like support, with flexible support for variable lead configurations, time scales, and visual styles.

## Examples

| 1×3 layout | 4×3 layout |
|:---:|:---:|
| ![1x3 ECG](example/artifacts/no-attention/1/1x3.png) | ![4x3 ECG](example/artifacts/no-attention/1/4x3.png) |

| Interval attention | Line-color attention | Background attention |
|:---:|:---:|:---:|
| ![Interval attention ECG](example/artifacts/attention/4x3-interval-signed.png) | ![Line-color attention ECG](example/artifacts/attention/4x3-line-color-signed.png) | ![Background attention ECG](example/artifacts/attention/4x3-background-signed.png) |

## Features

- **Paper-like Rendering**: Classic grid background with major/minor squares.
- **Flexible Layouts**: Plot any combination of leads using `pmecg.template_factory(...)` or custom configuration lists.
- **Diagnostic Metadata**: Print patient information (name, age, sex) and recording details directly on the plot.
- **Diagnostic Statistics**: Display ECG metrics (HR, SNR, HRV, QRS duration, etc.) directly on the plot.
- **Matplotlib-based**: Export to high-quality formats like PNG, PDF, or SVG.

## Installation

```bash
pip install pmecg
```

## Quick start

```python
import numpy as np
import pandas as pd
import pmecg

# 10-second, 12-lead ECG sampled at 500 Hz (synthetic example)
fs = 500
t = np.linspace(0, 10, int(fs * 10))
data = {name: np.random.randn(len(t)) * 0.1 for name in pmecg.SUPPORTED_LEADS}
df = pd.DataFrame(data)

# Create a plotter and render a standard 4x3 ECG plot
plotter = pmecg.ECGPlotter()
configuration = pmecg.template_factory("4x3", df, leads_map=None)
fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs)
fig.savefig("ecg.png", dpi=300, bbox_inches="tight")
```

## Advanced Usage

### Patient Metadata

You can annotate your plots with patient information and computed diagnostic stats:

```python
import pmecg

# 1. Define patient/recording information
info = pmecg.ECGInformation(
    hospital="General Hospital",
    patient_name="John Doe",
    age=45,
    sex="Male",
    date="2026-03-04",
    machine_model="ECG-Pro 3000"
)

# 2. Define statistics manually
stats = pmecg.ECGStats(bpm=72, rr_interval_ms=833, qrs_duration_ms=90)

# 3. Plot with information enabled
plotter = pmecg.ECGPlotter(print_information=True)
configuration = pmecg.template_factory("4x3", df, leads_map=None)
fig = plotter.plot(
    df, 
    configuration=configuration, 
    information=info, 
    stats=stats
)
```

### Lead Configurations

The `configuration` parameter in `ECGPlotter.plot` defines how leads are arranged on the plot:

- **`None` (default)**: Plots every lead present in the input ECG data on its own row for its entire duration.
- **Template-generated list**: Use `pmecg.template_factory(...)` to build a predefined layout. Supported templates include:
  - `1x1`, `1x2`, `1x3`, `1x4`, `1x6`, `1x8`, `1x12`: Single column where all leads are shown for their entire duration.
  - `2x4`, `2x6`, `4x3`: Standard multi-column layouts. `nxm` means that there are `n` rows (plus strip leads) and `m` columns (segments).
- **Custom list**: A list where each element represents a row:
  - A **single string** inside the list (e.g., `['V5', 'V6']`): Each string becomes one full-width row.
  - A **sub-list of strings** (e.g., `[['I', 'V1'], ['II', 'V2']]`): Leads in each sub-list are concatenated within that row. In this case, the first row would feature the first half of lead I, and the second half of lead V1. The second row would feature the first half of lead II and the second half of lead V2.
  - A **mix of sub-lists and strings** (e.g., `[['I', 'V1'], ['II', 'V2'], 'III']`) can be used to specify what should be in each row. Sub-lists specify what lead to print in each column of that row, while strings specify strip leads.


### Built-in Templates With `LeadsMap`

Built-in templates such as `4x3` use conventional lead names internally. Call
`pmecg.template_factory(...)` to expand the template into an explicit plotting
configuration. If your input data uses different column names, pass a
`pmecg.LeadsMap` so template slots still resolve correctly while keeping your
custom labels in the rendered plot:

```python
configuration = pmecg.template_factory(template, ecg_data, leads_map)
```

Arguments:

- `template`: the name of the built-in template to expand, such as `"4x3"` or `"2x6"`.
- `ecg_data`: the ECG input used to resolve the final lead names. This must be the same kind of object you later pass to `ECGPlotter.plot()`: either a `pd.DataFrame`, `(np.ndarray, list[str])`, or `(list[np.ndarray], list[str])`.
- `leads_map`: either `None`, when the input already uses conventional lead names such as `I`, `II`, `V1`, ..., or a `pmecg.LeadsMap` mapping conventional template leads to your custom input lead names.

In practice, this means that notable built-in templates are not passed directly to
`ECGPlotter.plot()`. You first generate an explicit configuration with
`pmecg.template_factory(...)`, then pass that resulting list to `plot()`.

Lead names are matched exactly. Lowercase labels such as `i`, `ii`, or `v1`
are treated as custom input names, so built-in templates require a
`pmecg.LeadsMap` if you want those columns to fill conventional lead names.

```python
import numpy as np
import pandas as pd
import pmecg

fs = 500
t = np.linspace(0, 10, int(fs * 10))

ecg_df = pd.DataFrame(
    {
        "Lead 1": np.random.randn(len(t)) * 0.1,
        "Lead 2": np.random.randn(len(t)) * 0.1,
        "Lead 3": np.random.randn(len(t)) * 0.1,
        "aVR-custom": np.random.randn(len(t)) * 0.1,
        "aVL-custom": np.random.randn(len(t)) * 0.1,
        "aVF-custom": np.random.randn(len(t)) * 0.1,
        "Chest-1": np.random.randn(len(t)) * 0.1,
        "Chest-2": np.random.randn(len(t)) * 0.1,
        "Chest-3": np.random.randn(len(t)) * 0.1,
        "Chest-4": np.random.randn(len(t)) * 0.1,
        "Chest-5": np.random.randn(len(t)) * 0.1,
        "Chest-6": np.random.randn(len(t)) * 0.1,
    }
)

leads_map = pmecg.LeadsMap(
    I="Lead 1",
    II="Lead 2",
    III="Lead 3",
    AVR="aVR-custom",
    AVL="aVL-custom",
    AVF="aVF-custom",
    V1="Chest-1",
    V2="Chest-2",
    V3="Chest-3",
    V4="Chest-4",
    V5="Chest-5",
    V6="Chest-6",
)

configuration = pmecg.template_factory("4x3", ecg_df, leads_map=leads_map)
fig = pmecg.ECGPlotter().plot(ecg_df, configuration=configuration, sampling_frequency=fs)
```

If you provide your own custom configuration, `leads_map` is not needed. In that case,
`ECGPlotter.plot()` matches the configuration directly against the input column names:

```python
fig = pmecg.ECGPlotter().plot(
    ecg_df,
    configuration=[["Lead 1", "Chest-1"], "Chest-6"],
    sampling_frequency=fs,
)
```

### Customizing the Plotter

The `ECGPlotter` class allows full control over the visual style:

```python
plotter = pmecg.ECGPlotter(
    speed=25.0,           # Paper speed in mm/s
    voltage=10.0,         # Amplitude in mm/mV
    row_distance=2.0,     # Vertical distance between zero lines of consecutive rows in mV
    line_width=0.7,       # Thickness of the signal
    grid_color="#e0e0e0", # Custom grid color
    show_time_axis=True   # Show time ticks at the bottom
)
```

### Attention Maps

Attention overlays are class-based. Instantiate one of
`pmecg.BackgroundAttentionMap`, `pmecg.IntervalAttentionMap`, or
`pmecg.LineColorAttentionMap`, then pass it to `ECGPlotter.plot()`.

Each attention class:

- accepts the attention data directly (`pd.DataFrame` or the same tuple formats accepted for ECG input),
- validates and aligns it against the plotted ECG leads,
- requires an explicit `polarity`:
  - `"positive"` for non-negative attention values, rendered with a single color,
  - `"signed"` for attention values spanning both negative and positive values, rendered with two colors,
- automatically rescales the prepared attention data with one global factor across all columns:
  - positive attention is divided by its global maximum only when that maximum exceeds `1`,
  - signed attention is divided by the global maximum absolute value only when that magnitude exceeds `1`,
- segments the attention values row-by-row so multi-column ECG layouts work automatically.

You can also generate attention inputs from sparse annotations before
instantiating an attention-map class:

- `pmecg.attention_map_from_indices_annotations(...)` fills a DataFrame from
  per-lead sample-index ranges.
- `pmecg.attention_map_from_time_annotations(...)` does the same from time
  ranges in seconds and internally converts them to sample-index ranges using
  the sampling frequency.

Non-interval attention maps that expose a gradient add a right-side color
scale automatically. Attention-aware layouts keep an expanded right margin so
the ECG trace keeps the same plotting area.

```python
signed_attention = pmecg.LineColorAttentionMap(
    data=pd.DataFrame({"I": np.linspace(-2.0, 1.5, len(df))}),
    polarity="signed",
    color=("blue", "red"),
)

positive_attention = pmecg.IntervalAttentionMap(
    data=pd.DataFrame({"I": np.linspace(0.0, 3.0, len(df))}),
    polarity="positive",
    color="darkorange",
    max_attention_mV=0.4,
    alpha=0.35,
)
```

```python
annotated_attention = pmecg.attention_map_from_time_annotations(
    ecg_df,
    fs=fs,
    I=[
        {"time_range": [0.25, 0.45], "attention_value": 1.0},
        {"time_range": [0.80, 1.10], "attention_value": 0.5},
    ],
    V2=[{"time_range": [0.30, 0.60], "attention_value": 0.8}],
)

indexed_attention = pmecg.attention_map_from_indices_annotations(
    ecg_df,
    I=[{"index_range": [125, 225], "attention_value": 1.0}],
    V2=[{"index_range": [150, 300], "attention_value": 0.8}],
)
```

If the attention input contains a single vector, it is broadcast to all ECG
leads before layout segmentation. Positive attention always uses the range
`[0, max(attention)]` after any automatic scaling. Signed attention keeps its
negative and positive extrema after any automatic scaling, so the color scale
still reflects the actual prepared range.

Constructor parameters:

- `pmecg.BackgroundAttentionMap(...)`: `ecg_data`, `polarity`, `color`, `show_colormap`
- `pmecg.IntervalAttentionMap(...)`: `ecg_data`, `polarity`, `color`, `max_attention_mV`, `alpha`, `show_colormap`, `smoothing_window`
- `pmecg.LineColorAttentionMap(...)`: `ecg_data`, `polarity`, `color`, `show_colormap`

The `color` parameter depends on `polarity`:

- `polarity="positive"` → pass a single matplotlib color string, such as `"red"` or `"#ff6600"`.
- `polarity="signed"` → pass a `(negative_color, positive_color)` tuple, such as `("blue", "red")`.

`LineColorAttentionMap` draws the regular black ECG trace first, then overlays a
gradient-colored line collection on top of it.

## Development

```bash
git clone https://github.com/bonassifabio/pmecg.git
cd pmecg
pixi install
pixi run lint
pixi run test-fast
```

Use `pixi run test` for the full suite in the default environment.
Use `pixi run test-all` to run the full suite across all configured Python
versions, and `pixi run test-all-fast` to do the same for the
non-integration suite.

For ad hoc commands, run the tool through Pixi directly, for example
`pixi run pytest tests/test_data.py::TestSegmentLeads -v`.

## License

Copyright (C) 2026 Fabio Bonassi

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3** as published by the Free Software Foundation.  
See [LICENSE](LICENSE) for the full text.
