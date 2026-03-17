---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  name: pmecg-docs
  display_name: 'Python 3 (pmecg docs)'
  language: python
---

# Attention Maps

Attention maps let you overlay a per-sample scalar score on top of the ECG
trace. A typical use-case is visualising the output of a neural network: which
time steps did the model attend to when it produced a particular prediction?

This page shows how to build attention data, choose the right map style, and
use the built-in annotation helpers.

## Setup

```{code-cell} python
import wfdb
import numpy as np
import pandas as pd
import pmecg

record = wfdb.rdrecord('00001_hr', pn_dir='ptb-xl/1.0.3/records500/00000/')
ecg_df = pd.DataFrame(record.p_signal, columns=record.sig_name)
fs = record.fs

n_samples = len(ecg_df)
```

## Creating Attention Data

Attention data must be a `pandas.DataFrame` whose columns match the ECG lead
names (or a `(array, lead_names)` tuple — the same formats accepted by
`ECGPlotter.plot`).

Here we simulate model attention using a sine wave on two leads. Oscillating
between −1 and +1 makes this a natural example of **signed** attention —
positive values mean the model attended strongly in one direction, negative in
the other.

```{code-cell} python
period = n_samples / 3  # one cycle every third of the recording → 3 full cycles
t = np.arange(n_samples)

# When the attention DataFrame has more than one column, every ECG lead must
# be present. We start from a zero-filled DataFrame (one column per lead) and
# then assign values only to the two leads of interest.
attention_df = pd.DataFrame(0.0, index=ecg_df.index, columns=ecg_df.columns)
attention_df['II'] = np.sin(2 * np.pi * t / period)            # full range [-1, 1]
attention_df['V5'] = np.sin(2 * np.pi * t / period + np.pi)   # same shape, phase-shifted
```

All three map types below accept this DataFrame directly.

---

## Interval Attention Map

`IntervalAttentionMap` draws a band **around the ECG trace** whose half-width
scales with the attention magnitude. This is the least visually intrusive style
— it leaves the trace itself untouched.

```{code-cell} python
# polarity='signed' → values may be negative or positive.
# color=(negative_color, positive_color) — two strings, one per sign.
interval_map = pmecg.IntervalAttentionMap(
    attention_df,
    polarity='signed',
    color=('steelblue', 'tomato'),   # blue for negative, red for positive
    max_attention_mV=0.3,            # band half-width at full attention strength
    alpha=0.4,
    smoothing_window=4,             # light smoothing to reduce visual noise
)

plotter = pmecg.ECGPlotter(grid_mode='cm')
fig = plotter.plot(
    ecg_df,
    configuration=pmecg.template_factory('4x3', ecg_df, leads_map=None),
    sampling_frequency=fs,
    attention_map=interval_map,
    show=True,
)
```

---

## Background Attention Map

`BackgroundAttentionMap` fills the **full height of each row** with a
semi-transparent color block. The opacity scales with the attention magnitude,
so high-attention regions are visually dominant.

```{code-cell} python
# show_colormap=True (default) adds a vertical color scale on the right margin.
# The plotter automatically widens the figure to preserve the ECG trace area.
background_map = pmecg.BackgroundAttentionMap(
    attention_df,
    polarity='signed',
    color=('steelblue', 'tomato'),
    show_colormap=True,
)

fig = plotter.plot(
    ecg_df,
    configuration=pmecg.template_factory('4x3', ecg_df, leads_map=None),
    sampling_frequency=fs,
    attention_map=background_map,
    show=True,
)
```

---

## Line Color Attention Map

`LineColorAttentionMap` **recolors the ECG trace itself**: each segment of the
line is assigned a color whose opacity reflects the local attention value.
Segments with zero attention are invisible, so the original black trace
disappears where the model did not attend.

```{code-cell} python
line_map = pmecg.LineColorAttentionMap(
    attention_df,
    polarity='signed',
    color=('steelblue', 'tomato'),
    show_colormap=True,
)

fig = plotter.plot(
    ecg_df,
    configuration=pmecg.template_factory('4x3', ecg_df, leads_map=None),
    sampling_frequency=fs,
    attention_map=line_map,
    show=True,
)
```

---

## Unipolar (Positive) Attention

When attention scores are strictly non-negative — for example, the output of a
softmax or a ReLU — use `polarity='positive'`. The `'positive'` polarity
requires that **all values are ≥ 0** and that at least one value is > 0. If
your raw scores can go negative, clip them first.

```{code-cell} python
# Clip the signed sine wave to [0, 1] before passing it to the map.
# Any sample where the original value was negative will now have zero attention.
positive_attention_df = attention_df.clip(lower=0.0, upper=1.0)

# polarity='positive' → color is a single string, not a tuple.
interval_map_positive = pmecg.IntervalAttentionMap(
    positive_attention_df,
    polarity='positive',
    color='tomato',
    max_attention_mV=0.35,
    alpha=0.5,
    smoothing_window=4,
)

fig = plotter.plot(
    ecg_df,
    configuration=pmecg.template_factory('4x3', ecg_df, leads_map=None),
    sampling_frequency=fs,
    attention_map=interval_map_positive,
    show=True,
)
```

---

## Using `attention_map_from_time_annotations`

Building a full attention array manually can be tedious when you only need to
highlight a handful of time windows. `attention_map_from_time_annotations`
accepts sparse annotations expressed in **seconds** and fills in the rest with
zeros.

Each annotation is a dict with two keys:

| Key | Type | Meaning |
|-----|------|---------|
| `time_range` | `(start_s, end_s)` | Half-open interval in seconds `[start, end)` |
| `attention_value` | `float` | Scalar score to assign to every sample in the window |

```{code-cell} python
# Highlight two suspicious windows on lead II and one on V2.
# All other leads and time steps are filled with 0 automatically.
annotation_map_df = pmecg.attention_map_from_time_annotations(
    ecg_df,
    fs,                             # sampling frequency, needed to convert seconds → indices
    II=[
        {'time_range': (1.2, 2.0), 'attention_value': 0.9},   # first window, strong attention
        {'time_range': (6.5, 7.5), 'attention_value': 0.5},   # second window, moderate
    ],
    V2=[
        {'time_range': (3.0, 4.5), 'attention_value': 0.75},
    ],
)

# The result is a plain DataFrame — inspect it like any other attention input.
print(annotation_map_df.describe())
```

Pass the DataFrame to any attention map class:

```{code-cell} python
annotation_interval = pmecg.IntervalAttentionMap(
    annotation_map_df,
    polarity='positive',   # all values are 0 or positive by construction
    color='tomato',
    smoothing_window=4,
)

fig = plotter.plot(
    ecg_df,
    configuration=pmecg.template_factory('4x3', ecg_df, leads_map=None),
    sampling_frequency=fs,
    attention_map=annotation_interval,
    show=True,
)
```

```{admonition} Tip
:class: tip

If you prefer to annotate by **sample index** rather than seconds (e.g. when
working with pre-segmented windows), use
`pmecg.attention_map_from_indices_annotations` with `index_range` instead of
`time_range`. The call signature is otherwise identical.
```
