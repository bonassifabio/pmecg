# Docs & Docstring Review

## Missing Docstrings

### `_plot_attention_color_scale` — no docstring at all
- **File:** `src/pmecg/utils/plot.py`
- Parameters `ax`, `attention_map`, `width_inches`, `right_margin_mm`, `top_inches`, `bottom_inches` are completely undocumented.

---

## Parameter Documentation Gaps

### `_print_information` — `strip_speed` not in docstring
- **File:** `src/pmecg/utils/plot.py`
- The `strip_speed` parameter appears in the function signature but is absent from the docstring.

### `_plot_row` — `attention_values` shape not specified
- **File:** `src/pmecg/utils/plot.py`
- Docstring says "Attention values for this row, optional" but doesn't state that the array must be 1-D with length equal to `len(signal)`.

### `_plot_row` — `segment_offsets` length constraint not documented
- **File:** `src/pmecg/utils/plot.py`
- Docstring doesn't mention that the list must have length equal to `len(leads)`.

### `attention_map_from_indices_annotations` — annotation dict structure not documented
- **File:** `src/pmecg/utils/attention.py`
- The kwargs accept `list[_IndexAnnotation]` but the required keys (`index_range`, `attention_value`) are never shown. An example in the docstring would fix this.

### `attention_map_from_time_annotations` — same issue
- **File:** `src/pmecg/utils/attention.py`
- Required keys (`time_range`, `attention_value`) are not documented in the docstring.

### `ECGPlotter.__init__` — `grid_mode` valid values not enumerated
- **File:** `src/pmecg/plot.py`
- Docstring mentions `'cm'` but doesn't say that only `None` and `'cm'` are accepted. Anything else raises an `AssertionError` (line 189).

### `_even_leads_split` — warning side-effect not mentioned
- **File:** `src/pmecg/utils/data.py`
- Issues a warning when samples don't divide evenly, but the docstring is silent about it.

### `_apply_configuration` — unequal row sample-count warning not mentioned
- **File:** `src/pmecg/utils/data.py`
- A warning is issued when segments in a row have different sample counts, but the docstring doesn't mention it.

---

## Missing / Incomplete Return-Type Documentation

### `_nice_tick_step` — return type not stated
- **File:** `src/pmecg/utils/plot.py`
- Docstring describes what the function does but never states it returns a `float` (seconds).

### `_resolve_template_lead` — return type missing
- **File:** `src/pmecg/utils/data.py`
- Docstring is otherwise good but doesn't explicitly state the return type is `str`.

### `_validate_configuration_row_definition` — complex return type not documented
- **File:** `src/pmecg/utils/data.py`
- Returns `list[str] | str | list[LeadSegment] | LeadSegment` depending on input, but the docstring doesn't explain this mirroring behaviour.

---

## Behaviour Documentation Gaps

### `AbstractAttentionMap.prepare()` — internal usage and call requirements not documented
- **File:** `src/pmecg/utils/attention.py`
- Docstring doesn't say that `ECGPlotter.plot()` calls this automatically, and that `.dataframe`, `.row_attentions`, and `.range` are only valid after calling it.

### `AbstractAttentionMap.dataframe` / `.range` properties — prerequisites missing
- **File:** `src/pmecg/utils/attention.py`
- Neither property docstring mentions that `.prepare()` must be called first, or documents the DataFrame shape / tuple structure of `.range`.

### `_plot_grid` — dead `'inch'` mode not documented
- **File:** `src/pmecg/utils/plot.py`
- The function raises `NotImplementedError` for `'inch'` mode, but this is not mentioned anywhere. The docstring and the public API (`ECGPlotter`) should both state that only `'cm'` is currently supported.

### `_resolve_configuration` — normalisation side-effect not mentioned
- **File:** `src/pmecg/utils/data.py`
- Docstring says "validate", but the function also normalises each entry via `_validate_configuration_row_definition`. The docstring should mention both roles.

### `expand_to_12_leads` — behaviour with duplicate column names not documented
- **File:** `src/pmecg/utils/data.py`
- No mention of what happens when the input already contains duplicate lead columns.

---

## Documentation / Example Issues

### `docs/examples/attention.md` — variable naming inconsistency
- A variable is named `annotation_background` but is assigned an `IntervalAttentionMap` instance. Should be `annotation_interval` (or similar) to match the class.

### `LeadsMap` NamedTuple — partial construction not highlighted
- **File:** `src/pmecg/utils/data.py`
- Docstring doesn't make it obvious that only a subset of the 12 fields needs to be set (the rest default to `None`). An example showing `LeadsMap(I="custom_I")` would help.
