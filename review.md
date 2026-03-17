# Code Review: Strip Leads Attention Feature

## Executive Summary

This diff adds `strip_leads_attention` parameter support to the attention map classes, allowing per-sample attention overlays on strip leads rendered via `StripLeadsConfig`. The implementation has **3 critical issues**, **3 important issues**, and several minor concerns.

---

## Critical Issues

### 1. Strip Attention Not Validated Against Polarity Constraints

**Location:** `src/pmecg/utils/attention.py` — `prepare()`, strip processing block

```python
self._strip_lead_attentions = {}
if self.strip_leads_attention is not None:
    strip_df_raw = _attention_to_dataframe(self.strip_leads_attention)
    strip_df_scaled = strip_df_raw.astype(float) / scale
    for col in strip_df_scaled.columns:
        self._strip_lead_attentions[str(col)] = strip_df_scaled[col].to_numpy(dtype=float)
```

The strip attention DataFrame is scaled but **never validated** for polarity constraints. When `polarity='positive'`, main data is required to be non-negative — strip attention receives no such check. When `polarity='signed'`, main data must span both signs — strip data doesn't. A user can pass negative values with `polarity='positive'` and they will render silently with undefined color behavior.

**Fix:** Apply the same polarity guard to strip data. At minimum, call `_finite_attention_bounds()` on each strip column and assert the constraint holds.

---

### 2. Length Mismatch Silently Skipped With No Feedback

**Location:** `src/pmecg/plot.py` — rendering loop

```python
if strip_attn is not None and len(strip_attn) == len(row_signal):
    row_attention = strip_attn
    row_attention_map = prepared_attention
```

When `len(strip_attn) != len(row_signal)`, the overlay is **silently dropped** with no feedback. This is inconsistent with the main layout, where a length mismatch raises `ValueError` in `prepare()`. Users will see a strip rendered without attention and have no idea why.

The test `test_strip_lead_attention_length_mismatch_skips_overlay` **validates the silent skip as correct behavior**, which encodes the wrong lesson.

**Fix:** Either validate length at `prepare()` time (requires strip ECG length to be known), or at minimum issue a `warnings.warn()` at render time when a mismatch is detected and the overlay is dropped.

---

### 3. `strip_lead_attentions` Property Returns Mutable Internal Dict

**Location:** `src/pmecg/utils/attention.py` — `strip_lead_attentions` property

```python
return self._strip_lead_attentions
```

The property hands out a direct reference to the internal mutable dict. A caller can do `attn.strip_lead_attentions['II'][:] = 0` and corrupt the prepared state. Every other prepared-state property in this class exposes immutable types: `dataframe` returns a copy, `row_attentions` is a tuple, `range` is a tuple. This breaks the pattern.

**Fix:** Return `dict(self._strip_lead_attentions)` (shallow copy — the np.ndarray values are still shared but the dict structure is protected), or `types.MappingProxyType(self._strip_lead_attentions)` for full read-only semantics.

---

## Important Issues

### 4. Non-Finite Values in Strip Attention Not Validated

**Location:** `src/pmecg/utils/attention.py` — `prepare()`, strip processing block

Strip attention is converted to float and divided by scale with no NaN/Inf check. The main data path calls `_finite_attention_bounds()` which raises if all values are non-finite. Strip data can be all-NaN without any error — it will silently produce an apparently-correct overlay that renders nothing.

**Fix:** Call `_finite_attention_bounds()` on the strip df (or each column) before storing, consistent with main data handling.

---

### 5. Double Computation of Bounds in `prepare()`

**Location:** `src/pmecg/utils/attention.py` — `prepare()`

```python
scale = _attention_scale(aligned_df, self.polarity)          # calls _finite_attention_bounds
scaled_df, resolved_range = _scale_attention_dataframe(...)  # calls _finite_attention_bounds again
```

`_attention_scale` and `_scale_attention_dataframe` both call `_finite_attention_bounds` on the same `aligned_df`. The reason `_attention_scale` was introduced as a separate helper is precisely because `_scale_attention_dataframe` didn't expose the scale factor — but the right fix is to make `_scale_attention_dataframe` return `(scaled_df, resolved_range, scale)` rather than adding a parallel function that duplicates the bounds computation. This avoids the waste and removes the need for `_attention_scale` entirely.

The refusal to change `_scale_attention_dataframe`'s return signature (citing two test callsites) is not a sufficient reason — fixing those two tests is a three-line change.

---

### 6. Documentation Example Correctness

**Location:** `docs/examples/attention.md` — Strip Lead Attention code cell

Two problems:

1. **Context dependency undocumented.** `ii_values = ecg_df['II'].to_numpy()` is used, but `ecg_df` is defined in the Setup section several pages up. There is no comment establishing this dependency. A reader encountering this section in isolation will not understand where `ecg_df` comes from.

2. **`attention_df` mismatch.** The example passes `attention_df.clip(lower=0.0)` as main layout attention. `attention_df` from the Setup section has columns `['II', 'V5']` with signed sine values. The `4x3` template uses all 12 leads; `prepare()` will fail because `attention_df` only has two columns and is not a single-column broadcast. Either the docs example is broken (it would raise `ValueError: Attention data is missing ECG leads`) or `attention_df` from context has all 12 columns, in which case the example silently clips V5 attention to zero and makes the strip-attention-only-for-II behavior appear more contrived than necessary. Needs clarification.

---

## Minor Issues

### 7. Missing Test Cases

The new tests cover the happy path and length-mismatch skip, but are missing:

- Strip attention with negative values and `polarity='positive'` — should raise (currently silently accepted).
- Strip attention with only-positive values and `polarity='signed'` — should raise (currently silent).
- Strip attention that is all-NaN — should raise or warn.
- Multiple strip leads where only a subset match (three-lead case) — not explicitly covered.

### 8. Rendering Test Assertions Are Coarse

`test_strip_lead_with_matching_attention_renders_overlay` asserts `len(ax.collections) >= 3`. This confirms *some* artist was added for the strip row but says nothing about whether the correct values were used. Contrast with `test_interval_attention_rendered_band_matches_scaled_positive_attention`, which verifies vertex positions against scaled attention values. The strip attention rendering tests should do the same.

### 9. Subclass Docstring Redundancy

All three subclass docstrings add `strip_leads_attention` with a cross-reference stub ("See :class:`AbstractAttentionMap` for full documentation") but then still include the "By default ``None``" sentence inline. Either omit the parameter from subclass docstrings entirely (Sphinx will inherit it), or document it fully without the stub. The current state is neither.

---

## Questions for Stakeholder

1. **Length mismatch behavior:** Should `prepare()` accept an optional `strip_n_samples` argument so length can be validated eagerly? Or is a `warnings.warn()` at render time acceptable?
2. **Polarity validation on strip data:** Strict (same rules as main) or relaxed (e.g., allow zero-only strip attention under `polarity='signed'`)?
3. **API placement:** Was putting `strip_leads_attention` on the attention map a deliberate choice over `StripLeadsConfig(ecg_data=..., attention=...)`? If strip attention is always conceptually paired with a specific strip dataset, `StripLeadsConfig` may be more cohesive.
4. **Mutable property:** Intentional for caller convenience, or should it return a read-only view?

---

## Required Fixes Before Merge

| # | Issue | Severity |
|---|-------|----------|
| 1 | Polarity validation missing for strip attention | Critical |
| 2 | Length mismatch silently skipped with no feedback | Critical |
| 3 | `strip_lead_attentions` returns mutable internal dict | Critical |
| 4 | Non-finite values in strip attention not validated | Important |
| 5 | Double `_finite_attention_bounds` computation — `_attention_scale` is the wrong fix | Important |
| 6 | Docs example context dependency and potential `prepare()` failure | Important |
| 7 | Missing polarity/NaN test cases | Important |
| 8 | Rendering tests assert only collection count, not values | Minor |
| 9 | Subclass docstring cross-reference redundancy | Minor |
