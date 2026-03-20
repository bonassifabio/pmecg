"""Unit tests for pmecg.utils.data."""

from __future__ import annotations

import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from pmecg.types import LeadSegment
from pmecg.utils.data import (
    SUPPORTED_LEADS,
    LeadsMap,
    _apply_configuration,
    _build_row_signal,
    _even_leads_split,
    _numpy_to_dataframe,
    _resolve_configuration,
    _template_configuration,
    cabrera_factory,
    expand_to_12_leads,
    template_factory,
)

# SUPPORTED_LEADS = ("I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6")
# Lead at index i has value (i+1.0) in all helper fixtures below.
LEAD_VALUE = {lead: float(i + 1) for i, lead in enumerate(SUPPORTED_LEADS)}

N_SAMPLES = 120  # divisible by 1, 2, 3, 4, 6, 8, 12
CUSTOM_LEADS_MAP = LeadsMap(
    I="LI",
    II="LII",
    III="LIII",
    aVR="aVR-custom",
    aVL="aVL-custom",
    aVF="aVF-custom",
    V1="Chest-1",
    V2="Chest-2",
    V3="Chest-3",
    V4="Chest-4",
    V5="Chest-5",
    V6="Chest-6",
)
CUSTOM_LEADS = [lead for lead in CUSTOM_LEADS_MAP if lead is not None]


def _make_ecg_array(leads: list[str]) -> np.ndarray:
    """Return shape (N_SAMPLES, len(leads)) array; column i is filled with (i+1.0)."""
    return np.column_stack([(i + 1.0) * np.ones(N_SAMPLES) for i in range(len(leads))])


def _make_12lead_df():
    """12-lead DataFrame where lead 'X' has the constant value LEAD_VALUE['X']."""
    leads = list(SUPPORTED_LEADS)
    return _numpy_to_dataframe(_make_ecg_array(leads), leads)


def _should_warn_divisible(configuration, n_samples):
    """Return True if any segment in the configuration does not evenly divide n_samples."""
    if configuration is None:
        return False
    # Normalize to list of rows, where each row is a list of leads
    rows = [[e] if isinstance(e, str) else e for e in configuration]
    for row in rows:
        if n_samples % len(row) != 0:
            return True
    return False


# ---------------------------------------------------------------------------
# _numpy_to_dataframe
# ---------------------------------------------------------------------------

ONE_ROW_TEMPLATES = ["1x1", "1x2", "1x3", "1x4", "1x6", "1x8", "1x12"]


@pytest.mark.parametrize("template_key", ONE_ROW_TEMPLATES)
class TestNumpyToDataframe:
    """Conversion to DataFrame for every 1xL template configuration."""

    # Checks that converting a template-shaped array preserves the expected 2D dimensions.
    def test_shape(self, template_key):
        leads = _template_configuration(template_key)
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        assert df.shape == (N_SAMPLES, len(leads))

    # Checks that the DataFrame columns exactly mirror the requested lead order.
    def test_columns_match_leads(self, template_key):
        leads = _template_configuration(template_key)
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        assert list(df.columns) == list(leads)

    # Checks that each output column keeps the constant value assigned to its source lead index.
    def test_values_match_lead_index(self, template_key):
        """Column i must equal (i+1.0) for all samples."""
        leads = _template_configuration(template_key)
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        for i, lead in enumerate(leads):
            np.testing.assert_array_equal(df[lead].values, float(i + 1))


class TestNumpyToDataframeDefaults:
    # Checks that 12 unnamed input columns default to the canonical 12-lead labels.
    def test_12lead_default_column_names(self):
        """With exactly 12 leads and no explicit names, columns use SUPPORTED_LEADS."""
        ecg_data = _make_ecg_array(list(SUPPORTED_LEADS))
        df = _numpy_to_dataframe(ecg_data)  # lead_names=None
        assert list(df.columns) == list(SUPPORTED_LEADS)

    # Checks that non-12-lead arrays without explicit names are rejected.
    def test_wrong_lead_count_raises(self):
        """Passing a 5-lead array without explicit names must raise AssertionError."""
        with pytest.raises(AssertionError):
            _numpy_to_dataframe(np.ones((N_SAMPLES, 5)))

    # Checks that the helper rejects a lead-name list whose length does not match the data width.
    def test_mismatched_names_raises(self):
        """Passing lead_names of wrong length must raise AssertionError."""
        with pytest.raises(AssertionError):
            _numpy_to_dataframe(np.ones((N_SAMPLES, 3)), ["I", "II"])

    # Checks that explicitly provided custom lead names are kept untouched.
    def test_custom_names_are_preserved(self):
        """Explicit non-canonical names must be preserved verbatim."""
        df = _numpy_to_dataframe(np.ones((N_SAMPLES, 3)), ["LI", "Lead Two", "Chest-1"])
        assert list(df.columns) == ["LI", "Lead Two", "Chest-1"]

    # Checks that lowercase canonical lead names are preserved verbatim.
    def test_lowercase_canonical_names_are_preserved(self):
        """Lowercase canonical names must not be altered implicitly."""
        lowercase_leads = ["i", "ii", "v1"]
        df = _numpy_to_dataframe(np.ones((N_SAMPLES, 3)), lowercase_leads)
        assert list(df.columns) == lowercase_leads


# ---------------------------------------------------------------------------
# _even_leads_split + _build_row_signal
# ---------------------------------------------------------------------------

SEGMENT_LEAD_GROUPS = [
    ["I"],
    ["I", "II"],
    ["I", "aVR", "V6"],
    ["I", "II", "III", "aVR"],
    ["V1", "V2", "V3", "V4", "V5", "V6"],
    # Adding a case that triggers the warning
    ["I", "II", "III", "aVR", "aVL", "aVF", "V1"],  # 7 leads, 120 % 7 != 0
]


def _segment_via_new_api(df, selected_leads, disconnect_segments):
    """Helper: replicate old _segment_leads via the new split + build functions."""
    configs = _even_leads_split(selected_leads, df.shape[0])
    return _build_row_signal(df, configs, disconnect_segments=disconnect_segments)


@pytest.mark.parametrize("selected_leads", SEGMENT_LEAD_GROUPS)
@pytest.mark.parametrize("disconnect", [True, False])
class TestSegmentLeads:
    """Cartesian product: lead groups × disconnect flag."""

    # Checks that segmenting any requested lead group returns a 1D signal of n_leads * segment_len samples.
    def test_output_shape(self, selected_leads, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            signal, _, _ = _segment_via_new_api(_make_12lead_df(), selected_leads, disconnect)
        expected_len = len(selected_leads) * (N_SAMPLES // len(selected_leads))
        assert signal.shape == (expected_len,)

    # Checks that the function reports back the same lead sequence it was asked to segment.
    def test_returned_leads(self, selected_leads, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            _, ret_leads, _ = _segment_via_new_api(_make_12lead_df(), selected_leads, disconnect)
        assert ret_leads == selected_leads

    # Checks that the interior samples of each segment come from the expected source lead.
    def test_interior_segment_values(self, selected_leads, disconnect):
        """All samples except the last in each segment equal the expected lead value."""
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            signal, _, _ = _segment_via_new_api(_make_12lead_df(), selected_leads, disconnect)
        seg = N_SAMPLES // len(selected_leads)
        for i, lead in enumerate(selected_leads):
            np.testing.assert_array_equal(signal[i * seg : (i + 1) * seg - 1], LEAD_VALUE[lead])

    # Checks that segment boundaries are NaN only when disconnection markers are enabled.
    def test_last_sample_per_segment(self, selected_leads, disconnect):
        """Last sample of each segment is NaN iff disconnect_segments=True."""
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            signal, _, _ = _segment_via_new_api(_make_12lead_df(), selected_leads, disconnect)
        seg = N_SAMPLES // len(selected_leads)
        for i, lead in enumerate(selected_leads):
            last = (i + 1) * seg - 1
            if disconnect:
                assert np.isnan(signal[last])
            else:
                assert signal[last] == LEAD_VALUE[lead]


# Checks that passing a single lead as a string behaves exactly like passing a one-item list.
def test_segment_leads_string_input():
    """A string lead name is treated as a one-element list."""
    df = _make_12lead_df()
    for lead in ["I", "II", "V5"]:
        sig_str, leads_str, _ = _segment_via_new_api(df, lead, disconnect_segments=False)
        sig_list, _, _ = _segment_via_new_api(df, [lead], disconnect_segments=False)
        assert leads_str == [lead]
        np.testing.assert_array_equal(sig_str, sig_list)


# ---------------------------------------------------------------------------
# _apply_configuration
# ---------------------------------------------------------------------------

# (config, expected_leads_per_row) — each entry in expected_leads_per_row
# represents a row in the final plot.
APPLY_CONFIG_CASES = [
    pytest.param(["V5"], [["V5"]], id="single-lead"),
    # exotic list configs
    pytest.param(
        [["I", "II", "III"], ["aVR", "aVL", "aVF"]],
        [["I", "II", "III"], ["aVR", "aVL", "aVF"]],
        id="exotic-2x3",
    ),
    pytest.param(
        [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]],
        [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]],
        id="exotic-3x4",
    ),
    pytest.param(
        [["I", "II"], ["III", "aVR"], ["aVL", "aVF"], ["V1", "V2"]],
        [["I", "II"], ["III", "aVR"], ["aVL", "aVF"], ["V1", "V2"]],
        id="exotic-4x2",
    ),
    # configuration with 6 leads in a row (will trigger warning with N_SAMPLES=120? 120/6=20, no)
    # let's add a custom configuration that triggers warning
    pytest.param(
        [["I", "II", "III", "aVR", "aVL", "aVF", "V1"]],
        [["I", "II", "III", "aVR", "aVL", "aVF", "V1"]],
        id="warn-7-leads",
    ),
    pytest.param(
        [["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3"]],
        [["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3"]],
        id="warn-9-leads",
    ),
    pytest.param(
        [["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5"]],
        [["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5"]],
        id="warn-11-leads",
    ),
    pytest.param(
        [
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1"],
        ],
        [
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1"],
        ],
        id="mixed-warn",
    ),
    pytest.param(
        [["I", "II", "III", "aVR"], ["aVL", "aVF", "V1", "V2"], "V3"],
        [["I", "II", "III", "aVR"], ["aVL", "aVF", "V1", "V2"], ["V3"]],
        id="mixed-with-rhythm-strip",
    ),
]


@pytest.mark.parametrize("config,expected_leads_per_row", APPLY_CONFIG_CASES)
@pytest.mark.parametrize("disconnect", [True, False])
class TestApplyConfiguration:
    """Cartesian product: configuration cases × disconnect flag."""

    # Checks that applying a configuration yields one output row per configured row.
    def test_row_count(self, config, expected_leads_per_row, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        assert len(result) == len(expected_leads_per_row)

    # Checks that every configured row produces a signal of the expected length.
    def test_signal_shapes(self, config, expected_leads_per_row, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        for (signal, _, _offsets, _segs), exp_leads in zip(result, expected_leads_per_row):
            n = len(exp_leads) if isinstance(exp_leads, list) else 1
            assert signal.shape == (n * (N_SAMPLES // n),)

    # Checks that each configured row keeps the expected lead names in order.
    def test_lead_names(self, config, expected_leads_per_row, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        for (_, ret_leads, _offsets, _segs), exp in zip(result, expected_leads_per_row):
            assert ret_leads == (exp if isinstance(exp, list) else [exp])

    # Checks that each configured segment pulls interior samples from the correct lead values.
    def test_interior_segment_values(self, config, expected_leads_per_row, disconnect):
        """Interior samples in each segment match the expected lead value."""
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        for (signal, _, _offsets, _segs), row_leads in zip(result, expected_leads_per_row):
            leads = row_leads if isinstance(row_leads, list) else [row_leads]
            seg = N_SAMPLES // len(leads)
            for i, lead in enumerate(leads):
                np.testing.assert_array_equal(signal[i * seg : (i + 1) * seg - 1], LEAD_VALUE[lead])

    # Checks that configured segment boundaries become NaN only when disconnection markers are requested.
    def test_last_sample_per_segment(self, config, expected_leads_per_row, disconnect):
        """Last sample of each segment is NaN iff disconnect_segments=True."""
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        for (signal, _, _offsets, _segs), row_leads in zip(result, expected_leads_per_row):
            leads = row_leads if isinstance(row_leads, list) else [row_leads]
            seg = N_SAMPLES // len(leads)
            for i, lead in enumerate(leads):
                last = (i + 1) * seg - 1
                if disconnect:
                    assert np.isnan(signal[last])
                else:
                    assert signal[last] == LEAD_VALUE[lead]


class TestApplyConfigurationDefault:
    # Checks that a missing configuration falls back to one full-duration row per input lead.
    def test_default_none_configuration(self):
        """None configuration should plot all leads in the DataFrame for their entire duration."""
        df = _make_12lead_df()
        # Disable disconnect_segments to allow direct array comparison
        result = _apply_configuration(df, configuration=None, disconnect_segments=False)

        # Expecting one row per lead
        assert len(result) == len(df.columns)

        for i, (signal, selected_leads, _offsets, _segs) in enumerate(result):
            # Each row should contain exactly one lead
            assert len(selected_leads) == 1
            assert selected_leads[0] == df.columns[i]
            # The signal should be the lead data
            np.testing.assert_array_equal(signal, df[df.columns[i]].values)


class TestResolveConfiguration:
    # Checks that template resolution returns the built-in conventional layout for conventional input data.
    def test_template_factory_returns_builtin_configuration_for_canonical_dataframe(self):
        resolved = template_factory("4x3", _make_12lead_df(), None)
        assert resolved == _template_configuration("4x3")

    # Checks that template resolution rewrites conventional template leads into the caller's custom lead names.
    def test_template_factory_resolves_to_custom_input_names(self):
        resolved = template_factory("4x3", _numpy_to_dataframe(_make_ecg_array(CUSTOM_LEADS), CUSTOM_LEADS), CUSTOM_LEADS_MAP)
        assert resolved == [
            ["LI", "aVR-custom", "Chest-1", "Chest-4"],
            ["LII", "aVL-custom", "Chest-2", "Chest-5"],
            ["LIII", "aVF-custom", "Chest-3", "Chest-6"],
        ]

    # Checks that user-defined configurations are accepted when they already use the custom input labels.
    def test_custom_configuration_accepts_custom_names(self):
        configuration = [["LI", "aVR-custom", "Chest-1"], "Chest-6"]
        resolved = _resolve_configuration(configuration, CUSTOM_LEADS)
        assert resolved == configuration

    # Checks that passing a bare template string to the resolver is rejected as an invalid configuration shape.
    def test_top_level_string_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="configuration must be a list"):
            _resolve_configuration("4x3", list(SUPPORTED_LEADS))

    # Checks that conventional lead names are rejected when the available input columns use custom labels instead.
    def test_custom_configuration_with_canonical_names_raises(self):
        configuration = [["I", "aVR", "V1"], "V6"]
        with pytest.raises(ValueError, match="Lead name 'I' in configuration is not present"):
            _resolve_configuration(configuration, CUSTOM_LEADS)

    # Checks that template expansion fails when a required conventional lead is missing from the custom map.
    def test_template_factory_missing_required_canonical_mapping_raises(self):
        partial_map = CUSTOM_LEADS_MAP._replace(aVR=None)
        with pytest.raises(ValueError, match="Template '4x3' requires conventional lead 'aVR'"):
            template_factory("4x3", _numpy_to_dataframe(_make_ecg_array(CUSTOM_LEADS), CUSTOM_LEADS), partial_map)

    # Checks that duplicate custom names in the lead map are rejected before template expansion.
    def test_template_factory_duplicate_custom_names_in_map_raise(self):
        duplicate_map = CUSTOM_LEADS_MAP._replace(III="LII")
        with pytest.raises(ValueError, match="Duplicate custom lead name 'LII'"):
            template_factory("4x3", _numpy_to_dataframe(_make_ecg_array(CUSTOM_LEADS), CUSTOM_LEADS), duplicate_map)

    # Checks that configurations referencing missing leads fail fast during validation.
    def test_unknown_configuration_lead_raises(self):
        with pytest.raises(ValueError, match="Lead name 'I' in configuration is not present"):
            _resolve_configuration([["I", "UnknownLead"]], CUSTOM_LEADS)

    # Checks that lowercase conventional input names do not resolve to built-in templates without an explicit map.
    def test_template_factory_lowercase_canonical_input_requires_leads_map(self):
        df = _numpy_to_dataframe(np.ones((N_SAMPLES, 12)), [lead.lower() for lead in SUPPORTED_LEADS])
        with pytest.raises(ValueError, match="Template '4x3' requires conventional lead 'I'"):
            template_factory("4x3", df, None)

    # Checks that lowercase conventional input names can still be resolved explicitly through LeadsMap.
    def test_template_factory_lowercase_canonical_input_resolves_with_leads_map(self):
        lowercase_leads = [lead.lower() for lead in SUPPORTED_LEADS]
        df = pd.DataFrame(np.ones((N_SAMPLES, 12)), columns=lowercase_leads)
        leads_map = LeadsMap(**{lead: lead.lower() for lead in SUPPORTED_LEADS})
        resolved = template_factory("4x3", df, leads_map)
        assert resolved == [
            ["i", "avr", "v1", "v4"],
            ["ii", "avl", "v2", "v5"],
            ["iii", "avf", "v3", "v6"],
        ]


class TestLeadsMapDeprecation:
    # Checks that using the old uppercase AVR/AVL/AVF kwargs emits DeprecationWarning.
    def test_deprecated_avr_kwarg_warns(self):
        with pytest.warns(DeprecationWarning, match="AVR.*deprecated"):
            lm = LeadsMap(AVR="custom_avr")
        assert lm.aVR == "custom_avr"

    def test_deprecated_avl_kwarg_warns(self):
        with pytest.warns(DeprecationWarning, match="AVL.*deprecated"):
            lm = LeadsMap(AVL="custom_avl")
        assert lm.aVL == "custom_avl"

    def test_deprecated_avf_kwarg_warns(self):
        with pytest.warns(DeprecationWarning, match="AVF.*deprecated"):
            lm = LeadsMap(AVF="custom_avf")
        assert lm.aVF == "custom_avf"

    # Checks that the new kwargs do not emit warnings.
    def test_new_kwargs_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            lm = LeadsMap(aVR="x", aVL="y", aVF="z")
        assert lm.aVR == "x"
        assert lm.aVL == "y"
        assert lm.aVF == "z"

    # Checks that the deprecated kwarg does not overwrite an explicitly provided new kwarg.
    def test_deprecated_avr_does_not_overwrite_avr(self):
        with pytest.warns(DeprecationWarning):
            lm = LeadsMap(aVR="new", AVR="old")
        assert lm.aVR == "new"


# ---------------------------------------------------------------------------
# +3 rhythm strip templates
# ---------------------------------------------------------------------------


class TestThreeStripTemplates:
    # Checks that 4x3+3 has the expected row/rhythm-strip layout.
    def test_4x3_plus3_configuration(self):
        from pmecg.utils.data import _template_configuration

        config = _template_configuration("4x3+3")
        assert config == [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
            "II",
            "V1",
            "V5",
        ]

    # Checks that 2x6+3 has the expected row/rhythm-strip layout.
    def test_2x6_plus3_configuration(self):
        from pmecg.utils.data import _template_configuration

        config = _template_configuration("2x6+3")
        assert config == [
            ["I", "V1"],
            ["II", "V2"],
            ["III", "V3"],
            ["aVR", "V4"],
            ["aVL", "V5"],
            ["aVF", "V6"],
            "II",
            "V1",
            "V5",
        ]

    # Checks that 2x4+3 has the expected row/rhythm-strip layout.
    def test_2x4_plus3_configuration(self):
        from pmecg.utils.data import _template_configuration

        config = _template_configuration("2x4+3")
        assert config == [
            ["I", "V3"],
            ["II", "V4"],
            ["III", "V5"],
            ["aVR", "V6"],
            "II",
            "V1",
            "V5",
        ]

    # Checks that template_factory resolves +3 templates against a canonical DataFrame.
    def test_template_factory_resolves_4x3_plus3(self):
        df = _make_12lead_df()
        resolved = template_factory("4x3+3", df, None)
        assert resolved == [
            ["I", "aVR", "V1", "V4"],
            ["II", "aVL", "V2", "V5"],
            ["III", "aVF", "V3", "V6"],
            "II",
            "V1",
            "V5",
        ]

    # Checks that template_factory resolves +3 templates with custom lead names.
    def test_template_factory_resolves_4x3_plus3_custom_names(self):
        df = _numpy_to_dataframe(_make_ecg_array(CUSTOM_LEADS), CUSTOM_LEADS)
        resolved = template_factory("4x3+3", df, CUSTOM_LEADS_MAP)
        assert resolved == [
            ["LI", "aVR-custom", "Chest-1", "Chest-4"],
            ["LII", "aVL-custom", "Chest-2", "Chest-5"],
            ["LIII", "aVF-custom", "Chest-3", "Chest-6"],
            "LII",
            "Chest-1",
            "Chest-5",
        ]

    # Checks that cabrera_factory works with 4x3+3 (all 6 limb leads present).
    def test_cabrera_factory_4x3_plus3(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("4x3+3", df)
        assert config == [
            ["aVL", "II", "V1", "V4"],
            ["I", "aVF", "V2", "V5"],
            ["-aVR", "III", "V3", "V6"],
            "II",
            "V1",
            "V5",
        ]

    # Checks that cabrera_factory works with 2x6+3 (all 6 limb leads present).
    def test_cabrera_factory_2x6_plus3(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("2x6+3", df)
        assert config == [
            ["aVL", "V1"],
            ["I", "V2"],
            ["-aVR", "V3"],
            ["II", "V4"],
            ["aVF", "V5"],
            ["III", "V6"],
            "II",
            "V1",
            "V5",
        ]

    # Checks that cabrera_factory rejects 2x4+3 (missing AVL and AVF).
    def test_cabrera_factory_rejects_2x4_plus3(self):
        df = _make_12lead_df()
        with pytest.raises(ValueError, match="Cabrera format requires all six limb leads"):
            cabrera_factory("2x4+3", df)


# ---------------------------------------------------------------------------
# cabrera_factory
# ---------------------------------------------------------------------------


class TestCabreraFactory:
    # Checks that 4x3 Cabrera reorders limb leads correctly and keeps the rhythm strip.
    def test_4x3_configuration(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("4x3", df)
        assert config == [
            ["aVL", "II", "V1", "V4"],
            ["I", "aVF", "V2", "V5"],
            ["-aVR", "III", "V3", "V6"],
        ]

    # Checks that 2x6 Cabrera reorders limb leads correctly and keeps the rhythm strip.
    def test_2x6_configuration(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("2x6", df)
        assert config == [
            ["aVL", "V1"],
            ["I", "V2"],
            ["-aVR", "V3"],
            ["II", "V4"],
            ["aVF", "V5"],
            ["III", "V6"],
        ]

    # Checks that 1x6 Cabrera reorders all limb leads (no rhythm strip).
    def test_1x6_configuration(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("1x6", df)
        assert config == ["aVL", "I", "-aVR", "II", "aVF", "III"]

    # Checks that 1x12 Cabrera reorders limb leads, keeps precordial order.
    def test_1x12_configuration(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("1x12", df)
        assert config == ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Checks that aVR is replaced by -aVR (negated) in the output DataFrame.
    def test_neg_avr_column_added_dataframe(self):
        df = _make_12lead_df()
        new_data, _ = cabrera_factory("4x3", df)
        assert isinstance(new_data, pd.DataFrame)
        assert "-aVR" in new_data.columns
        assert "aVR" not in new_data.columns
        np.testing.assert_array_equal(new_data["-aVR"].values, -df["aVR"].values)

    # Checks that the original DataFrame is not mutated.
    def test_does_not_mutate_original_dataframe(self):
        df = _make_12lead_df()
        original_columns = list(df.columns)
        cabrera_factory("4x3", df)
        assert list(df.columns) == original_columns

    # Checks that aVR is replaced by -aVR (negated) for tuple (ndarray, names) input.
    def test_neg_avr_column_added_numpy_tuple(self):
        arr = _make_ecg_array(list(SUPPORTED_LEADS))
        ecg_data = (arr, list(SUPPORTED_LEADS))
        new_data, _ = cabrera_factory("4x3", ecg_data)
        assert isinstance(new_data, tuple)
        new_arr, new_names = new_data
        assert "-aVR" in new_names
        assert "aVR" not in new_names
        assert len(new_names) == len(SUPPORTED_LEADS)  # no extra column added
        avr_idx = list(SUPPORTED_LEADS).index("aVR")
        np.testing.assert_array_equal(new_arr[:, avr_idx], -arr[:, avr_idx])

    # Checks that aVR is replaced by -aVR (negated) for tuple (list[ndarray], names) input.
    def test_neg_avr_column_added_list_of_arrays(self):
        leads = list(SUPPORTED_LEADS)
        arrays = [np.full(N_SAMPLES, float(i + 1)) for i in range(len(leads))]
        ecg_data = (arrays, leads)
        new_data, _ = cabrera_factory("4x3", ecg_data)
        new_arrays, new_names = new_data
        assert "-aVR" in new_names
        assert "aVR" not in new_names
        assert len(new_names) == len(leads)  # no extra array added
        avr_idx = leads.index("aVR")
        np.testing.assert_array_equal(new_arrays[avr_idx], -arrays[avr_idx])

    # Checks that templates missing some limb leads are rejected.
    def test_rejects_template_without_all_limb_leads(self):
        df = _make_12lead_df()
        with pytest.raises(ValueError, match="Cabrera format requires all six limb leads"):
            cabrera_factory("1x1", df)

    def test_rejects_1x8_template(self):
        df = _make_12lead_df()
        with pytest.raises(ValueError, match="Cabrera format requires all six limb leads"):
            cabrera_factory("1x8", df)

    def test_rejects_2x4_template(self):
        df = _make_12lead_df()
        with pytest.raises(ValueError, match="Cabrera format requires all six limb leads"):
            cabrera_factory("2x4", df)

    # Checks that data without aVR is rejected.
    def test_rejects_data_without_avr(self):
        leads = ["I", "II", "III", "X", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        df = pd.DataFrame(np.ones((N_SAMPLES, 12)), columns=leads)
        with pytest.raises(ValueError, match="requires 'aVR' lead"):
            cabrera_factory("1x6", df)

    # Checks that precordial leads (V1-V6) are not affected by the substitution.
    def test_precordial_leads_unchanged(self):
        df = _make_12lead_df()
        _, config = cabrera_factory("4x3", df)
        # Collect all V leads from config
        v_leads = []
        for entry in config:
            if isinstance(entry, list):
                v_leads.extend(lead for lead in entry if lead.startswith("V"))
        assert sorted(v_leads) == ["V1", "V2", "V3", "V4", "V5", "V6"]

    # Checks that returned data can be plotted with ECGPlotter.
    def test_returned_data_plottable(self):
        df = _make_12lead_df()
        new_data, config = cabrera_factory("4x3", df)
        # All leads in config must exist in new_data
        all_config_leads = set()
        for entry in config:
            if isinstance(entry, list):
                all_config_leads.update(entry)
            else:
                all_config_leads.add(entry)
        assert all_config_leads.issubset(set(new_data.columns))

    # Checks that a pre-negated aVR column (name starts with '-') is only renamed, not sign-flipped.
    def test_pre_negated_avr_dataframe(self):
        cols = list(SUPPORTED_LEADS)
        df = pd.DataFrame(np.ones((N_SAMPLES, 12)), columns=cols)
        df["aVR"] = 2.0
        df = df.rename(columns={"aVR": "-aVR"})
        new_data, _ = cabrera_factory("4x3", df, leads_map=LeadsMap(aVR="-aVR"))
        assert isinstance(new_data, pd.DataFrame)
        assert "-aVR" in new_data.columns
        # Values must be unchanged (no double-negation); column was renamed in-place
        np.testing.assert_array_equal(new_data["-aVR"].values, df["-aVR"].values)

    # Checks the same skip-flip behaviour for tuple (ndarray, names) input.
    def test_pre_negated_avr_numpy_tuple(self):
        leads = list(SUPPORTED_LEADS)
        arr = _make_ecg_array(leads)
        avr_idx = leads.index("aVR")
        arr[:, avr_idx] = 3.0
        neg_leads = ["-aVR" if n == "aVR" else n for n in leads]
        ecg_data = (arr, neg_leads)
        new_data, _ = cabrera_factory("4x3", ecg_data, leads_map=LeadsMap(aVR="-aVR"))
        new_arr, new_names = new_data
        assert "-aVR" in new_names
        np.testing.assert_array_equal(new_arr[:, avr_idx], arr[:, avr_idx])

    # Checks the same skip-flip behaviour for tuple (list[ndarray], names) input.
    def test_pre_negated_avr_list_of_arrays(self):
        leads = list(SUPPORTED_LEADS)
        avr_idx = leads.index("aVR")
        arrays = [np.full(N_SAMPLES, float(i + 1)) for i in range(len(leads))]
        neg_leads = ["-aVR" if n == "aVR" else n for n in leads]
        ecg_data = (arrays, neg_leads)
        new_data, _ = cabrera_factory("4x3", ecg_data, leads_map=LeadsMap(aVR="-aVR"))
        new_arrays, new_names = new_data
        assert "-aVR" in new_names
        np.testing.assert_array_equal(new_arrays[avr_idx], arrays[avr_idx])

    # Checks that leads_map is respected: config uses custom column names and -AVR is derived from the mapped AVR.
    def test_leads_map_resolves_custom_names(self):
        custom_cols = [
            "LI",
            "LII",
            "LIII",
            "aVR-custom",
            "aVL-custom",
            "aVF-custom",
            "Chest-1",
            "Chest-2",
            "Chest-3",
            "Chest-4",
            "Chest-5",
            "Chest-6",
        ]
        df = pd.DataFrame(np.ones((N_SAMPLES, 12)), columns=custom_cols)
        df["aVR-custom"] = 2.0  # give AVR a distinct value so -AVR is verifiable
        new_data, config = cabrera_factory("4x3", df, leads_map=CUSTOM_LEADS_MAP)
        assert config == [
            ["aVL-custom", "LII", "Chest-1", "Chest-4"],
            ["LI", "aVF-custom", "Chest-2", "Chest-5"],
            ["-aVR", "LIII", "Chest-3", "Chest-6"],
        ]
        assert isinstance(new_data, pd.DataFrame)
        assert "-aVR" in new_data.columns
        assert "aVR-custom" not in new_data.columns
        np.testing.assert_array_equal(new_data["-aVR"].values, -df["aVR-custom"].values)


# ---------------------------------------------------------------------------
# Advanced configuration (LeadSegment dicts)
# ---------------------------------------------------------------------------


class TestBuildRowSignal:
    # Checks that _build_row_signal produces the expected total length.
    def test_output_length(self):
        df = _make_12lead_df()
        configs = [
            LeadSegment(lead="I", start=0, end=60),
            LeadSegment(lead="II", start=0, end=60),
        ]
        signal, leads, offsets = _build_row_signal(df, configs, disconnect_segments=False)
        assert signal.shape == (120,)
        assert leads == ["I", "II"]
        assert offsets == [0, 60]

    # Checks that each segment pulls data from the correct lead and range.
    def test_segment_values(self):
        df = _make_12lead_df()
        configs = [
            LeadSegment(lead="I", start=0, end=40),
            LeadSegment(lead="V6", start=10, end=50),
        ]
        signal, leads, offsets = _build_row_signal(df, configs, disconnect_segments=False)
        np.testing.assert_array_equal(signal[:40], df["I"].values[0:40])
        np.testing.assert_array_equal(signal[40:80], df["V6"].values[10:50])
        assert offsets == [0, 40]

    # Checks that disconnection NaNs are placed at segment boundaries.
    def test_disconnect_segments(self):
        df = _make_12lead_df()
        configs = [
            LeadSegment(lead="I", start=0, end=60),
            LeadSegment(lead="II", start=0, end=60),
        ]
        signal, _, _ = _build_row_signal(df, configs, disconnect_segments=True)
        assert np.isnan(signal[59])
        assert np.isnan(signal[119])

    # Checks that no disconnection NaNs appear when disabled.
    def test_no_disconnect(self):
        df = _make_12lead_df()
        configs = [
            LeadSegment(lead="I", start=0, end=60),
            LeadSegment(lead="II", start=0, end=60),
        ]
        signal, _, _ = _build_row_signal(df, configs, disconnect_segments=False)
        assert not np.isnan(signal[59])
        assert not np.isnan(signal[119])

    # Checks that a single LeadSegment produces a single-lead segment.
    def test_single_lead_config(self):
        df = _make_12lead_df()
        configs = [LeadSegment(lead="III", start=10, end=50)]
        signal, leads, offsets = _build_row_signal(df, configs, disconnect_segments=False)
        assert signal.shape == (40,)
        assert leads == ["III"]
        assert offsets == [0]
        np.testing.assert_array_equal(signal, df["III"].values[10:50])


class TestApplyConfigurationAdvanced:
    # Checks that LeadSegment-based rows are applied correctly.
    def test_lead_segment_rows(self):
        df = _make_12lead_df()
        config = [
            [LeadSegment(lead="I", start=0, end=60), LeadSegment(lead="II", start=0, end=60)],
            [LeadSegment(lead="III", start=0, end=60), LeadSegment(lead="aVR", start=0, end=60)],
        ]
        result = _apply_configuration(df, config, disconnect_segments=False)
        assert len(result) == 2
        for signal, _, _, _ in result:
            assert signal.shape == (120,)

    # Checks that mixing string rows and LeadSegment rows raises an error.
    def test_mixed_string_and_lead_segment_rows(self):
        df = _make_12lead_df()
        config = [
            [LeadSegment(lead="I", start=0, end=N_SAMPLES)],
            "II",
        ]
        with pytest.raises(ValueError, match="mixes string-based and LeadSegment-based rows"):
            _apply_configuration(df, config, disconnect_segments=False)

    # Checks that a single LeadSegment entry as a full-width row works.
    def test_single_lead_segment_entry(self):
        df = _make_12lead_df()
        config = [LeadSegment(lead="I", start=0, end=N_SAMPLES)]
        result = _apply_configuration(df, config, disconnect_segments=False)
        assert len(result) == 1
        assert result[0][0].shape == (N_SAMPLES,)
        np.testing.assert_array_equal(result[0][0], df["I"].values)

    # Checks that mixing strings and LeadSegments within a row is rejected.
    def test_mixed_types_in_row_rejected(self):
        df = _make_12lead_df()
        config = [
            ["I", LeadSegment(lead="II", start=0, end=60)],
        ]
        with pytest.raises(ValueError, match="all entries must be the same type"):
            _apply_configuration(df, config, disconnect_segments=False)

    # Checks that a LeadSegment with missing 'end' is rejected at construction time.
    def test_missing_end_rejected(self):
        with pytest.raises(TypeError):
            LeadSegment(lead="I", start=0)  # type: ignore[call-arg]

    # Checks that end <= start is rejected at construction time.
    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError, match="must be greater than"):
            LeadSegment(lead="I", start=50, end=30)

    # Checks that negative start is rejected at construction time.
    def test_negative_start_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            LeadSegment(lead="I", start=-1, end=50)

    # Checks that an unknown lead name is rejected at configuration resolution time.
    def test_unknown_lead_rejected(self):
        df = _make_12lead_df()
        config = [
            [LeadSegment(lead="UNKNOWN", start=0, end=60)],
        ]
        with pytest.raises(ValueError, match="not present in the input data"):
            _resolve_configuration(config, list(df.columns))

    # Checks that unequal row lengths trigger a UserWarning.
    def test_unequal_row_lengths_warns(self):
        df = _make_12lead_df()
        config = [
            [LeadSegment(lead="I", start=0, end=100)],  # 100 samples
            [LeadSegment(lead="II", start=0, end=80)],  # 80 samples
        ]
        with pytest.warns(UserWarning, match="unequal total sample counts"):
            _apply_configuration(df, config, disconnect_segments=False)

    # Checks that equal row lengths do NOT trigger the unequal-length warning.
    def test_equal_row_lengths_no_warn(self):
        df = _make_12lead_df()
        config = [
            [LeadSegment(lead="I", start=0, end=100)],
            [LeadSegment(lead="II", start=0, end=100)],
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _apply_configuration(df, config, disconnect_segments=False)

    # Checks that the unequal-length warning is NOT raised for string-only configurations.
    def test_string_only_config_no_unequal_length_warn(self):
        df = _make_12lead_df()
        # 3-lead row (120 samples) + 7-lead row (119 samples) — lengths differ
        # but only the divisibility warning fires, not the unequal-length one.
        config = [["I", "II", "III"], ["V1", "V2", "V3", "V4", "V5", "V6", "aVR"]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _apply_configuration(df, config, disconnect_segments=False)
        messages = [str(w.message) for w in caught]
        assert not any("unequal total sample counts" in m for m in messages)


# ---------------------------------------------------------------------------
# expand_to_12_leads
# ---------------------------------------------------------------------------

_8_LEAD_NAMES = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _make_8lead_df() -> pd.DataFrame:
    """Create an 8-lead DataFrame with constant per-lead values for easy verification."""
    rng = np.random.default_rng(0)
    data = {lead: rng.standard_normal(N_SAMPLES) for lead in _8_LEAD_NAMES}
    return pd.DataFrame(data)


class TestExpandTo12Lead:
    # Checks that the output has 12 columns in the standard order.
    def test_output_has_12_leads(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        assert list(result.columns) == list(SUPPORTED_LEADS)

    # Checks that the output has the same number of samples as the input.
    def test_output_shape(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        assert result.shape == (N_SAMPLES, 12)

    # Checks that leads I and II are preserved unchanged.
    def test_original_leads_preserved(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        np.testing.assert_array_equal(result["I"].values, df["I"].values)
        np.testing.assert_array_equal(result["II"].values, df["II"].values)

    # Checks that precordial leads V1-V6 are preserved unchanged.
    def test_precordial_leads_preserved(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
            np.testing.assert_array_equal(result[lead].values, df[lead].values)

    # Checks III = II - I.
    def test_lead_III_formula(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        np.testing.assert_allclose(result["III"].values, df["II"].values - df["I"].values)

    # Checks aVR = -(I + II) / 2.
    def test_lead_AVR_formula(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        np.testing.assert_allclose(result["aVR"].values, -(df["I"].values + df["II"].values) / 2.0)

    # Checks aVL = I - II/2.
    def test_lead_AVL_formula(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        np.testing.assert_allclose(result["aVL"].values, df["I"].values - df["II"].values / 2.0)

    # Checks aVF = II - I/2.
    def test_lead_AVF_formula(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        np.testing.assert_allclose(result["aVF"].values, df["II"].values - df["I"].values / 2.0)

    # Checks that Einthoven's law holds: I + III = II (i.e. III = II - I).
    def test_einthoven_law(self):
        df = _make_8lead_df()
        result = expand_to_12_leads(df)
        np.testing.assert_allclose(result["I"].values + result["III"].values, result["II"].values, atol=1e-10)

    # Checks that leads_map correctly resolves custom column names.
    def test_leads_map_resolves_custom_names(self):
        custom_leads = ["lead_I", "lead_II", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.standard_normal((N_SAMPLES, 8)), columns=custom_leads)
        lm = LeadsMap(I="lead_I", II="lead_II", V1="ch1", V2="ch2", V3="ch3", V4="ch4", V5="ch5", V6="ch6")
        result = expand_to_12_leads(df, leads_map=lm)
        assert list(result.columns) == list(SUPPORTED_LEADS)
        np.testing.assert_array_equal(result["I"].values, df["lead_I"].values)
        np.testing.assert_array_equal(result["II"].values, df["lead_II"].values)

    # Checks that a missing required lead raises ValueError.
    def test_missing_lead_raises(self):
        df = _make_8lead_df().drop(columns=["V3"])
        with pytest.raises(ValueError, match="requires lead 'V3'"):
            expand_to_12_leads(df)

    # Checks that a numpy tuple input is accepted.
    def test_numpy_tuple_input(self):
        arr = np.random.default_rng(2).standard_normal((N_SAMPLES, 8))
        result = expand_to_12_leads((arr, _8_LEAD_NAMES))
        assert list(result.columns) == list(SUPPORTED_LEADS)
        assert result.shape == (N_SAMPLES, 12)


# Checks that derived limb leads match the recorded PTB-XL ground-truth leads within floating-point tolerance.
@pytest.mark.integration
def test_expand_to_12_leads_matches_ptbxl():
    """Derived leads III, aVR, aVL, aVF match PTB-XL ground-truth within 1.5e-3 (element-wise)."""
    from ptbxl_helper import get_ptbxl_record

    record = get_ptbxl_record(1)
    # PTB-XL uses uppercase "AVR"/"AVL"/"AVF"; rename to canonical names before processing.
    ecg_df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    ecg_df = ecg_df.rename(columns={"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"})

    eight_lead_df = ecg_df.drop(columns=["III", "aVR", "aVL", "aVF"])
    result = expand_to_12_leads(eight_lead_df)

    for lead in ("III", "aVR", "aVL", "aVF"):
        np.testing.assert_allclose(
            result[lead].values,
            ecg_df[lead].values,
            atol=1.5e-3,
            err_msg=f"Derived lead {lead} does not match PTB-XL ground truth",
        )
