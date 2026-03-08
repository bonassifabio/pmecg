"""Unit tests for pmecg.utils.data."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from pmecg.utils.data import (
    SUPPORTED_LEADS,
    LeadsMap,
    _apply_configuration,
    _numpy_to_dataframe,
    _resolve_configuration,
    _segment_leads,
    _template_configuration,
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
# _segment_leads
# ---------------------------------------------------------------------------

SEGMENT_LEAD_GROUPS = [
    ["I"],
    ["I", "II"],
    ["I", "AVR", "V6"],
    ["I", "II", "III", "AVR"],
    ["V1", "V2", "V3", "V4", "V5", "V6"],
    # Adding a case that triggers the warning
    ["I", "II", "III", "AVR", "AVL", "AVF", "V1"],  # 7 leads, 120 % 7 != 0
]


@pytest.mark.parametrize("selected_leads", SEGMENT_LEAD_GROUPS)
@pytest.mark.parametrize("disconnect", [True, False])
class TestSegmentLeads:
    """Cartesian product: lead groups × disconnect flag."""

    # Checks that segmenting any requested lead group always returns a 1D signal of the original length.
    def test_output_shape(self, selected_leads, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            signal, _ = _segment_leads(_make_12lead_df(), selected_leads, disconnect_segments=disconnect)
        assert signal.shape == (N_SAMPLES,)

    # Checks that the function reports back the same lead sequence it was asked to segment.
    def test_returned_leads(self, selected_leads, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            _, ret_leads = _segment_leads(_make_12lead_df(), selected_leads, disconnect_segments=disconnect)
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
            signal, _ = _segment_leads(_make_12lead_df(), selected_leads, disconnect_segments=disconnect)
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
            signal, _ = _segment_leads(_make_12lead_df(), selected_leads, disconnect_segments=disconnect)
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
        sig_str, leads_str = _segment_leads(df, lead, disconnect_segments=False)
        sig_list, _ = _segment_leads(df, [lead], disconnect_segments=False)
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
        [["I", "II", "III"], ["AVR", "AVL", "AVF"]],
        [["I", "II", "III"], ["AVR", "AVL", "AVF"]],
        id="exotic-2x3",
    ),
    pytest.param(
        [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"]],
        [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"]],
        id="exotic-3x4",
    ),
    pytest.param(
        [["I", "II"], ["III", "AVR"], ["AVL", "AVF"], ["V1", "V2"]],
        [["I", "II"], ["III", "AVR"], ["AVL", "AVF"], ["V1", "V2"]],
        id="exotic-4x2",
    ),
    # configuration with 6 leads in a row (will trigger warning with N_SAMPLES=120? 120/6=20, no)
    # let's add a custom configuration that triggers warning
    pytest.param(
        [["I", "II", "III", "AVR", "AVL", "AVF", "V1"]],
        [["I", "II", "III", "AVR", "AVL", "AVF", "V1"]],
        id="warn-7-leads",
    ),
    pytest.param(
        [["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3"]],
        [["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3"]],
        id="warn-9-leads",
    ),
    pytest.param(
        [["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5"]],
        [["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5"]],
        id="warn-11-leads",
    ),
    pytest.param(
        [
            ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            ["I", "II", "III", "AVR", "AVL", "AVF", "V1"],
        ],
        [
            ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            ["I", "II", "III", "AVR", "AVL", "AVF", "V1"],
        ],
        id="mixed-warn",
    ),
    pytest.param(
        [["I", "II", "III", "AVR"], ["AVL", "AVF", "V1", "V2"], "V3"],
        [["I", "II", "III", "AVR"], ["AVL", "AVF", "V1", "V2"], ["V3"]],
        id="mixed-with-strip",
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

    # Checks that every configured row produces a full-length 1D signal.
    def test_signal_shapes(self, config, expected_leads_per_row, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        for signal, _ in result:
            assert signal.shape == (N_SAMPLES,)

    # Checks that each configured row keeps the expected lead names in order.
    def test_lead_names(self, config, expected_leads_per_row, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        for (_, ret_leads), exp in zip(result, expected_leads_per_row):
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
        for (signal, _), row_leads in zip(result, expected_leads_per_row):
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
        for (signal, _), row_leads in zip(result, expected_leads_per_row):
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

        for i, (signal, selected_leads) in enumerate(result):
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
            "LII",
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
        configuration = [["I", "AVR", "V1"], "V6"]
        with pytest.raises(ValueError, match="Lead name 'I' in configuration is not present"):
            _resolve_configuration(configuration, CUSTOM_LEADS)

    # Checks that template expansion fails when a required conventional lead is missing from the custom map.
    def test_template_factory_missing_required_canonical_mapping_raises(self):
        partial_map = CUSTOM_LEADS_MAP._replace(AVR=None)
        with pytest.raises(ValueError, match="Template '4x3' requires conventional lead 'AVR'"):
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
            "ii",
        ]
