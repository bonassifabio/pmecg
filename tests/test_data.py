"""Unit tests for pmecg.utils.data."""

from contextlib import nullcontext

import numpy as np
import pytest

from pmecg.utils.data import (
    SUPPORTED_LEADS,
    TEMPLATE_CONFIGURATIONS,
    _apply_configuration,
    _numpy_to_dataframe,
    _segment_leads,
)

# SUPPORTED_LEADS = ("I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6")
# Lead at index i has value (i+1.0) in all helper fixtures below.
LEAD_VALUE = {lead: float(i + 1) for i, lead in enumerate(SUPPORTED_LEADS)}

N_SAMPLES = 120  # divisible by 1, 2, 3, 4, 6, 8, 12


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
    if isinstance(configuration, str):
        if configuration in TEMPLATE_CONFIGURATIONS:
            config = TEMPLATE_CONFIGURATIONS[configuration]
        else:
            # Single lead name string
            config = [configuration]
    else:
        config = configuration

    # Normalize to list of rows, where each row is a list of leads
    rows = [[e] if isinstance(e, str) else e for e in config]
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

    def test_shape(self, template_key):
        leads = TEMPLATE_CONFIGURATIONS[template_key]
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        assert df.shape == (N_SAMPLES, len(leads))

    def test_columns_match_leads(self, template_key):
        leads = TEMPLATE_CONFIGURATIONS[template_key]
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        assert list(df.columns) == [lead.upper() for lead in leads]

    def test_values_match_lead_index(self, template_key):
        """Column i must equal (i+1.0) for all samples."""
        leads = TEMPLATE_CONFIGURATIONS[template_key]
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        for i, lead in enumerate(leads):
            np.testing.assert_array_equal(df[lead.upper()].values, float(i + 1))


class TestNumpyToDataframeDefaults:
    def test_12lead_default_column_names(self):
        """With exactly 12 leads and no explicit names, columns use SUPPORTED_LEADS."""
        ecg_data = _make_ecg_array(list(SUPPORTED_LEADS))
        df = _numpy_to_dataframe(ecg_data)  # lead_names=None
        assert list(df.columns) == list(SUPPORTED_LEADS)

    def test_wrong_lead_count_raises(self):
        """Passing a 5-lead array without explicit names must raise AssertionError."""
        with pytest.raises(AssertionError):
            _numpy_to_dataframe(np.ones((N_SAMPLES, 5)))

    def test_mismatched_names_raises(self):
        """Passing lead_names of wrong length must raise AssertionError."""
        with pytest.raises(AssertionError):
            _numpy_to_dataframe(np.ones((N_SAMPLES, 3)), ["I", "II"])


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

    def test_output_shape(self, selected_leads, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            signal, _ = _segment_leads(_make_12lead_df(), selected_leads, disconnect_segments=disconnect)
        assert signal.shape == (N_SAMPLES,)

    def test_returned_leads(self, selected_leads, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if N_SAMPLES % len(selected_leads) != 0
            else nullcontext()
        )
        with ctx:
            _, ret_leads = _segment_leads(_make_12lead_df(), selected_leads, disconnect_segments=disconnect)
        assert ret_leads == selected_leads

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
    pytest.param("V5", [["V5"]], id="single-lead"),
    # 1xL template strings: they now produce one row per lead
    pytest.param("1x1", [["I"]], id="template-1x1"),
    pytest.param("1x2", [["I"], ["II"]], id="template-1x2"),
    pytest.param("1x3", [["I"], ["II"], ["V2"]], id="template-1x3"),
    pytest.param("1x4", [["I"], ["II"], ["III"], ["V2"]], id="template-1x4"),
    pytest.param("1x6", [["I"], ["II"], ["III"], ["AVR"], ["AVL"], ["AVF"]], id="template-1x6"),
    pytest.param("1x8", [["I"], ["II"], ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"]], id="template-1x8"),
    pytest.param(
        "1x12",
        [["I"], ["II"], ["III"], ["AVR"], ["AVL"], ["AVF"], ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"]],
        id="template-1x12",
    ),
    # multi-row template strings
    pytest.param(
        "4x3", [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"], ["II"]], id="template-4x3"
    ),
    pytest.param("2x4", [["I", "V3"], ["II", "V4"], ["III", "V5"], ["AVR", "V6"], ["II"]], id="template-2x4"),
    pytest.param(
        "2x6",
        [["I", "V1"], ["II", "V2"], ["III", "V3"], ["AVR", "V4"], ["AVL", "V5"], ["AVF", "V6"], ["II"]],
        id="template-2x6",
    ),
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

    def test_row_count(self, config, expected_leads_per_row, disconnect):
        ctx = (
            pytest.warns(UserWarning, match="is not evenly divisible")
            if _should_warn_divisible(config, N_SAMPLES)
            else nullcontext()
        )
        with ctx:
            result = _apply_configuration(_make_12lead_df(), config, disconnect_segments=disconnect)
        assert len(result) == len(expected_leads_per_row)

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
