"""Unit tests for pmecg.utils.data."""

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
        assert list(df.columns) == [l.upper() for l in leads]

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
        ecg_data = np.ones((N_SAMPLES, 5))
        with pytest.raises(AssertionError):
            _numpy_to_dataframe(ecg_data)

    def test_mismatched_names_raises(self):
        """Passing lead_names of wrong length must raise AssertionError."""
        ecg_data = np.ones((N_SAMPLES, 3))
        with pytest.raises(AssertionError):
            _numpy_to_dataframe(ecg_data, ["I", "II"])


# ---------------------------------------------------------------------------
# _segment_leads
# ---------------------------------------------------------------------------

class TestSegmentLeads:
    """Segmentation of leads into a single concatenated vector."""

    def test_single_lead_string(self):
        """A string lead name is treated as a one-element list."""
        df = _make_12lead_df()
        signal, leads = _segment_leads(df, "II")
        assert leads == ["II"]
        assert signal.shape == (N_SAMPLES,)
        np.testing.assert_array_equal(signal, LEAD_VALUE["II"])  # 2.0

    def test_single_lead_list(self):
        df = _make_12lead_df()
        signal, leads = _segment_leads(df, ["V6"])
        assert leads == ["V6"]
        np.testing.assert_array_equal(signal, LEAD_VALUE["V6"])  # 12.0

    def test_two_leads_segment_boundaries(self):
        """First half → lead[0], second half → lead[1]."""
        df = _make_12lead_df()
        signal, leads = _segment_leads(df, ["I", "II"])
        seg = N_SAMPLES // 2
        np.testing.assert_array_equal(signal[:seg], LEAD_VALUE["I"])   # 1.0
        np.testing.assert_array_equal(signal[seg:], LEAD_VALUE["II"])  # 2.0

    def test_three_leads_segment_boundaries(self):
        """Three equal segments, each from a different lead."""
        df = _make_12lead_df()
        signal, _ = _segment_leads(df, ["I", "AVR", "V6"])
        seg = N_SAMPLES // 3
        np.testing.assert_array_equal(signal[:seg],        LEAD_VALUE["I"])   # 1.0
        np.testing.assert_array_equal(signal[seg:2 * seg], LEAD_VALUE["AVR"]) # 4.0
        np.testing.assert_array_equal(signal[2 * seg:],    LEAD_VALUE["V6"])  # 12.0

    def test_four_leads_segment_boundaries(self):
        df = _make_12lead_df()
        selected = ["I", "II", "III", "AVR"]
        signal, _ = _segment_leads(df, selected)
        seg = N_SAMPLES // 4
        for i, lead in enumerate(selected):
            np.testing.assert_array_equal(
                signal[i * seg : (i + 1) * seg], LEAD_VALUE[lead]
            )

    def test_output_shape(self):
        df = _make_12lead_df()
        signal, _ = _segment_leads(df, ["V1", "V2", "V3", "V4", "V5", "V6"])
        assert signal.shape == (N_SAMPLES,)

    def test_returned_leads_preserved(self):
        df = _make_12lead_df()
        selected = ["I", "II", "III"]
        _, ret_leads = _segment_leads(df, selected)
        assert ret_leads == selected


# ---------------------------------------------------------------------------
# _apply_configuration
# ---------------------------------------------------------------------------

class TestApplyConfigurationTemplates:
    """Apply every 1xL template (passed as a string key)."""

    @pytest.mark.parametrize("template_key", ONE_ROW_TEMPLATES)
    def test_1xL_returns_single_row(self, template_key):
        leads = TEMPLATE_CONFIGURATIONS[template_key]
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        result = _apply_configuration(df, template_key)
        assert len(result) == 1

    @pytest.mark.parametrize("template_key", ONE_ROW_TEMPLATES)
    def test_1xL_signal_shape(self, template_key):
        leads = TEMPLATE_CONFIGURATIONS[template_key]
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        signal, _ = _apply_configuration(df, template_key)[0]
        assert signal.shape == (N_SAMPLES,)

    @pytest.mark.parametrize("template_key", ONE_ROW_TEMPLATES)
    def test_1xL_lead_names(self, template_key):
        leads = TEMPLATE_CONFIGURATIONS[template_key]
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        _, ret_leads = _apply_configuration(df, template_key)[0]
        assert ret_leads == leads

    def test_1x3_segment_values(self):
        """1x3 = ['I','II','V2']; verify all three segment values."""
        leads = TEMPLATE_CONFIGURATIONS["1x3"]   # ['I', 'II', 'V2']
        df = _numpy_to_dataframe(_make_ecg_array(leads), leads)
        signal, _ = _apply_configuration(df, "1x3")[0]
        seg = N_SAMPLES // 3
        np.testing.assert_array_equal(signal[:seg],         1.0)  # I
        np.testing.assert_array_equal(signal[seg:2 * seg],  2.0)  # II
        np.testing.assert_array_equal(signal[2 * seg:],     3.0)  # V2

    def test_2x4_row_count(self):
        """2x4 config has 4 lead rows + 1 rhythm strip → 5 rows total."""
        df = _make_12lead_df()
        result = _apply_configuration(df, TEMPLATE_CONFIGURATIONS["2x4"])
        assert len(result) == 5

    def test_2x6_row_count(self):
        """2x6 config has 6 lead rows + 1 rhythm strip → 7 rows total."""
        df = _make_12lead_df()
        result = _apply_configuration(df, TEMPLATE_CONFIGURATIONS["2x6"])
        assert len(result) == 7

    def test_4x3_row_count(self):
        """4x3 config has 3 lead rows + 1 rhythm strip → 4 rows total."""
        df = _make_12lead_df()
        result = _apply_configuration(df, TEMPLATE_CONFIGURATIONS["4x3"])
        assert len(result) == 4


class TestApplyConfigurationExotic:
    """Exotic configurations – same number of columns per row, not matching any template."""

    def test_two_rows_of_three_row_count(self):
        df = _make_12lead_df()
        result = _apply_configuration(df, [["I", "II", "III"], ["AVR", "AVL", "AVF"]])
        assert len(result) == 2

    def test_two_rows_of_three_segment_values(self):
        """[['I','II','III'], ['AVR','AVL','AVF']]: verify segment values in both rows."""
        df = _make_12lead_df()
        config = [["I", "II", "III"], ["AVR", "AVL", "AVF"]]
        result = _apply_configuration(df, config)
        seg = N_SAMPLES // 3

        (sig0, _) = result[0]
        np.testing.assert_array_equal(sig0[:seg],        LEAD_VALUE["I"])    # 1.0
        np.testing.assert_array_equal(sig0[seg:2 * seg], LEAD_VALUE["II"])   # 2.0
        np.testing.assert_array_equal(sig0[2 * seg:],    LEAD_VALUE["III"])  # 3.0

        (sig1, _) = result[1]
        np.testing.assert_array_equal(sig1[:seg],        LEAD_VALUE["AVR"])  # 4.0
        np.testing.assert_array_equal(sig1[seg:2 * seg], LEAD_VALUE["AVL"])  # 5.0
        np.testing.assert_array_equal(sig1[2 * seg:],    LEAD_VALUE["AVF"])  # 6.0

    def test_three_rows_of_four_row_count(self):
        df = _make_12lead_df()
        config = [
            ["I",   "AVR", "V1", "V4"],
            ["II",  "AVL", "V2", "V5"],
            ["III", "AVF", "V3", "V6"],
        ]
        result = _apply_configuration(df, config)
        assert len(result) == 3

    def test_three_rows_of_four_segment_values(self):
        """Row 0: I→1.0, AVR→4.0, V1→7.0, V4→10.0; row 1: II→2.0, AVL→5.0, V2→8.0, V5→11.0; row 2: III→3.0, AVF→6.0, V3→9.0, V6→12.0."""
        df = _make_12lead_df()
        config = [
            ["I",   "AVR", "V1", "V4"],
            ["II",  "AVL", "V2", "V5"],
            ["III", "AVF", "V3", "V6"],
        ]
        result = _apply_configuration(df, config)
        seg = N_SAMPLES // 4

        (sig0, _) = result[0]
        np.testing.assert_array_equal(sig0[:seg],            LEAD_VALUE["I"])    # 1.0
        np.testing.assert_array_equal(sig0[seg:2 * seg],     LEAD_VALUE["AVR"])  # 4.0
        np.testing.assert_array_equal(sig0[2 * seg:3 * seg], LEAD_VALUE["V1"])   # 7.0
        np.testing.assert_array_equal(sig0[3 * seg:],        LEAD_VALUE["V4"])   # 10.0

        (sig1, _) = result[1]
        np.testing.assert_array_equal(sig1[:seg],            LEAD_VALUE["II"])   # 2.0
        np.testing.assert_array_equal(sig1[seg:2 * seg],     LEAD_VALUE["AVL"])  # 5.0
        np.testing.assert_array_equal(sig1[2 * seg:3 * seg], LEAD_VALUE["V2"])   # 8.0
        np.testing.assert_array_equal(sig1[3 * seg:],        LEAD_VALUE["V5"])   # 11.0

        (sig2, _) = result[2]
        np.testing.assert_array_equal(sig2[:seg],            LEAD_VALUE["III"])  # 3.0
        np.testing.assert_array_equal(sig2[seg:2 * seg],     LEAD_VALUE["AVF"])  # 6.0
        np.testing.assert_array_equal(sig2[2 * seg:3 * seg], LEAD_VALUE["V3"])   # 9.0
        np.testing.assert_array_equal(sig2[3 * seg:],        LEAD_VALUE["V6"])   # 12.0

    def test_single_lead_string(self):
        """A bare lead string (full-duration rhythm strip) is a valid configuration."""
        df = _make_12lead_df()
        result = _apply_configuration(df, "V5")
        assert len(result) == 1
        signal, leads = result[0]
        assert leads == ["V5"]
        np.testing.assert_array_equal(signal, LEAD_VALUE["V5"])  # 11.0

    def test_mixed_rows_with_rhythm_strip(self):
        """[['I','II','III','AVR'], ['AVL','AVF','V1','V2'], 'V3'] – 2 rows + strip."""
        df = _make_12lead_df()
        config = [["I", "II", "III", "AVR"], ["AVL", "AVF", "V1", "V2"], "V3"]
        result = _apply_configuration(df, config)
        assert len(result) == 3
        seg = N_SAMPLES // 4

        (sig0, _) = result[0]
        np.testing.assert_array_equal(sig0[:seg],            LEAD_VALUE["I"])    # 1.0
        np.testing.assert_array_equal(sig0[seg:2 * seg],     LEAD_VALUE["II"])   # 2.0
        np.testing.assert_array_equal(sig0[2 * seg:3 * seg], LEAD_VALUE["III"])  # 3.0
        np.testing.assert_array_equal(sig0[3 * seg:],        LEAD_VALUE["AVR"])  # 4.0

        (sig1, _) = result[1]
        np.testing.assert_array_equal(sig1[:seg],            LEAD_VALUE["AVL"])  # 5.0
        np.testing.assert_array_equal(sig1[seg:2 * seg],     LEAD_VALUE["AVF"])  # 6.0
        np.testing.assert_array_equal(sig1[2 * seg:3 * seg], LEAD_VALUE["V1"])   # 7.0
        np.testing.assert_array_equal(sig1[3 * seg:],        LEAD_VALUE["V2"])   # 8.0

        # Rhythm strip row: V3 → value 9.0, full N_SAMPLES length
        sig_strip, leads_strip = result[2]
        assert leads_strip == ["V3"]
        np.testing.assert_array_equal(sig_strip, LEAD_VALUE["V3"])  # 9.0

    def test_four_rows_of_two(self):
        """[['I','II'], ['III','AVR'], ['AVL','AVF'], ['V1','V2']] – 4 rows of 2."""
        df = _make_12lead_df()
        config = [["I", "II"], ["III", "AVR"], ["AVL", "AVF"], ["V1", "V2"]]
        result = _apply_configuration(df, config)
        assert len(result) == 4
        seg = N_SAMPLES // 2

        (sig0, _) = result[0]  # row 0: I→1.0, II→2.0
        np.testing.assert_array_equal(sig0[:seg], LEAD_VALUE["I"])   # 1.0
        np.testing.assert_array_equal(sig0[seg:], LEAD_VALUE["II"])  # 2.0

        (sig1, _) = result[1]  # row 1: III→3.0, AVR→4.0
        np.testing.assert_array_equal(sig1[:seg], LEAD_VALUE["III"])  # 3.0
        np.testing.assert_array_equal(sig1[seg:], LEAD_VALUE["AVR"])  # 4.0

        (sig2, _) = result[2]  # row 2: AVL→5.0, AVF→6.0
        np.testing.assert_array_equal(sig2[:seg], LEAD_VALUE["AVL"])  # 5.0
        np.testing.assert_array_equal(sig2[seg:], LEAD_VALUE["AVF"])  # 6.0

        (sig3, _) = result[3]  # row 3: V1→7.0, V2→8.0
        np.testing.assert_array_equal(sig3[:seg], LEAD_VALUE["V1"])  # 7.0
        np.testing.assert_array_equal(sig3[seg:], LEAD_VALUE["V2"])  # 8.0
