"""Tests for attention map conversion, scaling, and rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection

import pmecg.utils.attention as attention_utils
from pmecg import (
    BackgroundAttentionMap,
    ECGPlotter,
    IntervalAttentionMap,
    LineColorAttentionMap,
    attention_map_from_indices_annotations,
    attention_map_from_time_annotations,
)
from pmecg.utils.plot import MM_PER_INCH, RIGHT_MARGIN_MM, _adjust_row_distance, _compute_figure_size


def _first_axes(fig):
    return fig.axes[0]


def test_attention_conversion_preserves_dataframe():
    attention_df = pd.DataFrame({"I": [0.0, 1.0, 2.0], "II": [2.0, 1.0, 0.0]})

    converted = attention_utils._attention_to_dataframe(attention_df)

    pd.testing.assert_frame_equal(converted, attention_df)


def test_attention_conversion_from_numpy_tuple():
    values = np.column_stack((np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])))

    converted = attention_utils._attention_to_dataframe((values, ["I", "II"]))

    expected = pd.DataFrame(values, columns=["I", "II"])
    pd.testing.assert_frame_equal(converted, expected)


def test_attention_conversion_from_list_tuple():
    converted = attention_utils._attention_to_dataframe(
        ([np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])], ["I", "II"])
    )

    expected = pd.DataFrame({"I": [0.0, 1.0, 2.0], "II": [2.0, 1.0, 0.0]})
    pd.testing.assert_frame_equal(converted, expected)


def test_attention_conversion_stringifies_dataframe_columns():
    attention_df = pd.DataFrame({0: [0.0, 1.0], 1: [2.0, 3.0]})

    converted = attention_utils._attention_to_dataframe(attention_df)

    assert list(converted.columns) == ["0", "1"]


def test_attention_conversion_from_1d_array_requires_single_lead_name():
    with pytest.raises(ValueError, match="requires exactly one lead name"):
        attention_utils._attention_to_dataframe((np.array([0.0, 1.0, 2.0]), ["I", "II"]))


@pytest.mark.parametrize(
    ("ecg_data", "expected"),
    [
        (pd.DataFrame({"I": [0.0, 1.0, 2.0]}), 3),
        ((np.zeros((4, 2)), ["I", "II"]), 4),
        (([np.zeros(5), np.ones(5)], ["I", "II"]), 5),
    ],
)
def test_extract_n_samples_supports_all_ecg_input_formats(ecg_data, expected):
    assert attention_utils._extract_n_samples(ecg_data) == expected


def test_resolve_indices_annotation_returns_bounds_and_value():
    resolved = attention_utils._resolve_indices_annotation(
        {"index_range": [1, 4], "attention_value": 0.75},
        lead_name="I",
        n_samples=8,
    )

    assert resolved == (1, 4, 0.75)


def test_resolve_indices_annotation_rejects_non_dictionary_payload():
    with pytest.raises(ValueError, match="must be a dictionary"):
        attention_utils._resolve_indices_annotation(["bad"], lead_name="I", n_samples=8)


def test_extract_attention_value_returns_float():
    value = attention_utils._extract_attention_value({"attention_value": np.float32(0.5)}, lead_name="I")

    assert value == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("annotation", "match"),
    [
        ({}, "must contain 'attention_value'"),
        ({"attention_value": np.inf}, "must be a finite number"),
        ({"attention_value": "bad"}, "must be a finite number"),
    ],
)
def test_extract_attention_value_rejects_malformed_values(annotation, match):
    with pytest.raises(ValueError, match=match):
        attention_utils._extract_attention_value(annotation, lead_name="I")


def test_extract_time_range_returns_original_payload():
    time_range = attention_utils._extract_time_range(
        {"time_range": (0.25, 0.75), "attention_value": 1.0},
        lead_name="I",
    )

    assert time_range == (0.25, 0.75)


@pytest.mark.parametrize(
    ("annotation", "match"),
    [
        (["bad"], "must be a dictionary"),
        ({"attention_value": 1.0}, "must contain 'time_range'"),
        (
            {"time_range": [0.0, 1.0], "attention_value": 1.0, "extra": True},
            "Unsupported time annotation keys",
        ),
    ],
)
def test_extract_time_range_rejects_malformed_annotations(annotation, match):
    with pytest.raises(ValueError, match=match):
        attention_utils._extract_time_range(annotation, lead_name="I")


def test_index_range_to_sample_bounds_accepts_lists_and_tuples():
    assert attention_utils._index_range_to_sample_bounds({"index_range": [0, 3], "attention_value": 1.0}, n_samples=4) == (
        0,
        3,
    )
    assert attention_utils._index_range_to_sample_bounds({"index_range": (1, 4), "attention_value": 1.0}, n_samples=4) == (
        1,
        4,
    )


@pytest.mark.parametrize(
    ("annotation", "match"),
    [
        ({"attention_value": 1.0}, "must contain 'index_range'"),
        (
            {"index_range": [0, 1], "attention_value": 1.0, "extra": True},
            "Unsupported index annotation keys",
        ),
        ({"index_range": [0], "attention_value": 1.0}, "exactly two integers"),
        ({"index_range": [0.0, 1], "attention_value": 1.0}, "must be integers"),
        ({"index_range": [-1, 1], "attention_value": 1.0}, "must be non-negative"),
        ({"index_range": [3, 2], "attention_value": 1.0}, "greater than or equal"),
        ({"index_range": [0, 6], "attention_value": 1.0}, "exceeds the ECG length"),
        ({"index_range": [2, 2], "attention_value": 1.0}, "span at least one ECG sample"),
    ],
)
def test_index_range_to_sample_bounds_rejects_invalid_ranges(annotation, match):
    with pytest.raises(ValueError, match=match):
        attention_utils._index_range_to_sample_bounds(annotation, n_samples=5)


def test_time_range_to_sample_bounds_uses_floor_and_ceil_rounding():
    bounds = attention_utils._time_range_to_sample_bounds(
        (0.24, 0.76),
        fs=10.0,
        recording_duration_seconds=1.0,
        n_samples=10,
    )

    assert bounds == (2, 8)


def test_time_range_to_sample_bounds_clamps_to_last_sample():
    bounds = attention_utils._time_range_to_sample_bounds(
        (0.8, 1.0),
        fs=10.0,
        recording_duration_seconds=1.0,
        n_samples=9,
    )

    assert bounds == (8, 9)


@pytest.mark.parametrize(
    ("time_range", "match"),
    [
        ([0.0], "exactly two numbers"),
        (("bad", 1.0), "must be numeric"),
        ((0.0, np.inf), "must be finite"),
        ((-0.1, 0.1), "must be non-negative"),
        ((0.5, 0.4), "greater than or equal"),
        ((0.0, 1.1), "exceeds the ECG duration"),
        ((0.3, 0.3), "span at least one ECG sample"),
    ],
)
def test_time_range_to_sample_bounds_rejects_invalid_ranges(time_range, match):
    with pytest.raises(ValueError, match=match):
        attention_utils._time_range_to_sample_bounds(
            time_range,
            fs=10.0,
            recording_duration_seconds=1.0,
            n_samples=10,
        )


def test_attention_map_from_indices_annotations_rejects_non_list_annotation_payload():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="must be provided as a list"):
        attention_map_from_indices_annotations(ecg_df, I={"index_range": [0, 1], "attention_value": 1.0})


def test_attention_map_from_indices_annotations_rejects_non_dictionary_annotation():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="must be a dictionary"):
        attention_map_from_indices_annotations(ecg_df, I=[["bad"]])


def test_attention_single_vector_is_broadcast_to_all_leads():
    attention = IntervalAttentionMap(data=(np.array([0.0, 0.5, 1.0]), ["shared"]), polarity="positive", color="red")

    attention.prepare(["I", "II"], 3, [["I"], ["II"]])

    expected = pd.DataFrame({"I": [0.0, 0.5, 1.0], "II": [0.0, 0.5, 1.0]})
    pd.testing.assert_frame_equal(attention.dataframe.reset_index(drop=True), expected)


def test_attention_map_from_indices_annotations_builds_expected_dataframe():
    ecg_df = pd.DataFrame({"I": np.zeros(8), "II": np.zeros(8)})

    attention_df = attention_map_from_indices_annotations(
        ecg_df,
        I=[
            {"index_range": [1, 3], "attention_value": 0.75},
            {"index_range": [4, 6], "attention_value": 0.25},
        ],
        II=[{"index_range": [0, 2], "attention_value": -0.5}],
    )

    expected = pd.DataFrame(
        {
            "I": [0.0, 0.75, 0.75, 0.0, 0.25, 0.25, 0.0, 0.0],
            "II": [-0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(attention_df, expected)


def test_attention_map_from_indices_annotations_supports_tuple_ecg_input():
    ecg_data = (np.zeros((6, 2)), ["I", "II"])

    attention_df = attention_map_from_indices_annotations(
        ecg_data,
        II=[{"index_range": [2, 5], "attention_value": 1.0}],
    )

    expected = pd.DataFrame(
        {
            "I": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "II": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(attention_df, expected)


def test_attention_map_from_indices_annotations_rejects_unknown_lead():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="not present in ecg_data"):
        attention_map_from_indices_annotations(
            ecg_df,
            II=[{"index_range": [0, 2], "attention_value": 1.0}],
        )


def test_attention_map_from_indices_annotations_rejects_empty_index_span():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="index_range must span at least one ECG sample"):
        attention_map_from_indices_annotations(
            ecg_df,
            I=[{"index_range": [1, 1], "attention_value": 1.0}],
        )


def test_attention_map_from_time_annotations_delegates_to_index_annotations():
    ecg_df = pd.DataFrame({"I": np.zeros(8), "II": np.zeros(8)})

    from_time = attention_map_from_time_annotations(
        ecg_df,
        fs=2.0,
        I=[
            {"time_range": [0.5, 1.5], "attention_value": 0.75},
            {"time_range": [2.0, 3.0], "attention_value": 0.25},
        ],
        II=[{"time_range": [0.0, 1.0], "attention_value": -0.5}],
    )
    from_indices = attention_map_from_indices_annotations(
        ecg_df,
        I=[
            {"index_range": [1, 3], "attention_value": 0.75},
            {"index_range": [4, 6], "attention_value": 0.25},
        ],
        II=[{"index_range": [0, 2], "attention_value": -0.5}],
    )

    pd.testing.assert_frame_equal(from_time, from_indices)


def test_attention_map_from_time_annotations_rejects_empty_time_span():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="time_range must span at least one ECG sample"):
        attention_map_from_time_annotations(
            ecg_df,
            fs=2.0,
            I=[{"time_range": [1.0, 1.0], "attention_value": 1.0}],
        )


def test_attention_map_from_time_annotations_rejects_invalid_sampling_frequency():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="fs must be a positive number"):
        attention_map_from_time_annotations(
            ecg_df,
            fs=0.0,
            I=[{"time_range": [0.0, 1.0], "attention_value": 1.0}],
        )


def test_attention_map_from_time_annotations_rejects_non_list_annotation_payload():
    ecg_df = pd.DataFrame({"I": np.zeros(4)})

    with pytest.raises(ValueError, match="must be provided as a list"):
        attention_map_from_time_annotations(ecg_df, fs=2.0, I={"time_range": [0.0, 1.0], "attention_value": 1.0})


def test_finite_attention_bounds_ignores_non_finite_values():
    bounds = attention_utils._finite_attention_bounds(pd.DataFrame({"I": [np.nan, -2.0], "II": [np.inf, 3.0]}))

    assert bounds == (-2.0, 3.0)


def test_finite_attention_bounds_requires_at_least_one_finite_value():
    with pytest.raises(ValueError, match="at least one finite value"):
        attention_utils._finite_attention_bounds(pd.DataFrame({"I": [np.nan, np.inf]}))


def test_positive_attention_scales_globally_when_max_exceeds_one():
    attention = IntervalAttentionMap(
        data=pd.DataFrame({"I": [0.0, 2.0, 4.0], "II": [0.0, 1.0, 3.0]}),
        polarity="positive",
        color="red",
    )

    attention.prepare(["I", "II"], 3, [["I"], ["II"]])

    expected = pd.DataFrame({"I": [0.0, 0.5, 1.0], "II": [0.0, 0.25, 0.75]})
    pd.testing.assert_frame_equal(attention.dataframe.reset_index(drop=True), expected)
    assert attention.range == (0.0, 1.0)


def test_scale_attention_dataframe_preserves_positive_values_already_within_range():
    scaled_df, attention_range = attention_utils._scale_attention_dataframe(
        pd.DataFrame({"I": [0.0, 0.25, 0.5]}),
        "positive",
    )

    expected = pd.DataFrame({"I": [0.0, 0.25, 0.5]})
    pd.testing.assert_frame_equal(scaled_df, expected)
    assert attention_range == (0.0, 0.5)


def test_signed_attention_scales_globally_across_columns():
    attention = IntervalAttentionMap(
        data=pd.DataFrame({"I": [-4.0, -2.0, 0.0], "II": [1.0, 2.0, 3.0]}),
        polarity="signed",
        color=("blue", "red"),
    )

    attention.prepare(["I", "II"], 3, [["I"], ["II"]])

    expected = pd.DataFrame({"I": [-1.0, -0.5, 0.0], "II": [0.25, 0.5, 0.75]})
    pd.testing.assert_frame_equal(attention.dataframe.reset_index(drop=True), expected)
    assert attention.range == (-1.0, 0.75)


def test_scale_attention_dataframe_preserves_signed_values_with_unit_magnitude():
    scaled_df, attention_range = attention_utils._scale_attention_dataframe(
        pd.DataFrame({"I": [-1.0, 0.0, 0.5]}),
        "signed",
    )

    expected = pd.DataFrame({"I": [-1.0, 0.0, 0.5]})
    pd.testing.assert_frame_equal(scaled_df, expected)
    assert attention_range == (-1.0, 0.5)


def test_positive_attention_rejects_negative_values():
    attention = IntervalAttentionMap(data=pd.DataFrame({"I": [-0.1, 0.2, 0.4]}), polarity="positive", color="red")

    with pytest.raises(ValueError, match="non-negative"):
        attention.prepare(["I"], 3, ["I"])


def test_positive_attention_rejects_zero_only_values():
    attention = IntervalAttentionMap(data=pd.DataFrame({"I": [0.0, 0.0, 0.0]}), polarity="positive", color="red")

    with pytest.raises(ValueError, match="greater than 0"):
        attention.prepare(["I"], 3, ["I"])


def test_signed_attention_requires_negative_and_positive_values():
    attention = IntervalAttentionMap(data=pd.DataFrame({"I": [0.1, 0.2, 0.4]}), polarity="signed", color=("blue", "red"))

    with pytest.raises(ValueError, match="span both negative and positive"):
        attention.prepare(["I"], 3, ["I"])


def test_positive_attention_requires_single_color_string():
    with pytest.raises(ValueError, match="non-empty matplotlib color string"):
        IntervalAttentionMap(data=pd.DataFrame({"I": [0.0, 0.5, 1.0]}), polarity="positive", color=("blue", "red"))


def test_signed_attention_requires_two_color_tuple():
    with pytest.raises(ValueError, match="tuple of two non-empty matplotlib color strings"):
        IntervalAttentionMap(data=pd.DataFrame({"I": [-1.0, 0.0, 1.0]}), polarity="signed", color="red")


def test_attention_map_constructor_rejects_non_boolean_show_colormap():
    with pytest.raises(ValueError, match="show_colormap must be a boolean"):
        LineColorAttentionMap(
            data=pd.DataFrame({"I": [-1.0, 0.0, 1.0]}),
            polarity="signed",
            color=("blue", "red"),
            show_colormap=1,
        )


@pytest.mark.parametrize(
    ("max_attention_mV", "alpha", "match"),
    [
        (-0.1, 0.5, "non-negative number"),
        ("bad", 0.5, "non-negative number"),
        (0.25, -0.1, "between 0 and 1"),
        (0.25, 1.1, "between 0 and 1"),
    ],
)
def test_interval_attention_constructor_rejects_invalid_visual_parameters(max_attention_mV, alpha, match):
    with pytest.raises(ValueError, match=match):
        IntervalAttentionMap(
            data=pd.DataFrame({"I": [0.0, 0.5, 1.0]}),
            polarity="positive",
            color="red",
            max_attention_mV=max_attention_mV,
            alpha=alpha,
        )


def test_attention_mismatched_samples_raises():
    attention = IntervalAttentionMap(data=(np.array([0.0, 0.5, 1.0]), ["shared"]), polarity="positive", color="red")

    with pytest.raises(ValueError, match="same number of samples"):
        attention.prepare(["I"], 4, ["I"])


def test_align_attention_dataframe_reorders_columns_to_match_ecg_leads():
    aligned = attention_utils._align_attention_dataframe(
        pd.DataFrame({"II": [0.0, 1.0], "I": [2.0, 3.0]}),
        ["I", "II"],
    )

    expected = pd.DataFrame({"I": [2.0, 3.0], "II": [0.0, 1.0]})
    pd.testing.assert_frame_equal(aligned, expected)


def test_align_attention_dataframe_rejects_missing_leads():
    with pytest.raises(ValueError, match="missing ECG leads"):
        attention_utils._align_attention_dataframe(
            pd.DataFrame({"I": [0.0, 1.0], "III": [2.0, 3.0]}),
            ["I", "II", "III"],
        )


def test_align_attention_dataframe_rejects_duplicate_lead_names():
    with pytest.raises(ValueError, match="Duplicate lead names are not allowed"):
        attention_utils._align_attention_dataframe(
            pd.DataFrame(np.zeros((2, 2)), columns=["I", "I"]),
            ["I", "II"],
        )


@pytest.mark.parametrize("polarity", ["positive", "signed"])
def test_validate_attention_polarity_accepts_supported_values(polarity):
    assert attention_utils._validate_attention_polarity(polarity) == polarity


def test_validate_attention_polarity_rejects_unknown_value():
    with pytest.raises(ValueError, match="must be either 'positive' or 'signed'"):
        attention_utils._validate_attention_polarity("invalid")


def test_validate_attention_color_uses_defaults_for_each_polarity():
    assert attention_utils._validate_attention_color(None, "positive") == attention_utils.DEFAULT_POSITIVE_COLOR
    assert attention_utils._validate_attention_color(None, "signed") == attention_utils.DEFAULT_SIGNED_COLORS


@pytest.mark.parametrize(
    ("color", "polarity", "match"),
    [
        (("", "red"), "positive", "non-empty matplotlib color string"),
        (("blue", ""), "signed", "tuple of two non-empty matplotlib color strings"),
        ("red", "signed", "tuple of two non-empty matplotlib color strings"),
    ],
)
def test_validate_attention_color_rejects_invalid_color_shapes(color, polarity, match):
    with pytest.raises(ValueError, match=match):
        attention_utils._validate_attention_color(color, polarity)


def test_clip_attention_value_limits_values_to_range():
    assert attention_utils._clip_attention_value(-2.0, (-1.0, 1.0)) == -1.0
    assert attention_utils._clip_attention_value(0.5, (-1.0, 1.0)) == 0.5
    assert attention_utils._clip_attention_value(2.0, (-1.0, 1.0)) == 1.0


def test_attention_strength_positive_mode_uses_upper_bound():
    strengths = attention_utils._attention_strength(np.array([-1.0, 0.0, 0.5, 2.0]), (0.0, 1.0), "positive")

    np.testing.assert_allclose(strengths, np.array([0.0, 0.0, 0.5, 1.0]))


def test_attention_strength_signed_mode_tracks_negative_and_positive_magnitudes():
    strengths = attention_utils._attention_strength(np.array([-2.0, -0.5, 0.0, 0.75, 2.0]), (-1.0, 1.0), "signed")

    np.testing.assert_allclose(strengths, np.array([1.0, 0.5, 0.0, 0.75, 1.0]))


def test_attention_strength_positive_mode_returns_zeros_for_non_positive_range():
    strengths = attention_utils._attention_strength(np.array([0.0, 1.0]), (0.0, 0.0), "positive")

    np.testing.assert_allclose(strengths, np.array([0.0, 0.0]))


def test_rgb_for_value_selects_expected_color_branch():
    np.testing.assert_allclose(attention_utils._rgb_for_value(0.25, "positive", "red"), mcolors.to_rgb("red"))
    np.testing.assert_allclose(
        attention_utils._rgb_for_value(-0.25, "signed", ("blue", "red")),
        mcolors.to_rgb("blue"),
    )
    np.testing.assert_allclose(
        attention_utils._rgb_for_value(0.25, "signed", ("blue", "red")),
        mcolors.to_rgb("red"),
    )


def test_interval_color_for_value_clips_and_hides_zero_values():
    assert attention_utils._interval_color_for_value(0.0, (0.0, 1.0), "positive", "red", 0.4) == (0.0, 0.0, 0.0, 0.0)

    expected = (*mcolors.to_rgb("red"), 0.4)
    assert attention_utils._interval_color_for_value(2.0, (0.0, 1.0), "positive", "red", 0.4) == expected


def test_line_overlay_color_for_value_scales_alpha_from_attention_strength():
    assert attention_utils._line_overlay_color_for_value(0.0, (-1.0, 1.0), "signed", ("blue", "red")) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    expected = (*mcolors.to_rgb("red"), 0.75)
    assert attention_utils._line_overlay_color_for_value(0.75, (-1.0, 1.0), "signed", ("blue", "red")) == expected


def test_background_color_for_value_uses_background_alpha_scale():
    expected = (*mcolors.to_rgb("blue"), 0.375)
    assert attention_utils._background_color_for_value(-0.5, (-1.0, 1.0), "signed", ("blue", "red")) == expected
    assert attention_utils._background_color_for_value(0.0, (-1.0, 1.0), "signed", ("blue", "red")) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_attention_row_segmentation_uses_only_relevant_lead_slice():
    attention = IntervalAttentionMap(
        data=pd.DataFrame({"I": [1.0, 2.0, 3.0, 4.0], "II": [10.0, 20.0, 30.0, 40.0]}),
        polarity="positive",
        color="red",
    )

    attention.prepare(["I", "II"], 4, [["I", "II"]])

    np.testing.assert_allclose(attention.row_attentions[0], np.array([0.025, 0.05, 0.25, 0.5]))


@pytest.mark.parametrize("attribute_name", ["dataframe", "row_attentions", "range"])
def test_attention_map_properties_require_prepare(attribute_name):
    attention = LineColorAttentionMap(data=pd.DataFrame({"I": [-1.0, 0.0, 1.0]}), polarity="signed", color=("blue", "red"))

    with pytest.raises(RuntimeError, match="has not been prepared yet"):
        getattr(attention, attribute_name)


def test_attention_map_colormap_rgba_matches_value_mapping():
    attention = LineColorAttentionMap(data=pd.DataFrame({"I": [-1.0, 0.0, 1.0]}), polarity="signed", color=("blue", "red"))

    attention.prepare(["I"], 3, ["I"])

    expected = np.array(
        [
            [[0.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 1.0]],
        ]
    )
    np.testing.assert_allclose(attention.colormap_rgba(n_steps=3), expected)


def test_interval_attention_build_artists_returns_empty_for_short_signal():
    attention = IntervalAttentionMap(data=pd.DataFrame({"I": [0.5, 1.0]}), polarity="positive", color="red")
    attention.prepare(["I"], 2, ["I"])

    fig, ax = plt.subplots()
    try:
        artists = attention.build_artists(
            ax,
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.5]),
            0.0,
            1.0,
            1.0,
            1.0,
        )

        assert artists == []
    finally:
        plt.close(fig)


def test_background_attention_build_artists_skips_zero_alpha_segments():
    attention = BackgroundAttentionMap(data=pd.DataFrame({"I": [-1.0, 0.0, 1.0]}), polarity="signed", color=("blue", "red"))
    attention.prepare(["I"], 3, ["I"])

    fig, ax = plt.subplots()
    try:
        artists = attention.build_artists(
            ax,
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            0.0,
            1.0,
            1.0,
            1.0,
        )

        assert artists == []
    finally:
        plt.close(fig)


def test_line_color_attention_build_artists_skips_non_finite_segments():
    attention = LineColorAttentionMap(data=pd.DataFrame({"I": [-1.0, 0.0, 1.0]}), polarity="signed", color=("blue", "red"))
    attention.prepare(["I"], 3, ["I"])

    fig, ax = plt.subplots()
    try:
        artists = attention.build_artists(
            ax,
            np.array([0.0, 1.0]),
            np.array([np.nan, np.nan]),
            np.array([-1.0, 1.0]),
            0.0,
            1.0,
            1.0,
            1.0,
        )

        assert artists == []
    finally:
        plt.close(fig)


def test_interval_attention_rendered_band_matches_scaled_positive_attention():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    attention_df = pd.DataFrame({"I": [0.0, 0.25, 0.5, 0.75, 1.0]})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=IntervalAttentionMap(
            data=attention_df,
            polarity="positive",
            color="red",
            max_attention_mV=0.5,
            alpha=0.4,
        ),
    )
    try:
        ax = _first_axes(fig)
        band_collection = next(collection for collection in ax.collections if isinstance(collection, PolyCollection))
        signal_line = ax.lines[0]
        x = signal_line.get_xdata()
        expected_half_band = attention_df["I"].to_numpy() * 0.5 * plotter.voltage / MM_PER_INCH

        for x_value, expected in zip(x, expected_half_band):
            y_values = np.concatenate(
                [path.vertices[np.isclose(path.vertices[:, 0], x_value), 1] for path in band_collection.get_paths()]
            )
            y_span = y_values.max() - y_values.min()
            np.testing.assert_allclose(y_span / 2.0, expected, atol=1e-12)
    finally:
        plt.close(fig)


def test_interval_attention_rendered_sign_colors_match_expected_colors():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    attention_df = pd.DataFrame({"I": [-1.0, -0.5, 0.0, 0.5, 1.0]})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=IntervalAttentionMap(
            data=attention_df,
            polarity="signed",
            color=("blue", "red"),
            max_attention_mV=0.5,
            alpha=0.4,
        ),
    )
    try:
        ax = _first_axes(fig)
        interval = next(collection for collection in ax.collections if isinstance(collection, PolyCollection))
        colors = interval.get_facecolors()
        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.4],
                [0.0, 0.0, 1.0, 0.4],
                [1.0, 0.0, 0.0, 0.4],
                [1.0, 0.0, 0.0, 0.4],
            ]
        )
        np.testing.assert_allclose(colors, expected, atol=1e-12)
    finally:
        plt.close(fig)


def test_line_color_attention_positive_mode_uses_single_color_and_scaled_alpha():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    attention_df = pd.DataFrame({"I": [0.0, 0.25, 0.5, 0.75, 1.0]})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=LineColorAttentionMap(data=attention_df, polarity="positive", color="red"),
    )
    try:
        ax = _first_axes(fig)
        line_collection = next(collection for collection in ax.collections if isinstance(collection, LineCollection))
        colors = line_collection.get_colors()
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.125],
                [1.0, 0.0, 0.0, 0.375],
                [1.0, 0.0, 0.0, 0.625],
                [1.0, 0.0, 0.0, 0.875],
            ]
        )
        np.testing.assert_allclose(colors, expected, atol=1e-12)
    finally:
        plt.close(fig)


def test_line_color_attention_signed_mode_matches_expected_colors():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    attention_df = pd.DataFrame({"I": [-1.0, -0.5, 0.0, 0.5, 1.0]})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=LineColorAttentionMap(data=attention_df, polarity="signed", color=("blue", "red")),
    )
    try:
        ax = _first_axes(fig)
        signal_lines = [line for line in ax.lines if len(line.get_xdata()) == len(ecg_df)]
        assert len(signal_lines) == 1
        assert signal_lines[0].get_color() == "black"
        line_collection = next(collection for collection in ax.collections if isinstance(collection, LineCollection))
        colors = line_collection.get_colors()
        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.75],
                [0.0, 0.0, 1.0, 0.25],
                [1.0, 0.0, 0.0, 0.25],
                [1.0, 0.0, 0.0, 0.75],
            ]
        )
        np.testing.assert_allclose(colors, expected, atol=1e-12)
    finally:
        plt.close(fig)


def test_color_attention_single_mask_is_broadcast_across_rendered_leads():
    ecg_df = pd.DataFrame({"I": np.zeros(5), "II": np.zeros(5)})
    shared_attention = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I", "II"],
        sampling_frequency=1.0,
        show=False,
        attention_map=LineColorAttentionMap(
            data=(shared_attention, ["shared"]),
            polarity="signed",
            color=("blue", "red"),
        ),
    )
    try:
        ax = _first_axes(fig)
        line_collections = [collection for collection in ax.collections if isinstance(collection, LineCollection)]
        assert len(line_collections) == 2

        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.75],
                [0.0, 0.0, 1.0, 0.25],
                [1.0, 0.0, 0.0, 0.25],
                [1.0, 0.0, 0.0, 0.75],
            ]
        )
        for collection in line_collections:
            np.testing.assert_allclose(collection.get_colors(), expected, atol=1e-12)
    finally:
        plt.close(fig)


def test_background_attention_rendered_alpha_matches_expected_values():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    attention_df = pd.DataFrame({"I": [-1.0, -0.5, 0.0, 0.5, 1.0]})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=BackgroundAttentionMap(data=attention_df, polarity="signed", color=("blue", "red")),
    )
    try:
        ax = _first_axes(fig)
        background = next(collection for collection in ax.collections if isinstance(collection, PolyCollection))
        colors = background.get_facecolors()
        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.5625],
                [0.0, 0.0, 1.0, 0.1875],
                [1.0, 0.0, 0.0, 0.1875],
                [1.0, 0.0, 0.0, 0.5625],
            ]
        )
        np.testing.assert_allclose(colors, expected, atol=1e-12)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("attention_map_factory", "attention_df", "expected_labels"),
    [
        (
            lambda data: LineColorAttentionMap(data=data, polarity="signed", color=("blue", "red")),
            pd.DataFrame({"I": [-1.0, -0.5, 0.0, 0.5, 1.0]}),
            {"-1", "0", "1"},
        ),
        (
            lambda data: BackgroundAttentionMap(data=data, polarity="signed", color=("blue", "red")),
            pd.DataFrame({"I": [-1.0, -0.5, 0.0, 0.5, 1.0]}),
            {"-1", "0", "1"},
        ),
    ],
)
def test_non_interval_attention_adds_color_scale_and_expands_right_margin(
    attention_map_factory, attention_df, expected_labels
):
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=attention_map_factory(attention_df),
    )
    try:
        adjusted_row_distance = _adjust_row_distance(plotter.row_distance, plotter.voltage)
        expected_width, _ = _compute_figure_size(
            1,
            len(ecg_df),
            1.0,
            plotter.speed,
            plotter.voltage,
            adjusted_row_distance,
            right_margin_mm=RIGHT_MARGIN_MM * 2.0,
        )

        width, _ = fig.get_size_inches()
        assert abs(width - expected_width) < 1e-6

        ax = _first_axes(fig)
        assert len(ax.images) == 1
        color_scale_labels = {text.get_text() for text in ax.texts}
        assert expected_labels.issubset(color_scale_labels)
    finally:
        plt.close(fig)


def test_non_interval_attention_can_hide_color_scale_while_keeping_extra_right_margin():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    attention_map = LineColorAttentionMap(
        data=pd.DataFrame({"I": [-1.0, -0.5, 0.0, 0.5, 1.0]}),
        polarity="signed",
        color=("blue", "red"),
        show_colormap=False,
    )

    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=attention_map,
    )
    try:
        adjusted_row_distance = _adjust_row_distance(plotter.row_distance, plotter.voltage)
        expected_width, _ = _compute_figure_size(
            1,
            len(ecg_df),
            1.0,
            plotter.speed,
            plotter.voltage,
            adjusted_row_distance,
            right_margin_mm=RIGHT_MARGIN_MM * 2.0,
        )

        width, _ = fig.get_size_inches()
        assert abs(width - expected_width) < 1e-6

        ax = _first_axes(fig)
        assert len(ax.images) == 0
        color_scale_labels = {text.get_text() for text in ax.texts}
        assert "-1" not in color_scale_labels
        assert "0" not in color_scale_labels
        assert "1" not in color_scale_labels
    finally:
        plt.close(fig)


def test_interval_attention_does_not_add_color_scale_but_keeps_extra_right_margin():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    attention_map = IntervalAttentionMap(
        data=pd.DataFrame({"I": [0.0, 0.25, 0.5, 0.75, 1.0]}),
        polarity="positive",
        color="red",
        max_attention_mV=0.5,
        alpha=0.4,
    )

    assert attention_map.show_colormap is False

    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=attention_map,
    )
    try:
        adjusted_row_distance = _adjust_row_distance(plotter.row_distance, plotter.voltage)
        expected_width, _ = _compute_figure_size(
            1,
            len(ecg_df),
            1.0,
            plotter.speed,
            plotter.voltage,
            adjusted_row_distance,
            right_margin_mm=RIGHT_MARGIN_MM * 2.0,
        )

        width, _ = fig.get_size_inches()
        assert abs(width - expected_width) < 1e-6

        ax = _first_axes(fig)
        assert len(ax.images) == 0
        color_scale_labels = {text.get_text() for text in ax.texts}
        assert "0" not in color_scale_labels
        assert "1" not in color_scale_labels
    finally:
        plt.close(fig)


def test_interval_attention_can_opt_in_to_color_scale():
    ecg_df = pd.DataFrame({"I": np.zeros(5)})
    plotter = ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        print_information=False,
        disconnect_segments=False,
    )
    attention_map = IntervalAttentionMap(
        data=pd.DataFrame({"I": [0.0, 0.25, 0.5, 0.75, 1.0]}),
        polarity="positive",
        color="red",
        max_attention_mV=0.5,
        alpha=0.4,
        show_colormap=True,
    )

    fig = plotter.plot(
        ecg_df,
        configuration=["I"],
        sampling_frequency=1.0,
        show=False,
        attention_map=attention_map,
    )
    try:
        adjusted_row_distance = _adjust_row_distance(plotter.row_distance, plotter.voltage)
        expected_width, _ = _compute_figure_size(
            1,
            len(ecg_df),
            1.0,
            plotter.speed,
            plotter.voltage,
            adjusted_row_distance,
            right_margin_mm=RIGHT_MARGIN_MM * 2.0,
        )

        width, _ = fig.get_size_inches()
        assert abs(width - expected_width) < 1e-6

        ax = _first_axes(fig)
        assert len(ax.images) == 1
        color_scale_labels = {text.get_text() for text in ax.texts}
        assert "0" in color_scale_labels
        assert "1" in color_scale_labels
    finally:
        plt.close(fig)
