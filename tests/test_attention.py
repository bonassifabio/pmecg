"""Tests for attention map conversion, scaling, and rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PolyCollection

from pmecg import (
    BackgroundAttentionMap,
    ECGPlotter,
    IntervalAttentionMap,
    LineColorAttentionMap,
    attention_map_from_indices_annotations,
    attention_map_from_time_annotations,
)
from pmecg.utils.attention import _attention_to_dataframe
from pmecg.utils.plot import MM_PER_INCH, RIGHT_MARGIN_MM, _adjust_row_distance, _compute_figure_size


def _first_axes(fig):
    return fig.axes[0]


def test_attention_conversion_preserves_dataframe():
    attention_df = pd.DataFrame({"I": [0.0, 1.0, 2.0], "II": [2.0, 1.0, 0.0]})

    converted = _attention_to_dataframe(attention_df)

    pd.testing.assert_frame_equal(converted, attention_df)


def test_attention_conversion_from_numpy_tuple():
    values = np.column_stack((np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])))

    converted = _attention_to_dataframe((values, ["I", "II"]))

    expected = pd.DataFrame(values, columns=["I", "II"])
    pd.testing.assert_frame_equal(converted, expected)


def test_attention_conversion_from_list_tuple():
    converted = _attention_to_dataframe(([np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])], ["I", "II"]))

    expected = pd.DataFrame({"I": [0.0, 1.0, 2.0], "II": [2.0, 1.0, 0.0]})
    pd.testing.assert_frame_equal(converted, expected)


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


def test_attention_mismatched_samples_raises():
    attention = IntervalAttentionMap(data=(np.array([0.0, 0.5, 1.0]), ["shared"]), polarity="positive", color="red")

    with pytest.raises(ValueError, match="same number of samples"):
        attention.prepare(["I"], 4, ["I"])


def test_attention_row_segmentation_uses_only_relevant_lead_slice():
    attention = IntervalAttentionMap(
        data=pd.DataFrame({"I": [1.0, 2.0, 3.0, 4.0], "II": [10.0, 20.0, 30.0, 40.0]}),
        polarity="positive",
        color="red",
    )

    attention.prepare(["I", "II"], 4, [["I", "II"]])

    np.testing.assert_allclose(attention.row_attentions[0], np.array([0.025, 0.05, 0.25, 0.5]))


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


def test_interval_attention_does_not_add_color_scale_but_keeps_extra_right_margin():
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
        attention_map=IntervalAttentionMap(
            data=pd.DataFrame({"I": [0.0, 0.25, 0.5, 0.75, 1.0]}),
            polarity="positive",
            color="red",
            max_attention_mV=0.5,
            alpha=0.4,
        ),
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
