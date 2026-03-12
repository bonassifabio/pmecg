from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Literal, TypedDict

import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PolyCollection

from .data import (
    ConfigurationDataType,
    ECGDataType,
    _apply_configuration,
    _extract_input_leads,
    _numpy_to_dataframe,
    _validate_input_lead_names,
)

AttentionArrayType = np.ndarray | list[np.ndarray]
AttentionDataType = tuple[AttentionArrayType, list[str]] | pd.DataFrame
AttentionPolarity = Literal["positive", "signed"]
AttentionColorType = str | tuple[str, str]
BACKGROUND_MAX_ALPHA = 0.75
DEFAULT_POSITIVE_COLOR = "red"
DEFAULT_SIGNED_COLORS = ("blue", "red")


class _TimeAnnotation(TypedDict):
    time_range: list[float] | tuple[float, float]
    attention_value: float


class _IndexAnnotation(TypedDict):
    index_range: list[int] | tuple[int, int]
    attention_value: float


class AbstractAttentionMap(ABC):
    """Abstract base class for attention-aware ECG overlays.

    Subclasses own the raw attention input and the rendering parameters for one
    visual style. The shared :meth:`prepare` step converts the input into a
    lead-aligned :class:`pandas.DataFrame`, validates the requested polarity,
    applies a single global scaling factor when values exceed magnitude 1, and
    segments the attention values according to the ECG layout.
    """

    def __init__(
        self,
        data: AttentionDataType,
        *,
        polarity: AttentionPolarity,
        show_colormap: bool = True,
    ) -> None:
        self.data = data
        self.polarity = _validate_attention_polarity(polarity)
        if not isinstance(show_colormap, bool):
            raise ValueError("show_colormap must be a boolean")
        self.show_colormap = show_colormap
        self._dataframe: pd.DataFrame | None = None
        self._row_attentions: tuple[np.ndarray, ...] = ()
        self._resolved_range: tuple[float, float] | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Prepared attention values aligned to the ECG leads."""
        if self._dataframe is None:
            raise RuntimeError("Attention map has not been prepared yet")
        return self._dataframe

    @property
    def row_attentions(self) -> tuple[np.ndarray, ...]:
        """Prepared attention values segmented to match each ECG row."""
        if self._resolved_range is None:
            raise RuntimeError("Attention map has not been prepared yet")
        return self._row_attentions

    @property
    def range(self) -> tuple[float, float]:
        """Resolved global attention range used for rendering."""
        if self._resolved_range is None:
            raise RuntimeError("Attention map has not been prepared yet")
        return self._resolved_range

    @property
    def shows_color_scale(self) -> bool:
        """Whether plotting this attention map requires the right-side color scale."""
        return True

    def prepare(
        self,
        ecg_leads: list[str],
        n_samples: int,
        configuration: ConfigurationDataType | None,
    ) -> None:
        """Convert, validate, scale, and segment the attention input for plotting."""
        df = _attention_to_dataframe(self.data)
        aligned_df = _align_attention_dataframe(df, ecg_leads)
        if aligned_df.shape[0] != n_samples:
            raise ValueError("Attention data must have the same number of samples as ecg_data")

        scaled_df, resolved_range = _scale_attention_dataframe(aligned_df, self.polarity)
        self._dataframe = scaled_df
        self._resolved_range = resolved_range
        self._row_attentions = tuple(
            signal for signal, _ in _apply_configuration(scaled_df, configuration, disconnect_segments=False)
        )

    def colormap_rgba(self, n_steps: int = 256) -> np.ndarray:
        """Build an RGBA image representing the prepared value-to-color mapping."""
        values = np.linspace(self.range[0], self.range[1], n_steps)
        colors = np.array([self._rgba_for_value(float(value)) for value in values], dtype=float)
        return colors.reshape(n_steps, 1, 4)

    @abstractmethod
    def _rgba_for_value(self, value: float) -> tuple[float, ...]:
        """Map one attention value to RGBA using the subclass visual semantics."""

    @abstractmethod
    def build_artists(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        attention_values: np.ndarray,
        y_offset: float,
        row_half_height_inches: float,
        mv_to_inches: float,
        line_width: float,
    ) -> list[Artist]:
        """Return Matplotlib artists for one ECG row using a common rendering signature."""


class IntervalAttentionMap(AbstractAttentionMap):
    """Render attention as a colored band around the ECG trace."""

    def __init__(
        self,
        data: AttentionDataType,
        *,
        polarity: AttentionPolarity,
        color: AttentionColorType | None = None,
        max_attention_mV: float = 0.25,
        alpha: float = 0.25,
        show_colormap: bool = False,
    ) -> None:
        super().__init__(data, polarity=polarity, show_colormap=show_colormap)
        if not isinstance(max_attention_mV, (int, float)) or float(max_attention_mV) < 0:
            raise ValueError("max_attention_mV must be a non-negative number")
        if not isinstance(alpha, (int, float)) or not 0 <= float(alpha) <= 1:
            raise ValueError("alpha must be between 0 and 1")

        self.max_attention_mV = float(max_attention_mV)
        self.color = _validate_attention_color(color, self.polarity)
        self.alpha = float(alpha)

    def _rgba_for_value(self, value: float) -> tuple[float, ...]:
        return _interval_color_for_value(value, self.range, self.polarity, self.color, self.alpha)

    def build_artists(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        attention_values: np.ndarray,
        y_offset: float,
        row_half_height_inches: float,
        mv_to_inches: float,
        line_width: float,
    ) -> list[Artist]:
        """Return a band-shaped PolyCollection that follows the ECG trace."""
        del ax, y_offset, row_half_height_inches, line_width
        if len(x) < 2:
            return []

        strengths = _attention_strength(attention_values, self.range, self.polarity)
        half_band = strengths * self.max_attention_mV * mv_to_inches

        vertices: list[list[tuple[float, float]]] = []
        facecolors: list[tuple[float, ...]] = []
        segment_attention = (attention_values[:-1] + attention_values[1:]) / 2.0

        for x0, x1, y0, y1, band0, band1, att in zip(
            x[:-1], x[1:], y[:-1], y[1:], half_band[:-1], half_band[1:], segment_attention
        ):
            if not all(np.isfinite(value) for value in (x0, x1, y0, y1, band0, band1, att)):
                continue
            vertices.append([(x0, y0 - band0), (x0, y0 + band0), (x1, y1 + band1), (x1, y1 - band1)])
            facecolors.append(self._rgba_for_value(float(att)))

        if not vertices:
            return []

        return [PolyCollection(vertices, facecolors=facecolors, edgecolors="none", linewidths=0, zorder=2)]


class BackgroundAttentionMap(AbstractAttentionMap):
    """Render attention as semi-transparent background blocks behind each ECG row."""

    def __init__(
        self,
        data: AttentionDataType,
        *,
        polarity: AttentionPolarity,
        color: AttentionColorType | None = None,
        show_colormap: bool = True,
    ) -> None:
        super().__init__(data, polarity=polarity, show_colormap=show_colormap)
        self.color = _validate_attention_color(color, self.polarity)

    def _rgba_for_value(self, value: float) -> tuple[float, ...]:
        return _background_color_for_value(value, self.range, self.polarity, self.color)

    def build_artists(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        attention_values: np.ndarray,
        y_offset: float,
        row_half_height_inches: float,
        mv_to_inches: float,
        line_width: float,
    ) -> list[Artist]:
        """Return background rectangles spanning the full height of the ECG row."""
        del ax, y, mv_to_inches, line_width
        if len(x) < 2:
            return []

        y_min = y_offset - row_half_height_inches
        y_max = y_offset + row_half_height_inches
        vertices: list[list[tuple[float, float]]] = []
        facecolors: list[tuple[float, ...]] = []

        for x0, x1, att0, att1 in zip(x[:-1], x[1:], attention_values[:-1], attention_values[1:]):
            if not (np.isfinite(att0) and np.isfinite(att1)):
                continue
            rgba = self._rgba_for_value(float((att0 + att1) / 2.0))
            if rgba[3] <= 0:
                continue
            vertices.append([(x0, y_min), (x1, y_min), (x1, y_max), (x0, y_max)])
            facecolors.append(rgba)

        if not vertices:
            return []

        return [PolyCollection(vertices, facecolors=facecolors, edgecolors="none", zorder=1)]


class LineColorAttentionMap(AbstractAttentionMap):
    """Render attention as a gradient-colored line drawn on top of the ECG trace."""

    def __init__(
        self,
        data: AttentionDataType,
        *,
        polarity: AttentionPolarity,
        color: AttentionColorType | None = None,
        show_colormap: bool = True,
    ) -> None:
        super().__init__(data, polarity=polarity, show_colormap=show_colormap)
        self.color = _validate_attention_color(color, self.polarity)

    def _rgba_for_value(self, value: float) -> tuple[float, ...]:
        return _line_overlay_color_for_value(value, self.range, self.polarity, self.color)

    def build_artists(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        attention_values: np.ndarray,
        y_offset: float,
        row_half_height_inches: float,
        mv_to_inches: float,
        line_width: float,
    ) -> list[Artist]:
        """Return a gradient LineCollection to overlay on the ECG trace."""
        del ax, y_offset, row_half_height_inches, mv_to_inches
        if len(x) < 2:
            return []

        points = np.column_stack((x, y))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        segment_attention = (attention_values[:-1] + attention_values[1:]) / 2.0
        finite_mask = np.isfinite(segments).all(axis=(1, 2)) & np.isfinite(segment_attention)
        if not np.any(finite_mask):
            return []

        segment_colors = [self._rgba_for_value(float(value)) for value in segment_attention[finite_mask]]
        return [
            LineCollection(
                segments[finite_mask],
                colors=segment_colors,
                linewidths=line_width,
                zorder=4,
                capstyle="round",
                joinstyle="round",
            )
        ]


def attention_map_from_indices_annotations(
    ecg_data: ECGDataType,
    **annotations_by_lead: list[_IndexAnnotation],
) -> pd.DataFrame:
    """Build an attention DataFrame from per-lead sample-index annotations.

    Parameters
    ----------
    ecg_data : ECGDataType
        ECG data used to determine the output lead names and number of samples.
        The helper accepts the same DataFrame and tuple formats supported by
        :class:`pmecg.ECGPlotter`.
    **annotations_by_lead : list[dict]
        Keyword arguments keyed by lead name. Each value must be a list of
        dictionaries with the keys ``index_range`` and ``attention_value``.
        ``index_range`` is interpreted as a half-open interval ``[start, end)``
        over ECG sample indices.

    Returns
    -------
    pd.DataFrame
        Attention values aligned to the ECG samples, with one column per lead
        and one row per ECG sample. Leads without annotations are filled with
        zeros.

    Raises
    ------
    ValueError
        If a lead name is missing from ``ecg_data``, if an annotation payload is
        malformed, or if an index range is invalid or falls outside the ECG
        recording length.
    """
    lead_names = _extract_input_leads(ecg_data)
    _validate_input_lead_names(lead_names)
    n_samples = _extract_n_samples(ecg_data)
    attention_df = pd.DataFrame(0.0, index=np.arange(n_samples), columns=lead_names)

    for lead_name, lead_annotations in annotations_by_lead.items():
        if lead_name not in attention_df.columns:
            raise ValueError(f"Lead name {lead_name!r} is not present in ecg_data")
        if not isinstance(lead_annotations, list):
            raise ValueError(f"Annotations for lead {lead_name!r} must be provided as a list")

        for annotation in lead_annotations:
            start_idx, end_idx, attention_value = _resolve_indices_annotation(
                annotation,
                lead_name=lead_name,
                n_samples=n_samples,
            )
            attention_df.iloc[start_idx:end_idx, attention_df.columns.get_loc(lead_name)] = attention_value

    return attention_df


def attention_map_from_time_annotations(
    ecg_data: ECGDataType,
    fs: float,
    **annotations_by_lead: list[_TimeAnnotation],
) -> pd.DataFrame:
    """Build an attention DataFrame from per-lead time annotations.

    Parameters
    ----------
    ecg_data : ECGDataType
        ECG data used to determine the output lead names and number of samples.
    fs : float
        Sampling frequency in Hz.
    **annotations_by_lead : list[dict]
        Keyword arguments keyed by lead name. Each value must be a list of
        dictionaries with the keys ``time_range`` and ``attention_value``.
        ``time_range`` is interpreted as a half-open interval ``[start, end)``
        expressed in seconds.

    Returns
    -------
    pd.DataFrame
        Attention values aligned to the ECG samples, with one column per lead
        and one row per ECG sample. Leads without annotations are filled with
        zeros.

    Raises
    ------
    ValueError
        If ``fs`` is invalid, if a lead name is missing from ``ecg_data``, if an
        annotation payload is malformed, or if a time range is invalid or falls
        outside the ECG recording duration.
    """
    if not isinstance(fs, Real) or float(fs) <= 0:
        raise ValueError("fs must be a positive number")

    n_samples = _extract_n_samples(ecg_data)
    sampling_frequency = float(fs)
    recording_duration_seconds = n_samples / sampling_frequency
    indices_annotations_by_lead: dict[str, list[_IndexAnnotation]] = {}

    for lead_name, lead_annotations in annotations_by_lead.items():
        if not isinstance(lead_annotations, list):
            raise ValueError(f"Annotations for lead {lead_name!r} must be provided as a list")

        indices_annotations_by_lead[lead_name] = [
            {
                "index_range": list(
                    _time_range_to_sample_bounds(
                        _extract_time_range(annotation, lead_name=lead_name),
                        fs=sampling_frequency,
                        recording_duration_seconds=recording_duration_seconds,
                        n_samples=n_samples,
                    )
                ),
                "attention_value": _extract_attention_value(annotation, lead_name=lead_name),
            }
            for annotation in lead_annotations
        ]

    return attention_map_from_indices_annotations(ecg_data, **indices_annotations_by_lead)


def _attention_to_dataframe(attention_data: AttentionDataType) -> pd.DataFrame:
    """Convert user attention data into a DataFrame without altering semantics."""
    if isinstance(attention_data, pd.DataFrame):
        df = attention_data.copy()
        df.columns = [str(column) for column in df.columns]
        return df

    values, lead_names = attention_data
    normalized_lead_names = [str(name) for name in lead_names]
    _validate_input_lead_names(normalized_lead_names)

    if isinstance(values, np.ndarray) and values.ndim == 1:
        if len(normalized_lead_names) != 1:
            raise ValueError("A 1D attention array requires exactly one lead name")
        return pd.DataFrame(values.reshape(-1, 1), columns=normalized_lead_names)

    return _numpy_to_dataframe(values, normalized_lead_names)


def _extract_n_samples(ecg_data: ECGDataType) -> int:
    """Return the number of samples stored in ``ecg_data``."""
    if isinstance(ecg_data, pd.DataFrame):
        return int(ecg_data.shape[0])

    values, lead_names = ecg_data
    return int(_numpy_to_dataframe(values, [str(name) for name in lead_names]).shape[0])


def _resolve_indices_annotation(
    annotation: _IndexAnnotation,
    *,
    lead_name: str,
    n_samples: int,
) -> tuple[int, int, float]:
    """Validate one index annotation and convert it to sample bounds."""
    if not isinstance(annotation, dict):
        raise ValueError(f"Each annotation for lead {lead_name!r} must be a dictionary")

    start_idx, end_idx = _index_range_to_sample_bounds(annotation, n_samples=n_samples)
    return start_idx, end_idx, _extract_attention_value(annotation, lead_name=lead_name)


def _extract_attention_value(annotation: dict[str, object], *, lead_name: str) -> float:
    """Return a validated annotation attention value."""
    if not isinstance(annotation, dict):
        raise ValueError(f"Each annotation for lead {lead_name!r} must be a dictionary")

    if "attention_value" not in annotation:
        raise ValueError(f"Each annotation for lead {lead_name!r} must contain 'attention_value'")

    attention_value = annotation["attention_value"]
    if not isinstance(attention_value, Real) or not np.isfinite(float(attention_value)):
        raise ValueError(f"attention_value for lead {lead_name!r} must be a finite number")
    return float(attention_value)


def _extract_time_range(annotation: dict[str, object], *, lead_name: str) -> object:
    """Return a validated raw time range payload."""
    if not isinstance(annotation, dict):
        raise ValueError(f"Each annotation for lead {lead_name!r} must be a dictionary")

    expected_keys = {"time_range", "attention_value"}
    missing_keys = expected_keys.difference(annotation)
    if missing_keys:
        raise ValueError(
            f"Each time annotation for lead {lead_name!r} must contain "
            + ", ".join(sorted(repr(key) for key in missing_keys))
        )

    unexpected_keys = set(annotation).difference(expected_keys)
    if unexpected_keys:
        raise ValueError(
            f"Unsupported time annotation keys for lead {lead_name!r}: "
            + ", ".join(sorted(repr(key) for key in unexpected_keys))
        )

    return annotation["time_range"]


def _index_range_to_sample_bounds(annotation: _IndexAnnotation, *, n_samples: int) -> tuple[int, int]:
    """Convert an index annotation into validated sample-index bounds."""
    expected_keys = {"index_range", "attention_value"}
    missing_keys = expected_keys.difference(annotation)
    if missing_keys:
        raise ValueError(
            "Each index annotation must contain " + ", ".join(sorted(repr(key) for key in missing_keys))
        )

    unexpected_keys = set(annotation).difference(expected_keys)
    if unexpected_keys:
        raise ValueError(
            "Unsupported index annotation keys: " + ", ".join(sorted(repr(key) for key in unexpected_keys))
        )

    index_range = annotation["index_range"]
    if not isinstance(index_range, (list, tuple)) or len(index_range) != 2:
        raise ValueError("index_range must be a list or tuple containing exactly two integers")

    start_idx, end_idx = index_range
    if not isinstance(start_idx, Integral) or not isinstance(end_idx, Integral):
        raise ValueError("index_range values must be integers")

    start_index = int(start_idx)
    end_index = int(end_idx)
    if start_index < 0 or end_index < 0:
        raise ValueError("index_range values must be non-negative")
    if end_index < start_index:
        raise ValueError("index_range end must be greater than or equal to its start")
    if end_index > n_samples:
        raise ValueError("index_range end exceeds the ECG length")
    if end_index <= start_index:
        raise ValueError("index_range must span at least one ECG sample")
    return start_index, end_index


def _time_range_to_sample_bounds(
    time_range: object,
    *,
    fs: float,
    recording_duration_seconds: float,
    n_samples: int,
) -> tuple[int, int]:
    """Convert a time range in seconds into sample-index bounds."""
    if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
        raise ValueError("time_range must be a list or tuple containing exactly two numbers")

    start_time, end_time = time_range
    if not isinstance(start_time, Real) or not isinstance(end_time, Real):
        raise ValueError("time_range values must be numeric")

    start_seconds = float(start_time)
    end_seconds = float(end_time)
    if not (np.isfinite(start_seconds) and np.isfinite(end_seconds)):
        raise ValueError("time_range values must be finite")
    if start_seconds < 0 or end_seconds < 0:
        raise ValueError("time_range values must be non-negative")
    if end_seconds < start_seconds:
        raise ValueError("time_range end must be greater than or equal to its start")
    if end_seconds > recording_duration_seconds:
        raise ValueError("time_range end exceeds the ECG duration")

    start_idx = int(np.floor(start_seconds * fs))
    end_idx = int(np.ceil(end_seconds * fs))
    start_idx = min(start_idx, n_samples)
    end_idx = min(end_idx, n_samples)
    if end_idx <= start_idx:
        raise ValueError("time_range must span at least one ECG sample")
    return start_idx, end_idx


def _scale_attention_dataframe(df: pd.DataFrame, polarity: AttentionPolarity) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Validate attention polarity and apply a single global scaling factor when needed."""
    lower, upper = _finite_attention_bounds(df)

    if polarity == "positive":
        if lower < 0:
            raise ValueError("Positive attention polarity requires non-negative values")
        if upper <= 0:
            raise ValueError("Positive attention polarity requires at least one value greater than 0")
        scale = max(upper, 1.0)
        scaled_df = df.astype(float) / scale
        return scaled_df, (0.0, upper / scale)

    if not (lower < 0 < upper):
        raise ValueError("Signed attention polarity requires values that span both negative and positive")

    scale = max(abs(lower), abs(upper), 1.0)
    scaled_df = df.astype(float) / scale
    return scaled_df, (lower / scale, upper / scale)


def _finite_attention_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Return the finite global min/max across every attention column."""
    values = df.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Attention data must contain at least one finite value")
    return float(finite_values.min()), float(finite_values.max())


def _align_attention_dataframe(df: pd.DataFrame, ecg_leads: list[str]) -> pd.DataFrame:
    """Align attention columns with ECG leads, broadcasting a single column when needed."""
    _validate_input_lead_names(list(df.columns))

    if df.shape[1] == 1:
        values = np.repeat(df.iloc[:, [0]].to_numpy(), repeats=len(ecg_leads), axis=1)
        return pd.DataFrame(values, columns=ecg_leads, index=df.index)

    missing_leads = [lead for lead in ecg_leads if lead not in df.columns]
    if missing_leads:
        raise ValueError("Attention data is missing ECG leads: " + ", ".join(repr(lead) for lead in missing_leads))
    return df.loc[:, ecg_leads].copy()


def _validate_attention_polarity(polarity: AttentionPolarity) -> AttentionPolarity:
    if polarity not in ("positive", "signed"):
        raise ValueError("polarity must be either 'positive' or 'signed'")
    return polarity


def _validate_attention_color(color: AttentionColorType | None, polarity: AttentionPolarity) -> AttentionColorType:
    if polarity == "positive":
        if color is None:
            return DEFAULT_POSITIVE_COLOR
        if not isinstance(color, str) or len(color) == 0:
            raise ValueError("Positive attention requires color to be a non-empty matplotlib color string")
        return color

    if color is None:
        return DEFAULT_SIGNED_COLORS
    if (
        not isinstance(color, tuple)
        or len(color) != 2
        or not all(isinstance(color_value, str) and len(color_value) > 0 for color_value in color)
    ):
        raise ValueError("Signed attention requires color to be a tuple of two non-empty matplotlib color strings")
    return color


def _clip_attention_value(value: float, attention_range: tuple[float, float]) -> float:
    return float(np.clip(value, attention_range[0], attention_range[1]))


def _attention_strength(
    values: np.ndarray,
    attention_range: tuple[float, float],
    polarity: AttentionPolarity,
) -> np.ndarray:
    """Map attention values to [0, 1] strengths for interval/background intensity."""
    clipped = np.clip(values.astype(float), attention_range[0], attention_range[1])
    lower, upper = attention_range

    if polarity == "positive":
        if upper <= 0:
            return np.zeros_like(clipped, dtype=float)
        return np.clip(clipped / upper, 0.0, 1.0)

    strength = np.zeros_like(clipped, dtype=float)
    if lower < 0:
        negative_mask = clipped < 0
        strength[negative_mask] = np.abs(clipped[negative_mask] / lower)
    if upper > 0:
        positive_mask = clipped > 0
        strength[positive_mask] = clipped[positive_mask] / upper
    return np.clip(strength, 0.0, 1.0)


def _rgb_for_value(value: float, polarity: AttentionPolarity, color: AttentionColorType) -> tuple[float, float, float]:
    if polarity == "positive":
        assert isinstance(color, str)
        rgb = mcolors.to_rgb(color)
        return float(rgb[0]), float(rgb[1]), float(rgb[2])

    assert isinstance(color, tuple)
    rgb = mcolors.to_rgb(color[0] if value < 0 else color[1])
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _interval_color_for_value(
    value: float,
    attention_range: tuple[float, float],
    polarity: AttentionPolarity,
    color: AttentionColorType,
    alpha: float,
) -> tuple[float, ...]:
    """Convert one attention value into an RGBA color for interval bands."""
    clipped = _clip_attention_value(value, attention_range)
    if np.isclose(clipped, 0.0):
        return 0.0, 0.0, 0.0, 0.0

    rgb = _rgb_for_value(clipped, polarity, color)
    return rgb[0], rgb[1], rgb[2], alpha


def _line_overlay_color_for_value(
    value: float,
    attention_range: tuple[float, float],
    polarity: AttentionPolarity,
    color: AttentionColorType,
) -> tuple[float, ...]:
    """Convert one attention value into an RGBA color for the line-color overlay."""
    clipped = _clip_attention_value(value, attention_range)
    alpha = float(_attention_strength(np.array([clipped]), attention_range, polarity)[0])
    if np.isclose(alpha, 0.0):
        return 0.0, 0.0, 0.0, 0.0

    rgb = _rgb_for_value(clipped, polarity, color)
    return rgb[0], rgb[1], rgb[2], alpha


def _background_color_for_value(
    value: float,
    attention_range: tuple[float, float],
    polarity: AttentionPolarity,
    color: AttentionColorType,
) -> tuple[float, ...]:
    """Convert one attention value into a semi-transparent background RGBA color."""
    clipped = _clip_attention_value(value, attention_range)
    alpha = float(BACKGROUND_MAX_ALPHA * _attention_strength(np.array([clipped]), attention_range, polarity)[0])
    if np.isclose(alpha, 0.0):
        return 0.0, 0.0, 0.0, 0.0

    rgb = _rgb_for_value(clipped, polarity, color)
    return rgb[0], rgb[1], rgb[2], alpha
