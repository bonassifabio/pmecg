from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection, PolyCollection

from .data import ConfigurationDataType, _apply_configuration, _numpy_to_dataframe, _validate_input_lead_names

AttentionArrayType = np.ndarray | list[np.ndarray]
AttentionDataType = tuple[AttentionArrayType, list[str]] | pd.DataFrame
AttentionPolarity = Literal["positive", "signed"]
AttentionColorType = str | tuple[str, str]
BACKGROUND_MAX_ALPHA = 0.75
DEFAULT_POSITIVE_COLOR = "red"
DEFAULT_SIGNED_COLORS = ("blue", "red")


class AbstractAttentionMap(ABC):
    """Abstract base class for attention-aware ECG overlays.

    Subclasses own the raw attention input and the rendering parameters for one
    visual style. The shared :meth:`prepare` step converts the input into a
    lead-aligned :class:`pandas.DataFrame`, validates the requested polarity,
    applies a single global scaling factor when values exceed magnitude 1, and
    segments the attention values according to the ECG layout.
    """

    def __init__(self, data: AttentionDataType, *, polarity: AttentionPolarity) -> None:
        self.data = data
        self.polarity = _validate_attention_polarity(polarity)
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
    ) -> None:
        super().__init__(data, polarity=polarity)
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
    ) -> None:
        super().__init__(data, polarity=polarity)
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
    ) -> None:
        super().__init__(data, polarity=polarity)
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
