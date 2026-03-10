"""Helpers for deterministic figure output generation in integration tests."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path

_EPOCH = datetime(1970, 1, 1, 0, 0, 0)
_STABLE_METADATA: dict[str, dict[str, object]] = {
    "png": {"Software": "pmecg", "Creation Time": _EPOCH.isoformat()},
    "pdf": {"Creator": "pmecg", "Producer": "pmecg", "CreationDate": _EPOCH},
}


def figure_bytes(fig, fmt: str) -> bytes:
    """Render *fig* to bytes with fixed metadata so the result is deterministic."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight", metadata=_STABLE_METADATA[fmt])
    return buf.getvalue()


def save_if_changed(fig, path: str | Path, fmt: str) -> bool:
    """Write a figure only when the newly rendered bytes differ from the file already on disk."""
    output_path = Path(path)
    new_bytes = figure_bytes(fig, fmt)
    new_hash = hashlib.md5(new_bytes).hexdigest()

    if output_path.exists():
        if hashlib.md5(output_path.read_bytes()).hexdigest() == new_hash:
            return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(new_bytes)
    return True
