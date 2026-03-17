# pmecg — Copilot Instructions

`pmecg` is a Python library for plotting high-quality, paper-like ECG signals using Matplotlib.

## Commands

```bash
pixi install                                # Install the default development environment
pixi run test                               # Run all tests in the default environment
pixi run test-fast                          # Unit + structural tests only (no network)
pixi run pytest tests/test_data.py::TestSegmentLeads -v  # Run a specific test class
pixi run pytest tests/test_data.py::TestNumpyToDataframe::test_shape -v  # Run a single test
pixi run lint                               # Lint + formatting check
pixi run ruff check . --fix                 # Lint with auto-fix
```

Always use `pixi run` (or named Pixi tasks) to invoke Python tools — never `python` or `pytest` directly.

Integration tests (`@pytest.mark.integration`) require network access to download the PTB-XL dataset and are slow; skip them with `-m "not integration"` for routine development.

## Architecture

The codebase has three layers:

1. **Public API** (`src/pmecg/plot.py`, re-exported via `src/pmecg/__init__.py`):
   - `ECGPlotter` — main class; instantiate with visual parameters, then call `.plot()`
   - `ECGStats`, `ECGInformation` — dataclasses for optional metadata overlays
   - `template_factory()`, `LeadsMap`, `SUPPORTED_LEADS` — configuration helpers

2. **Data layer** (`src/pmecg/utils/data.py`):
   - Normalizes ECG input (numpy arrays, lists of arrays, or DataFrames) into a DataFrame
   - Resolves and validates configuration (template expansion, lead name mapping)
   - Segments leads into rows for rendering

3. **Rendering layer** (`src/pmecg/utils/plot.py`):
   - Low-level Matplotlib logic: grid drawing, figure sizing, calibration pulse, row plotting
   - Uses `MM_PER_INCH = 25.4` and physical-unit constants (mm) for exact paper sizes

Data flows: `ECGPlotter.plot()` → data layer normalizes input and expands config → rendering layer produces a Matplotlib `Figure`.

## Key Conventions

**Configuration system:**
- A *configuration* is `list[list[str] | str] | list[list[LeadSegment] | LeadSegment]` — either a purely string-based layout or a purely `LeadSegment`-based layout (mixing is not allowed). Each row element is a string/`LeadSegment` for a full-width row, or a list for concatenated leads.
- Built-in templates (`"4x3"`, `"2x6"`, `"1x12"`, etc.) must be expanded via `template_factory()` before passing to `ECGPlotter.plot()` — `plot()` does not accept raw template strings.
- Custom lead names are mapped to canonical names (`"I"`, `"II"`, ..., `"V6"`) via `LeadsMap`.

**Types:**
```python
ECGDataType = tuple[list[np.ndarray] | np.ndarray, list[str]] | pd.DataFrame
ConfigurationDataType = list[list[str] | str] | list[list[LeadSegment] | LeadSegment]
```

**Naming:**
- Public symbols: `PascalCase` (`ECGPlotter`, `LeadsMap`)
- Internal helpers: leading underscore (`_segment_leads`, `_plot_row`, `_RenderContext`)
- Module-level constants: `UPPER_CASE` (`MM_PER_INCH`, `SUPPORTED_LEADS`, `CAL_PULSE_AMP_MV`)

**Typing:** Full type annotations are required throughout (`py.typed` marker is present). Docstrings follow NumPy/SciPy style.

**Linting:** Ruff with rules `E, W, F, I, UP, B` and line length 128, targeting Python 3.8+. The `pixi run lint` task runs `ruff check . && ruff format --check .`.

**Tests:** `test_plot_systematic.py` uses `matplotlib.use("Agg")` for headless rendering. Tests are heavily parametrized across all built-in templates. New layout or data-handling changes should be validated against the systematic tests.
