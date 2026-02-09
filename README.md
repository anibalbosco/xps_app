# XPS Analysis

X-ray Photoelectron Spectroscopy (XPS) spectrum analysis tool. Reads spectra from multiple file formats (CSV, VAMAS/.vms, plain text .xy/.dat/.txt), performs background subtraction (Shirley/Tougaard), fits constrained multi-peak models (pseudo-Voigt), and generates reports.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart — CLI

Generate a synthetic C 1s spectrum and fit it:

```bash
xps generate c1s --noise 0.02 --seed 42 --output examples/c1s_example.csv
xps fit examples/c1s_example.csv --config examples/c1s_config.json --output report.pdf
```

Plot a raw spectrum:

```bash
xps plot examples/c1s_example.csv --output spectrum.png
```

Inspect a file:

```bash
xps info examples/c1s_example.csv
```

## Quickstart — Python

```python
from xpsanalysis import generate_c1s, fit_spectrum, PeakSpec

spectrum = generate_c1s(noise_level=0.02, seed=42)

peaks = [
    PeakSpec(name="CC", center=284.8, sigma=0.6, fraction=0.3, amplitude=1000),
    PeakSpec(name="CO", center=286.3, sigma=0.6, fraction=0.3, amplitude=400),
    PeakSpec(name="CeqO", center=288.5, sigma=0.7, fraction=0.3, amplitude=200),
]

result = fit_spectrum(spectrum, peaks=peaks, background_method="shirley")
print(f"R² = {result.r_squared:.4f}")
for name, area in result.component_areas.items():
    print(f"  {name}: {area:.2f}")
```

## Web UI

```bash
streamlit run src/xpsanalysis/app.py
```

Upload a spectrum file, define peaks in the sidebar, and click "Run Fit".

## Supported File Formats

| Format | Extensions | Regions |
|--------|-----------|---------|
| CSV | `.csv` | Single |
| VAMAS | `.vms` | Multiple |
| Plain text | `.xy`, `.dat`, `.txt` | Single |

## Running Tests

```bash
pytest
```
