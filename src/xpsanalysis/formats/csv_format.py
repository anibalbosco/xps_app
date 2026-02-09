"""CSV reader for XPS spectra.

Expected format: two columns ``binding_energy_eV, intensity`` with an
optional header row.  Lines starting with ``#`` are parsed as
``# key: value`` metadata comments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xpsanalysis.io import SpectrumMetadata, XPSSpectrum

_META_MAP = {
    "pass_energy": ("pass_energy", float),
    "analyzer": ("analyzer", str),
    "photon_energy": ("photon_energy", float),
    "core_level": ("core_level", str),
    "sample_id": ("sample_id", str),
    "scan_mode": ("scan_mode", str),
}


def load_csv(path: str | Path) -> list[XPSSpectrum]:
    """Read a CSV file with ``binding_energy_eV, intensity`` columns."""
    path = Path(path)
    metadata = SpectrumMetadata()
    data_lines: list[str] = []

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                _parse_comment(stripped, metadata)
                continue
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data found in {path}")

    # Skip header row if first field is not numeric
    first_fields = data_lines[0].split(",")
    try:
        float(first_fields[0].strip())
    except ValueError:
        data_lines = data_lines[1:]

    if not data_lines:
        raise ValueError(f"No numeric data found in {path}")

    rows = []
    for line in data_lines:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        rows.append((float(parts[0].strip()), float(parts[1].strip())))

    arr = np.array(rows, dtype=np.float64)
    spectrum = XPSSpectrum(
        energy=arr[:, 0],
        intensity=arr[:, 1],
        metadata=metadata,
    )
    return [spectrum]


def _parse_comment(line: str, metadata: SpectrumMetadata) -> None:
    """Parse ``# key: value`` comment lines into metadata fields."""
    text = line.lstrip("#").strip()
    if ":" not in text:
        return
    key, _, value = text.partition(":")
    key = key.strip().lower().replace(" ", "_")
    value = value.strip()
    if key in _META_MAP:
        attr, converter = _META_MAP[key]
        try:
            setattr(metadata, attr, converter(value))
        except (ValueError, TypeError):
            metadata.extra[key] = value
    else:
        metadata.extra[key] = value
