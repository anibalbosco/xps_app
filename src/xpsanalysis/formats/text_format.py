"""Plain-text columnar reader for XPS spectra (.xy, .dat, .txt).

Handles generic two-column numeric data with whitespace or tab delimiters.
Comment lines starting with ``#`` or ``%`` are parsed as metadata.

Files may contain multiple spectra separated by blank lines or by a new
comment/header block appearing after numeric data.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from xpsanalysis.io import SpectrumMetadata, XPSSpectrum


def load_text(path: str | Path) -> list[XPSSpectrum]:
    """Read a plain-text columnar file, returning one XPSSpectrum per section.

    Sections are delimited by blank lines or by a comment/header line
    appearing after numeric data has already been read for the current
    section.
    """
    path = Path(path)

    sections: list[tuple[SpectrumMetadata, list[tuple[float, float]]]] = []
    current_meta = SpectrumMetadata()
    current_rows: list[tuple[float, float]] = []
    has_data = False  # whether current section has any numeric rows

    with open(path) as f:
        for line in f:
            stripped = line.strip()

            # Blank line: finalize current section if it has data
            if not stripped:
                if has_data:
                    sections.append((current_meta, current_rows))
                    current_meta = SpectrumMetadata()
                    current_rows = []
                    has_data = False
                continue

            # Comment / header line
            if stripped[0] in ("#", "%"):
                if has_data:
                    # New header after data → start a new section
                    sections.append((current_meta, current_rows))
                    current_meta = SpectrumMetadata()
                    current_rows = []
                    has_data = False
                _parse_comment(stripped, current_meta)
                continue

            # Try to parse as numeric data
            parts = re.split(r"[\s,]+", stripped)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                # Non-numeric line (e.g. column header) after data → new section
                if has_data:
                    sections.append((current_meta, current_rows))
                    current_meta = SpectrumMetadata()
                    current_rows = []
                    has_data = False
                continue

            current_rows.append((x, y))
            has_data = True

    # Finalize last section
    if has_data:
        sections.append((current_meta, current_rows))

    if not sections:
        raise ValueError(f"No numeric data found in {path}")

    spectra = []
    for meta, rows in sections:
        arr = np.array(rows, dtype=np.float64)
        spectra.append(XPSSpectrum(
            energy=arr[:, 0],
            intensity=arr[:, 1],
            metadata=meta,
        ))

    return spectra


def _parse_comment(line: str, metadata: SpectrumMetadata) -> None:
    """Try to extract metadata from comment lines."""
    text = line.lstrip("#%").strip()
    if ":" not in text:
        return
    key, _, value = text.partition(":")
    key = key.strip().lower().replace(" ", "_")
    value = value.strip()
    if key == "core_level":
        metadata.core_level = value
    elif key == "photon_energy":
        try:
            metadata.photon_energy = float(value)
        except ValueError:
            pass
    elif key == "pass_energy":
        try:
            metadata.pass_energy = float(value)
        except ValueError:
            pass
    else:
        metadata.extra[key] = value
