"""Unified loading interface and core data structures for XPS spectra."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpectrumMetadata:
    """Optional metadata associated with an XPS spectrum."""

    pass_energy: Optional[float] = None
    analyzer: Optional[str] = None
    photon_energy: Optional[float] = None
    core_level: Optional[str] = None
    sample_id: Optional[str] = None
    scan_mode: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class XPSSpectrum:
    """An XPS spectrum: energy axis, intensity array, and optional metadata."""

    energy: NDArray[np.float64]
    intensity: NDArray[np.float64]
    metadata: SpectrumMetadata = field(default_factory=SpectrumMetadata)

    def __post_init__(self) -> None:
        self.energy = np.asarray(self.energy, dtype=np.float64)
        self.intensity = np.asarray(self.intensity, dtype=np.float64)
        if self.energy.shape != self.intensity.shape:
            raise ValueError(
                f"energy and intensity must have the same shape, "
                f"got {self.energy.shape} and {self.intensity.shape}"
            )
        if self.energy.ndim != 1:
            raise ValueError(f"Expected 1-D arrays, got ndim={self.energy.ndim}")


def load_spectrum(path: str | Path) -> list[XPSSpectrum]:
    """Load XPS spectra from a file, auto-detecting format by extension.

    Returns a list because VAMAS files can contain multiple spectral regions.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        from xpsanalysis.formats.csv_format import load_csv

        return load_csv(path)
    elif ext == ".vms":
        from xpsanalysis.formats.vamas import load_vamas

        return load_vamas(path)
    elif ext in (".xy", ".dat", ".txt"):
        from xpsanalysis.formats.text_format import load_text

        return load_text(path)
    else:
        warnings.warn(
            f"Unknown extension '{ext}', attempting plain-text columnar parse.",
            stacklevel=2,
        )
        from xpsanalysis.formats.text_format import load_text

        return load_text(path)


def save_results_csv(path: str | Path, fit_result: "FitResult") -> None:  # noqa: F821
    """Write fit results to a CSV file."""
    from xpsanalysis.fitting import FitResult  # noqa: F811

    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult instance")

    path = Path(path)
    energy = fit_result.spectrum.energy
    intensity = fit_result.spectrum.intensity
    background = fit_result.background
    envelope = fit_result.model_result.best_fit + background

    header_parts = [
        "binding_energy_eV",
        "raw_intensity",
        "background",
        "envelope",
    ]
    columns = [energy, intensity, background, envelope]

    for name, curve in fit_result.component_curves.items():
        header_parts.append(name)
        columns.append(curve + background)

    header_parts.append("residuals")
    columns.append(fit_result.residuals)

    data = np.column_stack(columns)
    header = ",".join(header_parts)

    # Add fit summary as comments
    meta_lines = [
        f"# background_method: {fit_result.background_method}",
        f"# r_squared: {fit_result.r_squared:.6f}",
    ]
    for name, area in fit_result.component_areas.items():
        meta_lines.append(f"# area_{name}: {area:.4f}")
    for pname, pval in fit_result.fit_params.items():
        meta_lines.append(f"# param_{pname}: {pval:.6f}")

    meta_block = "\n".join(meta_lines) + "\n"

    with open(path, "w") as f:
        f.write(meta_block)
        f.write(header + "\n")
        for row in data:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
