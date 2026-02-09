"""Fitting orchestrator: background subtraction → model build → fit → results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import lmfit

from xpsanalysis.io import XPSSpectrum
from xpsanalysis.energy import ensure_increasing
from xpsanalysis.background import shirley_background, tougaard_background
from xpsanalysis.models import PeakSpec, DoubletSpec, build_composite_model


@dataclass
class FitResult:
    """Container for the complete fitting result."""

    spectrum: XPSSpectrum
    background: np.ndarray
    background_method: str
    model_result: lmfit.model.ModelResult
    component_curves: dict[str, np.ndarray]
    component_areas: dict[str, float]
    residuals: np.ndarray
    r_squared: float
    fit_params: dict[str, float]


def fit_spectrum(
    spectrum: XPSSpectrum,
    peaks: list[PeakSpec] | None = None,
    doublets: list[DoubletSpec] | None = None,
    shared_fwhm_groups: list[list[str]] | None = None,
    background_method: str = "shirley",
    background_kwargs: dict | None = None,
) -> FitResult:
    """Run the full fitting pipeline on a spectrum.

    Steps:
    1. Ensure energy is in ascending order.
    2. Subtract background (Shirley or Tougaard).
    3. Build composite model from peak/doublet specs.
    4. Fit the model to the net (background-subtracted) intensity.
    5. Evaluate each component curve and compute areas.
    6. Package results into a FitResult.
    """
    peaks = peaks or []
    doublets = doublets or []
    shared_fwhm_groups = shared_fwhm_groups or []
    background_kwargs = background_kwargs or {}

    # 1. Sort ascending
    spectrum = ensure_increasing(spectrum)
    energy = spectrum.energy
    intensity = spectrum.intensity

    # 2. Background subtraction
    if background_method.lower() == "shirley":
        bg, _converged = shirley_background(energy, intensity, **background_kwargs)
    elif background_method.lower() == "tougaard":
        bg = tougaard_background(energy, intensity, **background_kwargs)
    else:
        raise ValueError(f"Unknown background method: {background_method!r}")

    net_intensity = intensity - bg

    # Ensure net intensity has no negative values (clamp to 0)
    net_intensity = np.maximum(net_intensity, 0.0)

    # 3. Build model
    model, params = build_composite_model(peaks, doublets, shared_fwhm_groups)

    # 4. Fit
    result = model.fit(net_intensity, params, x=energy)

    # 5. Evaluate components and areas
    component_curves: dict[str, np.ndarray] = {}
    component_areas: dict[str, float] = {}
    components = result.eval_components(x=energy)
    for comp_name, curve in components.items():
        # Strip trailing underscore from prefix
        clean_name = comp_name.rstrip("_")
        component_curves[clean_name] = curve
        component_areas[clean_name] = float(np.trapz(curve, energy))

    # 6. Residuals and R²
    residuals = net_intensity - result.best_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((net_intensity - np.mean(net_intensity)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Extract parameter values
    fit_params = {name: par.value for name, par in result.params.items()}

    return FitResult(
        spectrum=spectrum,
        background=bg,
        background_method=background_method,
        model_result=result,
        component_curves=component_curves,
        component_areas=component_areas,
        residuals=residuals,
        r_squared=r_squared,
        fit_params=fit_params,
    )
