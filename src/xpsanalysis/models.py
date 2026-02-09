"""Peak model builders using lmfit: pseudo-Voigt lineshapes with constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import lmfit


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float = 1.0,
    center: float = 0.0,
    sigma: float = 1.0,
    fraction: float = 0.5,
) -> np.ndarray:
    """Pseudo-Voigt lineshape: (1-fraction)*Gaussian + fraction*Lorentzian.

    Both components are peak-normalized to *amplitude*.
    """
    dx = x - center
    gaussian = np.exp(-np.log(2) * (dx / sigma) ** 2)
    lorentzian = sigma**2 / (dx**2 + sigma**2)
    return amplitude * ((1 - fraction) * gaussian + fraction * lorentzian)


@dataclass
class PeakSpec:
    """Specification for a single pseudo-Voigt peak."""

    name: str
    center: float
    sigma: float = 0.6
    fraction: float = 0.3
    amplitude: float = 100.0
    bounds: dict = field(default_factory=dict)
    """Bounds per parameter, e.g. ``{"center": {"min": 284, "max": 286}}``."""


@dataclass
class DoubletSpec:
    """Specification for a spin-orbit doublet (two linked peaks)."""

    name: str
    center: float
    splitting: float
    ratio: float = 0.5
    sigma: float = 1.0
    fraction: float = 0.3
    amplitude: float = 100.0
    fix_splitting: bool = True
    fix_ratio: bool = True
    bounds: dict = field(default_factory=dict)


def _make_peak_model(prefix: str) -> lmfit.Model:
    """Create an lmfit Model from the pseudo_voigt function with a prefix."""
    return lmfit.Model(pseudo_voigt, prefix=prefix)


def build_composite_model(
    peaks: list[PeakSpec] | None = None,
    doublets: list[DoubletSpec] | None = None,
    shared_fwhm_groups: list[list[str]] | None = None,
) -> tuple[lmfit.CompositeModel | lmfit.Model, lmfit.Parameters]:
    """Build a composite lmfit model from peak and doublet specifications.

    Returns
    -------
    model : CompositeModel or Model
        The composite model (sum of all components).
    params : Parameters
        The initial parameter set with constraints applied.
    """
    peaks = peaks or []
    doublets = doublets or []
    shared_fwhm_groups = shared_fwhm_groups or []

    models: list[lmfit.Model] = []
    all_params = lmfit.Parameters()

    # --- Single peaks ---
    for spec in peaks:
        prefix = f"{spec.name}_"
        m = _make_peak_model(prefix)
        models.append(m)

        pars = m.make_params()
        pars[f"{prefix}center"].set(value=spec.center)
        pars[f"{prefix}sigma"].set(value=spec.sigma, min=0.01)
        pars[f"{prefix}fraction"].set(value=spec.fraction, min=0.0, max=1.0)
        pars[f"{prefix}amplitude"].set(value=spec.amplitude, min=0.0)

        # Apply user bounds
        _apply_bounds(pars, prefix, spec.bounds)
        all_params += pars

    # --- Doublets ---
    for dspec in doublets:
        prefix_a = f"{dspec.name}_a_"
        prefix_b = f"{dspec.name}_b_"

        m_a = _make_peak_model(prefix_a)
        m_b = _make_peak_model(prefix_b)
        models.extend([m_a, m_b])

        pars_a = m_a.make_params()
        pars_a[f"{prefix_a}center"].set(value=dspec.center)
        pars_a[f"{prefix_a}sigma"].set(value=dspec.sigma, min=0.01)
        pars_a[f"{prefix_a}fraction"].set(value=dspec.fraction, min=0.0, max=1.0)
        pars_a[f"{prefix_a}amplitude"].set(value=dspec.amplitude, min=0.0)
        _apply_bounds(pars_a, prefix_a, dspec.bounds)
        all_params += pars_a

        pars_b = m_b.make_params()
        # Center of peak b = center of peak a + splitting
        pars_b[f"{prefix_b}center"].set(
            expr=f"{prefix_a}center + {dspec.splitting}",
        )
        if dspec.fix_splitting:
            pass  # expr already fixes it
        else:
            # Add splitting as a free parameter
            all_params.add(
                f"{dspec.name}_splitting",
                value=dspec.splitting,
                min=0.0,
            )
            pars_b[f"{prefix_b}center"].set(
                expr=f"{prefix_a}center + {dspec.name}_splitting",
            )

        # Amplitude of peak b = amplitude of peak a * ratio
        if dspec.fix_ratio:
            pars_b[f"{prefix_b}amplitude"].set(
                expr=f"{prefix_a}amplitude * {dspec.ratio}",
            )
        else:
            all_params.add(
                f"{dspec.name}_ratio",
                value=dspec.ratio,
                min=0.0,
                max=1.0,
            )
            pars_b[f"{prefix_b}amplitude"].set(
                expr=f"{prefix_a}amplitude * {dspec.name}_ratio",
            )

        # Share sigma and fraction
        pars_b[f"{prefix_b}sigma"].set(expr=f"{prefix_a}sigma")
        pars_b[f"{prefix_b}fraction"].set(expr=f"{prefix_a}fraction")

        all_params += pars_b

    # --- Build composite ---
    if not models:
        raise ValueError("At least one peak or doublet must be specified")

    composite = models[0]
    for m in models[1:]:
        composite = composite + m

    # --- Shared FWHM groups ---
    for group in shared_fwhm_groups:
        if len(group) < 2:
            continue
        master_prefix = f"{group[0]}_"
        for follower_name in group[1:]:
            follower_prefix = f"{follower_name}_"
            param_name = f"{follower_prefix}sigma"
            if param_name in all_params:
                all_params[param_name].set(expr=f"{master_prefix}sigma")

    return composite, all_params


def _apply_bounds(
    params: lmfit.Parameters,
    prefix: str,
    bounds: dict,
) -> None:
    """Apply user-specified bounds to parameters."""
    for param_short, constraints in bounds.items():
        full_name = f"{prefix}{param_short}"
        if full_name not in params:
            continue
        if isinstance(constraints, dict):
            if "min" in constraints:
                params[full_name].set(min=constraints["min"])
            if "max" in constraints:
                params[full_name].set(max=constraints["max"])
            if "vary" in constraints:
                params[full_name].set(vary=constraints["vary"])
            if "value" in constraints:
                params[full_name].set(value=constraints["value"])
