"""Energy axis utilities: sorting, calibration, reversal."""

from __future__ import annotations

import numpy as np

from xpsanalysis.io import XPSSpectrum


def ensure_increasing(spectrum: XPSSpectrum) -> XPSSpectrum:
    """Return a copy of *spectrum* with energy sorted in ascending order."""
    if spectrum.energy.size < 2:
        return spectrum
    if spectrum.energy[0] <= spectrum.energy[-1]:
        # Already ascending (or single point)
        idx = np.argsort(spectrum.energy)
        if np.all(idx == np.arange(len(idx))):
            return spectrum
    else:
        idx = np.argsort(spectrum.energy)
    return XPSSpectrum(
        energy=spectrum.energy[idx].copy(),
        intensity=spectrum.intensity[idx].copy(),
        metadata=spectrum.metadata,
    )


def calibrate(spectrum: XPSSpectrum, shift_eV: float) -> XPSSpectrum:
    """Return a copy with the energy axis shifted by *shift_eV*."""
    return XPSSpectrum(
        energy=spectrum.energy + shift_eV,
        intensity=spectrum.intensity.copy(),
        metadata=spectrum.metadata,
    )


def reverse_energy(spectrum: XPSSpectrum) -> XPSSpectrum:
    """Flip the energy axis direction (and corresponding intensities)."""
    return XPSSpectrum(
        energy=spectrum.energy[::-1].copy(),
        intensity=spectrum.intensity[::-1].copy(),
        metadata=spectrum.metadata,
    )
