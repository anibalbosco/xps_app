"""Synthetic XPS spectrum generators for testing and demonstration."""

from __future__ import annotations

import numpy as np

from xpsanalysis.io import SpectrumMetadata, XPSSpectrum


def _pseudo_voigt(
    x: np.ndarray,
    center: float,
    sigma: float,
    fraction: float,
    amplitude: float,
) -> np.ndarray:
    """Pseudo-Voigt lineshape: (1-fraction)*Gaussian + fraction*Lorentzian.

    Both components are normalized so that the peak maximum equals *amplitude*.
    """
    dx = x - center
    gaussian = np.exp(-np.log(2) * (dx / sigma) ** 2)
    lorentzian = sigma**2 / (dx**2 + sigma**2)
    return amplitude * ((1 - fraction) * gaussian + fraction * lorentzian)


def _shirley_step(
    x: np.ndarray,
    left_level: float,
    right_level: float,
    center: float,
    width: float,
) -> np.ndarray:
    """Smooth step function (sigmoid-based) to simulate a Shirley background.

    Returns values transitioning from *left_level* (low BE) to *right_level*
    (high BE), centered around *center* with given *width*.
    """
    step = 1.0 / (1.0 + np.exp(-(x - center) / width))
    return left_level + (right_level - left_level) * step


def generate_c1s(
    noise_level: float = 0.02,
    seed: int | None = None,
) -> XPSSpectrum:
    """Generate a synthetic C 1s spectrum with three peaks on a Shirley background.

    Peaks:
    - C-C at 284.8 eV (amplitude 1000)
    - C-O at 286.3 eV (amplitude 400)
    - C=O at 288.5 eV (amplitude 200)

    Parameters
    ----------
    noise_level : float
        Standard deviation of Gaussian noise, as a fraction of the maximum
        peak intensity.
    seed : int or None
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    energy = np.linspace(282.0, 292.0, 500)

    # Peaks
    cc = _pseudo_voigt(energy, center=284.8, sigma=0.6, fraction=0.3, amplitude=1000)
    co = _pseudo_voigt(energy, center=286.3, sigma=0.6, fraction=0.3, amplitude=400)
    c_eq_o = _pseudo_voigt(energy, center=288.5, sigma=0.7, fraction=0.3, amplitude=200)
    peaks = cc + co + c_eq_o

    # Background: step from ~100 (low BE) to ~50 (high BE)
    bg = _shirley_step(energy, left_level=100, right_level=50, center=286, width=1.5)

    intensity = peaks + bg
    if noise_level > 0:
        intensity += rng.normal(0, noise_level * 1000, size=energy.shape)

    metadata = SpectrumMetadata(
        core_level="C 1s",
        photon_energy=1486.6,
    )
    return XPSSpectrum(energy=energy, intensity=intensity, metadata=metadata)


def generate_fe2p(
    noise_level: float = 0.02,
    seed: int | None = None,
) -> XPSSpectrum:
    """Generate a synthetic Fe 2p spectrum with a spin-orbit doublet.

    Peaks:
    - Fe 2p3/2 at 710.8 eV (amplitude 1000)
    - Fe 2p1/2 at 724.4 eV (amplitude 500, ratio ~0.5)

    Parameters
    ----------
    noise_level : float
        Standard deviation of Gaussian noise, as a fraction of the maximum
        peak intensity.
    seed : int or None
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    energy = np.linspace(705.0, 735.0, 600)

    p3_2 = _pseudo_voigt(energy, center=710.8, sigma=1.2, fraction=0.4, amplitude=1000)
    p1_2 = _pseudo_voigt(energy, center=724.4, sigma=1.4, fraction=0.4, amplitude=500)
    peaks = p3_2 + p1_2

    bg = _shirley_step(energy, left_level=150, right_level=80, center=717, width=3.0)

    intensity = peaks + bg
    if noise_level > 0:
        intensity += rng.normal(0, noise_level * 1000, size=energy.shape)

    metadata = SpectrumMetadata(
        core_level="Fe 2p",
        photon_energy=1486.6,
    )
    return XPSSpectrum(energy=energy, intensity=intensity, metadata=metadata)
