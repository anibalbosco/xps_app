"""Extract the analyzer transmission function T(KE) from a survey spectrum.

The background of a survey spectrum (secondary + inelastically scattered
electrons) encodes the transmission function.  After masking photoelectron
peaks, the remaining background envelope is:

    B(KE) ∝ S(KE) × T(KE)

where S(KE) ~ 1/KE² is the universal secondary-electron cascade.  Dividing
out S(KE) and fitting a power law yields T(KE) = a × KE^n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


@dataclass
class TransmissionResult:
    """Result of transmission function extraction."""
    a: float               # prefactor
    n: float               # exponent  T(KE) = a * KE^n
    a_err: float           # uncertainty on a
    n_err: float           # uncertainty on n
    ke_data: np.ndarray    # KE values where T was sampled
    t_data: np.ndarray     # T(KE) values (background / secondary cascade)
    ke_fit: np.ndarray     # KE array for the fitted curve
    t_fit: np.ndarray      # fitted T(KE) curve
    bg_ke: np.ndarray      # KE of the background envelope (peaks masked)
    bg_intensity: np.ndarray  # background intensity (peaks masked)


def _detect_and_mask_peaks(
    energy: np.ndarray,
    intensity: np.ndarray,
    mask_width_ev: float = 8.0,
    prominence_frac: float = 0.03,
) -> np.ndarray:
    """Return a boolean mask that is True for background (non-peak) points.

    Peaks are detected and a window of ±mask_width_ev is excluded around each.
    """
    i_range = np.max(intensity) - np.min(intensity)
    prominence = max(i_range * prominence_frac, 1.0)

    indices, _ = find_peaks(intensity, prominence=prominence, distance=5)

    mask = np.ones(len(energy), dtype=bool)
    for idx in indices:
        peak_e = energy[idx]
        mask &= np.abs(energy - peak_e) > mask_width_ev

    return mask


def _smooth_background(
    ke: np.ndarray,
    intensity: np.ndarray,
    window_ev: float = 30.0,
) -> np.ndarray:
    """Smooth background using Savitzky-Golay filter."""
    if len(ke) < 15:
        return intensity
    pts_per_ev = len(ke) / (ke.max() - ke.min()) if ke.max() > ke.min() else 1
    window_pts = int(window_ev * pts_per_ev)
    window_pts = max(5, window_pts)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = min(window_pts, len(ke) - 2)
    if window_pts % 2 == 0:
        window_pts -= 1
    if window_pts < 5:
        return intensity
    return savgol_filter(intensity, window_pts, polyorder=2)


def extract_transmission(
    energy_be: np.ndarray,
    intensity: np.ndarray,
    photon_energy: float = 1486.6,
    mask_width_ev: float = 8.0,
    ke_min: float = 50.0,
    ke_max: float | None = None,
) -> TransmissionResult:
    """Extract the transmission function from a survey spectrum.

    Parameters
    ----------
    energy_be : array
        Binding energy axis (eV).
    intensity : array
        Intensity counts.
    photon_energy : float
        X-ray source energy (eV), default Al Kα.
    mask_width_ev : float
        Half-width (eV) to mask around detected peaks.
    ke_min : float
        Minimum kinetic energy to use (avoids low-KE secondary electron
        peak where the 1/KE² model breaks down).
    ke_max : float or None
        Maximum kinetic energy (default: photon_energy - 20).

    Returns
    -------
    TransmissionResult with fitted parameters and diagnostic arrays.
    """
    if ke_max is None:
        ke_max = photon_energy - 20.0

    # Sort by binding energy ascending
    order = np.argsort(energy_be)
    be = energy_be[order]
    ints = intensity[order].astype(float)

    # Convert to kinetic energy
    ke_all = photon_energy - be

    # Mask peaks (work in BE space where peaks are positive-going)
    bg_mask = _detect_and_mask_peaks(be, ints, mask_width_ev=mask_width_ev)

    # Restrict to valid KE range
    ke_range_mask = (ke_all >= ke_min) & (ke_all <= ke_max)
    combined_mask = bg_mask & ke_range_mask

    if np.sum(combined_mask) < 20:
        raise ValueError(
            "Too few background points after peak masking. "
            "Try reducing mask_width_ev or adjusting KE range.")

    ke_bg = ke_all[combined_mask]
    int_bg = ints[combined_mask]

    # Sort by KE
    ke_order = np.argsort(ke_bg)
    ke_bg = ke_bg[ke_order]
    int_bg = int_bg[ke_order]

    # Smooth the background envelope
    int_smooth = _smooth_background(ke_bg, int_bg, window_ev=30.0)

    # Fit the background directly as B(KE) = a * KE^m + c
    # where m = n - 2  (since B = S*T = KE^(-2) * a*KE^n = a*KE^(n-2))
    # The constant c captures the flat baseline (detector dark counts, etc.)
    # Then T(KE) exponent n = m + 2.

    def _bg_model(ke, a, m, c):
        return a * np.power(ke, m) + c

    # Initial guesses from the data
    c_guess = np.percentile(int_smooth, 10)
    a_guess = np.max(int_smooth) - c_guess
    m_guess = -2.7  # typical for FAT mode with n~-0.7

    try:
        popt, pcov = curve_fit(
            _bg_model, ke_bg, int_smooth,
            p0=[a_guess, m_guess, c_guess],
            bounds=([0, -6, -np.inf], [np.inf, 0, np.inf]),
            maxfev=10000,
        )
        a_bg, m_fit, c_fit = popt
        perr = np.sqrt(np.diag(pcov))
        m_err = perr[1]
    except (RuntimeError, ValueError):
        # Fallback: estimate baseline from high-KE end and use log-log
        high_ke = ke_bg > np.percentile(ke_bg, 80)
        c_fit = np.percentile(int_smooth[high_ke], 10) if np.sum(high_ke) > 5 else np.min(int_smooth)
        corrected = np.maximum(int_smooth - c_fit, 1e-10)
        pos = corrected > np.max(corrected) * 0.01
        if np.sum(pos) < 10:
            raise ValueError("Too few valid background points for fitting.")
        log_ke = np.log(ke_bg[pos])
        log_bg = np.log(corrected[pos])
        coeffs = np.polyfit(log_ke, log_bg, 1)
        m_fit = coeffs[0]
        a_bg = np.exp(coeffs[1])
        m_err = 0.0

    n_fit = m_fit + 2.0  # T(KE) = B(KE) * KE² / normalization → exponent shifts by +2
    n_err = m_err  # same uncertainty

    # Build T(KE) data for display: subtract baseline, multiply by KE²
    int_corrected = int_smooth - c_fit
    above_floor = int_corrected > np.max(int_corrected) * 0.005
    ke_valid = ke_bg[above_floor]
    int_valid = int_corrected[above_floor]

    if len(ke_valid) < 10:
        raise ValueError("Too few valid background points above baseline.")

    t_raw = int_valid * ke_valid**2
    t_max = np.max(t_raw)
    t_norm = t_raw / t_max if t_max > 0 else t_raw

    # Prefactor: normalize so T(KE) curve matches the data scale
    a_fit = t_norm[len(t_norm) // 2] / np.power(ke_valid[len(ke_valid) // 2], n_fit)
    a_err = abs(a_fit) * n_err / max(abs(n_fit), 0.01)  # propagated

    # Generate fitted curve
    ke_curve = np.linspace(ke_min, ke_max, 500)
    t_curve = a_fit * np.power(ke_curve, n_fit)

    return TransmissionResult(
        a=a_fit,
        n=n_fit,
        a_err=a_err,
        n_err=n_err,
        ke_data=ke_valid,
        t_data=t_norm,
        ke_fit=ke_curve,
        t_fit=t_curve,
        bg_ke=ke_bg,
        bg_intensity=int_smooth,
    )
