"""Background subtraction algorithms for XPS spectra.

Implements Shirley iterative and Tougaard universal background methods.
Both functions expect energy in ascending order.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def shirley_background(
    energy: NDArray[np.float64],
    intensity: NDArray[np.float64],
    tol: float = 1e-5,
    max_iter: int = 50,
) -> tuple[NDArray[np.float64], bool]:
    """Compute the Shirley iterative background.

    Parameters
    ----------
    energy : 1-D array
        Binding energy in ascending order.
    intensity : 1-D array
        Measured intensity.
    tol : float
        Convergence tolerance (max relative change in background).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    background : 1-D array
        The Shirley background.
    converged : bool
        Whether the iteration converged within *max_iter* steps.

    Notes
    -----
    The algorithm iterates:

    1. Start with a flat background equal to the intensity at the high-BE end.
    2. Compute the cumulative integral of (intensity − background) from high BE
       to each point.
    3. Scale the integral so the background at the low-BE end matches the
       spectrum intensity at the low-BE end.
    4. Add the intensity at the high-BE end as the offset.
    5. Repeat until convergence.
    """
    energy = np.asarray(energy, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    n = len(energy)
    if n < 2:
        return intensity.copy(), True

    # Endpoint intensities
    i_left = intensity[0]   # low BE end (ascending order ⇒ index 0)
    i_right = intensity[-1]  # high BE end

    # Step height the background must account for
    step = i_left - i_right

    # Start with flat background
    bg = np.full(n, i_right, dtype=np.float64)
    converged = False

    for _iteration in range(max_iter):
        # Net intensity above background
        net = intensity - bg

        # Cumulative integral from high BE (right) to each point E.
        # Using reversed cumulative trapezoid integration.
        # cumulative_integral[i] = integral from energy[i] to energy[-1] of net(E') dE'
        cumulative = np.zeros(n, dtype=np.float64)
        for i in range(n - 2, -1, -1):
            cumulative[i] = cumulative[i + 1] + 0.5 * (net[i] + net[i + 1]) * (
                energy[i + 1] - energy[i]
            )

        # Scale factor so background at left end = i_left
        total_integral = cumulative[0]
        if abs(total_integral) < 1e-30:
            converged = True
            break

        k = step / total_integral
        new_bg = i_right + k * cumulative

        # Check convergence
        max_change = np.max(np.abs(new_bg - bg))
        denom = max(np.max(np.abs(new_bg)), 1e-30)
        if max_change / denom < tol:
            bg = new_bg
            converged = True
            break
        bg = new_bg

    return bg, converged


def tougaard_background(
    energy: NDArray[np.float64],
    intensity: NDArray[np.float64],
    B: float = 2866.0,
    C: float = 1643.0,
    C_prime: float = 1.0,
    D: float = 1.0,
    T_max: float = 50.0,
) -> NDArray[np.float64]:
    """Compute the Tougaard universal background.

    Parameters
    ----------
    energy : 1-D array
        Binding energy in ascending order.
    intensity : 1-D array
        Measured intensity.
    B, C, C_prime, D : float
        Tougaard cross-section parameters.  Default values correspond to the
        "universal" cross-section.
    T_max : float
        Maximum loss energy (eV) for the integration.

    Returns
    -------
    background : 1-D array
        The Tougaard background.

    Notes
    -----
    B(E) = ∫₀ᵀᵐᵃˣ  I(E+T) · K(T) dT

    where  K(T) = B·T / ((C − C'·T²)² + D·T²)
    """
    energy = np.asarray(energy, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    n = len(energy)

    if n < 2:
        return np.zeros_like(intensity)

    de = np.mean(np.diff(energy))  # average step
    bg = np.zeros(n, dtype=np.float64)

    # Number of loss-energy steps in the integration
    n_loss = int(T_max / de) + 1

    for i in range(n):
        integral = 0.0
        for j in range(1, n_loss + 1):
            T = j * de
            idx = i + j  # index corresponding to energy E + T
            if idx >= n:
                break
            # Universal cross-section
            denom = (C - C_prime * T**2) ** 2 + D * T**2
            if denom < 1e-30:
                continue
            K = B * T / denom
            integral += intensity[idx] * K * de
        bg[i] = integral

    return bg
