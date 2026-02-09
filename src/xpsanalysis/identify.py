"""Auto-identification of XPS spectral peaks from the reference database."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from xpsanalysis.io import XPSSpectrum
from xpsanalysis.models import PeakSpec, DoubletSpec
from xpsanalysis.reference import (
    CoreLevelRef,
    ChemicalState,
    REFERENCE_DB,
    get_core_levels_in_range,
    get_core_level,
    parse_core_level_label,
)


@dataclass
class PeakAssignment:
    """A single suggested peak assignment."""

    core_level_ref: CoreLevelRef
    chemical_state: ChemicalState | None
    suggested_position: float
    confidence: float  # 0.0 to 1.0


@dataclass
class IdentificationResult:
    """Result of auto-identifying a spectrum."""

    element_symbol: str | None
    core_level_label: str | None
    assignments: list[PeakAssignment]
    suggested_peaks: list[PeakSpec] = field(default_factory=list)
    suggested_doublets: list[DoubletSpec] = field(default_factory=list)


def _detect_peaks(
    energy: np.ndarray,
    intensity: np.ndarray,
    min_prominence_frac: float = 0.05,
) -> list[tuple[float, float]]:
    """Find local maxima in intensity.

    Returns list of (energy_position, height) tuples sorted by height descending.
    """
    if len(energy) < 5:
        return []

    # Estimate prominence threshold as fraction of intensity range
    i_range = np.max(intensity) - np.min(intensity)
    prominence = max(i_range * min_prominence_frac, 1.0)

    indices, props = find_peaks(intensity, prominence=prominence, distance=5)
    if len(indices) == 0:
        return []

    results = [(float(energy[i]), float(intensity[i])) for i in indices]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _score_candidate(
    ref: CoreLevelRef,
    energy: np.ndarray,
    intensity: np.ndarray,
    detected: list[tuple[float, float]],
) -> float:
    """Score how well a reference core level matches the spectrum.

    Higher score = better match. Based on:
    - Whether the expected BE is near the center of the energy range
    - Whether there's a detected peak near the expected BE
    - Sensitivity factor (more commonly observed lines score higher)
    """
    e_min, e_max = float(energy.min()), float(energy.max())
    e_center = (e_min + e_max) / 2
    e_span = e_max - e_min

    be = ref.binding_energy

    # Must be within range
    if be < e_min or be > e_max:
        return 0.0

    # Score: closeness to center (peaks near center of scan more likely intended)
    dist_to_center = abs(be - e_center) / (e_span / 2) if e_span > 0 else 1.0
    center_score = max(0.0, 1.0 - 0.5 * dist_to_center)

    # Score: proximity to a detected peak
    peak_score = 0.0
    if detected:
        min_dist = min(abs(be - p[0]) for p in detected)
        # Within 3 eV of a detected peak is good
        peak_score = max(0.0, 1.0 - min_dist / 3.0)

    # Score: sensitivity factor (log scale, normalized)
    rsf_score = min(1.0, ref.sensitivity_factor / 5.0) if ref.sensitivity_factor > 0 else 0.1

    return 0.3 * center_score + 0.5 * peak_score + 0.2 * rsf_score


def identify_spectrum(spectrum: XPSSpectrum) -> IdentificationResult:
    """Identify the core level and suggest peak assignments for a spectrum.

    If ``spectrum.metadata.core_level`` is set (e.g. "C 1s"), it is used
    directly.  Otherwise, the energy range is searched against the reference
    database and the best-matching core level is chosen.
    """
    energy = np.sort(spectrum.energy)
    intensity = spectrum.intensity[np.argsort(spectrum.energy)]
    e_min, e_max = float(energy.min()), float(energy.max())

    detected = _detect_peaks(energy, intensity)

    # --- Resolve the core level ---
    ref: CoreLevelRef | None = None

    if spectrum.metadata.core_level:
        parsed = parse_core_level_label(spectrum.metadata.core_level)
        if parsed:
            sym, orb = parsed
            ref = get_core_level(sym, orb)

    if ref is None:
        # Search by energy range
        candidates = get_core_levels_in_range(e_min, e_max)
        if candidates:
            scored = [(c, _score_candidate(c, energy, intensity, detected)) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored[0][1] > 0:
                ref = scored[0][0]

    if ref is None:
        return IdentificationResult(
            element_symbol=None,
            core_level_label=None,
            assignments=[],
        )

    # --- Build assignments ---
    element = ref.element_symbol
    label = f"{element} {ref.orbital}"
    assignments: list[PeakAssignment] = []

    if ref.chemical_states:
        for cs in ref.chemical_states:
            if e_min <= cs.binding_energy <= e_max:
                # Check if there's a detected peak nearby
                conf = 0.5
                if detected:
                    min_dist = min(abs(cs.binding_energy - p[0]) for p in detected)
                    conf = max(0.3, min(1.0, 1.0 - min_dist / 4.0))
                assignments.append(PeakAssignment(
                    core_level_ref=ref,
                    chemical_state=cs,
                    suggested_position=cs.binding_energy,
                    confidence=conf,
                ))
    else:
        # No chemical states—use the elemental BE
        conf = 0.5
        if detected:
            min_dist = min(abs(ref.binding_energy - p[0]) for p in detected)
            conf = max(0.3, min(1.0, 1.0 - min_dist / 4.0))
        assignments.append(PeakAssignment(
            core_level_ref=ref,
            chemical_state=None,
            suggested_position=ref.binding_energy,
            confidence=conf,
        ))

    assignments.sort(key=lambda a: a.confidence, reverse=True)

    # --- Build PeakSpec / DoubletSpec ---
    suggested_peaks: list[PeakSpec] = []
    suggested_doublets: list[DoubletSpec] = []

    if ref.is_doublet and ref.splitting and ref.branching_ratio:
        # Each chemical state gets its own doublet (5/2+3/2, 3/2+1/2, etc.)
        for asn in assignments:
            center = asn.suggested_position
            amp = _estimate_amplitude(center, energy, intensity)
            name = _clean_name(asn.chemical_state.name if asn.chemical_state else element)
            suggested_doublets.append(DoubletSpec(
                name=name,
                center=center,
                splitting=ref.splitting,
                ratio=ref.branching_ratio,
                sigma=ref.typical_sigma,
                fraction=0.3,
                amplitude=amp,
            ))
    else:
        for asn in assignments:
            amp = _estimate_amplitude(asn.suggested_position, energy, intensity)
            name = _clean_name(asn.chemical_state.name if asn.chemical_state else element)
            suggested_peaks.append(PeakSpec(
                name=name,
                center=asn.suggested_position,
                sigma=ref.typical_sigma,
                fraction=0.3,
                amplitude=amp,
            ))

    return IdentificationResult(
        element_symbol=element,
        core_level_label=label,
        assignments=assignments,
        suggested_peaks=suggested_peaks,
        suggested_doublets=suggested_doublets,
    )


def _estimate_amplitude(
    center: float,
    energy: np.ndarray,
    intensity: np.ndarray,
    window: float = 2.0,
) -> float:
    """Estimate peak amplitude from the intensity near *center*."""
    mask = np.abs(energy - center) < window
    if np.any(mask):
        local_max = float(np.max(intensity[mask]))
        local_min = float(np.min(intensity[mask]))
        return max(local_max - local_min, 10.0)
    return 100.0


def _clean_name(name: str) -> str:
    """Make a valid Python identifier from a chemical state name."""
    return name.replace("-", "_").replace("=", "eq").replace(" ", "_").replace("(", "").replace(")", "")
