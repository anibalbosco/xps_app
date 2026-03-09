"""XPS spectrum analysis: background subtraction, peak fitting, and reporting."""

from xpsanalysis.io import XPSSpectrum, SpectrumMetadata, load_spectrum, save_results_csv
from xpsanalysis.energy import ensure_increasing, calibrate, reverse_energy
from xpsanalysis.background import shirley_background, tougaard_background
from xpsanalysis.models import PeakSpec, DoubletSpec, build_composite_model, pseudo_voigt
from xpsanalysis.fitting import FitResult, fit_spectrum
from xpsanalysis.synthetic import generate_c1s, generate_fe2p
from xpsanalysis.reference import (
    CoreLevelRef, ChemicalState, REFERENCE_DB, CITATIONS,
    AugerLine, AUGER_DB, XRAY_SOURCES, search_peak,
)
from xpsanalysis.identify import IdentificationResult, identify_spectrum

__all__ = [
    "XPSSpectrum",
    "SpectrumMetadata",
    "load_spectrum",
    "save_results_csv",
    "ensure_increasing",
    "calibrate",
    "reverse_energy",
    "shirley_background",
    "tougaard_background",
    "PeakSpec",
    "DoubletSpec",
    "build_composite_model",
    "pseudo_voigt",
    "FitResult",
    "fit_spectrum",
    "generate_c1s",
    "generate_fe2p",
    "CoreLevelRef",
    "ChemicalState",
    "REFERENCE_DB",
    "CITATIONS",
    "IdentificationResult",
    "identify_spectrum",
]
