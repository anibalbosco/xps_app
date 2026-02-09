"""Tests for the fitting orchestrator."""

import numpy as np
import pytest

from xpsanalysis.fitting import fit_spectrum, FitResult
from xpsanalysis.models import PeakSpec, DoubletSpec


class TestFitSpectrum:
    def test_c1s_fit(self, c1s_spectrum):
        peaks = [
            PeakSpec(name="CC", center=284.8, sigma=0.6, fraction=0.3, amplitude=1000),
            PeakSpec(name="CO", center=286.3, sigma=0.6, fraction=0.3, amplitude=400),
            PeakSpec(name="CeqO", center=288.5, sigma=0.7, fraction=0.3, amplitude=200),
        ]
        result = fit_spectrum(c1s_spectrum, peaks=peaks, background_method="shirley")

        assert isinstance(result, FitResult)
        assert result.r_squared > 0.9

        # Check recovered centers within 0.3 eV
        params = result.fit_params
        assert abs(params["CC_center"] - 284.8) < 0.3
        assert abs(params["CO_center"] - 286.3) < 0.3
        assert abs(params["CeqO_center"] - 288.5) < 0.3

    def test_c1s_areas(self, c1s_spectrum):
        peaks = [
            PeakSpec(name="CC", center=284.8, sigma=0.6, fraction=0.3, amplitude=1000),
            PeakSpec(name="CO", center=286.3, sigma=0.6, fraction=0.3, amplitude=400),
            PeakSpec(name="CeqO", center=288.5, sigma=0.7, fraction=0.3, amplitude=200),
        ]
        result = fit_spectrum(c1s_spectrum, peaks=peaks, background_method="shirley")

        # CC should have the largest area
        assert result.component_areas["CC"] > result.component_areas["CO"]
        assert result.component_areas["CO"] > result.component_areas["CeqO"]

    def test_tougaard_background(self, c1s_spectrum):
        peaks = [
            PeakSpec(name="CC", center=284.8, sigma=0.6, fraction=0.3, amplitude=1000),
        ]
        result = fit_spectrum(c1s_spectrum, peaks=peaks, background_method="tougaard")
        assert isinstance(result, FitResult)
        assert result.background_method == "tougaard"

    def test_invalid_background(self, c1s_spectrum):
        peaks = [PeakSpec(name="CC", center=284.8)]
        with pytest.raises(ValueError, match="Unknown background"):
            fit_spectrum(c1s_spectrum, peaks=peaks, background_method="invalid")

    def test_residuals_shape(self, c1s_spectrum):
        peaks = [PeakSpec(name="CC", center=284.8, amplitude=1000)]
        result = fit_spectrum(c1s_spectrum, peaks=peaks)
        assert result.residuals.shape == c1s_spectrum.energy.shape

    def test_component_curves_present(self, c1s_spectrum):
        peaks = [
            PeakSpec(name="CC", center=284.8, amplitude=1000),
            PeakSpec(name="CO", center=286.3, amplitude=400),
        ]
        result = fit_spectrum(c1s_spectrum, peaks=peaks)
        assert "CC" in result.component_curves
        assert "CO" in result.component_curves
