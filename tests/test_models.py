"""Tests for peak model builders."""

import numpy as np

from xpsanalysis.models import (
    PeakSpec,
    DoubletSpec,
    build_composite_model,
    pseudo_voigt,
)


class TestPseudoVoigt:
    def test_peak_at_center(self):
        x = np.linspace(-5, 5, 1001)
        y = pseudo_voigt(x, amplitude=10.0, center=0.0, sigma=1.0, fraction=0.5)
        peak_idx = np.argmax(y)
        assert abs(x[peak_idx]) < 0.01
        np.testing.assert_allclose(y[peak_idx], 10.0, rtol=0.01)

    def test_pure_gaussian(self):
        x = np.linspace(-10, 10, 1001)
        y = pseudo_voigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.0)
        expected = np.exp(-np.log(2) * (x / 1.0) ** 2)
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_pure_lorentzian(self):
        x = np.linspace(-10, 10, 1001)
        y = pseudo_voigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=1.0)
        expected = 1.0 / (x**2 + 1.0)
        np.testing.assert_allclose(y, expected, atol=1e-10)


class TestBuildCompositeModel:
    def test_single_peak(self):
        peaks = [PeakSpec(name="CC", center=284.8)]
        model, params = build_composite_model(peaks)
        assert "CC_center" in params
        assert "CC_sigma" in params

    def test_two_peaks(self):
        peaks = [
            PeakSpec(name="CC", center=284.8),
            PeakSpec(name="CO", center=286.3),
        ]
        model, params = build_composite_model(peaks)
        assert "CC_center" in params
        assert "CO_center" in params

    def test_doublet_constraints(self):
        doublets = [DoubletSpec(name="Fe2p", center=710.8, splitting=13.6, ratio=0.5)]
        model, params = build_composite_model(doublets=doublets)
        # Peak b center should be constrained
        assert params["Fe2p_b_center"].expr is not None
        assert "Fe2p_a_center" in params["Fe2p_b_center"].expr
        # Peak b amplitude should be constrained
        assert params["Fe2p_b_amplitude"].expr is not None

    def test_shared_fwhm(self):
        peaks = [
            PeakSpec(name="CC", center=284.8, sigma=0.6),
            PeakSpec(name="CO", center=286.3, sigma=0.7),
        ]
        model, params = build_composite_model(peaks, shared_fwhm_groups=[["CC", "CO"]])
        # CO sigma should be constrained to CC sigma
        assert params["CO_sigma"].expr is not None
        assert "CC_sigma" in params["CO_sigma"].expr

    def test_fit_single_peak(self):
        x = np.linspace(280, 290, 200)
        y_true = pseudo_voigt(x, amplitude=500, center=284.8, sigma=0.6, fraction=0.3)
        y = y_true + np.random.default_rng(42).normal(0, 5, size=x.shape)

        peaks = [PeakSpec(name="CC", center=285.0, amplitude=400)]
        model, params = build_composite_model(peaks)
        result = model.fit(y, params, x=x)
        assert abs(result.params["CC_center"].value - 284.8) < 0.2

    def test_no_peaks_raises(self):
        import pytest
        with pytest.raises(ValueError, match="At least one peak"):
            build_composite_model()
