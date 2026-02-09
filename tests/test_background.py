"""Tests for background subtraction algorithms."""

import numpy as np
import pytest

from xpsanalysis.background import shirley_background, tougaard_background
from xpsanalysis.energy import ensure_increasing


class TestShirleyBackground:
    def test_converges(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg, converged = shirley_background(spec.energy, spec.intensity)
        assert converged

    def test_endpoints(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg, _ = shirley_background(spec.energy, spec.intensity)
        # Background at low-BE end should roughly match spectrum
        np.testing.assert_allclose(bg[0], spec.intensity[0], rtol=0.05)
        # Background at high-BE end should roughly match spectrum
        np.testing.assert_allclose(bg[-1], spec.intensity[-1], rtol=0.05)

    def test_monotonic_for_step_background(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg, _ = shirley_background(spec.energy, spec.intensity)
        # Background should be generally non-increasing from low to high BE
        # (allowing small deviations near peaks)
        diff = np.diff(bg)
        # Most points should show non-increasing trend
        assert np.sum(diff <= 0.1) > len(diff) * 0.8

    def test_shape(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg, _ = shirley_background(spec.energy, spec.intensity)
        assert bg.shape == spec.intensity.shape


class TestTougaardBackground:
    def test_shape(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg = tougaard_background(spec.energy, spec.intensity)
        assert bg.shape == spec.intensity.shape

    def test_non_negative(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg = tougaard_background(spec.energy, spec.intensity)
        assert np.all(bg >= 0)

    def test_smooth(self, c1s_noiseless):
        spec = ensure_increasing(c1s_noiseless)
        bg = tougaard_background(spec.energy, spec.intensity)
        # The background should be smoother than the raw data
        bg_roughness = np.std(np.diff(bg))
        data_roughness = np.std(np.diff(spec.intensity))
        assert bg_roughness < data_roughness
