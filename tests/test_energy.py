"""Tests for energy axis utilities."""

import numpy as np

from xpsanalysis.io import XPSSpectrum
from xpsanalysis.energy import ensure_increasing, calibrate, reverse_energy


class TestEnsureIncreasing:
    def test_already_ascending(self):
        s = XPSSpectrum(energy=[1, 2, 3], intensity=[10, 20, 30])
        result = ensure_increasing(s)
        np.testing.assert_array_equal(result.energy, [1, 2, 3])

    def test_descending(self):
        s = XPSSpectrum(energy=[3, 2, 1], intensity=[30, 20, 10])
        result = ensure_increasing(s)
        np.testing.assert_array_equal(result.energy, [1, 2, 3])
        np.testing.assert_array_equal(result.intensity, [10, 20, 30])

    def test_unsorted(self):
        s = XPSSpectrum(energy=[2, 3, 1], intensity=[20, 30, 10])
        result = ensure_increasing(s)
        np.testing.assert_array_equal(result.energy, [1, 2, 3])
        np.testing.assert_array_equal(result.intensity, [10, 20, 30])


class TestCalibrate:
    def test_shift(self):
        s = XPSSpectrum(energy=[284.0, 285.0, 286.0], intensity=[1, 2, 3])
        result = calibrate(s, 0.5)
        np.testing.assert_allclose(result.energy, [284.5, 285.5, 286.5])
        np.testing.assert_array_equal(result.intensity, [1, 2, 3])

    def test_negative_shift(self):
        s = XPSSpectrum(energy=[284.0, 285.0], intensity=[1, 2])
        result = calibrate(s, -1.0)
        np.testing.assert_allclose(result.energy, [283.0, 284.0])


class TestReverseEnergy:
    def test_reverse(self):
        s = XPSSpectrum(energy=[1, 2, 3], intensity=[10, 20, 30])
        result = reverse_energy(s)
        np.testing.assert_array_equal(result.energy, [3, 2, 1])
        np.testing.assert_array_equal(result.intensity, [30, 20, 10])
