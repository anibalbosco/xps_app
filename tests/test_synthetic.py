"""Tests for synthetic spectrum generation."""

import numpy as np

from xpsanalysis.synthetic import generate_c1s, generate_fe2p


class TestGenerateC1s:
    def test_shape(self):
        spec = generate_c1s(noise_level=0.0, seed=0)
        assert spec.energy.shape == (500,)
        assert spec.intensity.shape == (500,)

    def test_energy_range(self):
        spec = generate_c1s()
        assert spec.energy.min() >= 282.0
        assert spec.energy.max() <= 292.0

    def test_metadata(self):
        spec = generate_c1s()
        assert spec.metadata.core_level == "C 1s"
        assert spec.metadata.photon_energy == 1486.6

    def test_peak_near_284_8(self):
        spec = generate_c1s(noise_level=0.0, seed=0)
        peak_idx = np.argmax(spec.intensity)
        peak_energy = spec.energy[peak_idx]
        assert abs(peak_energy - 284.8) < 0.5

    def test_reproducible_with_seed(self):
        s1 = generate_c1s(noise_level=0.05, seed=123)
        s2 = generate_c1s(noise_level=0.05, seed=123)
        np.testing.assert_array_equal(s1.intensity, s2.intensity)

    def test_noise(self):
        quiet = generate_c1s(noise_level=0.0, seed=0)
        noisy = generate_c1s(noise_level=0.1, seed=0)
        # They should differ due to noise
        assert not np.allclose(quiet.intensity, noisy.intensity)


class TestGenerateFe2p:
    def test_shape(self):
        spec = generate_fe2p(noise_level=0.0, seed=0)
        assert spec.energy.shape == (600,)

    def test_energy_range(self):
        spec = generate_fe2p()
        assert spec.energy.min() >= 705.0
        assert spec.energy.max() <= 735.0

    def test_metadata(self):
        spec = generate_fe2p()
        assert spec.metadata.core_level == "Fe 2p"

    def test_doublet_structure(self):
        spec = generate_fe2p(noise_level=0.0, seed=0)
        # Should have two local maxima (roughly)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(spec.intensity, distance=50, prominence=100)
        assert len(peaks) >= 2
