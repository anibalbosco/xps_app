"""Tests for spectrum auto-identification."""

import numpy as np

from xpsanalysis.io import XPSSpectrum, SpectrumMetadata
from xpsanalysis.identify import identify_spectrum, _detect_peaks


class TestIdentifySpectrum:
    def test_c1s_with_metadata(self, c1s_spectrum):
        result = identify_spectrum(c1s_spectrum)
        assert result.element_symbol == "C"
        assert result.core_level_label is not None
        assert "C" in result.core_level_label
        assert len(result.assignments) >= 1

    def test_c1s_suggested_peaks(self, c1s_spectrum):
        result = identify_spectrum(c1s_spectrum)
        centers = [p.center for p in result.suggested_peaks]
        assert any(abs(c - 284.8) < 1.5 for c in centers)

    def test_fe2p_suggests_doublet(self, fe2p_spectrum):
        result = identify_spectrum(fe2p_spectrum)
        assert len(result.suggested_doublets) >= 1
        d = result.suggested_doublets[0]
        assert abs(d.splitting - 13.6) < 1.0

    def test_doublet_per_chemical_state(self, fe2p_spectrum):
        """Each chemical state in a doublet orbital gets its own DoubletSpec."""
        result = identify_spectrum(fe2p_spectrum)
        # Fe 2p has multiple chemical states; each should be a doublet
        assert len(result.suggested_doublets) == len(result.assignments)
        for d in result.suggested_doublets:
            assert d.splitting > 0
            assert d.ratio > 0

    def test_c1s_no_doublets(self, c1s_spectrum):
        """C 1s is not a doublet orbital — should produce only single peaks."""
        result = identify_spectrum(c1s_spectrum)
        assert len(result.suggested_doublets) == 0
        assert len(result.suggested_peaks) >= 1

    def test_no_metadata_narrow_scan(self):
        energy = np.linspace(280, 295, 300)
        intensity = 100 + 500 * np.exp(-0.5 * ((energy - 284.8) / 0.8) ** 2)
        spectrum = XPSSpectrum(energy=energy, intensity=intensity)
        result = identify_spectrum(spectrum)
        assert result.element_symbol == "C"

    def test_valid_confidence(self, c1s_spectrum):
        result = identify_spectrum(c1s_spectrum)
        for a in result.assignments:
            assert 0.0 <= a.confidence <= 1.0


class TestDetectPeaks:
    def test_find_peaks(self):
        energy = np.linspace(280, 295, 500)
        intensity = (300 * np.exp(-((energy - 284.8) / 0.6) ** 2)
                     + 100 * np.exp(-((energy - 286.5) / 0.6) ** 2))
        peaks = _detect_peaks(energy, intensity)
        positions = [p[0] for p in peaks]
        assert any(abs(p - 284.8) < 0.5 for p in positions)

    def test_empty_spectrum(self):
        assert _detect_peaks(np.array([1.0]), np.array([1.0])) == []
