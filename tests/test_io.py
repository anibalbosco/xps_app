"""Tests for I/O: CSV loading, VAMAS parsing, text format, and save_results_csv."""

import numpy as np
import pytest
from pathlib import Path

from xpsanalysis.io import XPSSpectrum, SpectrumMetadata, load_spectrum, save_results_csv
from xpsanalysis.formats.csv_format import load_csv
from xpsanalysis.formats.text_format import load_text
from xpsanalysis.formats.vamas import load_vamas


class TestXPSSpectrum:
    def test_valid(self):
        s = XPSSpectrum(energy=[1, 2, 3], intensity=[10, 20, 30])
        assert len(s.energy) == 3
        assert s.energy.dtype == np.float64

    def test_mismatched_shape(self):
        with pytest.raises(ValueError, match="same shape"):
            XPSSpectrum(energy=[1, 2], intensity=[1, 2, 3])

    def test_non_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            XPSSpectrum(energy=[[1, 2]], intensity=[[1, 2]])


class TestCSVFormat:
    def test_round_trip(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "# core_level: C 1s\n"
            "# photon_energy: 1486.6\n"
            "binding_energy_eV,intensity\n"
            "284.0,100\n"
            "285.0,200\n"
            "286.0,150\n"
        )
        spectra = load_csv(csv_path)
        assert len(spectra) == 1
        s = spectra[0]
        assert len(s.energy) == 3
        np.testing.assert_allclose(s.energy, [284.0, 285.0, 286.0])
        np.testing.assert_allclose(s.intensity, [100, 200, 150])
        assert s.metadata.core_level == "C 1s"
        assert s.metadata.photon_energy == 1486.6

    def test_no_header_row(self, tmp_path):
        csv_path = tmp_path / "noheader.csv"
        csv_path.write_text("284.0,100\n285.0,200\n")
        spectra = load_csv(csv_path)
        assert len(spectra[0].energy) == 2

    def test_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        with pytest.raises(ValueError):
            load_csv(csv_path)


class TestTextFormat:
    def test_whitespace_delimited(self, tmp_path):
        path = tmp_path / "test.xy"
        path.write_text(
            "# core_level: O 1s\n"
            "530.0  1000\n"
            "531.0  1200\n"
            "532.0  800\n"
        )
        spectra = load_text(path)
        assert len(spectra) == 1
        np.testing.assert_allclose(spectra[0].energy, [530.0, 531.0, 532.0])
        assert spectra[0].metadata.core_level == "O 1s"

    def test_tab_delimited(self, tmp_path):
        path = tmp_path / "test.dat"
        path.write_text("530.0\t1000\n531.0\t1200\n")
        spectra = load_text(path)
        assert len(spectra[0].energy) == 2

    def test_percent_comments(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("% This is a comment\n284.0 100\n285.0 200\n")
        spectra = load_text(path)
        assert len(spectra[0].energy) == 2

    def test_multi_spectrum_blank_line_separator(self, tmp_path):
        path = tmp_path / "multi.xy"
        path.write_text(
            "284.0 100\n"
            "285.0 200\n"
            "\n"
            "530.0 300\n"
            "531.0 400\n"
            "532.0 500\n"
        )
        spectra = load_text(path)
        assert len(spectra) == 2
        assert len(spectra[0].energy) == 2
        assert len(spectra[1].energy) == 3
        np.testing.assert_allclose(spectra[0].energy, [284.0, 285.0])
        np.testing.assert_allclose(spectra[1].energy, [530.0, 531.0, 532.0])

    def test_multi_spectrum_header_separator(self, tmp_path):
        path = tmp_path / "multi.xy"
        path.write_text(
            "# core_level: C 1s\n"
            "284.0 100\n"
            "285.0 200\n"
            "# core_level: O 1s\n"
            "530.0 300\n"
            "531.0 400\n"
        )
        spectra = load_text(path)
        assert len(spectra) == 2
        assert spectra[0].metadata.core_level == "C 1s"
        assert spectra[1].metadata.core_level == "O 1s"
        np.testing.assert_allclose(spectra[0].energy, [284.0, 285.0])
        np.testing.assert_allclose(spectra[1].energy, [530.0, 531.0])

    def test_multi_spectrum_blank_line_and_header(self, tmp_path):
        path = tmp_path / "multi.dat"
        path.write_text(
            "# core_level: C 1s\n"
            "284.0 100\n"
            "285.0 200\n"
            "\n"
            "# core_level: N 1s\n"
            "398.0 50\n"
            "399.0 80\n"
            "\n"
            "# core_level: O 1s\n"
            "530.0 300\n"
            "531.0 400\n"
        )
        spectra = load_text(path)
        assert len(spectra) == 3
        assert spectra[0].metadata.core_level == "C 1s"
        assert spectra[1].metadata.core_level == "N 1s"
        assert spectra[2].metadata.core_level == "O 1s"

    def test_single_spectrum_unchanged(self, tmp_path):
        """Existing single-spectrum files still return a single-element list."""
        path = tmp_path / "single.xy"
        path.write_text("# core_level: C 1s\n284.0 100\n285.0 200\n286.0 150\n")
        spectra = load_text(path)
        assert len(spectra) == 1


class TestVAMAS:
    def _make_vamas_content(self):
        """Build a minimal VAMAS file content with one block."""
        lines = [
            "VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4",
            "Test Institution",
            "Test Model",
            "Test Operator",
            "Test Experiment",
            "1",                    # number of comment lines
            "Test comment",
            "NORM",                 # experiment mode
            "REGULAR",              # scan mode
            "1",                    # number of spectral regions
            "1",                    # number of experimental variables
            "counts",              # exp var label
            "d",                   # exp var unit
            "0",                    # number of inclusion entries
            "0",                    # num manually entered items
            "0",                    # num future upgrade experiment entries
            "0",                    # num future upgrade block entries
            "1",                    # number of blocks
            # --- Block ---
            "C 1s",                # block identifier
            "Sample A",            # sample id
            "2024", "1", "15", "10", "30", "0", "0",  # date/time + tz
            "0",                    # block comment lines
            "XPS",                 # technique
            "Al Ka",               # param 0: source label
            "1486.6",              # param 1: source energy
            "300",                 # param 2: source strength
            "0", "0",              # params 3-4: beam widths x,y
            # params 5-10: FOV + linescan (MAP only, skipped for NORM)
            "45.0",                # param 11: incidence polar angle
            "0.0",                 # param 12: azimuth
            "FAT",                 # param 13: analyzer mode
            "20",                  # param 14: pass energy
            "1",                   # param 15: magnification
            "4.5",                 # param 16: work function
            "0",                   # param 17: target bias
            "Binding Energy",      # x label
            "eV",                  # x units
            "284.0",               # x start
            "0.1",                 # x step
            "1",                    # num corresponding variables
            "counts",              # y label
            "d",                   # y unit
            "pulse counting",      # signal mode
            "0.1",                 # dwell time
            "1",                    # num scans
            "1.0",                 # signal collection time
            "5",                    # num y values
            "0",                    # num additional params
            # y-values
            "100", "200", "300", "200", "100",
        ]
        return "\n".join(lines) + "\n"

    def test_parse_vamas(self, tmp_path):
        path = tmp_path / "test.vms"
        path.write_text(self._make_vamas_content())
        spectra = load_vamas(path)
        assert len(spectra) == 1
        s = spectra[0]
        assert len(s.energy) == 5
        np.testing.assert_allclose(s.energy, [284.0, 284.1, 284.2, 284.3, 284.4])
        np.testing.assert_allclose(s.intensity, [100, 200, 300, 200, 100])
        assert s.metadata.core_level == "C 1s"
        assert s.metadata.photon_energy == 1486.6
        assert s.metadata.pass_energy == 20.0
        assert s.metadata.analyzer == "FAT"
        assert s.metadata.sample_id == "Sample A"


class TestLoadSpectrum:
    def test_csv_dispatch(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("binding_energy_eV,intensity\n284.0,100\n285.0,200\n")
        spectra = load_spectrum(path)
        assert len(spectra) == 1

    def test_txt_dispatch(self, tmp_path):
        path = tmp_path / "test.xy"
        path.write_text("284.0 100\n285.0 200\n")
        spectra = load_spectrum(path)
        assert len(spectra) == 1

    def test_unknown_extension_warns(self, tmp_path):
        path = tmp_path / "test.foo"
        path.write_text("284.0 100\n285.0 200\n")
        with pytest.warns(UserWarning, match="Unknown extension"):
            spectra = load_spectrum(path)
        assert len(spectra) == 1
