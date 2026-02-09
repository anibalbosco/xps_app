"""File format readers for XPS spectra."""

from xpsanalysis.formats.csv_format import load_csv
from xpsanalysis.formats.vamas import load_vamas
from xpsanalysis.formats.text_format import load_text

__all__ = ["load_csv", "load_vamas", "load_text"]
