"""VAMAS / .vms reader (ISO 14976 standard).

Parses the ASCII VAMAS format line-by-line: header section followed by
N data blocks, each representing one spectral region.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xpsanalysis.io import SpectrumMetadata, XPSSpectrum

# Experiment modes that include spatial coordinates per block
_MAP_MODES = {"MAP", "MAPDP"}
_SDP_MODES = {"SDP", "SDPSV"}
_SPATIAL_MODES = _MAP_MODES | _SDP_MODES


def load_vamas(path: str | Path) -> list[XPSSpectrum]:
    """Parse a VAMAS file and return one XPSSpectrum per block."""
    path = Path(path)
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f]

    reader = _LineReader(lines)
    spectra = _parse_vamas(reader)
    if not spectra:
        raise ValueError(f"No spectral blocks found in {path}")
    return spectra


class _LineReader:
    """Stateful line-by-line reader with peek support."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._pos = 0

    def next(self) -> str:
        if self._pos >= len(self._lines):
            raise EOFError("Unexpected end of VAMAS file")
        line = self._lines[self._pos]
        self._pos += 1
        return line

    def next_int(self) -> int:
        return int(self.next().strip())

    def next_float(self) -> float:
        return float(self.next().strip())

    def peek(self) -> str:
        """Look at the next line without consuming it."""
        if self._pos >= len(self._lines):
            raise EOFError("Unexpected end of VAMAS file")
        return self._lines[self._pos]

    @property
    def remaining(self) -> int:
        return len(self._lines) - self._pos


def _parse_vamas(reader: _LineReader) -> list[XPSSpectrum]:
    """Parse the full VAMAS structure: header → blocks."""
    # --- Header ---
    format_id = reader.next()  # "VAMAS Surface Chemical Analysis Standard ..."
    institution_id = reader.next()
    instrument_model = reader.next()
    operator_id = reader.next()
    experiment_id = reader.next()

    # Number of comment lines in header, then skip them
    num_comment_lines = reader.next_int()
    for _ in range(num_comment_lines):
        reader.next()

    experiment_mode = reader.next().strip()  # "NORM", "MAP", "SDP", etc.
    scan_mode = reader.next().strip()  # "REGULAR" or "IRREGULAR"

    # Number of spectral regions (for MAP/SDP) or specific experimental entries
    num_spectral_regions = reader.next_int()

    # Number of experimental variables (usually 1)
    num_exp_variables = reader.next_int()
    # Read experimental variable labels and units
    for _ in range(num_exp_variables):
        reader.next()  # label
        reader.next()  # unit

    # Number of entries in parameter inclusion/exclusion list
    num_inclusion = reader.next_int()
    inclusion_list = []
    for _ in range(num_inclusion):
        inclusion_list.append(reader.next_int())

    # Number of manually entered items in block
    num_manual_items = reader.next_int()

    # Number of future upgrade experiment entries
    num_future_exp = reader.next_int()
    for _ in range(num_future_exp):
        reader.next()

    # Number of future upgrade block entries
    num_future_block = reader.next_int()

    # Number of blocks
    num_blocks = reader.next_int()

    # --- Parse each block ---
    spectra = []
    for _block_idx in range(num_blocks):
        spectrum = _parse_block(
            reader,
            experiment_mode=experiment_mode,
            scan_mode=scan_mode,
            num_exp_variables=num_exp_variables,
            inclusion_list=inclusion_list,
            num_future_block=num_future_block,
        )
        if spectrum is not None:
            spectra.append(spectrum)

    return spectra


def _included(inclusion_list: list[int], param_index: int) -> bool:
    """Check whether a parameter is included.

    If the inclusion list is empty, all parameters are included (default).
    Parameters are 0-indexed into the list.
    """
    if not inclusion_list:
        return True
    if param_index >= len(inclusion_list):
        return True
    return inclusion_list[param_index] != 0


def _parse_block(
    reader: _LineReader,
    experiment_mode: str,
    scan_mode: str,
    num_exp_variables: int,
    inclusion_list: list[int],
    num_future_block: int,
) -> XPSSpectrum | None:
    """Parse a single VAMAS data block per ISO 14976."""

    # --- Always-present fields ---
    block_id = reader.next().strip()       # Block identifier (often core level)
    sample_id = reader.next().strip()      # Sample identifier

    # Date/time fields
    year = reader.next_int()
    month = reader.next_int()
    day = reader.next_int()
    hours = reader.next_int()
    minutes = reader.next_int()
    seconds = reader.next_int()
    num_hours_advance_gmt = reader.next_int()

    # Block comments
    num_block_comments = reader.next_int()
    block_comments = []
    for _ in range(num_block_comments):
        block_comments.append(reader.next())

    technique = reader.next().strip()  # "XPS", "AES", "UPS", etc.

    # --- Mode-dependent: spatial coordinates ---
    if experiment_mode.upper() in _MAP_MODES:
        _x_coord = reader.next_int()
        _y_coord = reader.next_int()

    # Experimental variable values (for MAP/SDP modes)
    if experiment_mode.upper() in _SPATIAL_MODES:
        for _ in range(num_exp_variables):
            reader.next_float()

    # --- Inclusion-list-controlled parameters ---
    # Parameter 0: analysis source label
    source_label = ""
    if _included(inclusion_list, 0):
        source_label = reader.next().strip()

    # Parameter 1: analysis source characteristic energy
    source_energy = 0.0
    if _included(inclusion_list, 1):
        source_energy = reader.next_float()

    # Parameter 2: analysis source strength
    if _included(inclusion_list, 2):
        source_strength = reader.next_float()

    # Parameter 3: analysis source beam width x
    if _included(inclusion_list, 3):
        reader.next_float()

    # Parameter 4: analysis source beam width y
    if _included(inclusion_list, 4):
        reader.next_float()

    # Parameters 5-10: MAP/MAPDP only (field of view + linescan coordinates)
    if experiment_mode.upper() in _MAP_MODES:
        # Parameter 5: field of view x
        if _included(inclusion_list, 5):
            reader.next_float()
        # Parameter 6: field of view y
        if _included(inclusion_list, 6):
            reader.next_float()
        # Parameters 7-10: linescan positions
        if _included(inclusion_list, 7):
            reader.next_int()  # first linescan x
        if _included(inclusion_list, 8):
            reader.next_int()  # first linescan y
        if _included(inclusion_list, 9):
            reader.next_int()  # last linescan x
        if _included(inclusion_list, 10):
            reader.next_int()  # last linescan y

    # Parameter 11: analysis source polar angle of incidence
    if _included(inclusion_list, 11):
        reader.next_float()

    # Parameter 12: analysis source azimuth
    if _included(inclusion_list, 12):
        reader.next_float()

    # Parameter 13: analyzer mode
    analyzer_mode = ""
    if _included(inclusion_list, 13):
        analyzer_mode = reader.next().strip()  # "FAT", "FRR", etc.

    # Parameter 14: pass energy / retard ratio / kinetic energy
    pass_energy = 0.0
    if _included(inclusion_list, 14):
        pass_energy = reader.next_float()

    # Parameter 15: magnification of analyser transfer lens
    if _included(inclusion_list, 15):
        reader.next_float()

    # Parameter 16: analyser work function or acceptance energy
    if _included(inclusion_list, 16):
        reader.next_float()

    # Parameter 17: target bias
    if _included(inclusion_list, 17):
        reader.next_float()

    # --- X-axis definition (always present) ---
    x_label = reader.next().strip()
    x_units = reader.next().strip()
    x_start = reader.next_float()
    x_step = reader.next_float()

    # --- Corresponding variables (y-axis) ---
    num_corresponding_variables = reader.next_int()

    y_labels = []
    y_units = []
    for _ in range(num_corresponding_variables):
        y_labels.append(reader.next().strip())
        y_units.append(reader.next().strip())

    # --- Signal parameters ---
    signal_mode = reader.next().strip()
    dwell_time = reader.next_float()
    num_scans = reader.next_int()

    # Signal collection time
    signal_collection_time = reader.next_float()

    # --- Data dimensions ---
    num_y_values = reader.next_int()

    # Additional numeric parameters (min/max pairs per corresponding variable)
    num_additional = reader.next_int()
    for _ in range(num_additional):
        reader.next()  # min value
        reader.next()  # max value

    # Future upgrade block entries
    for _ in range(num_future_block):
        reader.next()

    # --- Read y-values ---
    y_values = np.empty(num_y_values, dtype=np.float64)
    for i in range(num_y_values):
        y_values[i] = reader.next_float()

    # --- Build energy axis ---
    energy = np.arange(num_y_values, dtype=np.float64) * x_step + x_start

    metadata = SpectrumMetadata(
        pass_energy=pass_energy if pass_energy > 0 else None,
        analyzer=analyzer_mode if analyzer_mode else None,
        photon_energy=source_energy if source_energy > 0 else None,
        core_level=block_id if block_id else None,
        sample_id=sample_id if sample_id else None,
        scan_mode=signal_mode if signal_mode else None,
    )

    return XPSSpectrum(energy=energy, intensity=y_values, metadata=metadata)
