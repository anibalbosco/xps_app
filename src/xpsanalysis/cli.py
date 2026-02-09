"""Typer CLI for XPS analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="xps",
    help="XPS spectrum analysis: background subtraction, peak fitting, and reporting.",
    add_completion=False,
)


@app.command()
def fit(
    spectrum_file: Path = typer.Argument(..., help="Path to the spectrum file (.csv, .vms, .xy, .dat, .txt)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="JSON config file with peak definitions"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-identify peaks when no config provided"),
    background: str = typer.Option("shirley", "--background", "-b", help="Background method: shirley or tougaard"),
    output: Path = typer.Option("report.pdf", "--output", "-o", help="Output report path (.pdf, .html, or .csv)"),
    region: int = typer.Option(0, "--region", "-r", help="Region index for multi-region files (e.g. VAMAS)"),
) -> None:
    """Fit peaks to an XPS spectrum and generate a report."""
    from xpsanalysis.io import load_spectrum, save_results_csv
    from xpsanalysis.models import PeakSpec, DoubletSpec
    from xpsanalysis.fitting import fit_spectrum
    from xpsanalysis.report import generate_pdf_report, generate_html_report

    spectra = load_spectrum(spectrum_file)
    if region >= len(spectra):
        typer.echo(f"Error: region index {region} out of range (file has {len(spectra)} region(s))", err=True)
        raise typer.Exit(code=1)
    spectrum = spectra[region]

    if config is not None:
        cfg = json.loads(config.read_text())
        peaks = [PeakSpec(**p) for p in cfg.get("peaks", [])]
        doublets = [DoubletSpec(**d) for d in cfg.get("doublets", [])]
        shared_fwhm_groups = cfg.get("shared_fwhm_groups", [])
    elif auto:
        from xpsanalysis.identify import identify_spectrum
        id_result = identify_spectrum(spectrum)
        peaks = id_result.suggested_peaks
        doublets = id_result.suggested_doublets
        shared_fwhm_groups = []
        label = id_result.core_level_label or "unknown"
        typer.echo(f"Auto-identified: {label} ({len(peaks)} peaks, {len(doublets)} doublets)")
    else:
        typer.echo("Error: provide --config or use --auto", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Fitting {spectrum_file} (region {region}) with {background} background...")
    result = fit_spectrum(
        spectrum,
        peaks=peaks,
        doublets=doublets,
        shared_fwhm_groups=shared_fwhm_groups,
        background_method=background,
    )

    typer.echo(f"R² = {result.r_squared:.6f}")
    for name, area in result.component_areas.items():
        typer.echo(f"  {name}: area = {area:.2f}")

    ext = output.suffix.lower()
    if ext == ".pdf":
        generate_pdf_report(result, output)
    elif ext in (".html", ".htm"):
        generate_html_report(result, output)
    elif ext == ".csv":
        save_results_csv(output, result)
    else:
        generate_pdf_report(result, output)

    typer.echo(f"Report saved to {output}")


@app.command()
def plot(
    spectrum_file: Path = typer.Argument(..., help="Path to the spectrum file"),
    output: Path = typer.Option("spectrum.png", "--output", "-o", help="Output plot path (.png, .pdf, .svg)"),
    region: int = typer.Option(0, "--region", "-r", help="Region index for multi-region files"),
) -> None:
    """Plot a raw XPS spectrum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xpsanalysis.io import load_spectrum

    spectra = load_spectrum(spectrum_file)
    if region >= len(spectra):
        typer.echo(f"Error: region index {region} out of range (file has {len(spectra)} region(s))", err=True)
        raise typer.Exit(code=1)
    spectrum = spectra[region]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(spectrum.energy, spectrum.intensity, "k-", linewidth=0.8)
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_ylabel("Intensity (arb. units)")
    ax.invert_xaxis()
    title = spectrum.metadata.core_level or spectrum_file.stem
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    typer.echo(f"Plot saved to {output}")


@app.command()
def generate(
    kind: str = typer.Argument(..., help="Spectrum type: c1s or fe2p"),
    noise: float = typer.Option(0.02, "--noise", "-n", help="Noise level (fraction of max intensity)"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    output: Path = typer.Option("synthetic.csv", "--output", "-o", help="Output CSV path"),
) -> None:
    """Generate a synthetic XPS spectrum."""
    from xpsanalysis.synthetic import generate_c1s, generate_fe2p

    kind_lower = kind.lower()
    if kind_lower == "c1s":
        spectrum = generate_c1s(noise_level=noise, seed=seed)
    elif kind_lower == "fe2p":
        spectrum = generate_fe2p(noise_level=noise, seed=seed)
    else:
        typer.echo(f"Error: unknown spectrum type '{kind}'. Choose c1s or fe2p.", err=True)
        raise typer.Exit(code=1)

    # Write as CSV
    import numpy as np

    header_lines = []
    if spectrum.metadata.core_level:
        header_lines.append(f"# core_level: {spectrum.metadata.core_level}")
    if spectrum.metadata.photon_energy:
        header_lines.append(f"# photon_energy: {spectrum.metadata.photon_energy}")

    with open(output, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("binding_energy_eV,intensity\n")
        for e, i in zip(spectrum.energy, spectrum.intensity):
            f.write(f"{e:.4f},{i:.4f}\n")

    typer.echo(f"Synthetic {kind} spectrum saved to {output}")


@app.command()
def info(
    spectrum_file: Path = typer.Argument(..., help="Path to the spectrum file"),
) -> None:
    """Print metadata, format, energy range, and point count for a spectrum file."""
    from xpsanalysis.io import load_spectrum

    spectra = load_spectrum(spectrum_file)
    ext = spectrum_file.suffix.lower()
    format_name = {
        ".csv": "CSV",
        ".vms": "VAMAS",
        ".xy": "Plain text",
        ".dat": "Plain text",
        ".txt": "Plain text",
    }.get(ext, f"Unknown ({ext})")

    typer.echo(f"File: {spectrum_file}")
    typer.echo(f"Format: {format_name}")
    typer.echo(f"Regions: {len(spectra)}")
    typer.echo()

    for idx, spec in enumerate(spectra):
        typer.echo(f"--- Region {idx} ---")
        typer.echo(f"  Points: {len(spec.energy)}")
        typer.echo(f"  Energy range: {spec.energy.min():.2f} – {spec.energy.max():.2f} eV")
        meta = spec.metadata
        if meta.core_level:
            typer.echo(f"  Core level: {meta.core_level}")
        if meta.photon_energy:
            typer.echo(f"  Photon energy: {meta.photon_energy} eV")
        if meta.pass_energy:
            typer.echo(f"  Pass energy: {meta.pass_energy} eV")
        if meta.analyzer:
            typer.echo(f"  Analyzer mode: {meta.analyzer}")
        if meta.sample_id:
            typer.echo(f"  Sample ID: {meta.sample_id}")


@app.command()
def identify(
    spectrum_file: Path = typer.Argument(..., help="Path to the spectrum file"),
    region: int = typer.Option(0, "--region", "-r", help="Region index for multi-region files"),
) -> None:
    """Auto-identify peaks in an XPS spectrum using the reference database."""
    from xpsanalysis.io import load_spectrum
    from xpsanalysis.identify import identify_spectrum

    spectra = load_spectrum(spectrum_file)
    if region >= len(spectra):
        typer.echo(f"Error: region index {region} out of range (file has {len(spectra)} region(s))", err=True)
        raise typer.Exit(code=1)
    spectrum = spectra[region]

    result = identify_spectrum(spectrum)
    if result.core_level_label:
        typer.echo(f"Identified: {result.core_level_label}")
    else:
        typer.echo("Could not identify core level.")
        return

    if result.assignments:
        typer.echo(f"\nSuggested assignments ({len(result.assignments)}):")
        for a in result.assignments:
            name = a.chemical_state.name if a.chemical_state else "elemental"
            typer.echo(f"  {name:20s}  {a.suggested_position:8.1f} eV  (confidence: {a.confidence:.0%})")

    if result.suggested_peaks:
        typer.echo(f"\nSuggested peaks ({len(result.suggested_peaks)}):")
        for p in result.suggested_peaks:
            typer.echo(f"  {p.name:20s}  center={p.center:.1f}  sigma={p.sigma:.2f}  amp={p.amplitude:.0f}")

    if result.suggested_doublets:
        typer.echo(f"\nSuggested doublets ({len(result.suggested_doublets)}):")
        for d in result.suggested_doublets:
            typer.echo(f"  {d.name:20s}  center={d.center:.1f}  split={d.splitting:.1f}  ratio={d.ratio:.2f}")


if __name__ == "__main__":
    app()
