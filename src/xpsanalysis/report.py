"""Report generation: matplotlib plots and HTML/PDF reports."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from xpsanalysis.fitting import FitResult

_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>XPS Fit Report</title>
<style>
body { font-family: sans-serif; max-width: 900px; margin: 2em auto; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: right; }
th { background: #f0f0f0; text-align: left; }
img { max-width: 100%; }
h1, h2 { color: #333; }
.meta { color: #666; font-size: 0.9em; }
</style>
</head>
<body>
<h1>XPS Fit Report</h1>
<p class="meta">Background method: {{ background_method }} &mdash;
R&sup2; = {{ r_squared }}</p>

<h2>Fit Plot</h2>
<img src="data:image/png;base64,{{ plot_b64 }}" alt="Fit plot">

<h2>Component Areas</h2>
<table>
<tr><th>Component</th><th>Area</th><th>Fraction (%)</th></tr>
{% for name, area in areas %}
<tr><td>{{ name }}</td><td>{{ "%.2f"|format(area) }}</td><td>{{ "%.1f"|format(frac[name]) }}</td></tr>
{% endfor %}
</table>

<h2>Fit Parameters</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
{% for name, value in params %}
<tr><td>{{ name }}</td><td>{{ "%.4f"|format(value) }}</td></tr>
{% endfor %}
</table>
</body>
</html>
""")


def plot_fit(result: FitResult, ax: plt.Axes | None = None) -> plt.Figure:
    """Plot the fit result: raw data, background, envelope, components, residuals.

    Uses inverted x-axis (XPS convention: high BE on left).
    """
    energy = result.spectrum.energy
    intensity = result.spectrum.intensity
    bg = result.background
    envelope = result.model_result.best_fit + bg

    if ax is not None:
        fig = ax.get_figure()
        ax_main = ax
        ax_res = None
    else:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True,
            gridspec_kw={"hspace": 0.05},
        )

    # Main plot
    ax_main.plot(energy, intensity, "k.", markersize=2, label="Data", zorder=5)
    ax_main.plot(energy, bg, "b--", linewidth=1, label="Background")
    ax_main.plot(energy, envelope, "r-", linewidth=1.5, label="Envelope")

    for name, curve in result.component_curves.items():
        ax_main.fill_between(energy, bg, curve + bg, alpha=0.3, label=name)

    ax_main.set_ylabel("Intensity (arb. units)")
    ax_main.legend(fontsize=8)
    ax_main.invert_xaxis()

    # Residuals subplot
    if ax_res is not None:
        ax_res.plot(energy, result.residuals, "g-", linewidth=0.8)
        ax_res.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax_res.set_xlabel("Binding Energy (eV)")
        ax_res.set_ylabel("Residual")
        # Don't call invert_xaxis() here — sharex=True inherits it from ax_main

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fig.tight_layout()
        except (ValueError, UserWarning):
            pass
    return fig


def generate_pdf_report(result: FitResult, output_path: str | Path) -> None:
    """Save the fit result as a PDF (matplotlib figure + parameter table)."""
    output_path = Path(output_path)

    fig = plot_fit(result)

    # Create a second page with parameters
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.axis("off")

    lines = [
        f"Background method: {result.background_method}",
        f"R² = {result.r_squared:.6f}",
        "",
        "Component Areas:",
    ]
    total_area = sum(result.component_areas.values())
    for name, area in result.component_areas.items():
        frac = 100.0 * area / total_area if total_area > 0 else 0.0
        lines.append(f"  {name}: {area:.2f}  ({frac:.1f}%)")
    lines.append("")
    lines.append("Fit Parameters:")
    for pname, pval in result.fit_params.items():
        lines.append(f"  {pname}: {pval:.4f}")

    ax2.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax2.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
    )

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)

    plt.close(fig)
    plt.close(fig2)


def generate_html_report(result: FitResult, output_path: str | Path) -> None:
    """Generate an HTML report with embedded plot and parameter tables."""
    output_path = Path(output_path)

    fig = plot_fit(result)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("ascii")

    total_area = sum(result.component_areas.values())
    frac = {}
    for name, area in result.component_areas.items():
        frac[name] = 100.0 * area / total_area if total_area > 0 else 0.0

    html = _HTML_TEMPLATE.render(
        background_method=result.background_method,
        r_squared=f"{result.r_squared:.6f}",
        plot_b64=plot_b64,
        areas=list(result.component_areas.items()),
        frac=frac,
        params=list(result.fit_params.items()),
    )

    output_path.write_text(html, encoding="utf-8")
