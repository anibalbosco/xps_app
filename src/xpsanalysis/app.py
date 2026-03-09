"""Streamlit web UI for XPS spectrum analysis."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="XPS Analysis", layout="wide")

# Periodic table layout: (row, col) for each element symbol
_PT_LAYOUT = [
    [("H",1),None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,("He",2)],
    [("Li",3),("Be",4),None,None,None,None,None,None,None,None,None,None,("B",5),("C",6),("N",7),("O",8),("F",9),("Ne",10)],
    [("Na",11),("Mg",12),None,None,None,None,None,None,None,None,None,None,("Al",13),("Si",14),("P",15),("S",16),("Cl",17),("Ar",18)],
    [("K",19),("Ca",20),("Sc",21),("Ti",22),("V",23),("Cr",24),("Mn",25),("Fe",26),("Co",27),("Ni",28),("Cu",29),("Zn",30),("Ga",31),("Ge",32),("As",33),("Se",34),("Br",35),("Kr",36)],
    [("Rb",37),("Sr",38),("Y",39),("Zr",40),("Nb",41),("Mo",42),("Tc",43),("Ru",44),("Rh",45),("Pd",46),("Ag",47),("Cd",48),("In",49),("Sn",50),("Sb",51),("Te",52),("I",53),("Xe",54)],
    [("Cs",55),("Ba",56),("La",57),("Hf",72),("Ta",73),("W",74),("Re",75),("Os",76),("Ir",77),("Pt",78),("Au",79),("Hg",80),("Tl",81),("Pb",82),("Bi",83),("Po",84),("At",85),("Rn",86)],
    [("Fr",87),("Ra",88),("Ac",89),("Th",90),("Pa",91),("U",92),None,None,None,None,None,None,None,None,None,None,None,None],
    [None,None,None,("Ce",58),("Pr",59),("Nd",60),("Pm",61),("Sm",62),("Eu",63),("Gd",64),("Tb",65),("Dy",66),("Ho",67),("Er",68),("Tm",69),("Yb",70),("Lu",71),None],
]


def _render_periodic_table():
    """Render clickable periodic table and element detail panel."""
    from xpsanalysis.reference import REFERENCE_DB, get_element
    from xpsanalysis.synthetic import _pseudo_voigt

    st.subheader("Periodic Table — XPS Reference")

    # Build set of elements in DB for highlighting
    db_symbols = {r.element_symbol for r in REFERENCE_DB}

    # Check loaded spectrum range for highlighting
    spec_range = st.session_state.get("_spectrum_range")

    if spec_range:
        from xpsanalysis.reference import get_core_levels_in_range
        in_range = {r.element_symbol for r in get_core_levels_in_range(*spec_range)}
    else:
        in_range = set()

    # Render grid
    for row in _PT_LAYOUT:
        cols = st.columns(18)
        for ci, cell in enumerate(row):
            if cell is None:
                cols[ci].write("")
                continue
            sym, z = cell
            if sym not in db_symbols:
                cols[ci].write(f":gray[{sym}]")
                continue
            label = f"**{sym}**" if sym in in_range else sym
            if cols[ci].button(label, key=f"pt_{sym}", use_container_width=True):
                st.session_state["_selected_element"] = sym

    # Element detail panel
    sel = st.session_state.get("_selected_element")
    if not sel:
        return

    refs = get_element(sel)
    if not refs:
        return

    st.markdown("---")
    r0 = refs[0]
    st.subheader(f"{r0.element_name} ({r0.element_symbol}) — Z={r0.atomic_number}")

    for ref in refs:
        dub_tag = " (doublet)" if ref.is_doublet else ""
        st.markdown(f"**{ref.orbital}** — {ref.binding_energy:.1f} eV{dub_tag}")
        if ref.is_doublet and ref.splitting:
            st.caption(f"Splitting: {ref.splitting:.1f} eV, ratio: {ref.branching_ratio}")

        if ref.chemical_states:
            from xpsanalysis.reference import CITATIONS
            rows = []
            for cs in ref.chemical_states:
                ref_short = cs.reference or ref.reference
                rows.append({"State": cs.name, "BE (eV)": f"{cs.binding_energy:.1f}",
                             "Description": cs.description,
                             "Reference": ref_short if ref_short else "Not known"})
            st.table(rows)

            # Show full citations for references used in this table
            used_refs = {cs.reference or ref.reference for cs in ref.chemical_states}
            used_refs.discard("")
            if used_refs:
                with st.expander("References"):
                    for key in sorted(used_refs):
                        citation = CITATIONS.get(key, key)
                        st.caption(f"**[{key}]** {citation}")

            # Plot synthetic reference
            e_min = min(cs.binding_energy for cs in ref.chemical_states) - 5
            e_max = max(cs.binding_energy for cs in ref.chemical_states) + 5
            x = np.linspace(e_min, e_max, 500)
            fig, ax = plt.subplots(figsize=(6, 2.5))
            total = np.zeros_like(x)
            for cs in ref.chemical_states:
                y = _pseudo_voigt(x, cs.binding_energy, ref.typical_sigma, 0.3, 100)
                ax.fill_between(x, y, alpha=0.3, label=cs.name)
                total += y
            ax.plot(x, total, "k-", linewidth=1, label="Sum")
            ax.set_xlabel("Binding Energy (eV)")
            ax.set_ylabel("Intensity (ref)")
            ax.invert_xaxis()
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(f"{sel} {ref.orbital} reference")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.markdown(
        "**Data sources:** Binding energies and chemical state assignments "
        "are compiled from the "
        "[NIST X-ray Photoelectron Spectroscopy Database](https://srdata.nist.gov/xps/) "
        "(Standard Reference Database 20, Version 4.1) and the "
        "*Handbook of X-ray Photoelectron Spectroscopy* "
        "(J.F. Moulder et al., Physical Electronics, 1995). "
        "These are secondary compilations — each entry in the NIST database "
        "includes full citations to the original experimental literature. "
        "For primary references behind a specific assignment, consult the "
        "NIST database entry for that element and core level."
    )


def _run_fit_section(spectrum, peaks, doublets, shared_fwhm_groups, bg_method,
                     cache_key, save_results_csv):
    """Render the fit button, results, and downloads."""
    from xpsanalysis.models import PeakSpec, DoubletSpec

    if st.button("Run Fit", type="primary"):
        if not peaks and not doublets:
            st.error("Define at least one peak or doublet.")
            return

        from xpsanalysis.fitting import fit_spectrum
        from xpsanalysis.report import plot_fit, generate_html_report

        with st.spinner("Fitting..."):
            result = fit_spectrum(
                spectrum, peaks=peaks, doublets=doublets,
                shared_fwhm_groups=shared_fwhm_groups,
                background_method=bg_method)

        st.session_state["last_fit_result"] = result
        st.session_state["last_fit_key"] = cache_key

    # Display stored result (persists across reruns)
    result = st.session_state.get("last_fit_result")
    fit_key = st.session_state.get("last_fit_key")
    if result is None or fit_key != cache_key:
        return

    from xpsanalysis.report import plot_fit, generate_html_report

    st.subheader("Fit Result")
    st.write(f"**R\u00b2** = {result.r_squared:.6f}")

    fig_fit = plot_fit(result)
    st.pyplot(fig_fit)
    plt.close(fig_fit)

    total_area = sum(result.component_areas.values())
    area_data = []
    for name, area in result.component_areas.items():
        frac = 100.0 * area / total_area if total_area > 0 else 0.0
        area_data.append({"Component": name, "Area": f"{area:.2f}",
                          "Fraction (%)": f"{frac:.1f}"})
    st.table(area_data)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f_csv:
            save_results_csv(f_csv.name, result)
            csv_content = Path(f_csv.name).read_text()
        st.download_button("Download CSV", csv_content,
                           file_name="fit_results.csv", mime="text/csv")
    with col_dl2:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f_html:
            generate_html_report(result, f_html.name)
            html_content = Path(f_html.name).read_text()
        st.download_button("Download HTML Report", html_content,
                           file_name="report.html", mime="text/html")


def main() -> None:
    tab_periodic, tab_analysis = st.tabs(["Periodic Table Reference", "Spectrum Analysis"])

    # ---- Sidebar: file upload and settings ----
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader(
            "Upload spectrum", type=["csv", "vms", "xy", "dat", "txt"])
        st.header("Background")
        bg_method = st.selectbox("Method", ["shirley", "tougaard"])

    # ---- Periodic Table tab (always available) ----
    with tab_periodic:
        _render_periodic_table()

    # ---- Analysis tab ----
    with tab_analysis:
        if uploaded_file is None:
            st.info("Upload a spectrum file to get started.")
            return
        _analysis_tab(uploaded_file, bg_method)


def _analysis_tab(uploaded_file, bg_method):
    from xpsanalysis.io import load_spectrum, save_results_csv
    from xpsanalysis.models import PeakSpec, DoubletSpec

    cache_key = f"spectra_{uploaded_file.name}_{uploaded_file.size}"
    if cache_key not in st.session_state:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.session_state[cache_key] = load_spectrum(tmp_path)

    spectra = st.session_state[cache_key]

    # Region selector
    region_idx = 0
    if len(spectra) > 1:
        labels = []
        for i, s in enumerate(spectra):
            lbl = f"Region {i}"
            if s.metadata.core_level:
                lbl += f" ({s.metadata.core_level})"
            labels.append(lbl)
        region_idx = st.sidebar.selectbox("Select region", range(len(spectra)),
                                          format_func=lambda i: labels[i])
    spectrum = spectra[region_idx]

    # Store range for periodic table highlighting
    st.session_state["_spectrum_range"] = (float(spectrum.energy.min()),
                                           float(spectrum.energy.max()))

    # ---- Spectrum plot + info ----
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Raw Spectrum")
        fig_raw, ax_raw = plt.subplots(figsize=(8, 4))
        ax_raw.plot(spectrum.energy, spectrum.intensity, "k-", linewidth=0.8)
        ax_raw.set_xlabel("Binding Energy (eV)")
        ax_raw.set_ylabel("Intensity")
        ax_raw.invert_xaxis()
        if spectrum.metadata.core_level:
            ax_raw.set_title(spectrum.metadata.core_level)
        st.pyplot(fig_raw)
        plt.close(fig_raw)
    with col2:
        st.subheader("Spectrum Info")
        st.write(f"**Points:** {len(spectrum.energy)}")
        st.write(f"**Energy range:** {spectrum.energy.min():.2f} – {spectrum.energy.max():.2f} eV")
        if spectrum.metadata.core_level:
            st.write(f"**Core level:** {spectrum.metadata.core_level}")
        if spectrum.metadata.photon_energy:
            st.write(f"**Photon energy:** {spectrum.metadata.photon_energy} eV")


    # ---- Auto-Identify (runs automatically on load) ----
    from xpsanalysis.identify import identify_spectrum

    id_key = f"auto_id_{cache_key}_{region_idx}"
    if id_key not in st.session_state:
        st.session_state[id_key] = identify_spectrum(spectrum)

    id_result = st.session_state[id_key]

    # Show identification results
    st.subheader("Peak Identification")
    if id_result and id_result.core_level_label:
        st.success(f"Identified: **{id_result.core_level_label}**")
        if id_result.assignments:
            rows = []
            for a in id_result.assignments:
                name = a.chemical_state.name if a.chemical_state else "elemental"
                rows.append({"Assignment": name,
                             "Position (eV)": f"{a.suggested_position:.1f}",
                             "Confidence": f"{a.confidence:.0%}"})
            st.table(rows)
    else:
        st.warning("Could not auto-identify peaks for this energy range.")

    # Use auto-identified peaks as defaults
    auto_peaks = id_result.suggested_peaks if id_result else []
    auto_doublets = id_result.suggested_doublets if id_result else []

    # ---- Sidebar: peak parameter forms (pre-populated from auto-id) ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Peak Parameters")

    default_n_peaks = len(auto_peaks) if auto_peaks else 0
    default_n_doublets = len(auto_doublets) if auto_doublets else 0

    # Use a version counter to force widget refresh when re-identifying
    pv = st.session_state.get("_param_version", 0)

    num_peaks = st.sidebar.number_input(
        "Single peaks", 0, 20, default_n_peaks, key=f"npk_v{pv}")
    num_doublets = st.sidebar.number_input(
        "Spin-orbit doublets", 0, 10, default_n_doublets, key=f"ndub_v{pv}")

    peaks: list[PeakSpec] = []
    doublets: list[DoubletSpec] = []
    shared_fwhm_groups: list[list[str]] = []
    peak_names: list[str] = []

    # Single peak forms
    for i in range(int(num_peaks)):
        ap = auto_peaks[i] if i < len(auto_peaks) else None
        d_name = ap.name if ap else f"P{i+1}"
        d_center = ap.center if ap else 285.0 + i * 1.5
        d_sigma = ap.sigma if ap else 0.6
        d_amp = ap.amplitude if ap else 500.0
        with st.sidebar.expander(f"Peak {i+1}: {d_name}", expanded=(i == 0)):
            name = st.text_input("Name", value=d_name, key=f"pn_{i}_v{pv}")
            center = st.number_input("Center (eV)", value=d_center, key=f"pc_{i}_v{pv}")
            sigma = st.number_input("Sigma (eV)", value=d_sigma, min_value=0.01, key=f"ps_{i}_v{pv}")
            fraction = st.slider("Fraction", 0.0, 1.0, 0.3, key=f"pf_{i}_v{pv}")
            amplitude = st.number_input("Amplitude", value=d_amp, min_value=0.0, key=f"pa_{i}_v{pv}")
            peaks.append(PeakSpec(name=name, center=center, sigma=sigma,
                                  fraction=fraction, amplitude=amplitude))
            peak_names.append(name)

    # Doublet forms
    for j in range(int(num_doublets)):
        ad = auto_doublets[j] if j < len(auto_doublets) else None
        d_name = ad.name if ad else f"D{j+1}"
        d_center = ad.center if ad else 710.0 + j * 15.0
        d_split = ad.splitting if ad else 13.0
        d_ratio = ad.ratio if ad else 0.5
        d_sigma = ad.sigma if ad else 1.0
        d_amp = ad.amplitude if ad else 1000.0
        with st.sidebar.expander(f"Doublet {j+1}: {d_name}", expanded=(j == 0)):
            name = st.text_input("Name", value=d_name, key=f"dn_{j}_v{pv}")
            center = st.number_input("Center (eV)", value=d_center, key=f"dc_{j}_v{pv}")
            split = st.number_input("Splitting (eV)", value=d_split, key=f"ds_{j}_v{pv}")
            ratio = st.number_input("Ratio", value=d_ratio,
                                    min_value=0.0, max_value=1.0, key=f"dr_{j}_v{pv}")
            sigma = st.number_input("Sigma (eV)", value=d_sigma,
                                    min_value=0.01, key=f"dsg_{j}_v{pv}")
            amplitude = st.number_input("Amplitude", value=d_amp,
                                        min_value=0.0, key=f"da_{j}_v{pv}")
            doublets.append(DoubletSpec(name=name, center=center, splitting=split,
                                        ratio=ratio, sigma=sigma, amplitude=amplitude))

    all_component_names = peak_names + [d.name for d in doublets]
    if len(all_component_names) >= 2:
        st.sidebar.markdown("---")
        if st.sidebar.checkbox("Share FWHM across components", key=f"sfwhm_v{pv}"):
            sel = st.sidebar.multiselect("Components to share", all_component_names,
                                         default=all_component_names[:2], key=f"sfwhm_sel_v{pv}")
            if len(sel) >= 2:
                shared_fwhm_groups.append(sel)

    # Re-identify button (forces fresh identification and resets widget values)
    if st.sidebar.button("Re-identify Peaks"):
        st.session_state[id_key] = identify_spectrum(spectrum)
        st.session_state["_param_version"] = pv + 1
        st.rerun()

    # ---- Run fit ----
    _run_fit_section(spectrum, peaks, doublets, shared_fwhm_groups, bg_method,
                     cache_key, save_results_csv)


main()
