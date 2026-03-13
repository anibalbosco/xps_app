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
    from xpsanalysis.reference import REFERENCE_DB, AUGER_DB, XRAY_SOURCES, get_element, get_auger_lines
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

            # Build markdown table with clickable reference links
            header = "| State | BE (eV) | Description | Reference |\n"
            header += "|---|---|---|---|\n"
            table_rows = ""
            for cs in ref.chemical_states:
                ref_key = cs.reference or ref.reference
                if ref_key:
                    entry = CITATIONS.get(ref_key)
                    doi = entry.get("doi") if entry else None
                    if doi:
                        ref_cell = f"[{ref_key}](https://doi.org/{doi})"
                    else:
                        ref_cell = ref_key
                else:
                    ref_cell = "Not known"
                desc = cs.description or ""
                table_rows += f"| {cs.name} | {cs.binding_energy:.1f} | {desc} | {ref_cell} |\n"
            st.markdown(header + table_rows)

            # Show full citations for references used in this table
            used_refs = {cs.reference or ref.reference for cs in ref.chemical_states}
            used_refs.discard("")
            if used_refs:
                for key in sorted(used_refs):
                    entry = CITATIONS.get(key)
                    if entry:
                        citation = entry["citation"]
                        doi = entry.get("doi")
                        if doi:
                            st.caption(
                                f"**[{key}]** {citation} "
                                f"[doi:{doi}](https://doi.org/{doi})"
                            )
                        else:
                            st.caption(f"**[{key}]** {citation}")

            # Plot synthetic reference
            e_min = min(cs.binding_energy for cs in ref.chemical_states) - 5
            e_max = max(cs.binding_energy for cs in ref.chemical_states) + 5
            if ref.is_doublet and ref.splitting:
                e_max = max(e_max, max(cs.binding_energy + ref.splitting
                            for cs in ref.chemical_states) + 5)
            x = np.linspace(e_min, e_max, 500)
            fig, ax = plt.subplots(figsize=(6, 2.5))
            total = np.zeros_like(x)
            for cs in ref.chemical_states:
                amp = 100
                y = _pseudo_voigt(x, cs.binding_energy, ref.typical_sigma, 0.3, amp)
                if ref.is_doublet and ref.splitting and ref.branching_ratio:
                    partner_amp = amp * ref.branching_ratio
                    y_partner = _pseudo_voigt(x, cs.binding_energy + ref.splitting,
                                             ref.typical_sigma, 0.3, partner_amp)
                    y_combined = y + y_partner
                    ax.fill_between(x, y_combined, alpha=0.3, label=cs.name)
                    total += y_combined
                else:
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

    # Auger lines for this element
    auger_lines = get_auger_lines(sel)
    if auger_lines:
        st.markdown("---")
        st.markdown("**Auger Lines**")
        photon_energy = XRAY_SOURCES["Al Kα (1486.6 eV)"]
        header = "| Transition | KE (eV) | Apparent BE (eV) | σ (eV) |\n"
        header += "|---|---|---|---|\n"
        rows = ""
        for aug in auger_lines:
            app_be = photon_energy - aug.kinetic_energy
            rows += (f"| {aug.element_symbol} {aug.transition} "
                     f"| {aug.kinetic_energy:.1f} | {app_be:.1f} "
                     f"| {aug.typical_sigma:.1f} |\n")
        st.markdown(header + rows)
        st.caption("Apparent BE calculated for Al Kα (1486.6 eV). "
                   "Kinetic energy is source-independent.")

        # Plot Auger reference peaks
        all_app_be = [photon_energy - a.kinetic_energy for a in auger_lines]
        e_min = min(all_app_be) - 10
        e_max = max(all_app_be) + 10
        x = np.linspace(e_min, e_max, 500)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        total = np.zeros_like(x)
        for aug in auger_lines:
            app_be = photon_energy - aug.kinetic_energy
            y = _pseudo_voigt(x, app_be, aug.typical_sigma, 0.5, 100 * aug.relative_intensity)
            ax.fill_between(x, y, alpha=0.3, label=f"{aug.transition} (KE={aug.kinetic_energy:.0f})")
            total += y
        ax.plot(x, total, "k-", linewidth=1, label="Sum")
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (ref)")
        ax.invert_xaxis()
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{sel} Auger lines (Al Kα)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown(
        "**Data sources:** Binding energies and chemical state assignments "
        "are compiled from the "
        "[NIST X-ray Photoelectron Spectroscopy Database](https://srdata.nist.gov/xps/) "
        "(Standard Reference Database 20, Version 4.1), the "
        "*Handbook of X-ray Photoelectron Spectroscopy* "
        "(J.F. Moulder et al., Physical Electronics, 1995), and primary "
        "literature sources cited in the reference column of each element's table. "
        "Where available, references link directly to the original publication via DOI. "
        "Entries marked 'Not known' have not yet been traced to a specific primary source."
    )


def _render_peak_search():
    """Render the peak search / unknown species identification tab."""
    from xpsanalysis.reference import search_peak, XRAY_SOURCES

    st.subheader("Peak Search — Identify Unknown Peaks")
    st.caption(
        "Enter the binding energy of an unknown peak to find matching "
        "core levels and Auger lines.")

    search_cols = st.columns([2, 1, 2])
    with search_cols[0]:
        search_pos = st.number_input(
            "Peak position (eV)", value=284.8, format="%.1f",
            key="peak_search_pos")
    with search_cols[1]:
        search_tol = st.number_input(
            "Tolerance ± (eV)", value=5.0, min_value=0.1, max_value=50.0,
            format="%.1f", key="peak_search_tol")
    with search_cols[2]:
        source_name = st.selectbox(
            "X-ray source", list(XRAY_SOURCES.keys()), key="xray_source")
    photon_energy = XRAY_SOURCES[source_name]

    if st.button("Search", type="primary", key="peak_search_btn"):
        matches = search_peak(search_pos, search_tol, photon_energy)
        st.session_state["_peak_search_results"] = matches
        st.session_state["_peak_search_query"] = (search_pos, search_tol, source_name)

    results = st.session_state.get("_peak_search_results")
    query = st.session_state.get("_peak_search_query")
    if results is not None and query:
        qpos, qtol, qsrc = query
        st.caption(
            f"Matches for **{qpos:.1f} eV** ± {qtol:.1f} eV "
            f"(source: {qsrc})")
        if not results:
            st.info("No matches found. Try increasing the tolerance.")
        else:
            core_matches = [m for m in results if m.match_type == "core_level"]
            auger_matches = [m for m in results if m.match_type == "auger"]

            if core_matches:
                st.markdown("**Core levels:**")
                rows = []
                for m in core_matches:
                    rows.append({
                        "Line": m.line_label,
                        "BE (eV)": f"{m.energy:.1f}",
                        "Δ (eV)": f"{m.delta:+.1f}",
                        "Element": f"{m.element_name} ({m.element_symbol})",
                    })
                st.table(rows)

            if auger_matches:
                st.markdown("**Auger lines:**")
                rows = []
                for m in auger_matches:
                    ke = photon_energy - m.energy
                    rows.append({
                        "Line": m.line_label,
                        "Apparent BE (eV)": f"{m.energy:.1f}",
                        "KE (eV)": f"{ke:.1f}",
                        "Δ (eV)": f"{m.delta:+.1f}",
                        "Element": f"{m.element_name} ({m.element_symbol})",
                    })
                st.table(rows)
                st.caption(
                    "Auger peak positions in binding energy depend on the X-ray source. "
                    "The kinetic energy (KE) is source-independent.")


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


def _parse_formula(name: str) -> dict[str, int]:
    """Parse chemical state name into element counts, e.g. 'Nb2O5' -> {'Nb':2,'O':5}."""
    import re as _re
    _MANUAL = {
        "C-C": {"C": 1}, "C-O": {"C": 1, "O": 1}, "C=O": {"C": 1, "O": 1},
        "O-C=O": {"C": 1, "O": 2}, "CO3": {"C": 1, "O": 3},
        "CF2": {"C": 1, "F": 2}, "CF3": {"C": 1, "F": 3},
        "Metal oxide": {"O": 1}, "Hydroxide": {"O": 1, "H": 1},
        "C-O": {"O": 1}, "Adsorbed H2O": {"H": 2, "O": 1},
        "Metal nitride": {"N": 1}, "Amine": {"N": 1}, "Amide": {"N": 1},
        "Pyrrolic": {"N": 1}, "Quaternary-N": {"N": 1},
        "N-oxide": {"N": 1, "O": 1}, "Nitrate": {"N": 1, "O": 3},
        "Metal fluoride": {"F": 1}, "Organic F": {"F": 1},
        "Sulfide": {"S": 1}, "Thiol": {"S": 1}, "S elemental": {"S": 1},
        "Sulfite": {"S": 1, "O": 3}, "Sulfonate": {"S": 1, "O": 3},
        "Sulfate": {"S": 1, "O": 4}, "Chloride": {"Cl": 1},
        "Organic Cl": {"Cl": 1}, "ClO3": {"Cl": 1, "O": 3},
        "Bromide": {"Br": 1}, "Br organic": {"Br": 1},
        "Iodide": {"I": 1}, "Iodate": {"I": 1, "O": 3},
        "Phosphate": {"P": 1, "O": 4}, "Satellite": {}, "Shake-up": {},
    }
    if name in _MANUAL:
        return _MANUAL[name]
    low = name.lower()
    if "metal" in low or "elemental" in low:
        return {}
    formula = name.strip()
    formula = _re.sub(r'\s*\d*[+-]$', '', formula)
    if '/' in formula:
        formula = formula.split('/')[0]
    if ' ' in formula:
        formula = formula.split(' ')[0]
    # Expand parentheses e.g. Ni(OH)2
    while '(' in formula:
        m = _re.search(r'\(([^)]+)\)(\d*)', formula)
        if not m:
            break
        inner, mult = m.group(1), int(m.group(2)) if m.group(2) else 1
        expanded = ''
        for em in _re.finditer(r'([A-Z][a-z]?)(\d*)', inner):
            n = int(em.group(2)) if em.group(2) else 1
            expanded += f"{em.group(1)}{n * mult}"
        formula = formula[:m.start()] + expanded + formula[m.end():]
    counts = {}
    for m in _re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        sym, n = m.group(1), int(m.group(2)) if m.group(2) else 1
        counts[sym] = counts.get(sym, 0) + n
    return counts


def _compute_composition(selected_states, amounts):
    """Compute atomic % from selected chemical states and their amounts.

    Parameters
    ----------
    selected_states : list of (element_symbol, ChemicalState, CoreLevelRef)
    amounts : dict mapping state name to relative amount of the *primary element*

    Returns
    -------
    dict mapping element symbol to atomic percent (summing to 100)
    """
    atom_counts: dict[str, float] = {}
    for elem_sym, cs, ref in selected_states:
        amt = amounts.get(cs.name, 0.0)
        if amt <= 0:
            continue
        formula = _parse_formula(cs.name)
        if not formula:
            # Pure element state
            atom_counts[elem_sym] = atom_counts.get(elem_sym, 0) + amt
        else:
            # Find how many of the primary element per formula unit
            primary_per_fu = formula.get(elem_sym, 1)
            # Scale: user sets amount of primary element atoms
            fu = amt / primary_per_fu
            for sym, count in formula.items():
                atom_counts[sym] = atom_counts.get(sym, 0) + fu * count
    total = sum(atom_counts.values())
    if total <= 0:
        return {}
    return {sym: 100.0 * v / total for sym, v in atom_counts.items()}


def _render_simulation_tab():
    """Render the spectrum simulation tab with element selection wizard."""
    from xpsanalysis.reference import REFERENCE_DB, get_element
    from xpsanalysis.synthetic import _pseudo_voigt, _shirley_step

    st.subheader("Spectrum Simulation")
    st.caption(
        "Select elements, choose oxidation states, set compositions, "
        "and generate a simulated XPS spectrum. Ligand content (O, N, etc.) "
        "is automatically constrained by compound stoichiometry.")

    # --- Build lookup of elements with chemical states ---
    db_symbols = set()
    elem_states: dict[str, list[tuple]] = {}  # sym -> [(cs, ref), ...]
    for ref in REFERENCE_DB:
        if not ref.chemical_states:
            continue
        db_symbols.add(ref.element_symbol)
        for cs in ref.chemical_states:
            elem_states.setdefault(ref.element_symbol, []).append((cs, ref))

    # --- Step 1: Element selection via periodic table ---
    st.markdown("### Step 1: Select Elements")
    if "_sim_elements" not in st.session_state:
        st.session_state["_sim_elements"] = set()
    sel_elems = st.session_state["_sim_elements"]

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
            is_sel = sym in sel_elems
            label = f"**[{sym}]**" if is_sel else sym
            if cols[ci].button(label, key=f"sim_pt_{sym}", use_container_width=True):
                if sym in sel_elems:
                    sel_elems.discard(sym)
                else:
                    sel_elems.add(sym)
                st.rerun()

    if not sel_elems:
        st.info("Click elements above to select them for simulation.")
        return

    sel_sorted = sorted(sel_elems, key=lambda s: next(
        (r.atomic_number for r in REFERENCE_DB if r.element_symbol == s), 0))
    st.write("**Selected:** " + ", ".join(sel_sorted))

    # --- Step 2: Oxidation state selection ---
    st.markdown("### Step 2: Choose Oxidation States")
    if "_sim_checked_states" not in st.session_state:
        st.session_state["_sim_checked_states"] = {}
    checked = st.session_state["_sim_checked_states"]

    active_states = []  # (elem_sym, cs, ref)
    for sym in sel_sorted:
        states = elem_states.get(sym, [])
        if not states:
            continue
        st.markdown(f"**{sym}**")
        for cs, ref in states:
            key = f"sim_cs_{sym}_{cs.name}"
            default = checked.get(key, False)
            formula = _parse_formula(cs.name)
            formula_str = ""
            if formula:
                ligands = {k: v for k, v in formula.items() if k != sym}
                if ligands:
                    formula_str = " — stoich: " + ", ".join(
                        f"{v} {k}" for k, v in ligands.items()) + f" per {formula.get(sym, 1)} {sym}"
            val = st.checkbox(
                f"{cs.name} ({cs.binding_energy:.1f} eV){formula_str}",
                value=default, key=key)
            checked[key] = val
            if val:
                active_states.append((sym, cs, ref))

    if not active_states:
        st.info("Select at least one oxidation state above.")
        return

    # --- Step 3: Set composition ---
    st.markdown("### Step 3: Set Composition (Atomic %)")
    st.caption(
        "Set the amount of each chemical state in atomic % of the primary element. "
        "Ligand content (e.g., O in oxides) is calculated from stoichiometry.")

    amounts = {}
    comp_cols = st.columns(min(len(active_states), 3))
    for i, (sym, cs, ref) in enumerate(active_states):
        with comp_cols[i % len(comp_cols)]:
            val = st.number_input(
                f"{cs.name} ({sym})", min_value=0.0, max_value=100.0,
                value=10.0, step=1.0, key=f"sim_amt_{cs.name}")
            amounts[cs.name] = val

    # Compute and display composition
    composition = _compute_composition(active_states, amounts)
    if composition:
        st.markdown("**Calculated composition (atomic %):**")
        comp_data = []
        for sym in sorted(composition, key=lambda s: -composition[s]):
            comp_data.append({"Element": sym, "Atomic %": f"{composition[sym]:.1f}"})
        st.table(comp_data)

    # --- Generate spectrum ---
    if st.button("Generate Spectrum", type="primary", key="sim_generate"):
        _generate_simulation(active_states, amounts, composition)

    # Show stored result
    if "_sim_result" in st.session_state:
        for fig in st.session_state["_sim_result"]:
            st.pyplot(fig)


def _generate_simulation(active_states, amounts, composition):
    """Generate and display simulated XPS spectra including Auger peaks."""
    from xpsanalysis.synthetic import _pseudo_voigt, _shirley_step
    from xpsanalysis.reference import REFERENCE_DB, AUGER_DB, XRAY_SOURCES

    if not composition:
        st.error("Set non-zero amounts for at least one state.")
        return

    photon_energy = XRAY_SOURCES["Al Kα (1486.6 eV)"]

    # Group states by core level (element + orbital)
    regions: dict[str, list] = {}  # "Sym orbital" -> [(cs, ref, at%)]
    for sym, cs, ref in active_states:
        amt = amounts.get(cs.name, 0.0)
        if amt <= 0:
            continue
        label = f"{sym} {ref.orbital}"
        regions.setdefault(label, []).append((cs, ref, amt))

    # Also add ligand core levels if they're in the composition
    # (O 1s, N 1s, etc.) — auto-generated from stoichiometry
    ligand_elements = set(composition.keys()) - {sym for sym, _, _ in active_states}
    for lig_sym in ligand_elements:
        for ref in REFERENCE_DB:
            if ref.element_symbol == lig_sym and ref.chemical_states:
                label = f"{lig_sym} {ref.orbital}"
                if label not in regions:
                    regions[label] = [(ref.chemical_states[0], ref,
                                       composition.get(lig_sym, 0))]
                break

    # Collect Auger lines for all elements in the composition
    auger_regions: dict[str, list] = {}  # "Sym TRANS Auger" -> [(aug, at%)]
    for sym, at_pct in composition.items():
        if at_pct <= 0:
            continue
        for aug in AUGER_DB:
            if aug.element_symbol == sym:
                label = f"{sym} {aug.transition} Auger"
                auger_regions.setdefault(label, []).append((aug, at_pct))

    figs = []

    # --- Survey spectrum overview ---
    survey_x = np.linspace(0, photon_energy, 3000)
    survey_y = np.zeros_like(survey_x)
    survey_labels = []  # (position, label) for annotation
    max_at = max(composition.values()) if composition else 1

    # Add all photoelectron peaks to survey
    for region_label, states_in_region in regions.items():
        for cs, ref, amt in states_in_region:
            at_pct = composition.get(ref.element_symbol, amt)
            amp = 500 * (at_pct / max_at) * ref.sensitivity_factor
            y = _pseudo_voigt(survey_x, cs.binding_energy, ref.typical_sigma * 1.5, 0.3, amp)
            if ref.is_doublet and ref.splitting and ref.branching_ratio:
                y += _pseudo_voigt(survey_x, cs.binding_energy + ref.splitting,
                                   ref.typical_sigma * 1.5, 0.3, amp * ref.branching_ratio)
            survey_y += y
        # Label at the primary peak position
        main_cs = states_in_region[0][0]
        survey_labels.append((main_cs.binding_energy, region_label))

    # Add all Auger peaks to survey
    for region_label, auger_in_region in auger_regions.items():
        for aug, at_pct in auger_in_region:
            app_be = photon_energy - aug.kinetic_energy
            if app_be < 0 or app_be > photon_energy:
                continue
            amp = 350 * (at_pct / max_at) * aug.relative_intensity
            survey_y += _pseudo_voigt(survey_x, app_be, aug.typical_sigma * 1.5, 0.5, amp)
        aug0 = auger_in_region[0][0]
        app_be0 = photon_energy - aug0.kinetic_energy
        if 0 < app_be0 < photon_energy:
            survey_labels.append((app_be0, region_label))

    # Sloping background typical of survey scans
    survey_bg = _shirley_step(survey_x, 200, 20, photon_energy / 2, 200)
    survey_y += survey_bg
    rng = np.random.default_rng(44)
    survey_y += rng.normal(0, 8, size=survey_x.shape)

    fig_survey, ax_s = plt.subplots(figsize=(12, 4))
    ax_s.plot(survey_x, survey_y, "k-", linewidth=0.6)
    ax_s.set_xlabel("Binding Energy (eV)")
    ax_s.set_ylabel("Intensity (arb. units)")
    ax_s.invert_xaxis()
    ax_s.set_title("Survey Spectrum — Simulated (Al Kα)")
    # Annotate peaks
    for pos, lbl in survey_labels:
        y_at = np.interp(pos, survey_x, survey_y)
        ax_s.annotate(lbl, xy=(pos, y_at), xytext=(0, 12),
                       textcoords="offset points", fontsize=6,
                       ha="center", va="bottom",
                       arrowprops=dict(arrowstyle="-", lw=0.5, color="0.5"))
    fig_survey.tight_layout()
    figs.append(fig_survey)

    # --- Photoelectron regions ---
    for region_label, states_in_region in sorted(regions.items()):
        all_be = []
        for cs, ref, _ in states_in_region:
            all_be.append(cs.binding_energy)
            if ref.is_doublet and ref.splitting:
                all_be.append(cs.binding_energy + ref.splitting)
        e_min = min(all_be) - 8
        e_max = max(all_be) + 8
        x = np.linspace(e_min, e_max, 600)

        fig, ax = plt.subplots(figsize=(7, 3))
        total_y = np.zeros_like(x)
        max_amp = 1000.0

        for cs, ref, amt in states_in_region:
            amp = max_amp * (amt / max(sum(a for _, _, a in states_in_region), 1))
            sigma = ref.typical_sigma
            y = _pseudo_voigt(x, cs.binding_energy, sigma, 0.3, amp)
            if ref.is_doublet and ref.splitting and ref.branching_ratio:
                y2 = _pseudo_voigt(x, cs.binding_energy + ref.splitting,
                                   sigma, 0.3, amp * ref.branching_ratio)
                y = y + y2
            ax.fill_between(x, y, alpha=0.3, label=cs.name)
            total_y += y

        bg_center = (e_min + e_max) / 2
        bg = _shirley_step(x, 80, 30, bg_center, 2.0)
        total_y += bg
        rng = np.random.default_rng(42)
        total_y += rng.normal(0, 0.02 * max_amp, size=x.shape)

        ax.plot(x, total_y, "k-", linewidth=1, label="Envelope")
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (arb. units)")
        ax.invert_xaxis()
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{region_label} — Simulated")
        fig.tight_layout()
        figs.append(fig)

    # --- Auger regions ---
    for region_label, auger_in_region in sorted(auger_regions.items()):
        all_app_be = [photon_energy - aug.kinetic_energy for aug, _ in auger_in_region]
        e_min = min(all_app_be) - 12
        e_max = max(all_app_be) + 12
        x = np.linspace(e_min, e_max, 600)

        fig, ax = plt.subplots(figsize=(7, 3))
        total_y = np.zeros_like(x)
        max_amp = 800.0
        total_at = max(sum(at for _, at in auger_in_region), 1)

        for aug, at_pct in auger_in_region:
            app_be = photon_energy - aug.kinetic_energy
            amp = max_amp * aug.relative_intensity * (at_pct / total_at)
            y = _pseudo_voigt(x, app_be, aug.typical_sigma, 0.5, amp)
            ax.fill_between(x, y, alpha=0.3,
                            label=f"{aug.transition} (KE={aug.kinetic_energy:.0f})")
            total_y += y

        bg_center = (e_min + e_max) / 2
        bg = _shirley_step(x, 60, 25, bg_center, 3.0)
        total_y += bg
        rng = np.random.default_rng(43)
        total_y += rng.normal(0, 0.02 * max_amp, size=x.shape)

        ax.plot(x, total_y, "k-", linewidth=1, label="Envelope")
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (arb. units)")
        ax.invert_xaxis()
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{region_label} — Simulated (Al Kα)")
        fig.tight_layout()
        figs.append(fig)

    st.session_state["_sim_result"] = figs
    st.rerun()


def _render_transmission_tab():
    """Render the transmission function extraction tab."""
    from xpsanalysis.reference import XRAY_SOURCES

    st.subheader("Transmission Function — T(KE)")
    st.caption(
        "Upload a survey spectrum to extract the analyzer transmission function. "
        "The background shape (secondary electrons) encodes T(KE). Peaks are "
        "automatically masked, the secondary cascade (∝ 1/KE²) is divided out, "
        "and a power law T(KE) = a × KE^n is fitted.")

    # File uploader specific to this tab
    survey_file = st.file_uploader(
        "Upload survey spectrum", type=["csv", "vms", "xy", "dat", "txt"],
        key="transmission_upload")

    if survey_file is None:
        st.info("Upload a survey spectrum (wide energy range, e.g. 0–1400 eV) to extract T(KE).")
        return

    from xpsanalysis.io import load_spectrum

    cache_key = f"tf_survey_{survey_file.name}_{survey_file.size}"
    if cache_key not in st.session_state:
        suffix = Path(survey_file.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(survey_file.read())
            tmp_path = tmp.name
        st.session_state[cache_key] = load_spectrum(tmp_path)

    spectra = st.session_state[cache_key]

    # Pick region if multiple
    region_idx = 0
    if len(spectra) > 1:
        labels = [f"Region {i}" + (f" ({s.metadata.core_level})" if s.metadata.core_level else "")
                  for i, s in enumerate(spectra)]
        region_idx = st.selectbox("Select region", range(len(spectra)),
                                  format_func=lambda i: labels[i], key="tf_region")
    spectrum = spectra[region_idx]

    # Energy scale and source selection
    param_cols = st.columns(4)
    with param_cols[0]:
        source_name = st.selectbox("X-ray source", list(XRAY_SOURCES.keys()),
                                   key="tf_source")
        photon_energy = XRAY_SOURCES[source_name]
    with param_cols[1]:
        # Auto-detect: if max energy > photon_energy, likely KE
        e_max_data = float(spectrum.energy.max())
        auto_guess = "Kinetic Energy" if e_max_data > photon_energy * 0.85 else "Binding Energy"
        energy_scale = st.selectbox(
            "Energy axis in file", ["Binding Energy", "Kinetic Energy"],
            index=0 if auto_guess == "Binding Energy" else 1,
            key="tf_energy_scale")

    # Convert to BE if needed
    if energy_scale == "Kinetic Energy":
        energy_be = photon_energy - spectrum.energy
        x_label_raw = "Kinetic Energy (eV)"
        x_data_raw = spectrum.energy
    else:
        energy_be = spectrum.energy
        x_label_raw = "Binding Energy (eV)"
        x_data_raw = spectrum.energy

    # Compute both scales for all plots
    ke_all = photon_energy - energy_be

    # Show raw survey in both scales
    fig_raw, (ax_be, ax_ke) = plt.subplots(1, 2, figsize=(14, 3))
    ax_be.plot(energy_be, spectrum.intensity, "k-", linewidth=0.6)
    ax_be.set_xlabel("Binding Energy (eV)")
    ax_be.set_ylabel("Intensity")
    ax_be.invert_xaxis()
    ax_be.set_title("Survey — Binding Energy")
    ax_ke.plot(ke_all, spectrum.intensity, "k-", linewidth=0.6)
    ax_ke.set_xlabel("Kinetic Energy (eV)")
    ax_ke.set_ylabel("Intensity")
    ax_ke.set_title("Survey — Kinetic Energy")
    fig_raw.tight_layout()
    st.pyplot(fig_raw)
    plt.close(fig_raw)

    # Parameters
    with param_cols[2]:
        mask_width = st.number_input("Peak mask width ± (eV)", value=8.0,
                                     min_value=2.0, max_value=30.0, step=1.0,
                                     key="tf_mask_width")
    with param_cols[3]:
        ke_min = st.number_input("Min KE (eV)", value=50.0, min_value=10.0,
                                 max_value=500.0, step=10.0, key="tf_ke_min")

    if st.button("Extract T(KE)", type="primary", key="tf_extract"):
        from xpsanalysis.transmission import extract_transmission
        try:
            result = extract_transmission(
                energy_be, spectrum.intensity,
                photon_energy=photon_energy,
                mask_width_ev=mask_width,
                ke_min=ke_min,
            )
            st.session_state["_tf_result"] = result
        except ValueError as e:
            st.error(str(e))
            return

    result = st.session_state.get("_tf_result")
    if result is None:
        return

    # Display results
    st.markdown("---")
    st.subheader("Results")

    st.markdown(
        f"**T(KE) = {result.a:.3e} × KE^({result.n:+.3f})**")
    st.caption(
        f"Exponent n = {result.n:.3f} ± {result.n_err:.3f} · "
        f"Prefactor a = {result.a:.3e} ± {result.a_err:.3e} "
        f"(normalized scale)")

    # Convert result arrays to both scales
    bg_be = photon_energy - result.bg_ke
    data_be = photon_energy - result.ke_data
    fit_be = photon_energy - result.ke_fit

    # Plot 1: Background with peaks masked — BE and KE
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 3.5))

    # BE scale
    ax1a.plot(energy_be, spectrum.intensity, color="0.7", linewidth=0.4, label="Raw survey")
    ax1a.plot(bg_be, result.bg_intensity, "b-", linewidth=1.0,
              label="Background (peaks masked)")
    ax1a.set_xlabel("Binding Energy (eV)")
    ax1a.set_ylabel("Intensity")
    ax1a.invert_xaxis()
    ax1a.legend(fontsize=7)
    ax1a.set_title("Peak masking — BE scale")

    # KE scale
    ax1b.plot(ke_all, spectrum.intensity, color="0.7", linewidth=0.4, label="Raw survey")
    ax1b.plot(result.bg_ke, result.bg_intensity, "b-", linewidth=1.0,
              label="Background (peaks masked)")
    ax1b.set_xlabel("Kinetic Energy (eV)")
    ax1b.set_ylabel("Intensity")
    ax1b.legend(fontsize=7)
    ax1b.set_title("Peak masking — KE scale")

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # Plot 2: T(KE) with fit — KE linear + KE log-log
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 4))

    ax2a.scatter(result.ke_data, result.t_data, s=3, alpha=0.3, color="steelblue",
                 label="B(KE) × KE² (data)")
    ax2a.plot(result.ke_fit, result.t_fit, "r-", linewidth=1.5,
              label=f"Fit: a·KE^({result.n:.2f})")
    ax2a.set_xlabel("Kinetic Energy (eV)")
    ax2a.set_ylabel("T(KE) (normalized)")
    ax2a.legend(fontsize=7)
    ax2a.set_title("Transmission Function — KE Linear")

    ax2b.scatter(result.ke_data, result.t_data, s=3, alpha=0.3, color="steelblue",
                 label="Data")
    ax2b.plot(result.ke_fit, result.t_fit, "r-", linewidth=1.5,
              label=f"n = {result.n:.3f}")
    ax2b.set_xscale("log")
    ax2b.set_yscale("log")
    ax2b.set_xlabel("Kinetic Energy (eV)")
    ax2b.set_ylabel("T(KE) (normalized)")
    ax2b.legend(fontsize=7)
    ax2b.set_title("Transmission Function — KE Log-Log")

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Plot 3: T vs BE scale
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 4))

    ax3a.scatter(data_be, result.t_data, s=3, alpha=0.3, color="steelblue",
                 label="B(KE) × KE² (data)")
    ax3a.plot(fit_be, result.t_fit, "r-", linewidth=1.5,
              label=f"Fit: a·KE^({result.n:.2f})")
    ax3a.set_xlabel("Binding Energy (eV)")
    ax3a.set_ylabel("T (normalized)")
    ax3a.invert_xaxis()
    ax3a.legend(fontsize=7)
    ax3a.set_title("Transmission Function — BE Linear")

    pos_data = result.t_data > 0
    pos_fit = result.t_fit > 0
    ax3b.scatter(data_be[pos_data], result.t_data[pos_data], s=3, alpha=0.3,
                 color="steelblue", label="Data")
    ax3b.plot(fit_be[pos_fit], result.t_fit[pos_fit], "r-", linewidth=1.5,
              label=f"n = {result.n:.3f}")
    ax3b.set_yscale("log")
    ax3b.set_xlabel("Binding Energy (eV)")
    ax3b.set_ylabel("T (normalized)")
    ax3b.invert_xaxis()
    ax3b.legend(fontsize=7)
    ax3b.set_title("Transmission Function — BE Semi-Log")

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Interpretation
    st.markdown("---")
    st.markdown("**Interpretation:**")
    if -1.5 < result.n < -0.3:
        st.success(
            f"Exponent n = {result.n:.2f} is consistent with a CHA analyzer "
            f"in FAT (Fixed Analyzer Transmission) mode. Typical values range "
            f"from −0.5 to −1.0.")
    elif abs(result.n) < 0.3:
        st.success(
            f"Exponent n = {result.n:.2f} is near zero, consistent with a CHA "
            f"analyzer in FRR (Fixed Retarding Ratio) mode or a well-corrected "
            f"FAT mode instrument.")
    else:
        st.warning(
            f"Exponent n = {result.n:.2f} is outside the typical range for "
            f"standard CHA analyzers. This may indicate non-standard operating "
            f"conditions, sample charging, or a different analyzer geometry.")


def main() -> None:
    tab_periodic, tab_search, tab_simulation, tab_transmission, tab_analysis = st.tabs([
        "Periodic Table Reference", "Peak Search", "Spectrum Simulation",
        "Transmission Function", "Spectrum Analysis"])

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

    # ---- Peak Search tab ----
    with tab_search:
        _render_peak_search()

    # ---- Simulation tab ----
    with tab_simulation:
        _render_simulation_tab()

    # ---- Transmission Function tab ----
    with tab_transmission:
        _render_transmission_tab()

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
