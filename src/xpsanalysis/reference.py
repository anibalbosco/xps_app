"""XPS binding energy reference database.

Standard binding energies from Moulder/PHI Handbook and NIST SRD 20.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ChemicalState:
    name: str
    binding_energy: float
    description: str = ""
    reference: str = ""


@dataclass
class CoreLevelRef:
    element_symbol: str
    element_name: str
    atomic_number: int
    orbital: str
    binding_energy: float
    is_doublet: bool = False
    splitting: float | None = None
    branching_ratio: float | None = None
    typical_sigma: float = 0.8
    chemical_states: list[ChemicalState] = field(default_factory=list)
    sensitivity_factor: float = 1.0
    reference: str = ""


# === CITATION REGISTRY ===
# Short keys used in reference fields map to full citations here.
# Each entry has "citation" (full text) and optional "doi" (just the DOI number).
CITATIONS: dict[str, dict[str, str]] = {
    "Moulder1995": {
        "citation": (
            "J.F. Moulder, W.F. Stickle, P.E. Sobol, K.D. Bomben, "
            "Handbook of X-ray Photoelectron Spectroscopy, "
            "Physical Electronics, Eden Prairie, MN, 1995."
        ),
    },
    "Beamson1992": {
        "citation": (
            "G. Beamson, D. Briggs, "
            "High Resolution XPS of Organic Polymers: The Scienta ESCA300 Database, "
            "John Wiley & Sons, Chichester, 1992."
        ),
    },
    "Biesinger2011": {
        "citation": (
            "M.C. Biesinger, B.P. Payne, A.P. Grosvenor, L.W.M. Lau, A.R. Gerson, R.St.C. Smart, "
            "Resolving surface chemical states in XPS analysis of first row transition metals, "
            "oxides and hydroxides: Cr, Mn, Fe, Co and Ni, "
            "Appl. Surf. Sci. 257 (2011) 2717-2730."
        ),
        "doi": "10.1016/j.apsusc.2010.10.051",
    },
    "Biesinger2017": {
        "citation": (
            "M.C. Biesinger, "
            "Advanced analysis of copper X-ray photoelectron spectra, "
            "Surf. Interface Anal. 49 (2017) 1325-1334."
        ),
        "doi": "10.1002/sia.6239",
    },
    "Biesinger2004": {
        "citation": (
            "M.C. Biesinger, C. Brown, J.R. Mycroft, R.D. Davidson, N.S. McIntyre, "
            "X-ray photoelectron spectroscopy studies of chromium compounds, "
            "Surf. Interface Anal. 36 (2004) 1550-1563."
        ),
        "doi": "10.1002/sia.1983",
    },
    "Biesinger2010": {
        "citation": (
            "M.C. Biesinger, L.W.M. Lau, A.R. Gerson, R.St.C. Smart, "
            "Resolving surface chemical states in XPS analysis of first row transition metals, "
            "oxides and hydroxides: Sc, Ti, V, Cu and Zn, "
            "Appl. Surf. Sci. 257 (2010) 887-898."
        ),
        "doi": "10.1016/j.apsusc.2010.07.086",
    },
    "Crist2000": {
        "citation": (
            "B.V. Crist, "
            "Handbook of Monochromatic XPS Spectra, "
            "XPS International LLC, 2000."
        ),
    },
    "NIST_SRD20": {
        "citation": (
            "NIST X-ray Photoelectron Spectroscopy Database, "
            "Standard Reference Database Number 20, "
            "National Institute of Standards and Technology, Gaithersburg MD, 20899 (2000)."
        ),
        "doi": "10.18434/T4T88K",
    },
    "Wagner1979": {
        "citation": (
            "C.D. Wagner, W.M. Riggs, L.E. Davis, J.F. Moulder, G.E. Muilenberg, "
            "Handbook of X-ray Photoelectron Spectroscopy, "
            "Perkin-Elmer Corp., Physical Electronics Division, Eden Prairie, MN, 1979."
        ),
    },
    "Grosvenor2004": {
        "citation": (
            "A.P. Grosvenor, B.A. Kobe, M.C. Biesinger, N.S. McIntyre, "
            "Investigation of multiplet splitting of Fe 2p XPS spectra and bonding in iron compounds, "
            "Surf. Interface Anal. 36 (2004) 1564-1574."
        ),
        "doi": "10.1002/sia.1984",
    },
    "Biesinger2009": {
        "citation": (
            "M.C. Biesinger, B.P. Payne, L.W.M. Lau, A. Gerson, R.St.C. Smart, "
            "X-ray photoelectron spectroscopic chemical state quantification of mixed nickel "
            "metal, oxide and hydroxide systems, "
            "Surf. Interface Anal. 41 (2009) 324-332."
        ),
        "doi": "10.1002/sia.3026",
    },
    "Payne2011": {
        "citation": (
            "B.P. Payne, M.C. Biesinger, N.S. McIntyre, "
            "X-ray photoelectron spectroscopy studies of reactions on chromium metal and "
            "chromium oxide surfaces, "
            "J. Electron Spectrosc. Relat. Phenom. 184 (2011) 29-37."
        ),
        "doi": "10.1016/j.elspec.2010.12.001",
    },
    "Himpsel1988": {
        "citation": (
            "F.J. Himpsel, F.R. McFeely, A. Taleb-Ibrahimi, J.A. Yarmoff, G. Hollinger, "
            "Microscopic structure of the SiO2/Si interface, "
            "Phys. Rev. B 38 (1988) 6084-6096."
        ),
        "doi": "10.1103/PhysRevB.38.6084",
    },
}


def _r(sym, name, z, orb, be, *, dub=False, sp=None, br=None, sig=0.8, cs=None, rsf=1.0,
       ref="Moulder1995"):
    """Shorthand constructor."""
    return CoreLevelRef(sym, name, z, orb, be, dub, sp, br, sig, cs or [], rsf, ref)


def _cs(name, be, desc="", ref=""):
    return ChemicalState(name, be, desc, ref)


# === REFERENCE DATABASE ===
# Values: Moulder/PHI Handbook, NIST SRD 20
# Doublet info: p→br=0.5, d→br=0.667, f→br=0.75

REFERENCE_DB: list[CoreLevelRef] = [
    # --- Z=1-2: H, He ---
    _r("H", "Hydrogen", 1, "1s", 13.6, rsf=0.01),
    _r("He", "Helium", 2, "1s", 24.6, rsf=0.01),
    # --- Z=3-4: Li, Be ---
    _r("Li", "Lithium", 3, "1s", 54.7, sig=0.9, rsf=0.02,
       cs=[_cs("Li metal", 54.7, "", "Moulder1995"),
           _cs("Li2O", 55.6, "", "Moulder1995")]),
    _r("Be", "Beryllium", 4, "1s", 111.5, sig=0.9, rsf=0.06,
       cs=[_cs("Be metal", 111.5, "", "Moulder1995"),
           _cs("BeO", 113.8, "", "Moulder1995")]),
    # --- Z=5: B ---
    _r("B", "Boron", 5, "1s", 187.5, sig=0.8, rsf=0.13,
       cs=[_cs("B elemental", 187.5, "", "Moulder1995"),
           _cs("B2O3", 193.0, "", "Moulder1995"),
           _cs("BN", 190.5, "", "Moulder1995")]),
    # --- Z=6: C ---
    _r("C", "Carbon", 6, "1s", 284.8, sig=0.6, rsf=0.25,
       cs=[_cs("C-C", 284.8, "Adventitious/graphitic", "Moulder1995"),
           _cs("C-O", 286.3, "Alcohol/ether", "Moulder1995"),
           _cs("C=O", 287.8, "Carbonyl", "Moulder1995"),
           _cs("O-C=O", 289.0, "Carboxyl/ester", "Moulder1995"),
           _cs("CO3", 290.1, "Carbonate", "Moulder1995"),
           _cs("CF2", 291.8, "PTFE-like", "Beamson1992"),
           _cs("CF3", 293.4, "Fluorocarbon", "Beamson1992")]),
    # --- Z=7: N ---
    _r("N", "Nitrogen", 7, "1s", 398.1, sig=0.8, rsf=0.42,
       cs=[_cs("Metal nitride", 397.2, "", "Moulder1995"),
           _cs("Amine", 399.5, "", "Moulder1995"),
           _cs("Amide", 400.0, "", "Beamson1992"),
           _cs("Pyrrolic", 400.3, "", "Moulder1995"),
           _cs("Quaternary-N", 401.5, "", "Moulder1995"),
           _cs("N-oxide", 402.0, "", "Moulder1995"),
           _cs("Nitrate", 407.2, "", "Moulder1995")]),
    # --- Z=8: O ---
    _r("O", "Oxygen", 8, "1s", 530.0, sig=0.8, rsf=0.66,
       cs=[_cs("Metal oxide", 529.8, "Lattice O2-", "Moulder1995"),
           _cs("Hydroxide", 531.2, "OH/C=O", "Moulder1995"),
           _cs("C-O", 532.5, "Ether/alcohol/water", "Moulder1995"),
           _cs("Adsorbed H2O", 533.5, "", "Moulder1995")]),
    # --- Z=9: F ---
    _r("F", "Fluorine", 9, "1s", 685.0, sig=0.9, rsf=1.00,
       cs=[_cs("Metal fluoride", 684.5, "", "Moulder1995"),
           _cs("Organic F", 688.0, "C-F bond", "Moulder1995")]),
    # --- Z=10: Ne ---
    _r("Ne", "Neon", 10, "1s", 870.2, rsf=0.01),
    # --- Z=11: Na ---
    _r("Na", "Sodium", 11, "1s", 1071.8, sig=1.0, rsf=2.30,
       cs=[_cs("Na metal", 1071.8, "", "Moulder1995"), _cs("Na2O", 1072.5, "", "Moulder1995"), _cs("NaCl", 1071.6, "", "Moulder1995")]),
    # --- Z=12: Mg ---
    _r("Mg", "Magnesium", 12, "2p3/2", 49.7, dub=True, sp=0.28, br=0.5, sig=0.7, rsf=0.12,
       cs=[_cs("Mg metal", 49.7, "", "Moulder1995"), _cs("MgO", 50.8, "", "Moulder1995")]),
    _r("Mg", "Magnesium", 12, "1s", 1303.0, sig=1.2, rsf=3.00),
    # --- Z=13: Al ---
    _r("Al", "Aluminum", 13, "2p3/2", 72.7, dub=True, sp=0.44, br=0.5, sig=0.7, rsf=0.18,
       cs=[_cs("Al metal", 72.7, "", "Moulder1995"), _cs("Al2O3", 74.5, "Alumina", "Moulder1995")]),
    # --- Z=14: Si ---
    _r("Si", "Silicon", 14, "2p3/2", 99.3, dub=True, sp=0.6, br=0.5, sig=0.7, rsf=0.27,
       cs=[_cs("Si elemental", 99.3, "", "Moulder1995"),
           _cs("SiO", 101.5, "", "Himpsel1988"),
           _cs("Si3N4", 101.8, "", "Moulder1995"),
           _cs("SiO2", 103.3, "Quartz/glass", "Himpsel1988")]),
    # --- Z=15: P ---
    _r("P", "Phosphorus", 15, "2p3/2", 130.0, dub=True, sp=0.87, br=0.5, sig=0.8, rsf=0.39,
       cs=[_cs("P elemental", 130.0, "", "Moulder1995"), _cs("Phosphate", 133.5, "PO4 3-", "Moulder1995")]),
    # --- Z=16: S ---
    _r("S", "Sulfur", 16, "2p3/2", 164.0, dub=True, sp=1.16, br=0.5, sig=0.7, rsf=0.54,
       cs=[_cs("Sulfide", 161.5, "S2-", "Moulder1995"),
           _cs("Thiol", 162.5, "R-SH", "Moulder1995"),
           _cs("S elemental", 164.0, "", "Moulder1995"),
           _cs("Sulfite", 166.5, "SO3 2-", "Moulder1995"),
           _cs("Sulfonate", 167.5, "R-SO3", "Beamson1992"),
           _cs("Sulfate", 169.0, "SO4 2-", "Moulder1995")]),
    # --- Z=17: Cl ---
    _r("Cl", "Chlorine", 17, "2p3/2", 200.0, dub=True, sp=1.6, br=0.5, sig=0.8, rsf=0.73,
       cs=[_cs("Chloride", 198.5, "Cl-", "Moulder1995"), _cs("Organic Cl", 200.5, "C-Cl", "Moulder1995"),
           _cs("ClO3", 207.0, "Chlorate", "Moulder1995")]),
    # --- Z=18: Ar ---
    _r("Ar", "Argon", 18, "2p3/2", 248.4, dub=True, sp=2.1, br=0.5, rsf=0.96),
    # --- Z=19: K ---
    _r("K", "Potassium", 19, "2p3/2", 293.0, dub=True, sp=2.8, br=0.5, sig=0.8, rsf=1.24,
       cs=[_cs("K metal", 293.0, "", "Moulder1995"), _cs("K2O", 293.6, "", "Moulder1995")]),
    # --- Z=20: Ca ---
    _r("Ca", "Calcium", 20, "2p3/2", 346.0, dub=True, sp=3.5, br=0.5, sig=0.9, rsf=1.58,
       cs=[_cs("Ca metal", 346.0, "", "Moulder1995"), _cs("CaO", 347.0, "", "Moulder1995"), _cs("CaCO3", 347.2, "", "Moulder1995")]),
    # --- Z=21-30: Sc through Zn (3d transition metals) ---
    _r("Sc", "Scandium", 21, "2p3/2", 398.5, dub=True, sp=4.0, br=0.5, sig=1.0, rsf=1.97),
    _r("Ti", "Titanium", 22, "2p3/2", 454.0, dub=True, sp=5.7, br=0.5, sig=0.9, rsf=2.00,
       cs=[_cs("Ti metal", 454.0, "", "Biesinger2010"),
           _cs("TiN", 455.8, "", "Moulder1995"),
           _cs("Ti2O3", 457.5, "", "Biesinger2010"),
           _cs("TiO2", 458.8, "Rutile/anatase", "Biesinger2010")]),
    _r("V", "Vanadium", 23, "2p3/2", 512.1, dub=True, sp=7.3, br=0.5, sig=1.0, rsf=2.12,
       cs=[_cs("V metal", 512.1, "", "Biesinger2010"),
           _cs("V2O3", 515.7, "", "Biesinger2010"),
           _cs("V2O5", 517.2, "", "Biesinger2010")]),
    _r("Cr", "Chromium", 24, "2p3/2", 574.2, dub=True, sp=9.0, br=0.5, sig=1.0, rsf=2.30,
       cs=[_cs("Cr metal", 574.2, "", "Biesinger2004"),
           _cs("Cr2O3", 576.4, "", "Biesinger2004"),
           _cs("CrO3", 579.2, "", "Biesinger2004")]),
    _r("Mn", "Manganese", 25, "2p3/2", 639.0, dub=True, sp=11.1, br=0.5, sig=1.0, rsf=2.66,
       cs=[_cs("Mn metal", 639.0, "", "Biesinger2011"),
           _cs("MnO", 641.0, "", "Biesinger2011"),
           _cs("MnO2", 642.2, "", "Biesinger2011")]),
    _r("Fe", "Iron", 26, "2p3/2", 706.8, dub=True, sp=13.6, br=0.5, sig=1.2, rsf=2.96,
       cs=[_cs("Fe metal", 706.8, "", "Biesinger2011"),
           _cs("FeO", 709.5, "Fe2+", "Grosvenor2004"),
           _cs("Fe2O3", 710.8, "Fe3+", "Grosvenor2004"),
           _cs("Satellite", 719.0, "Shake-up", "Grosvenor2004")]),
    _r("Co", "Cobalt", 27, "2p3/2", 778.2, dub=True, sp=15.0, br=0.5, sig=1.1, rsf=3.59,
       cs=[_cs("Co metal", 778.2, "", "Biesinger2011"),
           _cs("CoO", 780.3, "", "Biesinger2011"),
           _cs("Co3O4", 779.8, "", "Biesinger2011")]),
    _r("Ni", "Nickel", 28, "2p3/2", 852.7, dub=True, sp=17.3, br=0.5, sig=1.1, rsf=4.04,
       cs=[_cs("Ni metal", 852.7, "", "Biesinger2009"),
           _cs("NiO", 854.0, "", "Biesinger2009"),
           _cs("Ni(OH)2", 855.8, "", "Biesinger2009"),
           _cs("NiOOH", 856.1, "", "Biesinger2009")]),
    _r("Cu", "Copper", 29, "2p3/2", 932.6, dub=True, sp=19.8, br=0.5, sig=1.0, rsf=4.40,
       cs=[_cs("Cu metal", 932.6, "", "Biesinger2017"),
           _cs("Cu2O", 932.4, "Cu+", "Biesinger2017"),
           _cs("CuO", 933.8, "Cu2+", "Biesinger2017"),
           _cs("Cu(OH)2", 934.7, "", "Biesinger2017")]),
    _r("Zn", "Zinc", 30, "2p3/2", 1021.8, dub=True, sp=23.0, br=0.5, sig=1.0, rsf=4.80,
       cs=[_cs("Zn metal", 1021.8, "", "Biesinger2010"),
           _cs("ZnO", 1022.1, "", "Biesinger2010"),
           _cs("ZnS", 1022.0, "", "Biesinger2010")]),
    # --- Z=31-36: Ga through Kr ---
    _r("Ga", "Gallium", 31, "2p3/2", 1117.4, dub=True, sp=26.8, br=0.5, sig=1.1, rsf=5.40,
       cs=[_cs("Ga metal", 1117.4, "", "Moulder1995"), _cs("Ga2O3", 1118.5, "", "Moulder1995")]),
    _r("Ga", "Gallium", 31, "3d5/2", 18.7, dub=True, sp=0.45, br=0.667, sig=0.6, rsf=0.31),
    _r("Ge", "Germanium", 32, "2p3/2", 1217.1, dub=True, sp=31.1, br=0.5, sig=1.2, rsf=5.70),
    _r("Ge", "Germanium", 32, "3d5/2", 29.3, dub=True, sp=0.55, br=0.667, sig=0.6, rsf=0.41,
       cs=[_cs("Ge elemental", 29.3, "", "Moulder1995"), _cs("GeO2", 32.5, "", "Moulder1995")]),
    _r("As", "Arsenic", 33, "3d5/2", 41.7, dub=True, sp=0.7, br=0.667, sig=0.7, rsf=0.57,
       cs=[_cs("As elemental", 41.7, "", "Moulder1995"), _cs("As2O3", 44.3, "", "Moulder1995"), _cs("As2O5", 45.7, "", "Moulder1995")]),
    _r("Se", "Selenium", 34, "3d5/2", 55.5, dub=True, sp=0.86, br=0.667, sig=0.7, rsf=0.67,
       cs=[_cs("Se elemental", 55.5, "", "Moulder1995"), _cs("SeO2", 59.0, "", "Moulder1995")]),
    _r("Br", "Bromine", 35, "3d5/2", 70.4, dub=True, sp=1.04, br=0.667, sig=0.8, rsf=0.81,
       cs=[_cs("Bromide", 68.5, "", "Moulder1995"), _cs("Br organic", 70.4, "", "Moulder1995")]),
    _r("Kr", "Krypton", 36, "3d5/2", 93.8, dub=True, sp=1.22, br=0.667, rsf=0.97),
    # --- Z=37-48: Rb through Cd ---
    _r("Rb", "Rubidium", 37, "3d5/2", 110.3, dub=True, sp=1.5, br=0.667, sig=0.8, rsf=1.23),
    _r("Sr", "Strontium", 38, "3d5/2", 133.9, dub=True, sp=1.8, br=0.667, sig=0.9, rsf=1.50,
       cs=[_cs("Sr metal", 133.9, "", ""), _cs("SrO", 134.6, "", ""), _cs("SrTiO3", 133.5, "", "")]),
    _r("Y", "Yttrium", 39, "3d5/2", 156.0, dub=True, sp=2.1, br=0.667, sig=0.9, rsf=1.76,
       cs=[_cs("Y metal", 156.0, "", ""), _cs("Y2O3", 157.0, "", "")]),
    _r("Zr", "Zirconium", 40, "3d5/2", 178.8, dub=True, sp=2.4, br=0.667, sig=0.9, rsf=2.00,
       cs=[_cs("Zr metal", 178.8, "", ""), _cs("ZrO2", 182.2, "", "")]),
    _r("Nb", "Niobium", 41, "3d5/2", 202.4, dub=True, sp=2.7, br=0.667, sig=0.9, rsf=2.28,
       cs=[_cs("Nb metal", 202.4, "", ""), _cs("Nb2O5", 207.4, "", "")]),
    _r("Mo", "Molybdenum", 42, "3d5/2", 227.6, dub=True, sp=3.1, br=0.667, sig=0.8, rsf=2.55,
       cs=[_cs("Mo metal", 227.6, "", ""), _cs("MoO2", 229.3, "", ""), _cs("MoO3", 232.6, "", "")]),
    _r("Tc", "Technetium", 43, "3d5/2", 253.0, dub=True, sp=3.5, br=0.667, rsf=2.80),
    _r("Ru", "Ruthenium", 44, "3d5/2", 280.0, dub=True, sp=4.2, br=0.667, sig=0.8, rsf=3.05,
       cs=[_cs("Ru metal", 280.0, "", ""), _cs("RuO2", 280.8, "", "")]),
    _r("Rh", "Rhodium", 45, "3d5/2", 307.1, dub=True, sp=4.7, br=0.667, sig=0.8, rsf=3.36,
       cs=[_cs("Rh metal", 307.1, "", ""), _cs("Rh2O3", 308.5, "", "")]),
    _r("Pd", "Palladium", 46, "3d5/2", 335.3, dub=True, sp=5.3, br=0.667, sig=0.8, rsf=3.68,
       cs=[_cs("Pd metal", 335.3, "", ""), _cs("PdO", 336.8, "", "")]),
    _r("Ag", "Silver", 47, "3d5/2", 368.2, dub=True, sp=6.0, br=0.667, sig=0.8, rsf=4.05,
       cs=[_cs("Ag metal", 368.2, "", "Moulder1995"), _cs("Ag2O", 367.8, "", "Moulder1995"), _cs("AgCl", 368.3, "", "Moulder1995")]),
    _r("Cd", "Cadmium", 48, "3d5/2", 405.0, dub=True, sp=6.7, br=0.667, sig=0.9, rsf=4.35,
       cs=[_cs("Cd metal", 405.0, "", ""), _cs("CdO", 404.5, "", "")]),
    # --- Z=49-54: In through Xe ---
    _r("In", "Indium", 49, "3d5/2", 443.8, dub=True, sp=7.6, br=0.667, sig=0.9, rsf=4.63,
       cs=[_cs("In metal", 443.8, "", ""), _cs("In2O3", 444.5, "", "")]),
    _r("Sn", "Tin", 50, "3d5/2", 484.9, dub=True, sp=8.4, br=0.667, sig=0.9, rsf=4.95,
       cs=[_cs("Sn metal", 484.9, "", "Moulder1995"), _cs("SnO", 486.0, "", "Moulder1995"), _cs("SnO2", 486.6, "", "Moulder1995")]),
    _r("Sb", "Antimony", 51, "3d5/2", 528.2, dub=True, sp=9.4, br=0.667, sig=0.9, rsf=5.23,
       cs=[_cs("Sb metal", 528.2, "", ""), _cs("Sb2O3", 530.3, "", ""), _cs("Sb2O5", 530.8, "", "")]),
    _r("Te", "Tellurium", 52, "3d5/2", 573.0, dub=True, sp=10.4, br=0.667, sig=0.9, rsf=5.52,
       cs=[_cs("Te metal", 573.0, "", ""), _cs("TeO2", 576.2, "", "")]),
    _r("I", "Iodine", 53, "3d5/2", 619.3, dub=True, sp=11.5, br=0.667, sig=1.0, rsf=5.80,
       cs=[_cs("Iodide", 619.3, "", ""), _cs("Iodate", 624.0, "", "")]),
    _r("Xe", "Xenon", 54, "3d5/2", 676.4, dub=True, sp=12.6, br=0.667, rsf=6.10),
    # --- Z=55-57: Cs, Ba, La ---
    _r("Cs", "Cesium", 55, "3d5/2", 724.8, dub=True, sp=14.0, br=0.667, sig=1.0, rsf=6.40),
    _r("Ba", "Barium", 56, "3d5/2", 780.5, dub=True, sp=15.3, br=0.667, sig=1.0, rsf=6.80,
       cs=[_cs("Ba metal", 780.5, "", ""), _cs("BaO", 779.7, "", ""), _cs("BaCO3", 779.8, "", "")]),
    _r("La", "Lanthanum", 57, "3d5/2", 835.0, dub=True, sp=16.8, br=0.667, sig=1.2, rsf=7.20,
       cs=[_cs("La metal", 835.0, "", ""), _cs("La2O3", 834.6, "", "")]),
    # --- Z=58-71: Lanthanides ---
    _r("Ce", "Cerium", 58, "3d5/2", 882.5, dub=True, sp=18.3, br=0.667, sig=1.5, rsf=7.50,
       cs=[_cs("Ce metal", 882.5, "", ""), _cs("CeO2", 882.3, "Ce4+", ""), _cs("Ce2O3", 885.0, "Ce3+", "")]),
    _r("Pr", "Praseodymium", 59, "3d5/2", 932.0, dub=True, sp=20.0, br=0.667, sig=1.5, rsf=7.80),
    _r("Nd", "Neodymium", 60, "3d5/2", 982.0, dub=True, sp=21.5, br=0.667, sig=1.5, rsf=8.10,
       cs=[_cs("Nd metal", 982.0, "", ""), _cs("Nd2O3", 983.0, "", "")]),
    _r("Pm", "Promethium", 61, "3d5/2", 1032.0, dub=True, sp=23.0, br=0.667, rsf=8.40),
    _r("Sm", "Samarium", 62, "3d5/2", 1083.0, dub=True, sp=24.6, br=0.667, sig=1.5, rsf=8.70),
    _r("Eu", "Europium", 63, "3d5/2", 1135.0, dub=True, sp=29.2, br=0.667, sig=1.5, rsf=9.00,
       cs=[_cs("Eu metal", 1135.0, "", ""), _cs("Eu2O3", 1135.5, "", "")]),
    _r("Gd", "Gadolinium", 64, "3d5/2", 1186.0, dub=True, sp=31.0, br=0.667, sig=1.5, rsf=9.30),
    _r("Tb", "Terbium", 65, "3d5/2", 1241.0, dub=True, sp=33.0, br=0.667, sig=1.5, rsf=9.60),
    _r("Dy", "Dysprosium", 66, "3d5/2", 1296.0, dub=True, sp=35.0, br=0.667, sig=1.5, rsf=9.90),
    _r("Ho", "Holmium", 67, "3d5/2", 1351.0, dub=True, sp=37.0, br=0.667, sig=1.5, rsf=10.2),
    _r("Er", "Erbium", 68, "4d5/2", 167.3, dub=True, sp=4.7, br=0.667, sig=1.2, rsf=3.20),
    _r("Tm", "Thulium", 69, "4d5/2", 175.5, dub=True, sp=5.0, br=0.667, sig=1.2, rsf=3.40),
    _r("Yb", "Ytterbium", 70, "4d5/2", 185.0, dub=True, sp=5.3, br=0.667, sig=1.2, rsf=3.60),
    _r("Lu", "Lutetium", 71, "4d5/2", 196.0, dub=True, sp=5.6, br=0.667, sig=1.2, rsf=3.80),
    # --- Z=72-80: Hf through Hg ---
    _r("Hf", "Hafnium", 72, "4f7/2", 14.3, dub=True, sp=1.7, br=0.75, sig=0.6, rsf=1.90,
       cs=[_cs("Hf metal", 14.3, "", ""), _cs("HfO2", 17.1, "", "")]),
    _r("Ta", "Tantalum", 73, "4f7/2", 21.6, dub=True, sp=1.9, br=0.75, sig=0.6, rsf=2.20,
       cs=[_cs("Ta metal", 21.6, "", ""), _cs("Ta2O5", 26.4, "", "")]),
    _r("W", "Tungsten", 74, "4f7/2", 31.4, dub=True, sp=2.2, br=0.75, sig=0.5, rsf=2.50,
       cs=[_cs("W metal", 31.4, "", ""), _cs("WO3", 35.7, "", "")]),
    _r("Re", "Rhenium", 75, "4f7/2", 40.3, dub=True, sp=2.4, br=0.75, sig=0.6, rsf=2.80),
    _r("Os", "Osmium", 76, "4f7/2", 50.7, dub=True, sp=2.7, br=0.75, sig=0.6, rsf=3.10),
    _r("Ir", "Iridium", 77, "4f7/2", 60.9, dub=True, sp=3.0, br=0.75, sig=0.6, rsf=3.40,
       cs=[_cs("Ir metal", 60.9, "", ""), _cs("IrO2", 62.0, "", "")]),
    _r("Pt", "Platinum", 78, "4f7/2", 71.1, dub=True, sp=3.3, br=0.75, sig=0.6, rsf=3.80,
       cs=[_cs("Pt metal", 71.1, "", "Moulder1995"), _cs("PtO", 72.4, "", "Moulder1995"), _cs("PtO2", 74.5, "", "Moulder1995")]),
    _r("Au", "Gold", 79, "4f7/2", 84.0, dub=True, sp=3.7, br=0.75, sig=0.5, rsf=4.20,
       cs=[_cs("Au metal", 84.0, "", "Moulder1995"), _cs("Au2O3", 86.0, "", "Moulder1995")]),
    _r("Hg", "Mercury", 80, "4f7/2", 100.0, dub=True, sp=4.0, br=0.75, sig=0.7, rsf=4.50),
    # --- Z=81-92: Tl through U ---
    _r("Tl", "Thallium", 81, "4f7/2", 117.7, dub=True, sp=4.4, br=0.75, sig=0.8, rsf=4.85),
    _r("Pb", "Lead", 82, "4f7/2", 136.9, dub=True, sp=4.9, br=0.75, sig=0.8, rsf=5.20,
       cs=[_cs("Pb metal", 136.9, "", "Moulder1995"), _cs("PbO", 137.5, "", "Moulder1995"), _cs("PbO2", 137.1, "", "Moulder1995")]),
    _r("Bi", "Bismuth", 83, "4f7/2", 156.9, dub=True, sp=5.3, br=0.75, sig=0.8, rsf=5.60,
       cs=[_cs("Bi metal", 156.9, "", ""), _cs("Bi2O3", 159.0, "", "")]),
    _r("Po", "Polonium", 84, "4f7/2", 177.0, dub=True, sp=5.8, br=0.75, rsf=6.00),
    _r("At", "Astatine", 85, "4f7/2", 195.0, dub=True, sp=6.2, br=0.75, rsf=6.30),
    _r("Rn", "Radon", 86, "4f7/2", 214.0, dub=True, sp=6.6, br=0.75, rsf=6.60),
    _r("Fr", "Francium", 87, "4f7/2", 234.0, dub=True, sp=7.0, br=0.75, rsf=7.00),
    _r("Ra", "Radium", 88, "4f7/2", 254.0, dub=True, sp=7.4, br=0.75, rsf=7.30),
    _r("Ac", "Actinium", 89, "4f7/2", 272.0, dub=True, sp=7.8, br=0.75, rsf=7.60),
    _r("Th", "Thorium", 90, "4f7/2", 333.1, dub=True, sp=8.6, br=0.75, sig=0.8, rsf=8.00,
       cs=[_cs("Th metal", 333.1, "", ""), _cs("ThO2", 333.7, "", "")]),
    _r("Pa", "Protactinium", 91, "4f7/2", 360.0, dub=True, sp=9.0, br=0.75, rsf=8.30),
    _r("U", "Uranium", 92, "4f7/2", 377.3, dub=True, sp=10.8, br=0.75, sig=1.0, rsf=8.60,
       cs=[_cs("U metal", 377.3, "", ""), _cs("UO2", 380.0, "", ""), _cs("UO3", 381.5, "", "")]),
]


# === X-RAY SOURCES ===
XRAY_SOURCES: dict[str, float] = {
    "Al Kα (1486.6 eV)": 1486.6,
    "Mg Kα (1253.6 eV)": 1253.6,
}

# === AUGER LINE DATABASE ===
# Kinetic energies of principal Auger transitions (eV).
# These are source-independent; apparent BE = hν - KE.

@dataclass
class AugerLine:
    element_symbol: str
    element_name: str
    transition: str
    kinetic_energy: float
    typical_sigma: float = 2.5
    relative_intensity: float = 0.5


def _aug(sym, name, trans, ke, sig=2.5, ri=0.5):
    """Shorthand Auger constructor."""
    return AugerLine(sym, name, trans, ke, sig, ri)


AUGER_DB: list[AugerLine] = [
    # KLL transitions — sharp, intense
    _aug("C", "Carbon", "KLL", 263.0, 2.0, 0.6),
    _aug("N", "Nitrogen", "KLL", 379.0, 2.0, 0.6),
    _aug("O", "Oxygen", "KLL", 510.0, 2.5, 0.7),
    _aug("F", "Fluorine", "KLL", 656.0, 2.5, 0.5),
    _aug("Na", "Sodium", "KLL", 990.0, 2.5, 0.8),
    _aug("Mg", "Magnesium", "KLL", 1186.0, 2.5, 0.7),
    _aug("Al", "Aluminum", "KLL", 1393.0, 3.0, 0.6),
    _aug("Si", "Silicon", "KLL", 1619.0, 3.0, 0.5),
    # LMM transitions — broader, moderate intensity
    _aug("P", "Phosphorus", "LMM", 120.0, 2.5, 0.4),
    _aug("S", "Sulfur", "LMM", 152.0, 2.5, 0.5),
    _aug("Cl", "Chlorine", "LMM", 181.0, 2.5, 0.5),
    _aug("K", "Potassium", "LMM", 252.0, 3.0, 0.5),
    _aug("Ca", "Calcium", "LMM", 291.0, 3.0, 0.5),
    _aug("Ti", "Titanium", "LMM", 418.0, 3.0, 0.6),
    _aug("V", "Vanadium", "LMM", 473.0, 3.0, 0.5),
    _aug("Cr", "Chromium", "LMM", 529.0, 3.0, 0.6),
    _aug("Mn", "Manganese", "LMM", 589.0, 3.0, 0.6),
    _aug("Fe", "Iron", "LMM", 703.0, 3.5, 0.7),
    _aug("Co", "Cobalt", "LMM", 775.0, 3.5, 0.6),
    _aug("Ni", "Nickel", "LMM", 848.0, 3.5, 0.7),
    _aug("Cu", "Copper", "LMM", 918.0, 3.0, 0.8),
    _aug("Zn", "Zinc", "LMM", 988.0, 3.0, 0.7),
    _aug("Ga", "Gallium", "LMM", 1068.0, 3.0, 0.5),
    _aug("Ge", "Germanium", "LMM", 1147.0, 3.0, 0.5),
    _aug("As", "Arsenic", "LMM", 1228.0, 3.0, 0.5),
    # MNN transitions — broader
    _aug("Zr", "Zirconium", "MNN", 147.0, 3.5, 0.4),
    _aug("Nb", "Niobium", "MNN", 167.0, 3.5, 0.4),
    _aug("Mo", "Molybdenum", "MNN", 186.0, 3.5, 0.5),
    _aug("Ru", "Ruthenium", "MNN", 228.0, 3.5, 0.4),
    _aug("Rh", "Rhodium", "MNN", 253.0, 3.5, 0.4),
    _aug("Pd", "Palladium", "MNN", 279.0, 3.5, 0.5),
    _aug("Ag", "Silver", "MNN", 351.0, 3.0, 0.6),
    _aug("Cd", "Cadmium", "MNN", 376.0, 3.5, 0.4),
    _aug("In", "Indium", "MNN", 404.0, 3.5, 0.4),
    _aug("Sn", "Tin", "MNN", 430.0, 3.5, 0.5),
    _aug("Sb", "Antimony", "MNN", 454.0, 3.5, 0.4),
    _aug("Te", "Tellurium", "MNN", 483.0, 3.5, 0.4),
    _aug("Ba", "Barium", "MNN", 584.0, 4.0, 0.4),
    _aug("La", "Lanthanum", "MNN", 625.0, 4.0, 0.4),
    _aug("Ce", "Cerium", "MNN", 661.0, 4.0, 0.4),
    _aug("Pb", "Lead", "MNN", 94.0, 3.5, 0.5),
    _aug("Bi", "Bismuth", "MNN", 101.0, 3.5, 0.4),
    # NOO / NVV transitions — broad, lower intensity
    _aug("Hf", "Hafnium", "NOO", 176.0, 4.0, 0.3),
    _aug("Ta", "Tantalum", "NOO", 179.0, 4.0, 0.3),
    _aug("W", "Tungsten", "NOO", 179.0, 4.0, 0.3),
    _aug("Pt", "Platinum", "NOO", 168.0, 4.0, 0.4),
    _aug("Au", "Gold", "NOO", 165.0, 4.0, 0.4),
]


# === LOOKUP FUNCTIONS ===

def get_core_levels_in_range(e_min: float, e_max: float) -> list[CoreLevelRef]:
    """Return all core levels with binding energy in [e_min, e_max]."""
    return [r for r in REFERENCE_DB if e_min <= r.binding_energy <= e_max]


@dataclass
class PeakMatch:
    """A candidate match for an unknown peak position."""
    element_symbol: str
    element_name: str
    line_label: str
    energy: float
    delta: float
    match_type: str  # "core_level" or "auger"


def search_peak(position: float, tolerance: float = 5.0,
                photon_energy: float = 1486.6) -> list[PeakMatch]:
    """Search for core levels and Auger lines matching a peak position.

    Parameters
    ----------
    position : float
        Observed peak position in binding energy (eV).
    tolerance : float
        Search window ± this value (eV).
    photon_energy : float
        X-ray source energy (eV), needed for Auger apparent BE.

    Returns
    -------
    List of PeakMatch sorted by |delta|.
    """
    matches: list[PeakMatch] = []
    e_min = position - tolerance
    e_max = position + tolerance

    # Core levels (primary peak)
    for r in REFERENCE_DB:
        if e_min <= r.binding_energy <= e_max:
            matches.append(PeakMatch(
                element_symbol=r.element_symbol,
                element_name=r.element_name,
                line_label=f"{r.element_symbol} {r.orbital}",
                energy=r.binding_energy,
                delta=position - r.binding_energy,
                match_type="core_level",
            ))
        # Also check the doublet partner
        if r.is_doublet and r.splitting:
            partner_be = r.binding_energy + r.splitting
            if e_min <= partner_be <= e_max:
                # Determine partner orbital label
                orb = r.orbital
                if "3/2" in orb:
                    partner_orb = orb.replace("3/2", "1/2")
                elif "5/2" in orb:
                    partner_orb = orb.replace("5/2", "3/2")
                elif "7/2" in orb:
                    partner_orb = orb.replace("7/2", "5/2")
                else:
                    partner_orb = orb + "'"
                matches.append(PeakMatch(
                    element_symbol=r.element_symbol,
                    element_name=r.element_name,
                    line_label=f"{r.element_symbol} {partner_orb}",
                    energy=partner_be,
                    delta=position - partner_be,
                    match_type="core_level",
                ))

    # Auger lines (convert KE to apparent BE)
    for aug in AUGER_DB:
        apparent_be = photon_energy - aug.kinetic_energy
        if e_min <= apparent_be <= e_max:
            matches.append(PeakMatch(
                element_symbol=aug.element_symbol,
                element_name=aug.element_name,
                line_label=f"{aug.element_symbol} {aug.transition} Auger",
                energy=apparent_be,
                delta=position - apparent_be,
                match_type="auger",
            ))

    matches.sort(key=lambda m: abs(m.delta))
    return matches


def get_core_level(symbol: str, orbital: str) -> CoreLevelRef | None:
    """Look up a specific core level by element symbol and orbital."""
    for r in REFERENCE_DB:
        if r.element_symbol == symbol and r.orbital == orbital:
            return r
    return None


def get_element(symbol: str) -> list[CoreLevelRef]:
    """Return all core levels for an element."""
    return [r for r in REFERENCE_DB if r.element_symbol == symbol]


def get_auger_lines(symbol: str) -> list[AugerLine]:
    """Return all Auger lines for an element."""
    return [a for a in AUGER_DB if a.element_symbol == symbol]


def parse_core_level_label(label: str) -> tuple[str, str] | None:
    """Parse a core level label into (symbol, orbital).

    Examples::

        "C 1s"     -> ("C", "1s")
        "Fe 2p"    -> ("Fe", "2p3/2")
        "Fe 2p3/2" -> ("Fe", "2p3/2")
        "Au 4f7/2" -> ("Au", "4f7/2")
    """
    label = label.strip()
    m = re.match(r"([A-Z][a-z]?)\s+(\d[spdf])(\d/\d)?", label)
    if not m:
        return None
    symbol = m.group(1)
    base_orbital = m.group(2)
    j_part = m.group(3)  # e.g. "3/2" or None

    if j_part:
        orbital = base_orbital + j_part
    else:
        # Default to the primary (higher-j) component
        sub = base_orbital[-1]
        defaults = {"s": "", "p": "3/2", "d": "5/2", "f": "7/2"}
        suffix = defaults.get(sub, "")
        orbital = base_orbital + suffix if suffix else base_orbital

    return (symbol, orbital)
