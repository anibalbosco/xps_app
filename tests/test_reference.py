"""Tests for the XPS reference database."""

from xpsanalysis.reference import (
    REFERENCE_DB, get_core_level, get_core_levels_in_range,
    get_element, parse_core_level_label,
)


class TestDatabase:
    def test_not_empty(self):
        assert len(REFERENCE_DB) > 80

    def test_carbon_1s(self):
        ref = get_core_level("C", "1s")
        assert ref is not None
        assert abs(ref.binding_energy - 284.8) < 1.0
        assert len(ref.chemical_states) >= 5

    def test_iron_2p(self):
        ref = get_core_level("Fe", "2p3/2")
        assert ref is not None
        assert ref.is_doublet
        assert ref.splitting is not None and ref.splitting > 10

    def test_gold_4f(self):
        ref = get_core_level("Au", "4f7/2")
        assert ref is not None
        assert ref.is_doublet
        assert abs(ref.binding_energy - 84.0) < 1.0

    def test_all_have_valid_be(self):
        for ref in REFERENCE_DB:
            assert 0 < ref.binding_energy < 1500, f"{ref.element_symbol} {ref.orbital}"
            assert ref.atomic_number > 0

    def test_doublets_have_splitting(self):
        for ref in REFERENCE_DB:
            if ref.is_doublet:
                assert ref.splitting is not None and ref.splitting > 0
                assert ref.branching_ratio is not None and 0 < ref.branching_ratio < 1


class TestLookups:
    def test_range_query(self):
        hits = get_core_levels_in_range(280, 292)
        symbols = {h.element_symbol for h in hits}
        assert "C" in symbols

    def test_element_query(self):
        refs = get_element("Si")
        assert len(refs) >= 1
        orbitals = {r.orbital for r in refs}
        assert "2p3/2" in orbitals

    def test_missing_element(self):
        assert get_core_level("Xx", "1s") is None


class TestParseCoreLevel:
    def test_simple(self):
        assert parse_core_level_label("C 1s") == ("C", "1s")

    def test_p_default(self):
        assert parse_core_level_label("Fe 2p") == ("Fe", "2p3/2")

    def test_explicit_j(self):
        assert parse_core_level_label("Fe 2p3/2") == ("Fe", "2p3/2")

    def test_d_default(self):
        assert parse_core_level_label("Ag 3d") == ("Ag", "3d5/2")

    def test_f_default(self):
        assert parse_core_level_label("Au 4f") == ("Au", "4f7/2")

    def test_invalid(self):
        assert parse_core_level_label("garbage") is None
