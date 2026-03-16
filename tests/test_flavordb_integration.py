"""Tests for FlavorDB compound integration."""

import sys
import os
import json
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


class TestCompoundAffinity:
    @pytest.fixture
    def sample_flavordb(self):
        return {
            "tomato": ["linalool", "geraniol", "citral", "limonene", "eugenol"],
            "basil": ["linalool", "eugenol", "methyl chavicol", "geraniol"],
            "garlic": ["allicin", "diallyl disulfide", "linalool"],
            "chocolate": ["theobromine", "caffeine", "vanillin"],
            "cinnamon": ["cinnamaldehyde", "eugenol", "linalool"],
        }

    def test_overlap_identical(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        assert ca.overlap("tomato", "tomato") == 1.0

    def test_overlap_partial(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        score = ca.overlap("tomato", "basil")
        assert 0 < score < 1
        # Share linalool, geraniol, eugenol out of union of 6 unique compounds
        # Jaccard = 3/6 = 0.5
        assert abs(score - 0.5) < 0.01

    def test_overlap_no_shared(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        score = ca.overlap("chocolate", "garlic")
        assert score == 0.0

    def test_overlap_unknown_ingredient(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        assert ca.overlap("unicorn", "tomato") == 0.0

    def test_shared_compounds(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        shared = ca.shared_compounds("tomato", "basil")
        assert "linalool" in shared
        assert "eugenol" in shared
        assert "geraniol" in shared

    def test_most_similar(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        results = ca.most_similar("tomato", topn=3)
        assert len(results) > 0
        # Basil should be most similar (shares 3 compounds)
        assert results[0][0] == "basil"

    def test_most_similar_unknown(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        assert ca.most_similar("unicorn") == []

    def test_coverage(self, sample_flavordb):
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(sample_flavordb)
        assert ca.coverage == 5


class TestCombinedAffinityWithCompounds:
    def test_three_signal_blend(self):
        from affinity_models import CompoundAffinity, CombinedAffinity

        class MockF2V:
            def similarity(self, a, b): return 0.8
            def most_similar(self, ing, topn=10):
                return [("butter", 0.8)]

        class MockCF:
            def similar_ingredients(self, ing, topn=10):
                return [("butter", 0.7)]

        flavordb = {"garlic": ["linalool", "allicin"], "butter": ["linalool", "diacetyl"]}
        ca = CompoundAffinity(flavordb)

        combined = CombinedAffinity(MockF2V(), MockCF(), ca, alpha=0.4, beta=0.4, gamma=0.2)
        score = combined.affinity("garlic", "butter")
        # 0.4*0.8 + 0.4*0.7 + 0.2*(1/3) = 0.32 + 0.28 + 0.067 = 0.667
        assert 0.5 < score < 0.8

    def test_without_compounds_redistributes_weight(self):
        from affinity_models import CombinedAffinity

        class MockF2V:
            def similarity(self, a, b): return 0.8
            def most_similar(self, ing, topn=10): return [("b", 0.8)]

        class MockCF:
            def similar_ingredients(self, ing, topn=10): return [("b", 0.6)]

        combined = CombinedAffinity(MockF2V(), MockCF(), None, alpha=0.4, beta=0.4, gamma=0.2)
        # gamma redistributed: alpha=0.5, beta=0.5
        assert combined.alpha == 0.5
        assert combined.beta == 0.5
        assert combined.gamma == 0.0

    def test_top_affinities_includes_compound_candidates(self):
        from affinity_models import CompoundAffinity, CombinedAffinity

        class MockF2V:
            def similarity(self, a, b): return 0.5
            def most_similar(self, ing, topn=10):
                return [("a", 0.9)]

        class MockCF:
            def similar_ingredients(self, ing, topn=10):
                return [("b", 0.8)]

        flavordb = {
            "query": ["compound1", "compound2"],
            "a": ["compound1"],
            "b": ["compound2"],
            "c": ["compound1", "compound2"],  # highest compound overlap
        }
        ca = CompoundAffinity(flavordb)
        combined = CombinedAffinity(MockF2V(), MockCF(), ca)

        results = combined.top_affinities("query", topn=5)
        names = [r[0] for r in results]
        # "c" should appear because it's the compound neighbor
        assert "c" in names


class TestFlavorDBLoader:
    def test_load_real_flavordb(self):
        """Integration test — only runs if flavordb.json exists."""
        path = Path(__file__).parent.parent / "research" / "phase7" / "data" / "flavordb.json"
        if not path.exists():
            pytest.skip("flavordb.json not available")
        with open(path) as f:
            data = json.load(f)
        assert len(data) > 100
        # Tomato should have compounds
        assert "tomato" in data
        assert len(data["tomato"]) > 10

    def test_compound_affinity_on_real_data(self):
        """Integration test with real FlavorDB data."""
        path = Path(__file__).parent.parent / "research" / "phase7" / "data" / "flavordb.json"
        if not path.exists():
            pytest.skip("flavordb.json not available")
        with open(path) as f:
            flavordb = json.load(f)
        from affinity_models import CompoundAffinity
        ca = CompoundAffinity(flavordb)

        # Tomato and basil should have significant compound overlap
        score = ca.overlap("tomato", "basil")
        assert score > 0.1

        # Check shared compounds include known ones
        shared = ca.shared_compounds("tomato", "basil")
        assert len(shared) > 5
