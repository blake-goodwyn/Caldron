"""Tests for Phase 7 vocabulary canonicalization pipeline."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


# ── CanonicalMap construction ────────────────────────────────────────────

class TestCanonicalMap:
    def test_create_empty(self):
        from vocab_canonicalize import CanonicalMap
        cmap = CanonicalMap()
        assert len(cmap.synonyms) == 0
        assert len(cmap.blocklist) == 0
        assert len(cmap.compound_dishes) == 0

    def test_create_with_data(self):
        from vocab_canonicalize import CanonicalMap
        cmap = CanonicalMap(
            synonyms={"unsalted butter": "butter"},
            blocklist={"amounts"},
            compound_dishes={"hot buttered noodles"},
        )
        assert cmap.synonyms["unsalted butter"] == "butter"
        assert "amounts" in cmap.blocklist
        assert "hot buttered noodles" in cmap.compound_dishes

    def test_save_load_roundtrip(self, tmp_path):
        from vocab_canonicalize import CanonicalMap
        cmap = CanonicalMap(
            synonyms={"virgin olive oil": "olive oil", "unsalted butter": "butter"},
            blocklist={"amounts", "containers"},
            compound_dishes={"hot buttered noodles"},
        )
        path = tmp_path / "test_cmap.json"
        cmap.save(path)
        loaded = CanonicalMap.load(path)
        assert loaded.synonyms == cmap.synonyms
        assert loaded.blocklist == cmap.blocklist
        assert loaded.compound_dishes == cmap.compound_dishes


# ── Synonym collapse ────────────────────────────────────────────────────

class TestSynonymCollapse:
    def test_olive_oil_variants(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={
            "virgin olive oil": "olive oil",
            "light olive oil": "olive oil",
            "extra virgin olive oil": "olive oil",
        })
        assert canonicalize("virgin olive oil", cmap) == "olive oil"
        assert canonicalize("light olive oil", cmap) == "olive oil"
        assert canonicalize("extra virgin olive oil", cmap) == "olive oil"

    def test_butter_variants(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={
            "unsalted butter": "butter",
            "sweet butter": "butter",
            "salted butter": "butter",
        })
        assert canonicalize("unsalted butter", cmap) == "butter"
        assert canonicalize("sweet butter", cmap) == "butter"

    def test_cheese_variants(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={
            "shredded mozzarella": "mozzarella",
            "mozzarella cheese": "mozzarella",
        })
        assert canonicalize("shredded mozzarella", cmap) == "mozzarella"
        assert canonicalize("mozzarella cheese", cmap) == "mozzarella"

    def test_passthrough_valid_ingredient(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={"unsalted butter": "butter"})
        assert canonicalize("garlic", cmap) == "garlic"
        assert canonicalize("salt", cmap) == "salt"
        assert canonicalize("cumin", cmap) == "cumin"


# ── Noise/blocklist removal ─────────────────────────────────────────────

class TestBlocklist:
    def test_noise_tokens_return_none(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(blocklist={
            "amounts", "containers", "pink", "new", "sweet",
        })
        assert canonicalize("amounts", cmap) is None
        assert canonicalize("containers", cmap) is None
        assert canonicalize("pink", cmap) is None
        assert canonicalize("new", cmap) is None

    def test_valid_ingredients_not_blocked(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(blocklist={"amounts", "containers"})
        assert canonicalize("garlic", cmap) is not None
        assert canonicalize("butter", cmap) is not None


# ── Brand stripping ──────────────────────────────────────────────────────

class TestBrandStripping:
    def test_philadelphia_cream_cheese(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={"philadelphia cream cheese": "cream cheese"})
        assert canonicalize("philadelphia cream cheese", cmap) == "cream cheese"

    def test_parkay_margarine(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={"parkay margarine": "margarine"})
        assert canonicalize("parkay margarine", cmap) == "margarine"

    def test_kraft_cheese(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(synonyms={"kraft cheese": "cheese"})
        assert canonicalize("kraft cheese", cmap) == "cheese"


# ── Compound dish removal ────────────────────────────────────────────────

class TestCompoundDishes:
    def test_compound_dishes_return_none(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(compound_dishes={
            "hot buttered noodles",
            "broken spaghetti",
        })
        assert canonicalize("hot buttered noodles", cmap) is None
        assert canonicalize("broken spaghetti", cmap) is None

    def test_real_ingredients_not_removed(self):
        from vocab_canonicalize import canonicalize, CanonicalMap
        cmap = CanonicalMap(compound_dishes={"hot buttered noodles"})
        assert canonicalize("butter", cmap) == "butter"
        assert canonicalize("noodles", cmap) == "noodles"


# ── build_canonical_map ──────────────────────────────────────────────────

class TestBuildCanonicalMap:
    @pytest.fixture
    def sample_vocab(self):
        from data_pipeline import IngredientVocab
        recipes = [
            {"ingredients": ["butter", "unsalted butter", "sweet butter"]},
            {"ingredients": ["olive oil", "virgin olive oil", "light olive oil"]},
            {"ingredients": ["cream cheese", "philadelphia cream cheese"]},
            {"ingredients": ["mozzarella", "shredded mozzarella", "mozzarella cheese"]},
            {"ingredients": ["garlic", "salt", "pepper", "onion"]},
            {"ingredients": ["amounts", "containers", "pink"]},
        ] * 5
        return IngredientVocab(min_count=2).fit(recipes)

    def test_detects_substring_synonyms(self, sample_vocab):
        from vocab_canonicalize import build_canonical_map
        cmap = build_canonical_map(sample_vocab)
        # "unsalted butter" should map to "butter" (shorter, higher frequency)
        assert "unsalted butter" in cmap.synonyms
        canonical = cmap.synonyms["unsalted butter"]
        assert canonical == "butter"

    def test_keeps_shorter_form_as_canonical(self, sample_vocab):
        from vocab_canonicalize import build_canonical_map
        cmap = build_canonical_map(sample_vocab)
        # All olive oil variants should map to "olive oil"
        if "virgin olive oil" in cmap.synonyms:
            assert cmap.synonyms["virgin olive oil"] == "olive oil"

    def test_blocklist_populated(self, sample_vocab):
        from vocab_canonicalize import build_canonical_map
        cmap = build_canonical_map(sample_vocab)
        # Known noise tokens should be in blocklist
        assert len(cmap.blocklist) > 0

    def test_does_not_collapse_different_ingredients(self, sample_vocab):
        from vocab_canonicalize import build_canonical_map
        cmap = build_canonical_map(sample_vocab)
        # garlic, salt, pepper should NOT be synonyms of each other
        assert cmap.synonyms.get("garlic") is None
        assert cmap.synonyms.get("salt") is None

    def test_accepts_overrides(self, sample_vocab, tmp_path):
        from vocab_canonicalize import build_canonical_map
        import json
        overrides_path = tmp_path / "overrides.json"
        with open(overrides_path, "w") as f:
            json.dump({
                "synonyms": {"oleo": "margarine"},
                "blocklist": ["weird_token"],
                "compound_dishes": ["test dish"],
            }, f)
        cmap = build_canonical_map(sample_vocab, overrides_path=overrides_path)
        assert cmap.synonyms.get("oleo") == "margarine"
        assert "weird_token" in cmap.blocklist


# ── Integration with normalize_ingredient ────────────────────────────────

class TestNormalizeWithCanonical:
    def test_normalize_then_canonicalize(self):
        from data_pipeline import normalize_ingredient
        from vocab_canonicalize import CanonicalMap
        cmap = CanonicalMap(synonyms={"unsalted butter": "butter"})
        # normalize_ingredient strips quantities, then canonical map applies
        result = normalize_ingredient("2 tbsp unsalted butter", canonical_map=cmap)
        assert result == "butter"

    def test_normalize_without_canonical_unchanged(self):
        from data_pipeline import normalize_ingredient
        # Without canonical_map, behavior is unchanged (backward compat)
        result = normalize_ingredient("2 tbsp unsalted butter")
        assert "unsalted" in result or "butter" in result

    def test_normalize_blocklist_returns_empty(self):
        from data_pipeline import normalize_ingredient
        from vocab_canonicalize import CanonicalMap
        cmap = CanonicalMap(blocklist={"amounts"})
        result = normalize_ingredient("1 cup amounts", canonical_map=cmap)
        assert result == ""

    def test_normalize_compound_dish_returns_empty(self):
        from data_pipeline import normalize_ingredient
        from vocab_canonicalize import CanonicalMap
        cmap = CanonicalMap(compound_dishes={"hot buttered noodles"})
        result = normalize_ingredient("hot buttered noodles", canonical_map=cmap)
        assert result == ""
