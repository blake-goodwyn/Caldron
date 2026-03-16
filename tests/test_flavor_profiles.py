"""Tests for molecular flavor profiles (Phase 7 M2)."""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


# ── FlavorProfileAffinity ────────────────────────────────────────────────

class TestFlavorProfileAffinity:
    @pytest.fixture
    def sample_profiles(self):
        return {
            "tomato": {
                "descriptors": ["green", "fruity", "sweet", "acidic"],
                "compounds_with_profiles": {
                    "linalool": ["floral", "sweet"],
                    "geraniol": ["floral", "rose"],
                    "citral": ["citrus", "lemon"],
                },
            },
            "basil": {
                "descriptors": ["green", "sweet", "herbal", "spicy"],
                "compounds_with_profiles": {
                    "linalool": ["floral", "sweet"],
                    "eugenol": ["spicy", "clove"],
                },
            },
            "chocolate": {
                "descriptors": ["bitter", "roasted", "sweet"],
                "compounds_with_profiles": {
                    "vanillin": ["vanilla", "sweet"],
                },
            },
            "garlic": {
                "descriptors": ["pungent", "sulfurous"],
                "compounds_with_profiles": {
                    "allicin": ["garlic", "pungent"],
                },
            },
        }

    def test_descriptor_overlap(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        score = fpa.descriptor_overlap("tomato", "basil")
        # Shared: green, sweet. Union: green, fruity, sweet, acidic, herbal, spicy = 6
        # Jaccard = 2/6 = 0.333
        assert 0.3 < score < 0.4

    def test_descriptor_overlap_no_shared(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        score = fpa.descriptor_overlap("garlic", "chocolate")
        assert score == 0.0

    def test_descriptor_overlap_unknown(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        assert fpa.descriptor_overlap("unicorn", "basil") == 0.0

    def test_explain_pairing_structure(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        result = fpa.explain_pairing("tomato", "basil")
        assert "shared_descriptors" in result
        assert "shared_compounds" in result
        assert "compound_details" in result
        assert "descriptor_score" in result
        assert "n_shared_descriptors" in result
        assert "n_shared_compounds" in result

    def test_explain_pairing_content(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        result = fpa.explain_pairing("tomato", "basil")
        assert "green" in result["shared_descriptors"]
        assert "sweet" in result["shared_descriptors"]
        assert "linalool" in result["shared_compounds"]
        assert result["n_shared_compounds"] == 1

    def test_explain_pairing_no_overlap(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        result = fpa.explain_pairing("garlic", "chocolate")
        assert result["n_shared_descriptors"] == 0
        assert result["descriptor_score"] == 0.0

    def test_ingredients_with_profile(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        sweet = fpa.ingredients_with_profile("sweet")
        assert "tomato" in sweet
        assert "basil" in sweet
        assert "chocolate" in sweet
        assert "garlic" not in sweet

    def test_coverage(self, sample_profiles):
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(sample_profiles)
        assert fpa.coverage == 4


# ── Flavor profile parsing ───────────────────────────────────────────────

class TestFlavorProfileParsing:
    def test_parse_set_notation(self):
        import ast
        result = ast.literal_eval("{'sweet', 'umami'}")
        assert isinstance(result, set)
        assert "sweet" in result

    def test_parse_empty_set(self):
        import ast
        result = ast.literal_eval("set()")
        assert isinstance(result, set)
        assert len(result) == 0

    def test_parse_single_element(self):
        import ast
        result = ast.literal_eval("{'fruity'}")
        assert isinstance(result, set)
        assert "fruity" in result

    def test_malformed_graceful(self):
        import ast
        try:
            ast.literal_eval("not a set")
            assert False, "Should have raised"
        except (ValueError, SyntaxError):
            pass  # Expected


# ── ML service explain_pairing ───────────────────────────────────────────

class TestMLServiceExplainPairing:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from ml_service import CulinaryMLService
        CulinaryMLService.reset()
        yield
        CulinaryMLService.reset()

    def test_explain_pairing_with_mock(self):
        from ml_service import CulinaryMLService
        from affinity_models import FlavorProfileAffinity

        service = CulinaryMLService()
        profiles = {
            "tomato": {"descriptors": ["sweet", "green"], "compounds_with_profiles": {"linalool": ["floral"]}},
            "basil": {"descriptors": ["sweet", "herbal"], "compounds_with_profiles": {"linalool": ["floral"]}},
        }
        service._flavor_profiles = FlavorProfileAffinity(profiles)

        mock_f2v = MagicMock()
        mock_f2v.similarity.return_value = 0.65
        service._food2vec = mock_f2v

        with patch.object(service, '_normalize', side_effect=lambda x: x.lower()):
            result = service.explain_pairing("tomato", "basil")

        assert "shared_descriptors" in result
        assert "embedding_similarity" in result
        assert result["embedding_similarity"] == 0.65

    def test_returns_error_when_disabled(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._enabled = False
        result = service.explain_pairing("tomato", "basil")
        assert "error" in result


# ── Agent tool ───────────────────────────────────────────────────────────

class TestExplainPairingTool:
    def test_returns_valid_json(self):
        from agent_tools import explain_ingredient_pairing
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.explain_pairing.return_value = {
            "shared_descriptors": ["sweet", "floral"],
            "shared_compounds": ["linalool"],
            "descriptor_score": 0.45,
        }
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = explain_ingredient_pairing.invoke(
                {"ingredient_a": "tomato", "ingredient_b": "basil"}
            )
        parsed = json.loads(result)
        assert "shared_descriptors" in parsed

    def test_returns_error_when_unavailable(self):
        from agent_tools import explain_ingredient_pairing
        with patch("agent_tools._get_ml_service", return_value=None):
            result = explain_ingredient_pairing.invoke(
                {"ingredient_a": "tomato", "ingredient_b": "basil"}
            )
        parsed = json.loads(result)
        assert "error" in parsed

    def test_tool_wired_to_knowitall(self):
        from agent_defs import prompts_dict
        tools = prompts_dict["KnowItAll"]["tools"]
        tool_names = [t.name for t in tools]
        assert "explain_ingredient_pairing" in tool_names


# ── Integration with real FlavorDB data ──────────────────────────────────

class TestRealFlavorProfiles:
    def test_real_profiles_exist(self):
        path = Path(__file__).parent.parent / "research" / "phase7" / "data" / "ingredient_flavor_profiles.json"
        if not path.exists():
            pytest.skip("ingredient_flavor_profiles.json not available")
        with open(path) as f:
            profiles = json.load(f)
        assert len(profiles) > 100
        assert "tomato" in profiles
        assert len(profiles["tomato"]["descriptors"]) > 5

    def test_real_explain_pairing(self):
        path = Path(__file__).parent.parent / "research" / "phase7" / "data" / "ingredient_flavor_profiles.json"
        if not path.exists():
            pytest.skip("ingredient_flavor_profiles.json not available")
        with open(path) as f:
            profiles = json.load(f)
        from affinity_models import FlavorProfileAffinity
        fpa = FlavorProfileAffinity(profiles)
        result = fpa.explain_pairing("tomato", "basil")
        assert result["n_shared_compounds"] > 0
        assert result["n_shared_descriptors"] > 0
