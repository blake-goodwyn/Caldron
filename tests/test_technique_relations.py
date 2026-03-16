"""Tests for recipe technique relations (Phase 7 M1)."""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


# ── Technique extraction ─────────────────────────────────────────────────

class TestTechniqueExtraction:
    def test_multiple_techniques(self):
        from affinity_models import extract_techniques_from_instructions
        text = "Roast the chicken at 400F for 30 minutes. Meanwhile, saute the vegetables."
        techniques = extract_techniques_from_instructions(text)
        assert "roast" in techniques
        assert "saute" in techniques

    def test_no_techniques(self):
        from affinity_models import extract_techniques_from_instructions
        text = "Mix all ingredients and serve."
        techniques = extract_techniques_from_instructions(text)
        assert len(techniques) == 0 or "mix" not in techniques

    def test_handles_noisy_text(self):
        from affinity_models import extract_techniques_from_instructions
        text = "Step 1: Preheat oven. Step 2: Bake for 25 min!!!"
        techniques = extract_techniques_from_instructions(text)
        assert "bake" in techniques

    def test_handles_empty_string(self):
        from affinity_models import extract_techniques_from_instructions
        assert extract_techniques_from_instructions("") == []

    def test_conjugated_verbs(self):
        from affinity_models import extract_techniques_from_instructions
        text = "The bread was baked until golden. Simmer the sauce."
        techniques = extract_techniques_from_instructions(text)
        assert "bake" in techniques
        assert "simmer" in techniques


# ── Ingredient-technique matrix ──────────────────────────────────────────

class TestIngredientTechniqueMatrix:
    def test_builds_matrix(self):
        from affinity_models import build_ingredient_technique_matrix
        from data_pipeline import IngredientVocab

        recipes = [
            {"ingredients": ["chicken", "garlic"], "instructions": "Roast the chicken with garlic."},
            {"ingredients": ["chicken", "butter"], "instructions": "Saute chicken in butter."},
            {"ingredients": ["flour", "butter"], "instructions": "Bake at 350F for 30 minutes."},
        ] * 5
        vocab = IngredientVocab(min_count=1).fit(recipes)

        matrix, ing_names, tech_names = build_ingredient_technique_matrix(recipes, vocab)
        assert matrix.shape[0] > 0
        assert matrix.shape[1] > 0
        assert "roast" in tech_names or "saute" in tech_names

    def test_empty_instructions(self):
        from affinity_models import build_ingredient_technique_matrix
        from data_pipeline import IngredientVocab

        recipes = [{"ingredients": ["a", "b"]}]
        vocab = IngredientVocab(min_count=1).fit(recipes)
        matrix, ing_names, tech_names = build_ingredient_technique_matrix(recipes, vocab)
        assert len(tech_names) == 0


# ── TechniqueNMF ─────────────────────────────────────────────────────────

class TestTechniqueNMF:
    def test_fit_and_inspect(self):
        from affinity_models import TechniqueNMF
        import numpy as np

        # Synthetic matrix: 10 ingredients x 5 techniques
        matrix = np.random.rand(10, 5).astype(np.float32)
        ing_names = [f"ing_{i}" for i in range(10)]
        tech_names = ["bake", "roast", "fry", "boil", "steam"]

        nmf = TechniqueNMF(n_components=3)
        nmf.fit(matrix, ing_names, tech_names)

        components = nmf.inspect_components(top_n=3)
        assert len(components) == 3
        assert "top_ingredients" in components[0]
        assert "top_techniques" in components[0]

    def test_similar_by_technique_profile(self):
        from affinity_models import TechniqueNMF
        import numpy as np

        matrix = np.random.rand(10, 5).astype(np.float32)
        ing_names = [f"ing_{i}" for i in range(10)]
        tech_names = ["bake", "roast", "fry", "boil", "steam"]

        nmf = TechniqueNMF(n_components=3)
        nmf.fit(matrix, ing_names, tech_names)

        similar = nmf.similar_by_technique_profile("ing_0", topn=3)
        assert len(similar) == 3
        names = [s[0] for s in similar]
        assert "ing_0" not in names  # should exclude self


# ── KG cooked_by triples ─────────────────────────────────────────────────

class TestCookedByTriples:
    def test_adds_cooked_by_triples(self):
        from knowledge_graph import FoodKnowledgeGraph
        from data_pipeline import IngredientVocab

        recipes = [
            {"ingredients": ["chicken", "garlic"], "directions": "Roast the chicken with garlic."},
        ] * 10
        vocab = IngredientVocab(min_count=1).fit(recipes)

        kg = FoodKnowledgeGraph()
        kg.add_cooked_by_triples(recipes, vocab, min_count=3)

        assert kg.num_triples > 0
        # Should have chicken cooked_by technique:roast
        found = any(
            h == "chicken" and r == "cooked_by" and "roast" in t
            for h, r, t in kg.triples
        )
        assert found

    def test_respects_min_count(self):
        from knowledge_graph import FoodKnowledgeGraph
        from data_pipeline import IngredientVocab

        recipes = [
            {"ingredients": ["chicken"], "directions": "Roast chicken."},
        ] * 2  # Only 2 occurrences
        vocab = IngredientVocab(min_count=1).fit(recipes)

        kg = FoodKnowledgeGraph()
        kg.add_cooked_by_triples(recipes, vocab, min_count=5)
        assert kg.num_triples == 0  # Below threshold

    def test_no_directions_produces_no_triples(self):
        from knowledge_graph import FoodKnowledgeGraph
        from data_pipeline import IngredientVocab

        recipes = [{"ingredients": ["chicken"]}]
        vocab = IngredientVocab(min_count=1).fit(recipes)

        kg = FoodKnowledgeGraph()
        kg.add_cooked_by_triples(recipes, vocab, min_count=1)
        assert kg.num_triples == 0


# ── ML service suggest_techniques ────────────────────────────────────────

class TestMLServiceTechniques:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        from ml_service import CulinaryMLService
        CulinaryMLService.reset()
        yield
        CulinaryMLService.reset()

    def test_suggest_techniques_with_mock(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._technique_data = {
            "chicken": Counter({"roast": 50, "grill": 30, "fry": 20, "braise": 10}),
        }
        with patch.object(service, '_normalize', return_value='chicken'):
            results = service.suggest_techniques("chicken", n=3)
        assert len(results) == 3
        assert results[0]["technique"] == "roast"
        assert results[0]["source"] == "technique_cooccurrence"

    def test_returns_empty_when_disabled(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._enabled = False
        assert service.suggest_techniques("chicken") == []


# ── Agent tool ───────────────────────────────────────────────────────────

class TestSuggestTechniquesTool:
    def test_returns_valid_json(self):
        from agent_tools import suggest_techniques_for_ingredient
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.suggest_techniques.return_value = [
            {"technique": "roast", "score": 0.45, "source": "technique_cooccurrence"},
        ]
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = suggest_techniques_for_ingredient.invoke({"ingredient": "chicken"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["technique"] == "roast"

    def test_returns_error_when_unavailable(self):
        from agent_tools import suggest_techniques_for_ingredient
        with patch("agent_tools._get_ml_service", return_value=None):
            result = suggest_techniques_for_ingredient.invoke({"ingredient": "chicken"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_tool_wired_to_knowitall(self):
        from agent_defs import prompts_dict
        tools = prompts_dict["KnowItAll"]["tools"]
        tool_names = [t.name for t in tools]
        assert "suggest_techniques_for_ingredient" in tool_names
