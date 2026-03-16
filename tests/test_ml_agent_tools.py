"""Tests for ML-backed agent tools."""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


class TestSuggestIngredientSubstitution:
    def test_returns_valid_json(self):
        from agent_tools import suggest_ingredient_substitution
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.suggest_substitutions.return_value = [
            {"name": "margarine", "score": 0.93, "source": "food2vec"},
        ]
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = suggest_ingredient_substitution.invoke({"ingredient": "butter"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "margarine"

    def test_returns_error_when_unavailable(self):
        from agent_tools import suggest_ingredient_substitution
        with patch("agent_tools._get_ml_service", return_value=None):
            result = suggest_ingredient_substitution.invoke({"ingredient": "butter"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_returns_message_for_unknown_ingredient(self):
        from agent_tools import suggest_ingredient_substitution
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.suggest_substitutions.return_value = []
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = suggest_ingredient_substitution.invoke({"ingredient": "xyzzy"})
        parsed = json.loads(result)
        assert "message" in parsed

    def test_respects_count_parameter(self):
        from agent_tools import suggest_ingredient_substitution
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.suggest_substitutions.return_value = [
            {"name": "a", "score": 0.9, "source": "food2vec"},
            {"name": "b", "score": 0.8, "source": "food2vec"},
            {"name": "c", "score": 0.7, "source": "food2vec"},
        ]
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = suggest_ingredient_substitution.invoke(
                {"ingredient": "butter", "count": 3}
            )
        mock_service.suggest_substitutions.assert_called_once_with("butter", n=3)


class TestSuggestRecipeCompletion:
    def test_returns_valid_json(self):
        from agent_tools import suggest_recipe_completion
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.complete_recipe.return_value = [
            {"name": "flour", "score": 0.96, "source": "collaborative_filtering"},
            {"name": "vanilla", "score": 0.89, "source": "collaborative_filtering"},
        ]
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = suggest_recipe_completion.invoke(
                {"ingredients": ["chocolate", "butter", "sugar"]}
            )
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "flour"

    def test_returns_error_when_unavailable(self):
        from agent_tools import suggest_recipe_completion
        with patch("agent_tools._get_ml_service", return_value=None):
            result = suggest_recipe_completion.invoke({"ingredients": ["garlic"]})
        parsed = json.loads(result)
        assert "error" in parsed


class TestGetIngredientAffinity:
    def test_returns_score(self):
        from agent_tools import get_ingredient_affinity
        mock_service = MagicMock()
        mock_service.available = True
        mock_service.score_affinity.return_value = {
            "score": 0.73, "food2vec_score": 0.73, "source": "food2vec"
        }
        with patch("agent_tools._get_ml_service", return_value=mock_service):
            result = get_ingredient_affinity.invoke(
                {"ingredient_a": "garlic", "ingredient_b": "butter"}
            )
        parsed = json.loads(result)
        assert parsed["score"] == 0.73

    def test_returns_error_when_unavailable(self):
        from agent_tools import get_ingredient_affinity
        with patch("agent_tools._get_ml_service", return_value=None):
            result = get_ingredient_affinity.invoke(
                {"ingredient_a": "garlic", "ingredient_b": "butter"}
            )
        parsed = json.loads(result)
        assert "error" in parsed


class TestToolMetadata:
    """Verify tools have proper LangChain metadata for agent binding."""

    def test_substitution_tool_has_name(self):
        from agent_tools import suggest_ingredient_substitution
        assert suggest_ingredient_substitution.name == "suggest_ingredient_substitution"

    def test_completion_tool_has_name(self):
        from agent_tools import suggest_recipe_completion
        assert suggest_recipe_completion.name == "suggest_recipe_completion"

    def test_affinity_tool_has_name(self):
        from agent_tools import get_ingredient_affinity
        assert get_ingredient_affinity.name == "get_ingredient_affinity"

    def test_tools_importable_from_agent_defs(self):
        from agent_defs import prompts_dict
        knowitall_tools = prompts_dict["KnowItAll"]["tools"]
        tool_names = [t.name for t in knowitall_tools]
        assert "suggest_ingredient_substitution" in tool_names
        assert "suggest_recipe_completion" in tool_names
        assert "get_ingredient_affinity" in tool_names
