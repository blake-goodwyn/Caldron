"""Unit tests for agent_tools.py — LangChain tool functions."""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers — create real domain objects used by the tools
# ---------------------------------------------------------------------------

def _make_pot(recipes=None, urls=None):
    from class_defs import Pot
    pot = Pot()
    for r in (recipes or []):
        pot.add_recipe(r)
    for u in (urls or []):
        pot.add_url(u)
    return pot


def _make_graph(recipe=None):
    from class_defs import RecipeGraph
    graph = RecipeGraph()
    if recipe:
        graph.create_recipe_graph(recipe)
    return graph


def _make_mods_list(mods=None):
    from class_defs import ModsList
    ml = ModsList()
    for m in (mods or []):
        ml.suggest_mod(m)
    return ml


# ---------------------------------------------------------------------------
# Datetime tool
# ---------------------------------------------------------------------------

class TestGetDatetime:
    def test_returns_formatted_string(self):
        from agent_tools import get_datetime
        result = get_datetime.invoke({})
        # Should be YYYY-MM-DD HH:MM:SS format
        assert len(result) == 19
        assert result[4] == "-"
        assert result[10] == " "


# ---------------------------------------------------------------------------
# Pot tools
# ---------------------------------------------------------------------------

class TestGenerateIngredient:
    def test_returns_json_string(self):
        from agent_tools import generate_ingredient
        result = generate_ingredient.invoke(
            {"name": "flour", "quantity": 2.0, "unit": "cups"}
        )
        parsed = json.loads(result)
        assert parsed["name"] == "flour"
        assert parsed["quantity"] == 2.0


class TestGenerateRecipe:
    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_creates_recipe_and_adds_to_pot(self, mock_save, mock_load):
        from agent_tools import generate_recipe
        mock_load.return_value = _make_pot()
        result = generate_recipe.invoke({
            "name": "Bread",
            "ingredients": [{"name": "flour", "quantity": 2, "unit": "cups"}],
            "instructions": ["Mix", "Bake"],
            "tags": ["bread"],
            "sources": []
        })
        assert "Bread" in result
        mock_save.assert_called_once()


class TestGetRecipeFromPot:
    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_pop_recipe(self, mock_save, mock_load, sample_recipe):
        mock_load.return_value = _make_pot(recipes=[sample_recipe])
        from agent_tools import get_recipe_from_pot
        result = get_recipe_from_pot.invoke({"recipe_id": None})
        assert "Test Bread" in result

    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_pop_empty_returns_none_str(self, mock_save, mock_load):
        mock_load.return_value = _make_pot()
        from agent_tools import get_recipe_from_pot
        result = get_recipe_from_pot.invoke({"recipe_id": None})
        assert result == "None"


class TestAddUrlToPot:
    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_adds_url(self, mock_save, mock_load):
        mock_load.return_value = _make_pot()
        from agent_tools import add_url_to_pot
        result = add_url_to_pot.invoke({"url": "https://example.com/recipe"})
        assert "added" in result.lower()
        mock_save.assert_called_once()


class TestPopUrlFromPot:
    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_pops_url(self, mock_save, mock_load):
        mock_load.return_value = _make_pot(urls=["https://example.com"])
        from agent_tools import pop_url_from_pot
        result = pop_url_from_pot.invoke({})
        assert "example.com" in result


class TestExaminePot:
    @patch("agent_tools.load_pot_from_file")
    def test_returns_string_representation(self, mock_load, sample_recipe):
        mock_load.return_value = _make_pot(
            recipes=[sample_recipe], urls=["https://example.com"]
        )
        from agent_tools import examine_pot
        result = examine_pot.invoke({})
        assert "example.com" in result


class TestClearPot:
    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_clears(self, mock_save, mock_load, sample_recipe):
        pot = _make_pot(recipes=[sample_recipe])
        mock_load.return_value = pot
        from agent_tools import clear_pot
        result = clear_pot.invoke({})
        assert "cleared" in result.lower()
        assert len(pot.recipes) == 0


# ---------------------------------------------------------------------------
# Recipe Graph tools
# ---------------------------------------------------------------------------

class TestCreateRecipeGraph:
    @patch("agent_tools.load_graph_from_file")
    @patch("agent_tools.save_graph_to_file")
    def test_creates_graph(self, mock_save, mock_load, sample_recipe):
        mock_load.return_value = _make_graph()
        from agent_tools import create_recipe_graph
        result = create_recipe_graph.invoke({"recipe": sample_recipe.model_dump()})
        assert "created" in result.lower()
        mock_save.assert_called_once()


class TestGetRecipe:
    @patch("agent_tools.load_graph_from_file")
    def test_get_foundational(self, mock_load, sample_recipe):
        mock_load.return_value = _make_graph(sample_recipe)
        from agent_tools import get_recipe
        result = get_recipe.invoke({"node_id": None})
        assert "Test Bread" in result


class TestGetFoundationalRecipe:
    @patch("agent_tools.load_graph_from_file")
    def test_returns_recipe(self, mock_load, sample_recipe):
        mock_load.return_value = _make_graph(sample_recipe)
        from agent_tools import get_foundational_recipe
        result = get_foundational_recipe.invoke({})
        assert "Test Bread" in result


class TestGetGraphSize:
    @patch("agent_tools.load_graph_from_file")
    def test_returns_size(self, mock_load, sample_recipe):
        mock_load.return_value = _make_graph(sample_recipe)
        from agent_tools import get_graph_size
        result = get_graph_size.invoke({})
        assert "1" in result


# ---------------------------------------------------------------------------
# Modifications List tools
# ---------------------------------------------------------------------------

class TestSuggestMod:
    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.save_mods_list_to_file")
    def test_suggest_add_ingredient(self, mock_save, mock_load):
        mock_load.return_value = _make_mods_list()
        from agent_tools import suggest_mod
        result = suggest_mod.invoke({
            "priority": 1,
            "add_ingredient": {"name": "sugar", "quantity": 0.5, "unit": "cups"}
        })
        assert "successfully" in result.lower()
        mock_save.assert_called_once()

    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.save_mods_list_to_file")
    def test_suggest_add_instruction(self, mock_save, mock_load):
        mock_load.return_value = _make_mods_list()
        from agent_tools import suggest_mod
        result = suggest_mod.invoke({
            "priority": 2,
            "add_instruction": "Let cool for 10 minutes"
        })
        assert "successfully" in result.lower()


class TestGetModsList:
    @patch("agent_tools.load_mods_list_from_file")
    def test_returns_list_string(self, mock_load, sample_modification):
        mock_load.return_value = _make_mods_list([sample_modification])
        from agent_tools import get_mods_list
        result = get_mods_list.invoke({})
        assert isinstance(result, str)


class TestApplyMod:
    @patch("agent_tools.save_graph_to_file")
    @patch("agent_tools.save_mods_list_to_file")
    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.load_graph_from_file")
    def test_applies_mod(self, mock_load_graph, mock_load_mods,
                         mock_save_mods, mock_save_graph,
                         sample_recipe, sample_modification):
        mock_load_graph.return_value = _make_graph(sample_recipe)
        mock_load_mods.return_value = _make_mods_list([sample_modification])
        from agent_tools import apply_mod
        result = apply_mod.invoke({})
        assert "applied" in result.lower() or "success" in result.lower()

    @patch("agent_tools.save_graph_to_file")
    @patch("agent_tools.save_mods_list_to_file")
    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.load_graph_from_file")
    def test_empty_queue(self, mock_load_graph, mock_load_mods,
                         mock_save_mods, mock_save_graph, sample_recipe):
        mock_load_graph.return_value = _make_graph(sample_recipe)
        mock_load_mods.return_value = _make_mods_list()
        from agent_tools import apply_mod
        result = apply_mod.invoke({})
        assert "no modification" in result.lower() or "no mod" in result.lower()


class TestRankMod:
    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.save_mods_list_to_file")
    def test_rank_mod(self, mock_save, mock_load, sample_modification):
        mock_load.return_value = _make_mods_list([sample_modification])
        from agent_tools import rank_mod
        result = rank_mod.invoke({
            "mod_id": sample_modification._id,
            "new_priority": 5
        })
        assert "reprioritized" in result.lower()


# ---------------------------------------------------------------------------
# Scrape tool (mock external HTTP)
# ---------------------------------------------------------------------------

class TestAddNode:
    @patch("agent_tools.load_graph_from_file")
    @patch("agent_tools.save_graph_to_file")
    def test_adds_node(self, mock_save, mock_load, sample_recipe):
        mock_load.return_value = _make_graph(sample_recipe)
        from agent_tools import add_node
        result = add_node.invoke({
            "recipe_str": {
                "name": "V2 Bread",
                "ingredients": [{"name": "flour", "quantity": 3, "unit": "cups"}],
                "instructions": ["Mix more"],
                "tags": [],
                "sources": []
            }
        })
        assert "added" in result.lower() or "node" in result.lower()
        mock_save.assert_called_once()


class TestSetFoundationalRecipe:
    @patch("agent_tools.load_graph_from_file")
    @patch("agent_tools.save_graph_to_file")
    def test_sets_foundational(self, mock_save, mock_load, sample_recipe):
        graph = _make_graph(sample_recipe)
        node_id = graph.foundational_recipe_node
        mock_load.return_value = graph
        from agent_tools import set_foundational_recipe
        result = set_foundational_recipe.invoke({"node_id": node_id})
        assert "foundational" in result.lower()
        mock_save.assert_called_once()


class TestGetGraph:
    @patch("agent_tools.load_graph_from_file")
    def test_returns_graph_string(self, mock_load, sample_recipe):
        mock_load.return_value = _make_graph(sample_recipe)
        from agent_tools import get_graph
        result = get_graph.invoke({})
        assert "nodes" in result.lower() or "Node" in result
        assert "edge" in result.lower() or "Edge" in result


class TestRemoveMod:
    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.save_mods_list_to_file")
    def test_removes_existing_mod(self, mock_save, mock_load, sample_modification):
        mock_load.return_value = _make_mods_list([sample_modification])
        from agent_tools import remove_mod
        result = remove_mod.invoke({"mod_id": sample_modification._id})
        assert "successfully" in result.lower() or "removed" in result.lower()

    @patch("agent_tools.load_mods_list_from_file")
    @patch("agent_tools.save_mods_list_to_file")
    def test_remove_nonexistent_mod(self, mock_save, mock_load, sample_modification):
        mock_load.return_value = _make_mods_list([sample_modification])
        from agent_tools import remove_mod
        result = remove_mod.invoke({"mod_id": "nonexistent"})
        assert "failed" in result.lower()


class TestGetRecipeFromPotWithId:
    @patch("agent_tools.load_pot_from_file")
    @patch("agent_tools.save_pot_to_file")
    def test_get_by_specific_id(self, mock_save, mock_load, sample_recipe):
        mock_load.return_value = _make_pot(recipes=[sample_recipe])
        from agent_tools import get_recipe_from_pot
        result = get_recipe_from_pot.invoke({"recipe_id": sample_recipe._id})
        assert "Test Bread" in result


class TestScrapeRecipeInfo:
    @patch("agent_tools.scrape_me")
    def test_scrape_success(self, mock_scrape):
        scraper = MagicMock()
        scraper.title.return_value = "Mock Soup"
        scraper.ingredients.return_value = ["water", "salt"]
        scraper.instructions_list.return_value = ["Boil water", "Add salt"]
        mock_scrape.return_value = scraper

        from agent_tools import scrape_recipe_info
        result = scrape_recipe_info.invoke({"url": "https://example.com/soup"})
        assert result["name"] == "Mock Soup"
        assert "water" in result["ingredients"]
        assert result["source"] == "https://example.com/soup"

    @patch("agent_tools.scrape_me")
    def test_scrape_connection_error(self, mock_scrape):
        import requests
        mock_scrape.side_effect = requests.RequestException("timeout")
        from agent_tools import scrape_recipe_info
        result = scrape_recipe_info.invoke({"url": "https://bad.example.com"})
        assert result["source"] == "https://bad.example.com"
        assert "name" not in result

    @patch("agent_tools.scrape_me")
    def test_scrape_partial_failure(self, mock_scrape):
        """Ingredients succeed but instructions raise."""
        scraper = MagicMock()
        scraper.title.return_value = "Partial Soup"
        scraper.ingredients.return_value = ["water"]
        scraper.instructions_list.side_effect = AttributeError("no instructions")
        mock_scrape.return_value = scraper
        from agent_tools import scrape_recipe_info
        result = scrape_recipe_info.invoke({"url": "https://example.com/partial"})
        assert result["name"] == "Partial Soup"
        assert result["ingredients"] == ["water"]
        assert "instructions" not in result


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

class TestSuggestModErrors:
    @patch("agent_tools.load_mods_list_from_file")
    def test_file_not_found(self, mock_load):
        mock_load.side_effect = FileNotFoundError("missing file")
        from agent_tools import suggest_mod
        result = suggest_mod.invoke({
            "priority": 1,
            "add_tag": "test"
        })
        assert "failed" in result.lower()


class TestApplyModErrors:
    @patch("agent_tools.load_graph_from_file")
    def test_file_not_found(self, mock_load):
        mock_load.side_effect = FileNotFoundError("missing graph")
        from agent_tools import apply_mod
        result = apply_mod.invoke({})
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# Context manager integration tests (real files)
# ---------------------------------------------------------------------------

class TestContextManagers:
    def test_pot_context_roundtrip(self, state_dir):
        import os
        from class_defs import Pot, save_pot_to_file, load_pot_from_file
        pot_file = os.path.join(state_dir, "recipe_pot.json")
        # pot_context calls load/save with default args bound at import time,
        # so we patch the functions as seen by agent_tools
        with patch("agent_tools.load_pot_from_file", side_effect=lambda: load_pot_from_file(pot_file)), \
             patch("agent_tools.save_pot_to_file", side_effect=lambda p: save_pot_to_file(p, pot_file)):
            import agent_tools
            with agent_tools.pot_context() as pot:
                pot.add_url("https://example.com/ctx-test")
        loaded = load_pot_from_file(pot_file)
        assert "https://example.com/ctx-test" in loaded.urlList

    def test_graph_context_roundtrip(self, state_dir):
        import os
        from class_defs import load_graph_from_file, Recipe, Ingredient
        import agent_tools
        graph_file = os.path.join(state_dir, "recipe_graph.json")
        original = agent_tools.default_graph_file
        try:
            agent_tools.default_graph_file = graph_file
            with agent_tools.graph_context() as graph:
                r2 = Recipe(
                    name="Ctx Recipe",
                    ingredients=[Ingredient(name="water", quantity=1, unit="cup")],
                    instructions=["Pour"],
                    tags=[],
                    sources=[]
                )
                graph.add_node(r2)
            loaded = load_graph_from_file(graph_file)
            assert loaded.get_graph_size() == 2
        finally:
            agent_tools.default_graph_file = original

    def test_mods_context_roundtrip(self, state_dir):
        import os
        from class_defs import load_mods_list_from_file, RecipeModification
        import agent_tools
        mods_file = os.path.join(state_dir, "mods_list.json")
        original = agent_tools.default_mods_list_file
        try:
            agent_tools.default_mods_list_file = mods_file
            with agent_tools.mods_context() as mods:
                mods.suggest_mod(RecipeModification(priority=1, add_tag="ctx"))
            loaded = load_mods_list_from_file(mods_file)
            assert len(loaded.queue) == 1
        finally:
            agent_tools.default_mods_list_file = original
