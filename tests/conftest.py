"""Shared test fixtures for Caldron test suite."""

import sys
import os
import pytest

# Add cauldron-app to Python path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))

# Set dummy API keys so modules that validate at import time don't fail
os.environ.setdefault("TAVILY_API_KEY", "test-dummy-key")
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for state files."""
    return str(tmp_path)


@pytest.fixture
def sample_ingredient():
    """Create a sample Ingredient for testing."""
    from class_defs import Ingredient
    return Ingredient(name="flour", quantity=2.0, unit="cups")


@pytest.fixture
def sample_recipe(sample_ingredient):
    """Create a sample Recipe for testing."""
    from class_defs import Recipe
    return Recipe(
        name="Test Bread",
        ingredients=[sample_ingredient],
        instructions=["Mix ingredients", "Bake at 350F"],
        tags=["bread", "simple"],
        sources=["https://example.com/bread"]
    )


@pytest.fixture
def sample_modification():
    """Create a sample RecipeModification for testing."""
    from class_defs import RecipeModification, Ingredient
    return RecipeModification(
        priority=1,
        add_ingredient=Ingredient(name="sugar", quantity=0.5, unit="cups")
    )


@pytest.fixture
def sample_update_modification(sample_ingredient):
    """Create a RecipeModification that updates an existing ingredient."""
    from class_defs import RecipeModification, Ingredient
    return RecipeModification(
        priority=2,
        update_ingredient=Ingredient(name="flour", quantity=3.0, unit="cups")
    )


@pytest.fixture
def sample_pot(sample_recipe):
    """Create a Pot pre-loaded with a recipe and a URL."""
    from class_defs import Pot
    pot = Pot()
    pot.add_recipe(sample_recipe)
    pot.add_url("https://example.com/recipe")
    return pot


@pytest.fixture
def sample_graph(sample_recipe):
    """Create a RecipeGraph with a foundational recipe."""
    from class_defs import RecipeGraph
    graph = RecipeGraph()
    graph.create_recipe_graph(sample_recipe)
    return graph


@pytest.fixture
def sample_mods_list(sample_modification):
    """Create a ModsList with one pending modification."""
    from class_defs import ModsList
    mods = ModsList()
    mods.suggest_mod(sample_modification)
    return mods


@pytest.fixture
def state_dir(tmp_path, sample_recipe):
    """Create a tmp directory with pre-written JSON state files for tool tests."""
    from class_defs import (
        RecipeGraph, ModsList, Pot,
        save_graph_to_file, save_mods_list_to_file, save_pot_to_file,
    )
    graph = RecipeGraph()
    graph.create_recipe_graph(sample_recipe)
    save_graph_to_file(graph, str(tmp_path / "recipe_graph.json"))

    mods = ModsList()
    save_mods_list_to_file(mods, str(tmp_path / "mods_list.json"))

    pot = Pot()
    save_pot_to_file(pot, str(tmp_path / "recipe_pot.json"))

    return str(tmp_path)
