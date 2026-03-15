"""Shared test fixtures for Caldron test suite."""

import sys
import os
import pytest
import tempfile

# Add cauldron-app to Python path so tests can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))


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
