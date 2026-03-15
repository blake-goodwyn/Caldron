"""Unit tests for class_defs.py — data models and persistence."""

import json
import pytest
import networkx as nx


class TestIngredient:
    def test_creation(self, sample_ingredient):
        assert sample_ingredient.name == "flour"
        assert sample_ingredient.quantity == 2.0
        assert sample_ingredient.unit == "cups"

    def test_str_returns_json(self, sample_ingredient):
        result = str(sample_ingredient)
        parsed = json.loads(result)
        assert parsed["name"] == "flour"

    def test_to_json(self, sample_ingredient):
        result = sample_ingredient.to_json()
        parsed = json.loads(result)
        assert parsed["quantity"] == 2.0

    def test_from_json_roundtrip(self, sample_ingredient):
        from class_defs import Ingredient
        json_str = sample_ingredient.model_dump_json()
        restored = Ingredient.model_validate_json(json_str)
        assert restored.name == sample_ingredient.name
        assert restored.quantity == sample_ingredient.quantity


class TestRecipeModification:
    def test_creation(self, sample_modification):
        assert sample_modification.priority == 1
        assert sample_modification.add_ingredient is not None
        assert sample_modification.add_ingredient.name == "sugar"

    def test_serialization_roundtrip(self, sample_modification):
        from class_defs import RecipeModification
        json_str = sample_modification.model_dump_json()
        restored = RecipeModification.model_validate_json(json_str)
        assert restored.priority == 1
        assert restored.add_ingredient.name == "sugar"


class TestRecipe:
    def test_creation(self, sample_recipe):
        assert sample_recipe.name == "Test Bread"
        assert len(sample_recipe.ingredients) == 1
        assert len(sample_recipe.instructions) == 2

    def test_get_id(self, sample_recipe):
        id_val = sample_recipe.get_ID()
        assert isinstance(id_val, str)
        assert len(id_val) > 0

    def test_new_id(self, sample_recipe):
        old_id = sample_recipe.get_ID()
        sample_recipe.new_ID()
        assert sample_recipe.get_ID() != old_id

    def test_tiny(self, sample_recipe):
        tiny = sample_recipe.tiny()
        assert "Test Bread" in tiny

    def test_serialization_roundtrip(self, sample_recipe):
        from class_defs import Recipe
        json_str = sample_recipe.model_dump_json()
        restored = Recipe.model_validate_json(json_str)
        assert restored.name == "Test Bread"
        assert len(restored.ingredients) == 1

    def test_apply_modification_add_ingredient(self, sample_recipe, sample_modification):
        result = sample_recipe.apply_modification(sample_modification)
        assert result is True
        assert len(sample_recipe.ingredients) == 2
        assert sample_recipe.ingredients[-1].name == "sugar"

    def test_apply_modification_remove_ingredient(self, sample_recipe):
        from class_defs import RecipeModification, Ingredient
        mod = RecipeModification(
            priority=1,
            remove_ingredient=Ingredient(name="flour", quantity=0, unit="")
        )
        result = sample_recipe.apply_modification(mod)
        assert result is True
        assert len(sample_recipe.ingredients) == 0

    def test_apply_modification_add_instruction(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1, add_instruction="Let cool")
        result = sample_recipe.apply_modification(mod)
        assert result is True
        assert "Let cool" in sample_recipe.instructions

    def test_apply_modification_remove_instruction(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1, remove_instruction="Bake at 350F")
        result = sample_recipe.apply_modification(mod)
        assert result is True
        assert "Bake at 350F" not in sample_recipe.instructions

    def test_apply_modification_remove_missing_instruction(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1, remove_instruction="Nonexistent step")
        # Should not crash — just log a warning
        result = sample_recipe.apply_modification(mod)
        assert result is True

    def test_apply_modification_add_tag(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1, add_tag="vegan")
        result = sample_recipe.apply_modification(mod)
        assert result is True
        assert "vegan" in sample_recipe.tags

    def test_apply_modification_remove_tag(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1, remove_tag="bread")
        result = sample_recipe.apply_modification(mod)
        assert result is True
        assert "bread" not in sample_recipe.tags

    def test_apply_modification_remove_missing_tag(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1, remove_tag="nonexistent")
        # Should not crash
        result = sample_recipe.apply_modification(mod)
        assert result is True

    def test_apply_modification_update_ingredient_quantity(self, sample_recipe, sample_update_modification):
        result = sample_recipe.apply_modification(sample_update_modification)
        assert result is True
        assert sample_recipe.ingredients[0].quantity == 3.0
        assert sample_recipe.ingredients[0].unit == "cups"

    def test_apply_modification_update_ingredient_unit(self, sample_recipe):
        from class_defs import RecipeModification, Ingredient
        mod = RecipeModification(
            priority=1,
            update_ingredient=Ingredient(name="flour", quantity=2.0, unit="grams")
        )
        result = sample_recipe.apply_modification(mod)
        assert result is True
        assert sample_recipe.ingredients[0].unit == "grams"

    def test_apply_modification_update_nonexistent_ingredient(self, sample_recipe):
        from class_defs import RecipeModification, Ingredient
        mod = RecipeModification(
            priority=1,
            update_ingredient=Ingredient(name="butter", quantity=1.0, unit="tbsp")
        )
        result = sample_recipe.apply_modification(mod)
        assert result is True
        # No ingredient named "butter" — no change should occur
        assert len(sample_recipe.ingredients) == 1
        assert sample_recipe.ingredients[0].name == "flour"

    def test_apply_modification_remove_tag_when_tags_empty(self):
        from class_defs import Recipe, RecipeModification, Ingredient
        recipe = Recipe(
            name="No Tags",
            ingredients=[Ingredient(name="water", quantity=1, unit="cup")],
            instructions=["Boil"],
            tags=[],
            sources=[]
        )
        mod = RecipeModification(priority=1, remove_tag="ghost")
        result = recipe.apply_modification(mod)
        assert result is True
        assert recipe.tags == []

    def test_apply_modification_add_tag_when_tags_empty(self):
        from class_defs import Recipe, RecipeModification, Ingredient
        recipe = Recipe(
            name="No Tags",
            ingredients=[Ingredient(name="water", quantity=1, unit="cup")],
            instructions=["Boil"],
            tags=[],
            sources=[]
        )
        mod = RecipeModification(priority=1, add_tag="new")
        result = recipe.apply_modification(mod)
        assert result is True
        assert recipe.tags == ["new"]

    def test_apply_modification_empty(self, sample_recipe):
        from class_defs import RecipeModification
        mod = RecipeModification(priority=1)
        result = sample_recipe.apply_modification(mod)
        assert result is False


class TestRecipeGraph:
    def test_init_empty(self):
        from class_defs import RecipeGraph
        graph = RecipeGraph()
        assert graph.get_graph_size() == 0
        assert graph.foundational_recipe_node is None

    def test_create_recipe_graph(self, sample_recipe):
        from class_defs import RecipeGraph
        graph = RecipeGraph()
        node_id = graph.create_recipe_graph(sample_recipe)
        assert graph.get_graph_size() == 1
        assert graph.foundational_recipe_node == node_id

    def test_add_node(self, sample_recipe):
        from class_defs import RecipeGraph, Recipe, Ingredient
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        new_recipe = Recipe(
            name="Modified Bread",
            ingredients=[Ingredient(name="flour", quantity=3, unit="cups")],
            instructions=["Mix more"],
            tags=["bread"],
            sources=[]
        )
        node_id = graph.add_node(new_recipe)
        assert graph.get_graph_size() == 2
        assert graph.foundational_recipe_node == node_id

    def test_get_foundational_recipe(self, sample_recipe):
        from class_defs import RecipeGraph
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        recipe = graph.get_foundational_recipe()
        assert recipe is not None
        assert recipe.name == "Test Bread"

    def test_to_dict_from_dict_roundtrip(self, sample_recipe):
        from class_defs import RecipeGraph
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        data = graph.to_dict()
        restored = RecipeGraph.from_dict(data)
        assert restored.get_graph_size() == 1
        assert restored.foundational_recipe_node == graph.foundational_recipe_node

    def test_get_recipe_empty_graph(self):
        from class_defs import RecipeGraph
        graph = RecipeGraph()
        assert graph.get_recipe() is None

    def test_get_recipe_nonexistent_node(self, sample_graph):
        with pytest.raises(KeyError):
            sample_graph.get_recipe("nonexistent-node-id")

    def test_set_foundational_recipe_not_in_graph(self, sample_graph):
        from class_defs import Recipe, Ingredient
        orphan = Recipe(
            name="Orphan",
            ingredients=[Ingredient(name="air", quantity=1, unit="breath")],
            instructions=["Breathe"],
            tags=[],
            sources=[]
        )
        result = sample_graph.set_foundational_recipe(orphan)
        assert result == "Node not found in graph."

    def test_set_foundational_recipe_empty_graph(self):
        from class_defs import RecipeGraph, Recipe, Ingredient
        graph = RecipeGraph()
        recipe = Recipe(
            name="First",
            ingredients=[Ingredient(name="water", quantity=1, unit="cup")],
            instructions=["Pour"],
            tags=[],
            sources=[]
        )
        graph.set_foundational_recipe(recipe)
        assert graph.get_graph_size() == 1

    def test_get_graph_returns_digraph(self, sample_graph):
        g = sample_graph.get_graph()
        assert isinstance(g, nx.DiGraph)
        assert g.number_of_nodes() == 1

    def test_add_node_creates_edge(self, sample_recipe):
        from class_defs import RecipeGraph, Recipe, Ingredient
        graph = RecipeGraph()
        first_id = graph.create_recipe_graph(sample_recipe)
        new_recipe = Recipe(
            name="V2",
            ingredients=[Ingredient(name="flour", quantity=3, unit="cups")],
            instructions=["Mix more"],
            tags=[],
            sources=[]
        )
        second_id = graph.add_node(new_recipe)
        assert graph.get_graph().has_edge(first_id, second_id)

    def test_to_dict_from_dict_multinode_roundtrip(self, sample_recipe):
        from class_defs import RecipeGraph, Recipe, Ingredient
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        r2 = Recipe(
            name="V2",
            ingredients=[Ingredient(name="flour", quantity=3, unit="cups")],
            instructions=["Mix"],
            tags=[],
            sources=[]
        )
        r3 = Recipe(
            name="V3",
            ingredients=[Ingredient(name="flour", quantity=4, unit="cups")],
            instructions=["Knead"],
            tags=[],
            sources=[]
        )
        graph.add_node(r2)
        graph.add_node(r3)
        data = graph.to_dict()
        restored = RecipeGraph.from_dict(data)
        assert restored.get_graph_size() == 3
        assert restored.get_graph().number_of_edges() == 2

    def test_json_persistence(self, sample_recipe, tmp_path):
        from class_defs import RecipeGraph, save_graph_to_file, load_graph_from_file
        filepath = str(tmp_path / "test_graph.json")
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        save_graph_to_file(graph, filepath)
        loaded = load_graph_from_file(filepath)
        assert loaded.get_graph_size() == 1
        recipe = loaded.get_foundational_recipe()
        assert recipe.name == "Test Bread"


class TestModsList:
    def test_init_empty(self):
        from class_defs import ModsList
        mods = ModsList()
        assert len(mods.queue) == 0

    def test_suggest_mod(self, sample_modification):
        from class_defs import ModsList
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        assert len(mods.queue) == 1

    def test_apply_mod(self, sample_recipe, sample_modification):
        from class_defs import ModsList, RecipeGraph
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        mod, success = mods.apply_mod(graph)
        assert success is True
        assert mod is not None
        assert len(mods.queue) == 0

    def test_apply_mod_empty_queue(self, sample_recipe):
        from class_defs import ModsList, RecipeGraph
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        mods = ModsList()
        mod, success = mods.apply_mod(graph)
        assert success is False
        assert mod is None

    def test_remove_mod_existing(self, sample_modification):
        from class_defs import ModsList
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        assert mods.remove_mod(sample_modification._id) is True
        assert len(mods.queue) == 0

    def test_remove_mod_nonexistent(self, sample_modification):
        from class_defs import ModsList
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        assert mods.remove_mod("nonexistent-id") is False
        assert len(mods.queue) == 1

    def test_push_mod_success(self, sample_recipe, sample_modification):
        from class_defs import ModsList, RecipeGraph
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        mod, success = mods.push_mod(graph)
        assert success is True
        assert mod is not None

    def test_push_mod_empty_queue(self, sample_recipe):
        from class_defs import ModsList, RecipeGraph
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        mods = ModsList()
        mod, success = mods.push_mod(graph)
        assert success is False

    def test_rank_mod_nonexistent(self, sample_modification):
        from class_defs import ModsList
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        # Should be a no-op — no crash
        mods.rank_mod("nonexistent-id", 10)
        assert len(mods.queue) == 1

    def test_get_mods_list_sorted_by_priority(self):
        from class_defs import ModsList, RecipeModification, Ingredient
        mods = ModsList()
        low = RecipeModification(priority=1, add_tag="low")
        high = RecipeModification(priority=10, add_tag="high")
        mid = RecipeModification(priority=5, add_tag="mid")
        mods.suggest_mod(low)
        mods.suggest_mod(high)
        mods.suggest_mod(mid)
        result = mods.get_mods_list()
        tags = [m.add_tag for m in result]
        # get_mods_list sorts by ascending priority number (1=highest priority first)
        assert tags == ["low", "mid", "high"]

    def test_json_persistence(self, sample_modification, tmp_path):
        from class_defs import ModsList, save_mods_list_to_file, load_mods_list_from_file
        filepath = str(tmp_path / "test_mods.json")
        mods = ModsList()
        mods.suggest_mod(sample_modification)
        save_mods_list_to_file(mods, filepath)
        loaded = load_mods_list_from_file(filepath)
        assert len(loaded.queue) == 1


class TestPot:
    def test_init_empty(self):
        from class_defs import Pot
        pot = Pot()
        assert len(pot.recipes) == 0
        assert len(pot.urlList) == 0

    def test_add_recipe(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        assert len(pot.recipes) == 1

    def test_pop_recipe(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        recipe = pot.pop_recipe()
        assert recipe.name == "Test Bread"
        assert len(pot.recipes) == 0

    def test_pop_empty(self):
        from class_defs import Pot
        pot = Pot()
        assert pot.pop_recipe() is None

    def test_add_url(self):
        from class_defs import Pot
        pot = Pot()
        pot.add_url("https://example.com/recipe")
        assert len(pot.urlList) == 1

    def test_add_url_invalid(self):
        from class_defs import Pot
        pot = Pot()
        with pytest.raises(ValueError):
            pot.add_url("not-a-url")

    def test_add_url_duplicate(self):
        from class_defs import Pot
        pot = Pot()
        pot.add_url("https://example.com/recipe")
        pot.add_url("https://example.com/recipe")  # should not raise, just warn
        assert len(pot.urlList) == 1

    def test_clear_pot(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        pot.add_url("https://example.com")
        pot.clear_pot()
        assert len(pot.recipes) == 0
        assert len(pot.urlList) == 0

    def test_remove_recipe_existing(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        assert pot.remove_recipe(sample_recipe._id) is True
        assert len(pot.recipes) == 0

    def test_remove_recipe_nonexistent(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        assert pot.remove_recipe("nonexistent-id") is False
        assert len(pot.recipes) == 1

    def test_get_recipe_existing(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        found = pot.get_recipe(sample_recipe._id)
        assert found is not None
        assert found.name == "Test Bread"

    def test_get_recipe_nonexistent(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        assert pot.get_recipe("nonexistent-id") is None

    def test_remove_url_existing(self):
        from class_defs import Pot
        pot = Pot()
        pot.add_url("https://example.com/recipe")
        assert pot.remove_url("https://example.com/recipe") is True
        assert len(pot.urlList) == 0

    def test_remove_url_nonexistent(self):
        from class_defs import Pot
        pot = Pot()
        assert pot.remove_url("https://example.com/nope") is False

    def test_get_url_existing(self):
        from class_defs import Pot
        pot = Pot()
        pot.add_url("https://example.com/recipe")
        assert pot.get_url("https://example.com/recipe") == "https://example.com/recipe"

    def test_get_url_nonexistent(self):
        from class_defs import Pot
        pot = Pot()
        assert pot.get_url("https://example.com/nope") is None

    def test_get_all_recipes(self, sample_recipe):
        from class_defs import Pot
        pot = Pot()
        pot.add_recipe(sample_recipe)
        recipes = pot.get_all_recipes()
        assert len(recipes) == 1
        assert recipes[0].name == "Test Bread"

    def test_get_all_urls(self):
        from class_defs import Pot
        pot = Pot()
        pot.add_url("https://a.com")
        pot.add_url("https://b.com")
        urls = pot.get_all_urls()
        assert len(urls) == 2

    def test_pop_url(self):
        from class_defs import Pot
        pot = Pot()
        pot.add_url("https://example.com/recipe")
        url = pot.pop_url()
        assert url == "https://example.com/recipe"
        assert len(pot.urlList) == 0

    def test_pop_url_empty(self):
        from class_defs import Pot
        pot = Pot()
        assert pot.pop_url() is None

    def test_json_persistence(self, sample_recipe, tmp_path):
        from class_defs import Pot, save_pot_to_file, load_pot_from_file
        filepath = str(tmp_path / "test_pot.json")
        pot = Pot()
        pot.add_recipe(sample_recipe)
        save_pot_to_file(pot, filepath)
        loaded = load_pot_from_file(filepath)
        assert len(loaded.recipes) == 1
        assert loaded.recipes[0].name == "Test Bread"


class TestFileIOErrors:
    def test_load_graph_missing_file(self, tmp_path):
        from class_defs import load_graph_from_file
        with pytest.raises(FileNotFoundError):
            load_graph_from_file(str(tmp_path / "nope.json"))

    def test_load_mods_list_missing_file(self, tmp_path):
        from class_defs import load_mods_list_from_file
        with pytest.raises(FileNotFoundError):
            load_mods_list_from_file(str(tmp_path / "nope.json"))

    def test_load_pot_missing_file(self, tmp_path):
        from class_defs import load_pot_from_file
        with pytest.raises(FileNotFoundError):
            load_pot_from_file(str(tmp_path / "nope.json"))

    def test_graph_save_load_roundtrip(self, sample_recipe, tmp_path):
        from class_defs import RecipeGraph, save_graph_to_file, load_graph_from_file, Recipe, Ingredient
        filepath = str(tmp_path / "graph.json")
        graph = RecipeGraph()
        graph.create_recipe_graph(sample_recipe)
        r2 = Recipe(
            name="V2",
            ingredients=[Ingredient(name="flour", quantity=3, unit="cups")],
            instructions=["Mix"],
            tags=[],
            sources=[]
        )
        graph.add_node(r2)
        save_graph_to_file(graph, filepath)
        loaded = load_graph_from_file(filepath)
        assert loaded.get_graph_size() == 2
        assert loaded.get_graph().number_of_edges() == 1

    def test_mods_list_save_load_roundtrip(self, tmp_path):
        from class_defs import ModsList, RecipeModification, Ingredient, save_mods_list_to_file, load_mods_list_from_file
        filepath = str(tmp_path / "mods.json")
        mods = ModsList()
        mods.suggest_mod(RecipeModification(priority=1, add_tag="test"))
        mods.suggest_mod(RecipeModification(priority=5, add_ingredient=Ingredient(name="salt", quantity=1, unit="tsp")))
        save_mods_list_to_file(mods, filepath)
        loaded = load_mods_list_from_file(filepath)
        assert len(loaded.queue) == 2

    def test_pot_save_load_roundtrip(self, sample_recipe, tmp_path):
        from class_defs import Pot, save_pot_to_file, load_pot_from_file
        filepath = str(tmp_path / "pot.json")
        pot = Pot()
        pot.add_recipe(sample_recipe)
        pot.add_url("https://example.com")
        save_pot_to_file(pot, filepath)
        loaded = load_pot_from_file(filepath)
        assert len(loaded.recipes) == 1
        assert len(loaded.urlList) == 1
