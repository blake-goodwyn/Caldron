from contextlib import contextmanager
from contextvars import ContextVar
from langchain_core.tools import tool
from typing import Dict, List, Optional, Annotated, Any
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import requests
from recipe_scrapers import scrape_me
from langchain_core.messages import HumanMessage
from class_defs import load_graph_from_file, save_graph_to_file, default_graph_file, default_mods_list_file, default_pot_file, load_mods_list_from_file, save_mods_list_to_file, load_pot_from_file, save_pot_to_file, Recipe, Ingredient, RecipeModification, RecipeGraph
from logging_util import logger
from datetime import datetime

tavily_search_tool = TavilySearchResults()

# Per-session state file paths (defaults to module-level constants for CLI compatibility)
_graph_file: ContextVar[str] = ContextVar('_graph_file', default=default_graph_file)
_mods_file: ContextVar[str] = ContextVar('_mods_file', default=default_mods_list_file)
_pot_file: ContextVar[str] = ContextVar('_pot_file', default=default_pot_file)


# Context managers to reduce duplicated load/save patterns
@contextmanager
def pot_context():
    f = _pot_file.get()
    pot = load_pot_from_file(f)
    yield pot
    save_pot_to_file(pot, f)

@contextmanager
def graph_context():
    f = _graph_file.get()
    recipe_graph = load_graph_from_file(f)
    yield recipe_graph
    save_graph_to_file(recipe_graph, f)

@contextmanager
def mods_context():
    f = _mods_file.get()
    mods_list = load_mods_list_from_file(f)
    yield mods_list
    save_mods_list_to_file(mods_list, f)

## Datetime Tool (mainly for dummy use)

@tool
def get_datetime() -> Annotated[str, "The current date and time."]:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_user_input() -> Annotated[str, "The user's input."]:
    """Get the user's input."""
    return HumanMessage(content=input("Enter your input: "))

## Recipe Manipulation Tools

@tool
def scrape_recipe_info(
    url: Annotated[str, "The URL of the recipe to scrape."]
) -> Annotated[Dict[str, Optional[List[str]]], "A dictionary containing the recipe's name, ingredients, instructions, and tags."]:
    """
    Fetches recipe information from a given URL.

    Args:
        url (str): The URL of the recipe to scrape.

    Returns:
        dict: A dictionary containing the recipe's name, ingredients, instructions, and tags.
    """

    out: Dict[str, Optional[List[str]]] = {}
    out["source"] = url
    try:
        scraper = scrape_me(url, wild_mode=True)
    except (requests.RequestException, ConnectionError) as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return out

    try:
        out["ingredients"] = scraper.ingredients()
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to get ingredients: {e}")

    try:
        out["instructions"] = scraper.instructions_list()
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to get instructions: {e}")

    try:
        out["name"] = scraper.title()
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to get name: {e}")

    return out

@tool
def generate_ingredient(
    name: Annotated[str, "The name of the ingredient."],
    quantity: Annotated[float, "The quantity of the ingredient."],
    unit: Annotated[str, "The unit of the ingredient."]
) -> Annotated[str, "The JSON representation of the ingredient."]:
    """Generate a JSON representation of an Ingredient object."""
    logger.debug("Generating JSON representation of Ingredient object.")
    logger.debug(f"Name: {name}, Quantity: {quantity}, Unit: {unit}")
    ingredient = Ingredient(name=name, quantity=quantity, unit=unit)
    return str(ingredient)

@tool
def generate_recipe(
    name: Annotated[str, "The name of the recipe."],
    ingredients: Annotated[List[Dict[str, Any]], "A list of ingredient dicts. Example: [{'name': 'flour', 'quantity': 2, 'unit': 'cups'}]"],
    instructions: Annotated[List[str], "A list of recipe instructions."],
    tags: Annotated[List[str], "A list of recipe tags."] = None,
    sources: Annotated[List[str], "A list of web sources, book references, or other inspirations."] = None
) -> Annotated[str, "A string representation of the Recipe."]:
    """Generate a representation of a Recipe object and adds it to the Pot."""
    logger.debug("Generating representation of Recipe object.")
    parsed_ingredients = [Ingredient(**ing) if isinstance(ing, dict) else ing for ing in ingredients]
    recipe = Recipe(name=name, ingredients=parsed_ingredients, instructions=instructions, tags=tags, sources=sources)
    with pot_context() as pot:
        pot.add_recipe(recipe)
    return recipe.tiny()

@tool
def get_recipe_from_pot(
    recipe_id: Annotated[Optional[str], "The ID of the recipe to retrieve."]
) -> Annotated[str, "String representation of the Recipe object."]:
    """Get the Recipe object with the specified ID from the Pot."""
    logger.debug("Getting recipe from pot.")
    with pot_context() as pot:
        if recipe_id:
            out = str(pot.get_recipe(recipe_id))
        else:
            out = str(pot.pop_recipe())
    return out

@tool
def add_url_to_pot(
    url: Annotated[str, "The URL of the recipe to add to the Pot."]
) -> Annotated[str, "Message indicating success or failure."]:
    """Add a URL to the Pot."""
    logger.debug("Adding URL to pot.")
    with pot_context() as pot:
        pot.add_url(url)
    return "URL added to Pot."

@tool
def pop_url_from_pot() -> Annotated[Optional[str], "The URL popped from the Pot."]:
    """Returns a URL from the Pot."""
    logger.debug("Popping URL from pot.")
    with pot_context() as pot:
        url = pot.pop_url()
    return str(url)

@tool
def examine_pot() -> Annotated[str, "The string representation of the Pot's contents."]:
    """Get the contents of the Pot."""
    logger.debug("Dumping pot.")
    pot = load_pot_from_file()
    return str(''.join([str(pot.get_all_recipes()),str(pot.get_all_urls())]))

@tool
def clear_pot() -> Annotated[str, "Message indicating success or failure."]:
    """Clear the Pot of all recipes."""
    logger.debug("Clearing pot.")
    with pot_context() as pot:
        pot.clear_pot()
    return "Pot cleared."

## Recipe Graph Tools ##
@tool
def create_recipe_graph(
    recipe: Annotated[Dict[str, Any], "The representation of the Recipe object of the foundational recipe."],
) -> Annotated[str, "ID of the newly created foundational recipe node."]:
    """Create a new recipe graph with the provided foundational recipe. Typically used to start a new recipe graph."""
    logger.debug("Creating recipe graph with foundational recipe.")
    parsed_recipe = Recipe.model_validate(recipe) if isinstance(recipe, dict) else recipe
    with graph_context() as recipe_graph:
        node_id = recipe_graph.create_recipe_graph(parsed_recipe)
    return f"Recipe graph created with foundational recipe node ID: {node_id}"

@tool
def get_recipe(
    node_id: Annotated[Optional[str], "The node ID of the recipe to retrieve. If not provided, retrieves the foundational recipe."],
) -> Annotated[str, "String representation of the Recipe object."]:
    """Get the Recipe object at the specified node ID."""
    logger.debug("Getting recipe from recipe graph.")
    recipe_graph = load_graph_from_file(_graph_file.get())
    recipe = recipe_graph.get_recipe(node_id)
    return str(recipe)

@tool
def add_node(
    recipe_str: Annotated[Dict[str, Any], "The representation of the Recipe object of the recipe."],
) -> Annotated[str, "ID of the newly added recipe node."]:
    """Add a new node to the recipe graph with the provided recipe and create an edge from the current foundational recipe."""
    logger.debug("Adding node to recipe graph.")
    with graph_context() as recipe_graph:
        recipe = Recipe.model_validate(recipe_str) if isinstance(recipe_str, dict) else recipe_str
        node_id = recipe_graph.add_node(recipe)
    return f"New recipe node added with ID: {node_id}"

@tool
def get_node_id(
    recipe: Annotated[str, "The JSON representation of the dictionary of the Recipe object of the recipe."],
) -> Annotated[Optional[str], "The node ID of the recipe."]:
    """Get the node ID of the foundational recipe."""
    logger.debug("Getting node ID from recipe graph.")
    recipe_graph = load_graph_from_file(_graph_file.get())
    # TODO - see if the given recipe matches any recipe in the graph
    return str(recipe_graph.get_node_id())

@tool
def get_foundational_recipe() -> Annotated[str, "String representation of the current foundational recipe."]:
    """Get the current foundational recipe."""
    logger.debug("Getting foundational recipe from recipe graph.")
    recipe_graph = load_graph_from_file(_graph_file.get())
    recipe = recipe_graph.get_foundational_recipe()
    return str(recipe)

@tool
def set_foundational_recipe(
    node_id: Annotated[str, "The node ID of the recipe to set as foundational."],
) -> Annotated[str, "Message indicating success or failure."]:
    """Set the recipe with the specified node ID as the foundational recipe."""
    logger.debug("Setting foundational recipe in recipe graph.")
    with graph_context() as recipe_graph:
        recipe = recipe_graph.get_recipe(node_id)
        recipe_graph.set_foundational_recipe(recipe)
    return f"Foundational recipe set to node ID: {node_id}"

@tool
def get_graph() -> Annotated[str, "A representation of the current recipe graph."]:
    """Get a representation of the current recipe graph."""
    logger.debug("Getting recipe graph.")
    recipe_graph = load_graph_from_file(_graph_file.get())
    graph = recipe_graph.get_graph()
    nodes = [(node, data['recipe'].to_json()) for node, data in graph.nodes(data=True)]
    edges = list(graph.edges(data=True))
    return f"Recipe Graph: Nodes - {nodes}, Edges - {edges}"

@tool
def get_graph_size() -> Annotated[str, "The number of nodes in the recipe graph."]:
    """Get the number of nodes in the recipe graph."""
    logger.debug("Getting the number of nodes in the recipe graph.")
    recipe_graph = load_graph_from_file(_graph_file.get())
    return f"Number of nodes in recipe graph: {recipe_graph.get_graph_size()}"

## Modifications List Tools ##

@tool
def suggest_mod(
    priority: Annotated[int, "The priority of the modification."],
    add_ingredient: Annotated[Optional[Dict[str, Any]], "The ingredient to add."] = None,
    remove_ingredient: Annotated[Optional[Dict[str, Any]], "The ingredient to remove."] = None,
    update_ingredient: Annotated[Optional[Dict[str, Any]], "The ingredient to update."] = None,
    add_instruction: Annotated[Optional[str], "The instruction to add."] = None,
    remove_instruction: Annotated[Optional[str], "The instruction to remove."] = None,
    add_tag: Annotated[Optional[str], "The tag to add."] = None,
    remove_tag: Annotated[Optional[str], "The tag to remove."] = None
) -> Annotated[str, "The result of the suggestion."]:
    """
    Suggest a modification to be added to the modification list.
    """
    try:
        modification = RecipeModification(
            priority=priority,
            add_ingredient=add_ingredient,
            remove_ingredient=remove_ingredient,
            update_ingredient=update_ingredient,
            add_instruction=add_instruction,
            remove_instruction=remove_instruction,
            add_tag=add_tag,
            remove_tag=remove_tag
        )
        with mods_context() as mods_list:
            mods_list.suggest_mod(modification)
        return f"Modification suggested successfully. Mod ID: {modification._id}"
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Failed to suggest modification: {e}")
        return f"Failed to suggest modification: {e}"

@tool
def get_mods_list() -> Annotated[str, "String representation of the current list of suggested modifications."]:
    """Get the current list of suggested modifications."""
    logger.debug("Getting mods list.")
    mods_list = load_mods_list_from_file(_mods_file.get())
    current_mods_list = mods_list.get_mods_list()
    return str(current_mods_list)

@tool
def apply_mod() -> Annotated[str, "The result of applying the modification."]:
    """
    Apply a modification from the modification list to the recipe graph.

    Returns:
        str: A message describing the result of applying the modification.
    """
    try:
        gf, mf = _graph_file.get(), _mods_file.get()
        recipe_graph = load_graph_from_file(gf)
        mods_list = load_mods_list_from_file(mf)
        mod, success = mods_list.apply_mod(recipe_graph)
        save_mods_list_to_file(mods_list, mf)
        save_graph_to_file(recipe_graph, gf)
        if mod is not None and success:
            return f"Modification applied successfully: {mod}"
        elif mod is not None:
            return f"Modification failed to apply: {mod}"
        else:
            return "No modifications in queue to apply."
    except (ValueError, FileNotFoundError, KeyError) as e:
        logger.error(f"Failed to apply modification: {e}")
        return f"Error applying modification: {e}"

@tool
def rank_mod(
    mod_id: Annotated[str, "The ID of the modification to reprioritize."],
    new_priority: Annotated[int, "The new priority for the modification (1 = highest priority, larger numbers = lower priority)."],
) -> Annotated[str, "String representation of the updated list of modifications."]:
    """Reprioritize a given modification within the mods list.

    The priority ranking options are as follows:
    - A lower numerical value indicates higher priority (e.g., 1 is the highest priority).
    - Larger numerical values indicate lower priority.
    """
    logger.debug("Ranking modification in mods list.")
    with mods_context() as mods_list:
        mods_list.rank_mod(mod_id, new_priority)
        updated_mods_list = mods_list.get_mods_list()
    return f"Modification reprioritized: {updated_mods_list}"

@tool
def remove_mod(
    mod_id: Annotated[str, "The ID of the modification to remove."],
) -> Annotated[str, "Message indicating whether the modification was successfully removed."]:
    """Remove a modification from the mods list."""
    logger.debug("Removing modification from mods list.")
    with mods_context() as mods_list:
        result = mods_list.remove_mod(mod_id)
    if not result:
        return f"Failed to remove modification: {mod_id}"
    else:
        return f"Successfully removed modification: {mod_id}"

## Recipe Analysis Tools (ML-backed) ##

def _get_ml_service():
    """Get the ML service singleton. Returns None if unavailable."""
    try:
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        if not service.available:
            return None
        return service
    except Exception:
        return None


@tool
def suggest_ingredient_substitution(
    ingredient: Annotated[str, "The ingredient to find substitutes for."],
    count: Annotated[int, "Number of substitutions to return."] = 5,
) -> Annotated[str, "JSON list of substitution suggestions with confidence scores."]:
    """Find ingredient substitutions using ML embeddings. Use when a user asks
    'what can I use instead of X?', 'I don't have X', or 'substitute for X'."""
    service = _get_ml_service()
    if service is None:
        return json.dumps({"error": "ML models not available. Please train models first."})
    results = service.suggest_substitutions(ingredient, n=count)
    if not results:
        return json.dumps({"message": f"No substitutions found for '{ingredient}'."})
    return json.dumps(results)


@tool
def suggest_recipe_completion(
    ingredients: Annotated[List[str], "Current list of ingredient names in the recipe."],
    count: Annotated[int, "Number of ingredient suggestions to return."] = 5,
) -> Annotated[str, "JSON list of suggested ingredients to add, with confidence scores."]:
    """Suggest ingredients to complete a recipe using collaborative filtering.
    Use when building a recipe and wondering what's missing, or when asked
    'what else does this recipe need?' or 'what goes with these ingredients?'."""
    service = _get_ml_service()
    if service is None:
        return json.dumps({"error": "ML models not available. Please train models first."})
    results = service.complete_recipe(ingredients, n=count)
    if not results:
        return json.dumps({"message": "No suggestions found for the given ingredients."})
    return json.dumps(results)


@tool
def get_ingredient_affinity(
    ingredient_a: Annotated[str, "First ingredient."],
    ingredient_b: Annotated[str, "Second ingredient."],
) -> Annotated[str, "Affinity score and breakdown between two ingredients."]:
    """Check how well two ingredients pair together. Returns an affinity score
    from 0 (unrelated) to 1 (strongly paired). Use when asked 'do X and Y
    go together?' or 'how well does X pair with Y?'."""
    service = _get_ml_service()
    if service is None:
        return json.dumps({"error": "ML models not available. Please train models first."})
    result = service.score_affinity(ingredient_a, ingredient_b)
    return json.dumps(result)


@tool
def suggest_techniques_for_ingredient(
    ingredient: Annotated[str, "The ingredient to find cooking techniques for."],
    count: Annotated[int, "Number of technique suggestions to return."] = 5,
) -> Annotated[str, "JSON list of cooking technique suggestions with scores."]:
    """Suggest cooking techniques for an ingredient based on recipe data.
    Use when asked 'how should I cook X?', 'what techniques work with X?',
    or 'should I roast or braise X?'."""
    service = _get_ml_service()
    if service is None:
        return json.dumps({"error": "ML models not available. Please train models first."})
    results = service.suggest_techniques(ingredient, n=count)
    if not results:
        return json.dumps({"message": f"No technique data found for '{ingredient}'."})
    return json.dumps(results)