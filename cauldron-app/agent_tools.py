from langchain_core.tools import tool
from typing import Dict, List, Optional, Annotated, Any
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import json
from dotenv import load_dotenv
from recipe_scrapers import scrape_me
from langchain_core.messages import HumanMessage
from class_defs import load_graph_from_file, save_graph_to_file, default_graph_file, default_pot_file, default_mods_list_file, load_mods_list_from_file, save_mods_list_to_file, load_pot_from_file, save_pot_to_file, Recipe, Ingredient, RecipeModification, RecipeGraph
from logging_util import logger
from datetime import datetime

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search_tool = TavilySearchResults()

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

        try:
            ing: List[str] = scraper.ingredients()
            out["ingredients"] = ing
        except Exception as e:
            logger.error(f"Failed to get ingredients: {e}")

        try:
            inst: List[str] = scraper.instructions_list()
            out["instructions"] = inst
        except Exception as e:
            logger.error(f"Failed to get instructions: {e}")

        try:
            name: str = scraper.title()
            out["name"] = name
        except Exception as e:
            logger.error(f"Failed to get name: {e}")

        return out
    except Exception as e:
        logger.error(f"Failed to scrape recipe: {e}")
        return out

@tool("generate_ingredient", args_schema=Ingredient)
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

@tool("generate_recipe", args_schema=Recipe)
def generate_recipe(
    name: Annotated[str, "The name of the recipe."],
    ingredients: Annotated[List[Ingredient], "A list of Ingredient objects. Example: [{'name': 'flour', 'quantity': 2, 'unit': 'cups'}, {'name': 'sugar', 'quantity': 1, 'unit': 'cup'}]"],
    instructions: Annotated[List[str], "A list of recipe instructions."],
    tags: Annotated[List[str], "A list of recipe tags."] = None,
    sources: Annotated[List[str], "A list of web sources, book references, or other inspirations."] = None
) -> Annotated[str, "A string representation of the Recipe."]:
    """Generate a representation of a Recipe object and adds it to the Pot."""
    logger.debug("Generating representation of Recipe object.")
    logger.debug(f"Name: {name}, Ingredients: {ingredients}, Instructions: {instructions}, Tags: {tags}, Sources: {sources}")
    recipe = Recipe(name=name, ingredients=ingredients, instructions=instructions, tags=tags, sources=sources)
    pot = load_pot_from_file()
    pot.add_recipe(recipe)
    save_pot_to_file(pot)
    return recipe.tiny()

@tool
def get_recipe_from_pot(
    recipe_id: Annotated[Optional[str], "The ID of the recipe to retrieve."]
) -> Annotated[Optional[Recipe], "The Recipe object."]:
    """Get the Recipe object with the specified ID from the Pot."""
    logger.debug("Getting recipe from pot.")
    pot = load_pot_from_file()
    if recipe_id:
        out = str(pot.get_recipe(recipe_id))
    else:
        out = str(pot.pop_recipe())
    save_pot_to_file(pot)
    return out

@tool
def add_url_to_pot(
    url: Annotated[str, "The URL of the recipe to add to the Pot."]
) -> Annotated[str, "Message indicating success or failure."]:
    """Add a URL to the Pot."""
    logger.debug("Adding URL to pot.")
    pot = load_pot_from_file()
    pot.add_url(url)
    save_pot_to_file(pot)
    return "URL added to Pot."

@tool
def pop_url_from_pot() -> Annotated[Optional[str], "The URL popped from the Pot."]:
    """Returns a URL from the Pot."""
    logger.debug("Popping URL from pot.")
    pot = load_pot_from_file()
    url = pot.pop_url()
    save_pot_to_file(pot)
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
    pot = load_pot_from_file()
    pot.clear_pot()
    save_pot_to_file(pot)
    return "Pot cleared."

## Recipe Graph Tools ##
@tool
def create_recipe_graph() -> Annotated[str, "ID of the newly created foundational recipe node."]:
    """Create a new recipe graph with the provided foundational recipe. Typically used to start a new recipe graph."""
    logger.debug("Creating recipe graph with foundational recipe.")
    recipe_graph = load_graph_from_file(default_graph_file)
    pot = load_pot_from_file(default_pot_file)
    node_id = recipe_graph.create_recipe_graph(pot.pop_recipe())
    save_graph_to_file(recipe_graph, default_graph_file)
    return f"Recipe graph created with foundational recipe node ID: {node_id}"

@tool
def get_recipe(
    node_id: Annotated[Optional[str], "The node ID of the recipe to retrieve. If not provided, retrieves the foundational recipe."],
) -> Annotated[Optional[Recipe], "The Recipe object."]:
    """Get the Recipe object at the specified node ID."""
    logger.debug("Getting recipe from recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    recipe = recipe_graph.get_recipe(node_id)
    return str(recipe)

@tool
def add_node(
    recipe_str: Annotated[Recipe, "The representation of the Recipe object of the recipe."],
) -> Annotated[str, "ID of the newly added recipe node."]:
    """Add a new node to the recipe graph with the provided recipe and create an edge from the current foundational recipe."""
    logger.debug("Adding node to recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    recipe = Recipe.from_json(recipe_str)
    node_id = recipe_graph.add_node(recipe)
    save_graph_to_file(recipe_graph, default_graph_file)
    return f"New recipe node added with ID: {node_id}"

@tool
def get_node_id(
    recipe: Annotated[str, "The JSON representation of the dictionary of the Recipe object of the recipe."],
) -> Annotated[Optional[str], "The node ID of the recipe."]:
    """Get the node ID of the foundational recipe."""
    logger.debug("Getting node ID from recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    # TODO - see if the given recipe matches any recipe in the graph
    return str(recipe_graph.get_node_id())

@tool
def get_foundational_recipe() -> Annotated[Optional[Recipe], "The current foundational recipe."]:
    """Get the current foundational recipe."""
    logger.debug("Getting foundational recipe from recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    recipe = recipe_graph.get_foundational_recipe()
    return str(recipe)

@tool
def set_foundational_recipe(
    node_id: Annotated[str, "The node ID of the recipe to set as foundational."],
) -> Annotated[str, "Message indicating success or failure."]:
    """Set the recipe with the specified node ID as the foundational recipe."""
    logger.debug("Setting foundational recipe in recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    recipe = recipe_graph.get_recipe(node_id)
    recipe_graph.set_foundational_recipe(recipe)
    save_graph_to_file(recipe_graph, default_graph_file)
    return f"Foundational recipe set to node ID: {node_id}"

@tool
def get_graph() -> Annotated[str, "A representation of the current recipe graph."]:
    """Get a representation of the current recipe graph."""
    logger.debug("Getting recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    graph = recipe_graph.get_graph()
    nodes = [(node, data['recipe'].to_json()) for node, data in graph.nodes(data=True)]
    edges = list(graph.edges(data=True))
    return f"Recipe Graph: Nodes - {nodes}, Edges - {edges}"

@tool
def get_graph_size() -> Annotated[str, "The number of nodes in the recipe graph."]:
    """Get the number of nodes in the recipe graph."""
    logger.debug("Getting the number of nodes in the recipe graph.")
    recipe_graph = load_graph_from_file(default_graph_file)
    return f"Number of nodes in recipe graph: {recipe_graph.get_graph_size()}"

## Modifications List Tools ##

@tool("suggest_modification", args_schema=RecipeModification)
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
        mods_list = load_mods_list_from_file(default_mods_list_file)
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
        mods_list.suggest_mod(modification)
        save_mods_list_to_file(mods_list, default_mods_list_file)
        return f"Modification suggested successfully. Mod ID: {modification._id}"
    except Exception as e:
        logger.error(f"Failed to suggest modification: {e}")
        return "Failed to suggest modification."

@tool
def get_mods_list() -> Annotated[List[RecipeModification], "The current list of suggested modifications."]:
    """Get the current list of suggested modifications."""
    logger.debug("Getting mods list.")
    mods_list = load_mods_list_from_file(default_mods_list_file)
    current_mods_list = mods_list.get_mods_list()
    return str(current_mods_list)

@tool
def apply_mod() -> Annotated[Dict[str, Any], "The result of applying the modification."]:
    """
    Apply a modification from the modification list to the recipe graph.

    Returns:
        dict: The result of applying the modification.
    """
    try:
        recipe_graph = load_graph_from_file(default_graph_file)
        mods_list = load_mods_list_from_file(default_mods_list_file)
        mod, success = mods_list.apply_mod(recipe_graph)
        save_mods_list_to_file(mods_list, default_mods_list_file)
        save_graph_to_file(recipe_graph, default_graph_file)
        if mod is not None:
            return {"modification": mod.to_dict(), "success": success}
        else:
            return {"error": "No modification was applied."}
    except Exception as e:
        logger.error(f"Failed to apply modification: {e}")
        return {"error": str(e)}

@tool
def rank_mod(
    mod_id: Annotated[str, "The ID of the modification to reprioritize."],
    new_priority: Annotated[int, "The new priority for the modification (1 = highest priority, larger numbers = lower priority)."],
) -> Annotated[List[RecipeModification], "The updated list of modifications."]:
    """Reprioritize a given modification within the mods list.

    The priority ranking options are as follows:
    - A lower numerical value indicates higher priority (e.g., 1 is the highest priority).
    - Larger numerical values indicate lower priority.
    """
    logger.debug("Ranking modification in mods list.")
    mods_list = load_mods_list_from_file(default_mods_list_file)
    mods_list.rank_mod(mod_id, new_priority)
    save_mods_list_to_file(mods_list, default_mods_list_file)
    updated_mods_list = mods_list.get_mods_list()
    return f"Modification reprioritized: {updated_mods_list}"

@tool
def remove_mod(
    mod_id: Annotated[str, "The ID of the modification to remove."],
) -> Annotated[bool, "Indicates whether the modification was successfully removed."]:
    """Remove a modification from the mods list."""
    logger.debug("Removing modification from mods list.")
    mods_list = load_mods_list_from_file(default_mods_list_file)
    result = mods_list.remove_mod(mod_id)
    save_mods_list_to_file(mods_list, default_mods_list_file)
    if not result:
        return f"Failed to remove modification: {mod_id}"
    else:
        return f"Successfully removed modification: {mod_id}"

## Recipe Analysis Tools ##

## TODO - Analyze nutritional information

## TODO - Analyze recipe complexity

## TODO - Suggest potential ingredient substitutions

## TODO - Suggest potential ingredient add-in's

## TODO - Calculate recipe cost

## TODO - Calculate recipe "trendiness"
