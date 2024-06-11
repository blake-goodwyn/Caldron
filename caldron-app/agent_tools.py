from langchain_core.tools import tool
from typing import Dict, List, Optional, Annotated, Any
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import json
from dotenv import load_dotenv
from recipe_scrapers import scrape_me
from langchain_core.messages import HumanMessage
from class_defs import load_graph_from_file, save_graph_to_file, default_graph_file, default_mods_list_file, load_mods_list_from_file, save_mods_list_to_file, load_pot_from_file, save_pot_to_file, Recipe, Ingredient, RecipeModification, RecipeGraph
from logging_util import logger
from datetime import datetime
from custom_print import printer
from langchain.pydantic_v1 import BaseModel, Field, root_validator

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search_tool = TavilySearchResults()

@tool
def get_datetime() -> Annotated[str, "The current datetime."]:
    """Get the current datetime."""
    try:
        logger.debug("Getting current datetime.")
        return str(datetime.now())
    except Exception as e:
        logger.error(f"Failed to get current datetime: {e}")
        return "Failed to get current datetime."

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
    try:
        logger.debug("Generating JSON representation of Ingredient object.")
        logger.debug(f"Name: {name}, Quantity: {quantity}, Unit: {unit}")
        ingredient = Ingredient(name=name, quantity=quantity, unit=unit)
        return str(ingredient)
    except Exception as e:
        logger.error(f"Failed to generate Ingredient: {e}")
        return "Failed to generate Ingredient."


class GenerateRecipeSchema(BaseModel):
    name: str = Field(..., title="The name of the recipe.")
    ingredients: List[Ingredient] = Field(..., title="A list of Ingredient objects.")
    instructions: List[str] = Field(..., title="A list of recipe instructions.")
    tags: Optional[List[str]] = Field(default=None, title="A list of recipe tags.")
    sources: Optional[List[str]] = Field(default=None, title="A list of web sources, book references, or other inspirations.")

    @root_validator
    def validate_inputs(cls, values):
        if not values.get("ingredients"):
            raise ValueError("Ingredients must be provided.")
        if not values.get("instructions"):
            raise ValueError("Instructions must be provided.")
        return values

@tool("generate_recipe", args_schema=GenerateRecipeSchema)
def generate_recipe(
    name: Annotated[str, "The name of the recipe."],
    ingredients: Annotated[List[Ingredient], "A list of Ingredient objects. Example: [{'name': 'flour', 'quantity': 2, 'unit': 'cups'}, {'name': 'sugar', 'quantity': 1, 'unit': 'cup'}]"],
    instructions: Annotated[List[str], "A list of recipe instructions."],
    tags: Annotated[List[str], "A list of recipe tags."] = None,
    sources: Annotated[List[str], "A list of web sources, book references, or other inspirations."] = None
) -> Annotated[str, "A string representation of the Recipe."]:
    """Generate a representation of a Recipe object and adds it to the Pot."""
    try:
        logger.debug("Generating representation of Recipe object.")
        logger.debug(f"Name: {name}, Ingredients: {ingredients}, Instructions: {instructions}, Tags: {tags}, Sources: {sources}")
        recipe = Recipe(name=name, ingredients=ingredients, instructions=instructions, tags=tags, sources=sources)
        pot = load_pot_from_file()
        pot.add_recipe(recipe)
        save_pot_to_file(pot)
        return recipe.tiny()
    except Exception as e:
        logger.error(f"Failed to generate Recipe: {e}")
        return "Failed to generate Recipe."

@tool
def get_recipe_from_pot(
    recipe_id: Annotated[Optional[str], "The ID of the recipe to retrieve."]
) -> Annotated[Optional[Recipe], "The Recipe object."]:
    """Get the Recipe object with the specified ID from the Pot."""
    try:
        logger.debug("Getting recipe from pot.")
        pot = load_pot_from_file()
        if recipe_id:
            out = pot.get_recipe(recipe_id).tiny()
        else:
            out = pot.pop_recipe().tiny()
        save_pot_to_file(pot)
        return out
    except Exception as e:
        logger.error(f"Failed to get recipe from Pot: {e}")
        return None

@tool
def move_recipe_to_graph(
    recipe_id: Annotated[str, "The ID of the recipe to move to the recipe graph."],
) -> Annotated[str, "Message indicating success or failure."]:
    """Move the recipe with the specified ID from the Pot to the recipe graph."""
    try:
        logger.debug("Moving recipe from pot to recipe graph.")
        pot = load_pot_from_file()
        recipe = pot.get_recipe(recipe_id)
        recipe_graph = load_graph_from_file(default_graph_file)
        node_id = recipe_graph.add_node(recipe)
        save_graph_to_file(recipe_graph, default_graph_file)
        pot.remove_recipe(recipe_id)
        save_pot_to_file(pot)
        return f"Recipe moved to recipe graph with node ID: {node_id}"
    except Exception as e:
        logger.error(f"Failed to move recipe to recipe graph: {e}")
        return "Failed to move recipe to recipe graph."

@tool
def add_url_to_pot(
    url: Annotated[str, "The URL of the recipe to add to the Pot."]
) -> Annotated[str, "Message indicating success or failure."]:
    """Add a URL to the Pot."""
    try:
        logger.debug("Adding URL to pot.")
        pot = load_pot_from_file()
        pot.add_url(url)
        save_pot_to_file(pot)
        return "URL added to Pot."
    except Exception as e:
        logger.error(f"Failed to add URL to Pot: {e}")
        return "Failed to add URL to Pot."

@tool
def pop_url_from_pot() -> Annotated[Optional[str], "The URL popped from the Pot."]:
    """Returns a URL from the Pot."""
    try:
        logger.debug("Popping URL from pot.")
        pot = load_pot_from_file()
        url = pot.pop_url()
        save_pot_to_file(pot)
        return str(url)
    except Exception as e:
        logger.error(f"Failed to pop URL from Pot: {e}")
        return "Failed to pop URL from Pot."

@tool
def examine_pot() -> Annotated[str, "The string representation of the Pot's contents."]:
    """Get the contents of the Pot."""
    try:
        logger.debug("Dumping pot.")
        pot = load_pot_from_file()
        return str(''.join([str(pot.get_all_recipes()),str(pot.get_all_urls())]))
    except Exception as e:
        logger.error(f"Failed to examine Pot: {e}")
        return "Failed to examine Pot."

@tool
def clear_pot() -> Annotated[str, "Message indicating success or failure."]:
    """Clear the Pot of all recipes."""
    try:
        logger.debug("Clearing pot.")
        pot = load_pot_from_file()
        pot.clear_pot()
        save_pot_to_file(pot)
        return "Pot cleared."
    except Exception as e:
        logger.error(f"Failed to clear Pot: {e}")
        return "Failed to clear Pot."
    
@tool
def validate_recipe() -> Annotated[str, "Full print of recipe or error message."]:
    """Validate the recipe in the Pot."""
    try:
        logger.debug("Validating recipe in Pot.")
        graph = load_graph_from_file()
        recipe = graph.get_foundational_recipe()
        if recipe:
            return str(printer.pprint(recipe))
        else:
            return "No recipe to validate."
    except Exception as e:
        logger.error(f"Failed to validate recipe in Pot: {e}")
        return "Failed to validate recipe in Pot."

## Recipe Graph Tools ##
@tool
def create_recipe_graph(
    recipe: Annotated[Recipe, "The representation of the Recipe object of the foundational recipe."],
) -> Annotated[str, "ID of the newly created foundational recipe node."]:
    """Create a new recipe graph with the provided foundational recipe. Typically used to start a new recipe graph."""
    try:
        logger.debug("Creating recipe graph with foundational recipe.")
        recipe_graph = load_graph_from_file(default_graph_file)
        node_id = recipe_graph.create_recipe_graph(recipe)
        save_graph_to_file(recipe_graph, default_graph_file)
        return f"Recipe graph created with foundational recipe node ID: {node_id}"
    except Exception as e:
        logger.error(f"Failed to create recipe graph: {e}")
        return "Failed to create recipe graph."

@tool
def get_recipe(
    node_id: Annotated[Optional[str], "The node ID of the recipe to retrieve. If not provided, retrieves the foundational recipe."],
) -> Annotated[Recipe, "The Recipe object."]:
    """Get the Recipe object at the specified node ID."""
    try:
        logger.debug("Getting recipe from recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        recipe = recipe_graph.get_recipe(node_id)
        if isinstance(recipe, Recipe):
            return recipe
        elif recipe is None and get_graph_size() == 0:
            return "No recipe found. Recipe graph is empty."
        else:
            return "No recipe found at the specified node ID."
    except Exception as e:
        logger.error(f"Failed to get recipe from recipe graph: {e}")
        return "Failed to get recipe from recipe graph."

@tool
def add_node(
    recipe_str: Annotated[Recipe, "The representation of the Recipe object of the recipe."],
) -> Annotated[str, "ID of the newly added recipe node."]:
    """Add a new node to the recipe graph with the provided recipe and create an edge from the current foundational recipe."""
    try:
        logger.debug("Adding node to recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        recipe = Recipe.from_dict(recipe_str)
        node_id = recipe_graph.add_node(recipe)
        save_graph_to_file(recipe_graph, default_graph_file)
        return f"New recipe node added with ID: {node_id}"
    except Exception as e:
        logger.error(f"Failed to add node to recipe graph: {e}")
        return "Failed to add node to recipe graph."

@tool
def get_node_id(
    recipe: Annotated[str, "The JSON representation of the dictionary of the Recipe object of the recipe."],
) -> Annotated[Optional[str], "The node ID of the recipe."]:
    """Get the node ID of the foundational recipe."""
    try:
        logger.debug("Getting node ID from recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        # TODO - see if the given recipe matches any recipe in the graph
        return str(recipe_graph.get_node_id())
    except Exception as e:
        logger.error(f"Failed to get node ID from recipe graph: {e}")
        return "Failed to get node ID from recipe graph."

@tool
def get_foundational_recipe() -> Annotated[Optional[Recipe], "The current foundational recipe."]:
    """Get the current foundational recipe."""
    try:
        logger.debug("Getting foundational recipe from recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        recipe = recipe_graph.get_foundational_recipe()
        return str(recipe)
    except Exception as e:
        logger.error(f"Failed to get foundational recipe from recipe graph: {e}")
        return None

@tool
def set_foundational_recipe(
    node_id: Annotated[str, "The node ID of the recipe to set as foundational."],
) -> Annotated[str, "Message indicating success or failure."]:
    """Set the recipe with the specified node ID as the foundational recipe."""
    try:
        logger.debug("Setting foundational recipe in recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        recipe = recipe_graph.get_recipe(node_id)
        recipe_graph.set_foundational_recipe(recipe)
        save_graph_to_file(recipe_graph, default_graph_file)
        return f"Foundational recipe set to node ID: {node_id}"
    except Exception as e:
        logger.error(f"Failed to set foundational recipe in recipe graph: {e}")
        return "Failed to set foundational recipe in recipe graph."

@tool
def get_graph() -> Annotated[str, "A representation of the current recipe graph."]:
    """Get a representation of the current recipe graph."""
    try:
        logger.debug("Getting recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        graph = recipe_graph.get_graph()
        nodes = [(node, data['recipe'].to_dict()) for node, data in graph.nodes(data=True)]
        edges = list(graph.edges(data=True))
        return f"Recipe Graph: Nodes - {nodes}, Edges - {edges}"
    except Exception as e:
        logger.error(f"Failed to get recipe graph: {e}")
        return "Failed to get recipe graph."

@tool
def get_graph_size() -> Annotated[str, "The number of nodes in the recipe graph."]:
    """Get the number of nodes in the recipe graph."""
    try:
        logger.debug("Getting the number of nodes in the recipe graph.")
        recipe_graph = load_graph_from_file(default_graph_file)
        return f"Number of nodes in recipe graph: {recipe_graph.get_graph_size()}"
    except Exception as e:
        logger.error(f"Failed to get graph size: {e}")
        return "Failed to get graph size."

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
def get_mods_list() -> Annotated[str, "The current list of suggested modifications."]:
    """Get the current list of suggested modifications."""
    try:
        logger.debug("Getting mods list.")
        mods_list = load_mods_list_from_file(default_mods_list_file)
        current_mods_list = mods_list.get_mods_list()
        return str(current_mods_list)
    except Exception as e:
        logger.error(f"Failed to get mods list: {e}")
        return "Failed to get mods list."

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
) -> Annotated[str, "The updated list of modifications."]:
    """Reprioritize a given modification within the mods list.

    The priority ranking options are as follows:
    - A lower numerical value indicates higher priority (e.g., 1 is the highest priority).
    - Larger numerical values indicate lower priority.
    """
    try:
        logger.debug("Ranking modification in mods list.")
        mods_list = load_mods_list_from_file(default_mods_list_file)
        mods_list.rank_mod(mod_id, new_priority)
        save_mods_list_to_file(mods_list, default_mods_list_file)
        updated_mods_list = mods_list.get_mods_list()
        return f"Modification reprioritized: {updated_mods_list}"
    except Exception as e:
        logger.error(f"Failed to rank modification: {e}")
        return "Failed to rank modification."

@tool
def remove_mod(
    mod_id: Annotated[str, "The ID of the modification to remove."],
) -> Annotated[str, "Indicates whether the modification was successfully removed."]:
    """Remove a modification from the mods list."""
    try:
        logger.debug("Removing modification from mods list.")
        mods_list = load_mods_list_from_file(default_mods_list_file)
        result = mods_list.remove_mod(mod_id)
        save_mods_list_to_file(mods_list, default_mods_list_file)
        if not result:
            return f"Failed to remove modification: {mod_id}"
        else:
            return f"Successfully removed modification: {mod_id}"
    except Exception as e:
        logger.error(f"Failed to remove modification: {e}")
        return "Failed to remove modification."

## Recipe Analysis Tools ##

## TODO - Analyze nutritional information

## TODO - Analyze recipe complexity

## TODO - Suggest potential ingredient substitutions

## TODO - Suggest potential ingredient add-in's

## TODO - Calculate recipe cost

## TODO - Calculate recipe "trendiness"