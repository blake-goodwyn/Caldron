import uuid
import pickle
import os
import json
import networkx as nx
import heapq
from typing import List, Dict, Optional, Any
from logging_util import logger
from langchain.pydantic_v1 import BaseModel, Field
from pprint import pformat

default_mods_list_file = "mods_list.pkl"
default_graph_file="recipe_graph.pkl"

class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    quantity: float = Field(description="Quantity of the ingredient")
    unit: str = Field(description="Unit of measurement for the ingredient")

    def __str__(self):
        return json.dumps(self.dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of Ingredient object.")
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ingredient':
        logger.debug("Creating Ingredient object from dictionary.")
        return Ingredient.parse_raw(str(data))

class RecipeModification(BaseModel):
    id: str = Field(description="Unique identifier for the modification")
    priority: int = Field(description="Priority of the modification. Lower values have higher priority.")
    add_ingredient: Optional[Dict[str, Any]] = None
    remove_ingredient: Optional[Dict[str, Any]] = None
    update_ingredient: Optional[Dict[str, Any]] = None
    add_instruction: Optional[str] = None
    remove_instruction: Optional[str] = None
    add_tag: Optional[str] = None
    remove_tag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of RecipeModification object.")
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecipeModification':
        logger.debug("Creating RecipeModification object from dictionary.")
        return RecipeModification.parse_raw(str(data))

class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    id: Optional[str] = Field(description="Unique identifier for the recipe")
    ingredients: List[Ingredient] = Field(description="List of ingredients required for the recipe")
    instructions: List[str] = Field(description="List of instructions to prepare the recipe")
    tags: Optional[List[str]] = None  # New field for tags
    sources: Optional[List[str]] = None  # New field for sources or inspirations

    def __str__(self):
        return json.dumps(self.dict(), indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of Recipe object.")
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        logger.debug("Creating Recipe object from dictionary.")
        return Recipe.parse_raw(str(data))
    
    def apply_modification(self, modification: RecipeModification) -> None:
        logger.debug("Applying modification to Recipe object.")
        if modification.attributes.get('add_ingredient'):
            logger.debug("Adding ingredient to Recipe object.")
            self.ingredients.append(Ingredient.from_dict(modification.attributes['add_ingredient']))
        if modification.attributes.get('remove_ingredient'):
            logger
            self.ingredients = [ing for ing in self.ingredients if ing.name != modification.attributes['remove_ingredient']['name']]
        if modification.attributes.get('update_ingredient'):
            logger.debug("Updating ingredient in Recipe object.")
            for ing in self.ingredients:
                if ing.name == modification.attributes['update_ingredient']['name']:
                    ing.quantity = modification.attributes['update_ingredient'].get('quantity', ing.quantity)
                    ing.unit = modification.attributes['update_ingredient'].get('unit', ing.unit)
        if modification.attributes.get('add_instruction'):
            logger.debug("Adding instruction to Recipe object.")
            self.instructions.append(modification.attributes['add_instruction'])
        if modification.attributes.get('remove_instruction'):
            logger.debug("Removing instruction from Recipe object.")
            self.instructions.remove(modification.attributes['remove_instruction'])
        if modification.attributes.get('add_tag'):
            logger.debug("Adding tag to Recipe object.")
            self.tags.append(modification.attributes['add_tag'])
        if modification.attributes.get('remove_tag'):
            logger.debug("Removing tag from Recipe object.")
            self.tags.remove(modification.attributes['remove_tag'])

    def __pretty__(self, p, cycle):
        if cycle:
            p.text(f"<Recipe(name={self.name})>")
            return
        p.begin_group(1, f"<Recipe(name={self.name})>")
        p.breakable()
        p.text("Ingredients:")
        p.breakable()
        for ingredient in self.ingredients:
            p.pretty(ingredient)
            p.breakable()
        p.text("Instructions:")
        p.breakable()
        p.pretty(self.instructions)
        p.breakable()
        if self.tags:
            p.text("Tags:")
            p.breakable()
            p.pretty(self.tags)
            p.breakable()
        if self.sources:
            p.text("Sources:")
            p.breakable()
            p.pretty(self.sources)
            p.breakable()
        p.end_group(1, "")

class RecipeGraph:
    def __init__(self) -> None:
        logger.info("Initializing RecipeGraph object.")
        self.graph = nx.DiGraph()
        self.foundational_recipe_node: Optional[Recipe] = None

    def get_graph_size(self) -> int:
        logger.debug("Getting the number of nodes in the recipe graph.")
        return self.graph.number_of_nodes()

    def create_recipe_graph(self, recipe: Recipe) -> str:
        logger.debug("Creating recipe graph with foundational recipe.")
        node_id = str(uuid.uuid4())
        self.graph.add_node(node_id, recipe=recipe)
        self.foundational_recipe_node = node_id
        return node_id

    def get_recipe(self, node_id: Optional[str] = None) -> Optional[Recipe]:
        logger.debug("Getting recipe from recipe graph.")
        if node_id is None:
            node_id = self.foundational_recipe_node
        if self.get_graph_size() == 0:
            return None
        return self.graph.nodes[node_id].get('recipe', None)
    
    def get_node_id(self, node_id: Optional[str] = None) -> Optional[str]:
        logger.debug("Getting node ID from recipe graph.")
        if node_id is None:
            node_id = self.foundational_recipe_node
        return node_id

    def add_node(self, recipe: Recipe) -> str:
        logger.debug("Adding node to recipe graph.")
        node_id = str(uuid.uuid4())
        logger.debug(f"Node ID: {node_id}")
        self.graph.add_node(node_id, recipe=recipe)
        if self.foundational_recipe_node is not None:
            self.graph.add_edge(self.foundational_recipe_node, node_id)
        self.foundational_recipe_node = node_id
        return node_id

    def get_foundational_recipe(self) -> Optional[Recipe]:
        logger.debug("Getting foundational recipe from recipe graph.")
        return self.get_recipe(self.foundational_recipe_node)
    
    def set_foundational_recipe(self, node_id: str) -> None:
        logger.debug("Setting foundational recipe in recipe graph.")
        self.foundational_recipe_node = node_id
        # TODO - establish foundational recipe node when there is none

    def get_graph(self) -> nx.DiGraph:
        logger.debug("Getting recipe graph.")
        return self.graph

class ModsList:
    def __init__(self) -> None:
        logger.info("Initializing ModsList object.")
        self.queue: List[tuple[int, RecipeModification]] = []

    def suggest_mod(self, mod: RecipeModification) -> None:
        logger.debug("Suggesting modification to mods list.")
        heapq.heappush(self.queue, (-mod.priority, mod))

    def get_mods_list(self) -> List[RecipeModification]:
        logger.debug("Getting mods list.")
        return [mod for _, mod in sorted(self.queue, key=lambda x: -x[0])]

    def push_mod(self) -> Optional[RecipeModification]:
        logger.debug("Pushing modification from mods list.")
        if self.queue:
            return heapq.heappop(self.queue)[1]
        return None

    def rank_mod(self, mod_id: str, new_priority: int) -> None:
        logger.debug("Ranking modification in mods list.")
        for i, (_, mod) in enumerate(self.queue):
            if mod.id == mod_id:
                self.queue[i] = (-new_priority, mod)
                heapq.heapify(self.queue)
                break

    def remove_mod(self, mod_id: str) -> bool:
        logger.debug("Removing modification from mods list.")
        for i, (_, mod) in enumerate(self.queue):
            if mod.id == mod_id:
                self.queue.pop(i)
                heapq.heapify(self.queue)
                return True
        return False
    
##Graph Functions

def fresh_graph(filename):
    logger.info("Creating a new recipe graph.")
    recipe_graph = RecipeGraph()
    with open(filename, 'wb') as file:
        pickle.dump(recipe_graph, file)
    return filename

def save_graph_to_file(recipe_graph, filename):
    logger.info("Saving recipe graph to file.")
    with open(filename, 'wb') as file:
        pickle.dump(recipe_graph, file)

def load_graph_from_file(filename):
    logger.info("Loading recipe graph from file.")
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError("File does not exist.")

## Modifications Functions

def fresh_mods_list(filename: str) -> None:
    logger.info("Creating a new mods list.")
    mods_list = ModsList()
    save_mods_list_to_file(mods_list, filename)
    return filename

def load_mods_list_from_file(filename: str) -> ModsList:
    logger.info("Loading mods list from file.")
    try:
        with open(filename, 'rb') as file:
            mods_list = pickle.load(file)
        return mods_list
    except FileNotFoundError:
        return ModsList()

def save_mods_list_to_file(mods_list: ModsList, filename: str) -> None:
    logger.info("Saving mods list to file.")
    with open(filename, 'wb') as file:
        pickle.dump(mods_list, file)##Graph Tools

recipe_graph_tool_validation =  """

Formatting Instructions for Tools:

1. **Ingredient**: Each ingredient is represented by a JSON string with fields `name` (string), `quantity` (float), and `unit` (string). Example:
    
    {"name": "Flour", "quantity": 2.5, "unit": "cups"}

Ensure that the generate_info tool is provided with a name (str), quantity (float), and unit (str) as input. Example:

    generate_ingredient(name="Flour", quantity=2.5, unit="cups")

The output will be a JSON string representing the ingredient following the above format.
    

2. **RecipeModification**: Each modification must be a JSON string with fields `id` (string), `priority` (int), and optional fields `add_ingredient`, `remove_ingredient`, `update_ingredient`, `add_instruction`, `remove_instruction`, `add_tag`, and `remove_tag`. Example:
    
    {"id": "mod1", "priority": 1, "add_ingredient": {"name": "Sugar", "quantity": 0.5, "unit": "cups"}, "add_instruction": "Stir in the sugar"}

Ensure that the generate_mod tool is provided with an id (str), priority (int), and optional fields for the modification. Example:

    generate_mod(id="mod1", priority=1, add_ingredient='{"name": "Sugar", "quantity": 0.5, "unit": "cups"}', add_instruction="Stir in the sugar")

The output will be a JSON string representing the modification following the above format.    

3. **Recipe**: Each recipe is represented by a JSON string with fields `name` (string), `ingredients` (list of JSON strings following Ingredient format given above), `instructions` (list of strings), and optional fields `tags` (list of strings) and `sources` (list of strings). Example:

    {
        "name": "Pancakes",
        "ingredients": [
            '{"name": "Flour", "quantity": 2.0, "unit": "cups"}',
            '{"name": "Milk", "quantity": 1.5, "unit": "cups"}',
            '{"name": "Eggs", "quantity": 2, "unit": "units"}'
        ],
        "instructions": [
            "Mix all ingredients together.",
            "Cook on a hot griddle until golden brown."
        ],
        "tags": ["breakfast", "easy"],
        "sources": ["https://example.com/pancakes"]
    }

Ensure that the generate_recipe tool is provided with a name (str), ingredients (list of JSON strings), instructions (list of strings), and optional tags (list of strings) and sources (list of strings) as input. Example:

    generate_recipe(
        name="Pancakes",
        ingredients=[
            '{"name": "Flour", "quantity": 2.0, "unit": "cups"}',
            '{"name": "Milk", "quantity": 1.5, "unit": "cups"}',
            '{"name": "Eggs", "quantity": 2, "unit": "units"}'
        ],
        instructions=[
            "Mix all ingredients together.",
            "Cook on a hot griddle until golden brown."
        ],
        tags=["breakfast", "easy"],
        sources=["https://example.com/pancakes"]
    )

Always validate JSON strings for proper formatting after using these tools.
"""