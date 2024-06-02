import uuid
import pickle
import os
import networkx as nx
import heapq
from typing import List, Dict, Optional, Any
from logging_util import logger
from pydantic import BaseModel

default_mods_list_file = "mods_list.pkl"
default_graph_file="recipe_graph.pkl"

class Ingredient(BaseModel):
    name: str
    quantity: float
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of Ingredient object.")
        return {
            'name': self.name,
            'quantity': self.quantity,
            'unit': self.unit
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ingredient':
        logger.debug("Creating Ingredient object from dictionary.")
        return cls(
            name=data['name'],
            quantity=data['quantity'],
            unit=data['unit']
        )

class RecipeModification(BaseModel):
    id: str
    priority: int
    add_ingredient: Optional[Dict[str, Any]] = None
    remove_ingredient: Optional[Dict[str, Any]] = None
    update_ingredient: Optional[Dict[str, Any]] = None
    add_instruction: Optional[str] = None
    remove_instruction: Optional[str] = None
    add_tag: Optional[str] = None
    remove_tag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of RecipeModification object.")
        return {
            'id': self.id,
            'priority': self.priority,
            'add_ingredient': self.add_ingredient,
            'remove_ingredient': self.remove_ingredient,
            'update_ingredient': self.update_ingredient,
            'add_instruction': self.add_instruction,
            'remove_instruction': self.remove_instruction,
            'add_tag': self.add_tag,
            'remove_tag': self.remove_tag
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecipeModification':
        logger.debug("Creating RecipeModification object from dictionary.")
        return cls(
            id=data['id'],
            priority=data['priority'],
            add_ingredient=data.get('add_ingredient'),
            remove_ingredient=data.get('remove_ingredient'),
            update_ingredient=data.get('update_ingredient'),
            add_instruction=data.get('add_instruction'),
            remove_instruction=data.get('remove_instruction'),
            add_tag=data.get('add_tag'),
            remove_tag=data.get('remove_tag')
        )

class Recipe(BaseModel):
    name: str
    ingredients: List[Ingredient]
    instructions: List[str]
    tags: Optional[List[str]] = None
    sources: Optional[List[str]] = None  # New field for sources or inspirations

    def to_dict(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of Recipe object.")
        return {
            'name': self.name,
            'ingredients': [ing.to_dict() for ing in self.ingredients],
            'instructions': self.instructions,
            'tags': self.tags,
            'sources': self.sources  # Include sources in the dictionary representation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        logger.debug("Creating Recipe object from dictionary.")
        ingredients = [Ingredient.from_dict(ing) for ing in data['ingredients']]
        return cls(
            name=data['name'],
            ingredients=ingredients,
            instructions=data['instructions'],
            tags=data.get('tags', []),
            sources=data.get('sources', [])
        )
    
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

class RecipeGraph:
    def __init__(self) -> None:
        logger.info("Initializing RecipeGraph object.")
        self.graph = nx.DiGraph()
        self.foundational_recipe_node: Optional[str] = None

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

def save_graph_to_file(recipe_graph, filename):
    logger.info("Saving recipe graph to file.")
    with open(filename, 'wb') as file:
        pickle.dump(recipe_graph, file)

def load_graph_from_file(filename):
    logger.info("Loading recipe graph from file.")
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return RecipeGraph()

## Modifications Functions

def fresh_mods_list(filename: str) -> None:
    logger.info("Creating a new mods list.")
    mods_list = ModsList()
    save_mods_list_to_file(mods_list, filename)

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