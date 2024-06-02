import json
import uuid
import pickle
import os
from langchain_core.tools import tool
import networkx as nx
import heapq
from typing import List, Dict, Optional, Any, Annotated
from logging_util import logger
from pydantic import BaseModel, ValidationError

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
        self.graph.add_node(node_id, recipe=recipe)
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

@tool
def generate_ingredient(
    name: Annotated[str, "The name of the ingredient."],
    quantity: Annotated[float, "The quantity of the ingredient."],
    unit: Annotated[str, "The unit of the ingredient."]
) -> Annotated[str, "The JSON representation of the ingredient."]:
    """Generate a JSON representation of an Ingredient object."""
    logger.debug("Generating JSON representation of Ingredient object.")
    print(name, quantity, unit)
    ingredient = Ingredient(name=name, quantity=quantity, unit=unit)
    return json.dumps(ingredient.to_dict(), indent=2)

@tool
def generate_recipe(
    name: Annotated[str, "The name of the recipe."],
    ingredients: Annotated[List[str], "A list of JSON representations of Ingredient objects. Example: ['{\"name\": \"Flour\", \"quantity\": 2.5, \"unit\": \"cups\"}']"],
    instructions: Annotated[List[str], "A list of recipe instructions. Example: ['Preheat the oven to 350F', 'Mix the flour and sugar']"],
    tags: Annotated[Optional[List[str]], "A list of recipe tags."] = None,
    sources: Annotated[Optional[List[str]], "A list of sources or inspirations."] = None
) -> Annotated[str, "The JSON representation of the recipe."]:
    """Generate a JSON representation of a Recipe object."""
    logger.debug("Generating JSON representation of Recipe object.")
    ingredients_objs = [Ingredient.from_dict(json.loads(ing)) for ing in ingredients]
    recipe = Recipe(name=name, ingredients=ingredients_objs, instructions=instructions, tags=tags, sources=sources)
    return json.dumps(recipe.to_dict(), indent=2)

@tool
def create_recipe_graph(
    recipe: Annotated[str, "The JSON representation of the dictionary of the Recipe object of the foundational recipe."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "ID of the newly created foundational recipe node."]:
    """Create a new recipe graph with the provided foundational recipe. Typically used to start a new recipe graph."""
    logger.debug("Creating recipe graph with foundational recipe.")
    recipe_graph = load_graph_from_file(graph_file)
    node_id = recipe_graph.create_recipe_graph(recipe)
    save_graph_to_file(recipe_graph, graph_file)
    return f"Recipe graph created with foundational recipe node ID: {node_id}"

@tool
def get_recipe(
    node_id: Annotated[Optional[str], "The node ID of the recipe to retrieve. If not provided, retrieves the foundational recipe."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file,
) -> Annotated[Optional[Recipe], "The Recipe object."]:
    """Get the Recipe object at the specified node ID."""
    logger.debug("Getting recipe from recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    recipe = recipe_graph.get_recipe(node_id)
    return recipe

@tool
def add_node(
    recipe: Annotated[str, "The JSON representation of the dictionary of the Recipe object of the recipe."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "ID of the newly added recipe node."]:
    """Add a new node to the recipe graph with the provided recipe and create an edge from the current foundational recipe."""
    logger.debug("Adding node to recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    node_id = recipe_graph.add_node(recipe)
    save_graph_to_file(recipe_graph, graph_file)
    return f"New recipe node added with ID: {node_id}"

@tool
def get_node_id(
    recipe: Annotated[str, "The JSON representation of the dictionary of the Recipe object of the recipe."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[Optional[str], "The node ID of the recipe."]:
    """Get the node ID of the foundational recipe."""
    logger.debug("Getting node ID from recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    # TODO - see if the given recipe matches any recipe in the graph
    return recipe_graph.get_node_id()

@tool
def get_foundational_recipe(
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[Optional[Recipe], "The current foundational recipe."]:
    """Get the current foundational recipe."""
    logger.debug("Getting foundational recipe from recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    recipe = recipe_graph.get_foundational_recipe()
    return recipe

@tool
def set_foundational_recipe(
    node_id: Annotated[str, "The node ID of the recipe to set as foundational."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "Message indicating success or failure."]:
    """Set the recipe with the specified node ID as the foundational recipe."""
    logger.debug("Setting foundational recipe in recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    recipe_graph.set_foundational_recipe(node_id)
    save_graph_to_file(recipe_graph, graph_file)
    return f"Foundational recipe set to node ID: {node_id}"

@tool
def get_graph(
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "A representation of the current recipe graph."]:
    """Get a representation of the current recipe graph."""
    logger.debug("Getting recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    graph = recipe_graph.get_graph()
    nodes = [(node, data['recipe'].to_dict()) for node, data in graph.nodes(data=True)]
    edges = list(graph.edges(data=True))
    return f"Recipe Graph: Nodes - {nodes}, Edges - {edges}"

@tool
def get_graph_size(
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "The number of nodes in the recipe graph."]:
    """Get the number of nodes in the recipe graph."""
    logger.debug("Getting the number of nodes in the recipe graph.")
    recipe_graph = load_graph_from_file(graph_file)
    return f"Number of nodes in recipe graph: {recipe_graph.get_graph_size()}"

## Modifications Tools
@tool
def generate_mod(
    mod_id: Annotated[str, "The ID of the modification."],
    priority: Annotated[int, "The priority of the modification."],
    add_ingredient: Annotated[Optional[Dict[str, Any]], "The ingredient to add."]=None,
    remove_ingredient: Annotated[Optional[Dict[str, Any]], "The ingredient to remove."]=None,
    update_ingredient: Annotated[Optional[Dict[str, Any]], "The ingredient to update."]=None,
    add_instruction: Annotated[Optional[str], "The instruction to add."]=None,
    remove_instruction: Annotated[Optional[str], "The instruction to remove."]=None,
    add_tag: Annotated[Optional[str], "The tag to add."]=None,
    remove_tag: Annotated[Optional[str], "The tag to remove."]=None
) -> Annotated[str, "The JSON representation of the modification."]:
    """Generate a JSON representation of a RecipeModification object."""
    logger.debug("Generating JSON representation of RecipeModification object.")
    mod = RecipeModification(mod_id, priority, add_ingredient, remove_ingredient, update_ingredient, add_instruction, remove_instruction, add_tag, remove_tag)
    return json.dumps(mod.to_dict(), indent=2)

@tool
def suggest_mod(
    mod: Annotated[str, "The JSON representation of the dictionary of the RecipeModification object to be suggested."],
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[List[RecipeModification], "The updated list of modifications."]:
    """Suggest a new modification to be added to the mods list."""
    logger.debug("Suggesting modification to mods list.")
    mod_dict = json.loads(mod)
    recipe_mod = RecipeModification.from_dict(mod_dict)
    
    mods_list = load_mods_list_from_file(mods_list_file)
    mods_list.suggest_mod(recipe_mod)
    save_mods_list_to_file(mods_list, mods_list_file)
    
    updated_mods_list = mods_list.get_mods_list()
    return updated_mods_list

@tool
def get_mods_list(
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[List[RecipeModification], "The current list of suggested modifications."]:
    """Get the current list of suggested modifications."""
    logger.debug("Getting mods list.")
    mods_list = load_mods_list_from_file(mods_list_file)
    current_mods_list = mods_list.get_mods_list()
    return current_mods_list

@tool
def push_mod(
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[Optional[RecipeModification], "The modification that was applied, if available."]:
    """Apply the top modification from the mods list."""
    logger.debug("Pushing modification from mods list.")
    mods_list = load_mods_list_from_file(mods_list_file)
    mod_to_apply = mods_list.push_mod()
    save_mods_list_to_file(mods_list, mods_list_file)
    return mod_to_apply

@tool
def rank_mod(
    mod_id: Annotated[str, "The ID of the modification to reprioritize."],
    new_priority: Annotated[int, "The new priority for the modification (1 = highest priority, larger numbers = lower priority)."],
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[List[RecipeModification], "The updated list of modifications."]:
    """Reprioritize a given modification within the mods list.

    The priority ranking options are as follows:
    - A lower numerical value indicates higher priority (e.g., 1 is the highest priority).
    - Larger numerical values indicate lower priority.
    """
    logger.debug("Ranking modification in mods list.")
    mods_list = load_mods_list_from_file(mods_list_file)
    mods_list.rank_mod(mod_id, new_priority)
    save_mods_list_to_file(mods_list, mods_list_file)
    updated_mods_list = mods_list.get_mods_list()
    return updated_mods_list

@tool
def remove_mod(
    mod_id: Annotated[str, "The ID of the modification to remove."],
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[bool, "Indicates whether the modification was successfully removed."]:
    """Remove a modification from the mods list."""
    logger.debug("Removing modification from mods list.")
    mods_list = load_mods_list_from_file(mods_list_file)
    result = mods_list.remove_mod(mod_id)
    save_mods_list_to_file(mods_list, mods_list_file)
    return result

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