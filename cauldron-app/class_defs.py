import uuid
import pickle
import os
import ujson
import networkx as nx
import heapq
from typing import List, Dict, Optional, Any, Tuple, Type, TypeVar, Set
from logging_util import logger
from langchain.pydantic_v1 import BaseModel, Field

T = TypeVar('T')
default_mods_list_file = "mods_list.pkl"
default_graph_file="recipe_graph.pkl"
default_pot_file="recipe_pot.pkl"

class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    quantity: float = Field(description="Quantity of the ingredient")
    unit: str = Field(description="Unit of measurement for the ingredient")

    class Config:
        json_loads = ujson.loads

    def __str__(self) -> str:
        return self.json()

    def to_json(self) -> Dict[str, Any]:
        logger.debug("Creating dictionary representation of Ingredient object.")
        return self.json()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Ingredient':
        logger.debug("Creating Ingredient object from JSON string.")
        return Ingredient.parse_raw(str(data))

class RecipeModification(BaseModel):
    priority: int = Field(description="Priority of the modification. Lower values have higher priority.")
    add_ingredient: Optional[Ingredient] = None
    remove_ingredient: Optional[Ingredient] = None
    update_ingredient: Optional[Ingredient] = None
    add_instruction: Optional[str] = None
    remove_instruction: Optional[str] = None
    add_tag: Optional[str] = None
    remove_tag: Optional[str] = None
    id: Optional[str] = Field(default=str(uuid.uuid4()),description="Unique identifier for the modification")

    class Config:
        json_loads = ujson.loads

    def __str__(self) -> str:
        return self.json()

    def to_json(self) -> Dict[str, Any]:
        logger.debug("Creating JSON representation of RecipeModification object.")
        return self.json()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'RecipeModification':
        logger.debug("Creating RecipeModification object from JSON string.")
        return RecipeModification.parse_raw(str(data))

class Recipe(BaseModel):
    name: str = Field(description="Name of the recipe")
    ingredients: List[Ingredient] = Field(description="List of ingredients required for the recipe")
    instructions: List[str] = Field(description="List of instructions to prepare the recipe")
    tags: Optional[List[str]] = None  # New field for tags
    sources: Optional[List[str]] = None  # New field for sources or inspirations
    id: Optional[str] = Field(default=str(uuid.uuid4()),description="Unique identifier for the recipe")
    
    class Config:
        json_loads = ujson.loads
    
    def __str__(self) -> str:
        return self.json()

    def to_json(self) -> Dict[str, Any]:
        logger.debug("Creating JSON representation of Recipe object.")
        return self.json()
    
    def get_ID(self) -> str:
        return self.id
    
    def tiny(self) -> str:
        return f"{self.name} ({self.id})"
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Recipe':
        logger.debug("Creating Recipe object from JSON string.")
        return Recipe.parse_raw(str(data))

    def apply_modification(self, modification: RecipeModification) -> bool:
        logger.debug("Applying modification to Recipe object.")
        if modification.attributes.get('add_ingredient'):
            logger.debug("Adding ingredient to Recipe object.")
            self.ingredients.append(Ingredient.from_json(modification.attributes['add_ingredient']))
            return True
        if modification.attributes.get('remove_ingredient'):
            logger.debug("Removing ingredient from Recipe object.")
            self.ingredients = [ing for ing in self.ingredients if ing.name != modification.attributes['remove_ingredient']['name']]
            return True
        if modification.attributes.get('update_ingredient'):
            logger.debug("Updating ingredient in Recipe object.")
            for ing in self.ingredients:
                if ing.name == modification.attributes['update_ingredient']['name']:
                    ing.quantity = modification.attributes['update_ingredient'].get('quantity', ing.quantity)
                    ing.unit = modification.attributes['update_ingredient'].get('unit', ing.unit)
            return True
        if modification.attributes.get('add_instruction'):
            logger.debug("Adding instruction to Recipe object.")
            self.instructions.append(modification.attributes['add_instruction'])
            return True
        if modification.attributes.get('remove_instruction'):
            logger.debug("Removing instruction from Recipe object.")
            self.instructions.remove(modification.attributes['remove_instruction'])
            return True
        if modification.attributes.get('add_tag'):
            logger.debug("Adding tag to Recipe object.")
            self.tags.append(modification.attributes['add_tag'])
            return True
        if modification.attributes.get('remove_tag'):
            logger.debug("Removing tag from Recipe object.")
            self.tags.remove(modification.attributes['remove_tag'])
            return True
        return False

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
        return Recipe.from_json(self.graph.nodes[node_id].get('recipe', None))
    
    def get_node_id(self, node_id: Optional[str] = None) -> Optional[str]:
        logger.debug("Getting node ID from recipe graph.")
        if node_id is None:
            node_id = self.foundational_recipe_node
        return node_id

    def add_node(self, recipe: Recipe) -> str:
        logger.debug("Adding node to recipe graph.")
        node_id = recipe.id
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

class ModsList(BaseModel):
    queue: List[Tuple[int, RecipeModification]] = Field(default=[], description="Priority queue of recipe modifications")

    def __init__(self, **data):
        super().__init__(**data)
        logger.info("Initializing ModsList object.")

    def __str__(self) -> str:
        return self.json()
        
    def suggest_mod(self, mod: RecipeModification) -> None:
        logger.debug("Suggesting modification to mods list.")
        heapq.heappush(self.queue, (-mod.priority, mod))

    def get_mods_list(self) -> List[RecipeModification]:
        logger.debug("Getting mods list.")
        return [mod for _, mod in sorted(self.queue, key=lambda x: -x[0])]

    def push_mod(self, recipe_graph: RecipeGraph) -> Tuple[RecipeModification, bool]:
        logger.debug("Pushing modification from mods list.")
        if self.queue:
            mod = heapq.heappop(self.queue)[1]
            recipe = recipe_graph.get_foundational_recipe()
            if recipe is not None:
                return (mod, recipe.apply_modification(mod))
        return (None, False)

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
    
class Pot(BaseModel):
    contents: List[Recipe] = Field(default=[], description="Set of recipes in the pot")

    def __init__(self, **data):
        super().__init__(**data)
        logger.info("Initializing Pot object.")

    def __str__(self) -> str:
        return self.json()
    
    def add_recipe(self, recipe: Recipe) -> None:
        logger.debug("Adding recipe to pot.")
        self.contents.append(recipe)

    def remove_recipe(self, recipe_id: str) -> bool:
        logger.debug("Removing recipe from pot.")
        for i, recipe in enumerate(self.contents):
            if recipe.id == recipe_id:
                self.contents.pop(i)
                return True
        return False
    
    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        logger.debug("Getting recipe from pot.")
        for recipe in self.contents:
            if recipe.id == recipe_id:
                return recipe
        return None
    
    def pop_recipe(self) -> Optional[Recipe]:
        logger.debug("Popping recipe from pot.")
        if self.contents:
            return self.contents.pop()
        return None
    
    def get_all_recipes(self) -> List[Recipe]:
        logger.debug("Getting all recipes from pot.")
        return self.contents
    
    def clear_pot(self) -> None:
        logger.debug("Clearing pot.")
        self.contents = []

# General utility functions
def save_to_file(obj: T, filename: str) -> None:
    logger.info(f"Saving {obj.__class__.__name__} to file.")
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_from_file(cls: Type[T], filename: str) -> T:
    logger.info(f"Loading {cls.__name__} from file.")
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"{filename} does not exist.")

def fresh_instance(cls: Type[T], filename: str) -> T:
    logger.info(f"Creating a new instance of {cls.__name__}.")
    instance = cls()
    save_to_file(instance, filename)
    return filename

## Graph Wrapper Functions
def fresh_graph(filename: str=default_graph_file) -> RecipeGraph:
    return fresh_instance(RecipeGraph, filename)

def save_graph_to_file(recipe_graph: RecipeGraph, filename: str=default_graph_file) -> None:
    save_to_file(recipe_graph, filename)

def load_graph_from_file(filename: str=default_graph_file) -> RecipeGraph:
    return load_from_file(RecipeGraph, filename)

## ModsList Wrapper Functions
def fresh_mods_list(filename: str=default_mods_list_file) -> ModsList:
    return fresh_instance(ModsList, filename)

def save_mods_list_to_file(mods_list: ModsList, filename: str=default_mods_list_file) -> None:
    save_to_file(mods_list, filename)

def load_mods_list_from_file(filename: str=default_mods_list_file) -> ModsList:
    return load_from_file(ModsList, filename)

## Pot Wrapper Functions
def fresh_pot(filename: str=default_pot_file) -> Pot:
    return fresh_instance(Pot, filename)

def save_pot_to_file(pot: Pot, filename: str=default_pot_file) -> None:
    save_to_file(pot, filename)

def load_pot_from_file(filename: str=default_pot_file) -> Pot:
    return load_from_file(Pot, filename)