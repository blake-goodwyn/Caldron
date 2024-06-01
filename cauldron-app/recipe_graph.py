import json
import uuid
import pickle
import os
from langchain_core.tools import tool
from typing import Annotated, Optional
import networkx as nx

default_graph_file="recipe_graph.pkl"

class RecipeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.foundational_recipe_node = None

    def get_graph_size(self):
        return self.graph.number_of_nodes()

    def create_recipe_graph(self, recipe_json):
        node_id = str(uuid.uuid4())
        self.graph.add_node(node_id, recipe=recipe_json)
        self.foundational_recipe_node = node_id
        return node_id

    def get_recipe(self, node_id=None):
        if node_id is None:
            node_id = self.foundational_recipe_node
        return self.graph.nodes[node_id].get('recipe', None)

    def add_node(self, recipe_json):
        node_id = str(uuid.uuid4())
        self.graph.add_node(node_id, recipe=recipe_json)
        self.graph.add_edge(self.foundational_recipe_node, node_id)
        self.foundational_recipe_node = node_id
        return node_id

    def get_foundational_recipe(self):
        return self.get_recipe(self.foundational_recipe_node)
    
    def set_foundational_recipe(self, node_id):
        self.foundational_recipe_node = node_id
        #TODO - establish foundational recipe node when there is none

    def get_graph(self):
        return self.graph

def fresh_graph(filename):
    recipe_graph = RecipeGraph()
    with open(filename, 'wb') as file:
        pickle.dump(recipe_graph, file)

def save_graph_to_file(recipe_graph, filename):
    with open(filename, 'wb') as file:
        pickle.dump(recipe_graph, file)

def load_graph_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return RecipeGraph()

@tool
def create_recipe_graph(
    recipe_json: Annotated[str, "The JSON representation of the foundational recipe."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "ID of the newly created foundational recipe node."]:
    """Create a new recipe graph with the provided foundational recipe."""
    try:
        recipe_data = json.loads(recipe_json)
        recipe_graph = load_graph_from_file(graph_file)
        node_id = recipe_graph.create_recipe_graph(recipe_data)
        save_graph_to_file(recipe_graph, graph_file)
        return f"Recipe graph created with foundational recipe node ID: {node_id}"
    except json.JSONDecodeError:
        return "Failed to create recipe graph. Invalid JSON input."

@tool
def get_recipe(
    node_id: Annotated[Optional[str], "The node ID of the recipe to retrieve. If not provided, retrieves the foundational recipe."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file,
) -> Annotated[str, "The JSON representation of the recipe."]:
    """Get the JSON representation of the recipe at the specified node ID."""
    recipe_graph = load_graph_from_file(graph_file)
    recipe = recipe_graph.get_recipe(node_id)
    if recipe is not None:
        return json.dumps(recipe, indent=2)
    else:
        return "Recipe not found."

@tool
def add_node(
    recipe_json: Annotated[str, "The JSON representation of the recipe to add."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "ID of the newly added recipe node."]:
    """Add a new node to the recipe graph with the provided recipe and create an edge from the current foundational recipe."""
    try:
        recipe_data = json.loads(recipe_json)
        recipe_graph = load_graph_from_file(graph_file)
        node_id = recipe_graph.add_node(recipe_data)
        save_graph_to_file(recipe_graph, graph_file)
        return f"New recipe node added with ID: {node_id}"
    except json.JSONDecodeError:
        return "Failed to add recipe node. Invalid JSON input."

@tool
def get_foundational_recipe(
    graph_file: Annotated[str, "The filename for the recipe graph."]= default_graph_file
) -> Annotated[str, "The JSON representation of the current foundational recipe."]:
    """Get the JSON representation of the current foundational recipe."""
    recipe_graph = load_graph_from_file(graph_file)
    recipe = recipe_graph.get_foundational_recipe()
    if recipe is not None:
        return json.dumps(recipe, indent=2)
    else:
        return "Foundational recipe not found."
    
@tool
def set_foundational_recipe(
    node_id: Annotated[str, "The node ID of the recipe to set as foundational."],
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "Message indicating success or failure."]:
    """Set the recipe with the specified node ID as the foundational recipe."""
    recipe_graph = load_graph_from_file(graph_file)
    recipe_graph.set_foundational_recipe(node_id)
    save_graph_to_file(recipe_graph, graph_file)
    return f"Foundational recipe set to node ID: {node_id}"

@tool
def get_graph(
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "A representation of the current recipe graph."]:
    """Get a representation of the current recipe graph."""
    recipe_graph = load_graph_from_file(graph_file)
    graph = recipe_graph.get_graph()
    nodes = list(graph.nodes(data=True))
    edges = list(graph.edges(data=True))
    return f"Recipe Graph: Nodes - {nodes}, Edges - {edges}"

@tool
def get_graph_size(
    graph_file: Annotated[str, "The filename for the recipe graph."] = default_graph_file
) -> Annotated[str, "The number of nodes in the recipe graph."]:
    """Get the number of nodes in the recipe graph."""
    recipe_graph = load_graph_from_file(graph_file)
    return f"Number of nodes in recipe graph: {recipe_graph.get_graph_size()}"