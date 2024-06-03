from cauldron_app import CauldronApp
from util import db_path, llm_model
from langchain_util import HumanMessage
from recipe_graph import load_graph_from_file, Recipe
import pprint


app = CauldronApp(db_path, llm_model)
i = input("Enter a message: ")
while True:
    for s in app.chain.stream(
        {
            "messages": [
                HumanMessage(
                    content=i
                )
            ],
            'sender': 'user',
            'next': 'ConductorAgent'
        },
        {"recursion_limit": 50}
    ):
        pass
        
    graph = load_graph_from_file(app.recipe_graph)
    pprint.pprint(Recipe.from_dict(graph.get_foundational_recipe()))
    i = input("Enter a message: ")