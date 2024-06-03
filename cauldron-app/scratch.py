from cauldron_app import CauldronApp
from util import db_path, llm_model
from langchain_util import HumanMessage
from recipe_graph import load_graph_from_file
from custom_print import printer

app = CauldronApp(db_path, llm_model)
i = input("Enter a message: ")
while i != "exit":
    for s in app.chain.stream(
        {
            "messages": [
                HumanMessage(
                    content=i
                )
            ],
            'sender': 'user',
            'next': 'CauldronRouter'
        },
        {"recursion_limit": 50}
    ):
        print("--------------------")
        if 'SummaryAgent' in s.keys():
            print(s['SummaryAgent']['messages'][0].content)
        else:
            print(s)
        
    graph = load_graph_from_file(app.recipe_graph)
    printer.pprint(graph.get_foundational_recipe())
    i = input("Enter a message: ")