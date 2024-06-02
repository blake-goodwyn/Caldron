from cauldron_app import CauldronApp
from util import db_path, llm_model
from langchain_util import HumanMessage


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
        #print(s)
        #print("----")
    i = input("Enter a message: ")