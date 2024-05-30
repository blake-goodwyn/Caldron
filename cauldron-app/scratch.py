from cauldron_app import CauldronApp
from util import db_path, llm_model
from langchain_util import HumanMessage


app = CauldronApp(db_path, llm_model)
for s in app.chain.stream(
    {
        "messages": [
            HumanMessage(
                content="I want to make gluten-free bread with xanthan gum."
            )
        ]
    },
    {"recursion_limit": 50}
):
    print(s)
    print("----")