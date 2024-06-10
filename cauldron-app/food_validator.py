from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from dotenv import load_dotenv
import os
from logging_util import logger

load_dotenv()
LANGCHAIN_TRACING_V2=True
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

food_validator_conditional_edges = [
    "TRUE": "Caldron\nPostman",
    "FALSE": "Frontman",
]

def createFoodValidator(name, system_prompt, llm: ChatOpenAI) -> str:
    """A LLM-based food-discernment router."""
    glutton_fx = {
        "name": "discernFood",
        "description": "Return whether or not the item is food.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "result": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": ["TRUE", "FALSE"]},
                    ],
                },
            },
            "required": ["result"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Given the request above, is the request a valid food? [TRUE/FALSE]")
        ]
    )
    return (
        prompt
        | llm.bind_functions(functions=[glutton_fx], function_call="discernFood")
        | JsonOutputFunctionsParser()
    )