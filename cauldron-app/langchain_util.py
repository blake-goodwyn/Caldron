# langchain_util.py

import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.agent import RunnableAgent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX, SQL_PREFIX
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph
from openai import OpenAI

from dotenv import load_dotenv
import os
from logging_util import logger

load_dotenv()
LANGCHAIN_TRACING_V2=True
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def quickResponse(request):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are the helpful recipe development assistant, Caldron. While a user is waiting for their recipe request, provide a brief, polite message letting the user know that their request is underway. If the request is empty or inappropriate, politely ask to that they make another request"},
            {"role": "user", "content": request},
        ],
    )
    # Extract the generated text from the response
    return response.choices[0].message.content

def createAgent(
    name: str,
    system_prompt: str,
    llm: ChatOpenAI,
    tools: list,
) -> str:
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","""
             You are an agent within a multi-agent architecture.\n
             "Keep all language concise but detailed as necessary.\n
             "You have the following role: \n{system_message}\n\n
             "You have access to the following tools: {tool_names}.\n\n 
             """),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    prompt = prompt.partial(system_message=system_prompt)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    agent = RunnableAgent(
            runnable=create_openai_tools_agent(llm, tools=tools, prompt=prompt),
            input_keys_arg=["messages"],
            return_keys_arg=["output"]
        )
    return AgentExecutor(name=name, agent=agent, tools=tools)

    """An LLM-based router."""
    if exit:
        members.append("FINISH")
    route_fx = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": members},
                    ],
                },
                "sender": {
                    "title": "Sender",
                    "anyOf": [
                        {"string": name},
                    ]
                }
            },
            "required": ["next", "sender"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            )
        ]
    ).partial(options=str(members), team_members=", ".join(members))
    logger.debug(f"Router options: {members}");
    return (
        prompt
        | llm.bind_functions(functions=[route_fx], function_call="route")
        | JsonOutputFunctionsParser()
    )

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    try:
        result = agent.invoke(state)
        #logger.info(f"Agent {name} invoked with state: {state}")
        if "output" in result.keys(): # If the agent has an output
            result = AIMessage(content=result["output"], name=name)
            return {
                "messages": [result],
                "sender": name,
            }
        else: # If it routed to another agent
            return {
                "sender": name,
                "next": result["next"]
            }
    except Exception as e:
        logger.error(e)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    next: str

def workflow():
    return StateGraph(AgentState)

def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results
