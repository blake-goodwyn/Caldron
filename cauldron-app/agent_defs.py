### Definition Strings for chains of various Agents

# Agent Definitions
# -----------------

import functools
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from logging_util import logger
from langchain_util import createAgent, createRouter, agent_node, createBookworm
from langgraph.graph import END
from util import db_path, llm_model
from agent_tools import get_datetime, tavily_search_tool, scrape_recipe_info, generate_recipe, clear_pot, create_recipe_graph, get_recipe, get_recipe_from_pot, examine_pot, add_node, get_foundational_recipe, set_foundational_recipe, get_graph, suggest_mod, get_mods_list, apply_mod, rank_mod, remove_mod, pop_url_from_pot, add_url_to_pot

prompts_dict = {
    "Frontman": {
        "type": "agent",
        "label": "User\nRep",
        "prompt": "You are Caldron, an intelligent assistant for recipe development. You will be friendly and chipper in your responses. Through the use of other agents in the architecture, you are capable of finding recipe information, aiding in ideation, adapting recipes given specific constraints, and integrating recipe feedback. Your task is primarily to handle interactions with the user and to summarize the entirety of the given message chain from other agents to deliver a concise explanation of changes to the user.\nQuestions that come up that require user feedback will be sent to you. Please pose them to the user. Assume the user has no information beyond what they explicitly give to you.\nYou will make no mention of the tools used to do so such as the names of agents, the names of tools, or the Pot. Prior to completing, run the clear_pot tool to ensure that all recent recipes are cleared from short-term memory.",
        "tools": [get_datetime]
    },
    "Tavily": {
        "type": "agent",
        "label": "Web\nSearch",
        "prompt": """
        You are Tavily. Your task is to search the internet for relevant recipes that match the user's request. Some actions may be:\n
        1. Search the internet. Use the tavily_search_tool to find a recipe that matches the user's request.\n
        2. Add a URL to the Pot. Use the add_url_to_pot tool to add a URL to the Pot for further examination.\n
        Make sure all URLs are added to the Pot for further examination by the Sleuth. Once all URLs have been identified, pass your results to the Research\nPostman.
        """,
        "tools": [tavily_search_tool, add_url_to_pot]
    },
    "Sleuth": {
        "type": "agent",
        "label": "Web\nScraper",
        "prompt": """
        You are Sleuth. Your task is to scrape recipe data from the internet. Some actions may be:\n
        1. Grab URLs from the Pot. Use the pop_url_from_pot tool to retrieve a URL from the Pot.\n
        2. Get recipe information. Use the scrape_recipe_info tool to find information about a specific recipe given its URL.\n
        3. Generate a recipe. Use the generate_recipe tool to summarize the recipe found and add it to the Pot.\n
        You MUST use the scrape_recipe_info tool on URLs in the Pot given to you. You will then use generate_recipe with that information. Esnure that you have examined all recipe URLs identified before proceeding. Once all recipes have been assessed, pass your results to the Research\nPostman.
        """,
        "tools": [pop_url_from_pot, scrape_recipe_info, generate_recipe],
        "tool_choice": {"type": "function", "function": {"name": "generate_recipe"}}
    },
    "Spinnaret": {
        "type": "agent",
        "label": "Dev\nTracker",
        "prompt": """
        You are Spinnaret. Your task is simple: call create_recipe_graph to make a recipe graph."
        """,
        "tools": [create_recipe_graph],
    },
}

direct_edges = [
    ("Tavily", "Sleuth"),
    ("Sleuth", "Spinnaret"),
    ("Spinnaret", "Frontman"),
    ("Frontman", END)
]

def create_all_agents(llm: ChatOpenAI, prompts_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Creating all agents.")
    agents = {}

    for name, d in prompts_dict.items():
        if d["type"] == "supervisor":
            logger.info(f"Creating supervisor agent: {name}")
            if name == "Caldron\nPostman":
                agent = createRouter(name, d["prompt"], llm, members=d["members"], exit=True)
            else:
                agent = createRouter(name, d["prompt"], llm, members=d["members"])

        elif d["type"] == "sql":
            logger.info(f"Creating SQL agent: {name}")
            agent = createBookworm(name, d["prompt"], llm_model, db_path, verbose=True)

        elif d["type"] == "agent":
            logger.info(f"Creating agent: {name}")
            if "tool_choice" in d:
                agent = createAgent(name, d["prompt"], llm, d["tools"]) #TODO - add tool_choice
            else:
                agent = createAgent(name, d["prompt"], llm, d["tools"])

        agents[name] = functools.partial(agent_node, agent=agent, name=name)
        logger.info(f"Agent {name} created.")

    logger.info("All agents created.")
    return agents

def form_edges(flow_graph):
    for source, target in direct_edges:
        flow_graph.add_edge(source, target)
    return direct_edges
