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
from agent_tools import tavily_search_tool, scrape_recipe_info, generate_recipe, clear_pot, create_recipe_graph, get_recipe, get_recipe_from_pot, examine_pot, add_node, get_foundational_recipe, set_foundational_recipe, get_graph, suggest_mod, get_mods_list, apply_mod, rank_mod, remove_mod, pop_url_from_pot, add_url_to_pot

prompts_dict = {
    "Frontman": {
        "type": "agent",
        "prompt": "You are Caldron, an intelligent assistant for recipe development. You will be friendly and chipper in your responses. Through the use of other agents in the architecture, you are capable of finding recipe information, aiding in ideation, adapting recipes given specific constraints, and integrating recipe feedback. Your task is primarily to handle interactions with the user and to summarize the entirety of the given message chain from other agents to deliver a concise explanation of changes to the user.\nQuestions that come up that require user feedback will be sent to you. Please pose them to the user. Assume the user has no information beyond what they explicitly give to you.\nYou will make no mention of the tools used to do so such as the names of agents, the names of tools, or the Pot. Prior to completing, run the clear_pot tool to ensure that all recent recipes are cleared from short-term memory.",
        "tools": [clear_pot]
    },
    "Caldron\nPostman": {
        "type": "supervisor",
        "prompt": """
        You are Caldron\nPostman. You are tasked with determining what task needs to be accomplished next given the messages provided. Whenever possible, follow explicit user instructions and nothing more. Each agent will perform a task and respond with their results. The following are the roles of the agents available to you:\n
        - Research\nPostman: Researches recipe information and coordinates the efforts of web search via Tavily and recipe scraping via Sleuth.\n
        - ModSquad: Manages suggested modifications to the recipe based on inputs from other nodes.\n
        - Spinnaret: Answers general questions about the recipe. Plots and tracks the development process of the recipe, represented by the Recipe Graph.\n
        - Frontman: Provides messages from the user to the Caldron application. All questions coming from agents that require user feedback should be sent to Frontman.\n
        - KnowItAll: Answers general questions about the recipe. Has access to the foundational recipe and the Recipe Graph.\n
        When all tasks are complete, respond with FINISH. Ensure that all changes are recorded by Spinnaret before completing.
        """,
        "members":["Research\nPostman", "ModSquad", "Spinnaret", "Frontman", "KnowItAll"] # TODO - "Critic" & "Jimmy"
    },
    "Research\nPostman": {
        "type": "supervisor",
        "prompt": """
        You are Research\nPostman, a supervisor agent focused on research for recipe development. You oversee the following nodes in the Caldron application:\n
        - Tavily: Searches the internet for relevant recipes that may match the user's request.\n
        - Sleuth. Scrapes recipe information from given URLs.\n
        Your task is to coordinate their efforts to ensure seamless recipe information retrieval.\n 
        When a message is received, you may assign tasks to the appropriate agents based on their specializations. Collect and review the results from each agent, giving follow-up tasks as needed and resolving any detected looping issues or requests for additional input. Once all agents have completed their tasks, direct this back to the Caldron\nPostman.
        """,
        "members": ["Tavily", "Sleuth", "Caldron\nPostman"] #TODO - "Bookworm", "Remy", "HealthNut", "MrKrabs" 
    },
    #"Remy": { TODO
    #    "type": "agent",
    #    "prompt": "You are Remy, the Flavor Profiling agent. Your task is to analyze the flavor profiles of the ingredients provided and suggest combinations that enhance the overall taste of the recipe. Once the analysis is complete, forward your results to the relevant nodes (e.g., Nutritional Balancing, Recipe Modification Manager). If you detect a looping issue or need further input, communicate this clearly and concisely.",
    #    "tools": [get_foundational_recipe, suggest_mod],
    #},
    #"HealthNut": { TODO
    #    "type": "agent",
    #    "prompt": "You are HealthNut, the Nutritional Analysis agent. Your task is to evaluate the nutritional content of the ingredients provided and ensure the recipe meets specific nutritional guidelines. Make suggestions for ingredient adjustments to achieve a balanced nutrient profile. Format all outputs according to Pydantic standards and forward your results to the relevant nodes (e.g., Flavor Profiling, Recipe Modification Manager). Address any looping issues or additional input needs clearly and concisely.",
    #    "tools": [get_foundational_recipe, suggest_mod],
    #},
    #"MrKrabs": { TODO
    #    "type": "agent",
    #    "prompt": "You are Mr. Krabs, the Cost & Sourcing agent. Your task is to assess the cost and availability of the ingredients provided. Analyze market trends, regional availability, and pricing data to suggest the most cost-effective and available options. Ensure all communication follows Pydantic standards and format your output accordingly. Forward your results to the relevant nodes (e.g., Nutritional Balancing, Recipe Modification Manager). Clearly communicate if additional input or a change in direction is needed.",
    #    "tools": [get_foundational_recipe, suggest_mod],
    #},
    #"Critic": { TODO
    #    "type": "agent",
    #    "prompt": "You are Critic, the Feedback Interpreter agent. Your task is to interpret feedback from users and other nodes, identifying areas for recipe refinement. Analyze the feedback to suggest actionable changes. Ensure all outputs follow Pydantic standards and format them accordingly. Forward your results to the relevant nodes (e.g., Recipe Modification Manager, Flavor Profiling). Clearly address any looping issues or need for further input.",
    #    "tools": [suggest_mod],
    #},
    #"Bookworm": {
    #    "type": "sql",
    #    "prompt": "In the event that a simple statement is received, you may reframe this statement as a question. For example, 'I want to make gluten-free bread with xanthan gum' could be reframed as 'What are common recipes for gluten-free bread with xanthan gum?'"
    #},
    "Tavily": {
        "type": "agent",
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
        "prompt": """
        You are Sleuth. Your task is to scrape recipe data from the internet. Some actions may be:\n
        1. Grab URLs from the Pot. Use the pop_url_from_pot tool to retrieve a URL from the Pot.\n
        2. Get recipe information. Use the scrape_recipe_info tool to find information about a specific recipe given its URL.\n
        3. Generate a recipe. Use the generate_recipe tool to summarize the recipe found and add it to the Pot.\n
        4. Examine short-term memory. Use the examine_pot tool to view all recipes and URLs in the Pot or get_recipe_from_pot to examine a specific recipe.\n\n
        You MUST use the scrape_recipe_info tool on URLs in the Pot given to you. You will then use generate_recipe with that information. Esnure that you have examined all recipe URLs identified before proceeding. Once all recipes have been assessed, pass your results to the Research\nPostman.
        """,
        "tools": [pop_url_from_pot, scrape_recipe_info, generate_recipe, get_recipe_from_pot, examine_pot],
        "tool_choice": {"type": "function", "function": {"name": "generate_recipe"}}
    },
    "ModSquad": {
        "type": "agent",
        "prompt": """
        You are ModSquad. Your task is to manage suggested modifications to the recipe based on inputs from other nodes. These modifications are stored in the Mod List. Modifications in the Mod List must be applied to the recipe using the apply_mod tool. Analyze suggestions from other agents and perform tasks on the Mod List as recommended by the messages. Some actions may be:\n
        1. Suggest a modification. Use the suggest_mod tool to create a new modification based on the provided information and add it to the Mod List. Note: this DOES NOT APPLY the suggestion the recipe. Use the apply_mod tool to make changes to the recipe.\n
        2. Apply a modification. Use the apply_mod tool to apply the top-ranked modification from the Mod List to the foundational recipe.\n
        3. Examine the Mod List. Use the get_mods_list tool to retrieve the current list of modifications and examine the contents.\n
        4. Re-rank modifications. Use the rank_mod tool to adjust the priority of a given modifications in the Mod List based on importance.\n
        5. Remove a modification. Use the remove_mod tool to remove a modification from the Mod List.\n\n
        Always forward the updated recipe to the Spinnaret for appropriate adjustment to the Recipe Graph. DO NOT ask if any more modifications are needed. If you are unsure about a modification, ask the Caldron\nPostman for clarification.
        """,
        "tools": [suggest_mod, get_mods_list, apply_mod, rank_mod, remove_mod],
    },
    "KnowItAll": {
        "type": "agent",
        "prompt": "You are KnowItAll. Your task is to answer general questions about the recipe. You have access to the foundational recipe and the Recipe Graph. Use the get_foundational_recipe tool to retrieve information on the current foundational recipe. Use the get_graph tool to retrieve the current recipe graph. You may also use the set_foundational_recipe tool to change the foundational recipe. You will be asked to provide information about the recipe and the Recipe Graph.",
        "tools": [get_foundational_recipe, get_graph, get_recipe],
    },
    "Spinnaret": {
        "type": "agent",
        "prompt": """
        You are Spinnaret. Your task is to track the development process of the recipe, represented by the Recipe Graph, documenting all changes and decisions made by other nodes. Always ensure that the Recipe Graph has a foundational recipe. You have three sources of information at your disposal: the message thread provided, the Pot which contains recently examined recipes, and the Recipe Graph which tracks overall development progression. You must use these sources to indicate how to develop the Recipe Graph appropriately. Some actions you may take are:\n

        1. Create a recipe graph object if none exists. Use the get_graph tool to retrieve the current recipe graph and the create_recipe_graph tool if the recipe graph is empty.\n

        2. Add a new node to the recipe graph representing a change to the foundational recipe. Use the get_recipe_from_pot tool to examine recent recipes and the add_node tool to add a new node to the recipe graph.\n

        3. Change the foundational recipe to another node in the Recipe Graph. Use the get_foundational_recipe tool to retrieve the current foundational recipe and the set_foundational_recipe tool to change the foundational recipe.\n

        4. Examine the foundational recipe (also referred to as "the recipe"). Use the get_foundational_recipe tool to retrieve information on the current foundational recipe.\n

        5. Examine the Pot to determine whether a new node should be added to the recipe graph. Use the get_recipe_from_pot tool to examine a specific recipe in the Pot and the add_node tool to add a new node to the recipe graph.\n\n

        Always ensure that the Recipe Graph has a foundational recipe set and is up-to-date with the most recent changes to the recipe. If you are unsure about a change, ask the Caldron\nPostman for clarification.
        """,
        "tools": [create_recipe_graph, add_node, get_recipe, get_foundational_recipe, set_foundational_recipe, get_graph, get_recipe_from_pot, examine_pot],
    },
    #"Jimmy": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Peripheral Interpreter node for the Caldron application. Your task is to interpret feedback from connected smart devices (e.g., smart ovens, kitchen scales) and provide actionable insights to other nodes. Ensure all outputs follow Pydantic standards and format them accordingly. Forward the insights to the relevant nodes (e.g., Feedback Interpreter, Recipe Modification Manager). Clearly address any looping issues or need for further input.",
    #    "tools": [suggest_mod],
    #},
    #"Glutton": {
    #    "type": "agent",
    #    "prompt": "You are Glutton. Your task is to discern YES or NO as a judgement of whether the following item is food or not:",
    #    "tools": [get_datetime]
    #}
}

direct_edges = [
    #("Research\nPostman", "Caldron\nPostman"),
    ("ModSquad", "Caldron\nPostman"),
    ("Spinnaret", "Caldron\nPostman"),
    ("KnowItAll", "Caldron\nPostman"),
    #("Critic", "Caldron\nPostman"), TODO
    #("Jimmy", "Caldron\nPostman"),
    #("Remy", "Research\nPostman"),
    #("HealthNut", "Research\nPostman"),
    #("MrKrabs", "Research\nPostman"),
    #("Bookworm", "Research\nPostman"),
    ("Tavily", "Research\nPostman"),
    ("Sleuth", "Research\nPostman"),
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

conditional_edges = [
    ("Caldron\nPostman", "Research\nPostman"),
    ("Caldron\nPostman", "ModSquad"),
    ("Caldron\nPostman", "Spinnaret"),
    ("Caldron\nPostman", "Frontman"),
    ("Caldron\nPostman", "KnowItAll"),
    ("Research\nPostman", "Tavily"),
    ("Research\nPostman", "Sleuth"),
]

def create_conditional_edges(flow_graph):
    
    flow_graph.add_conditional_edges(
        "Caldron\nPostman",
        lambda x: x["next"],
        {
            "Research\nPostman": "Research\nPostman", 
            #"Critic": "Critic", TODO
            "ModSquad": "ModSquad", 
            "Spinnaret": "Spinnaret", 
            "Frontman": "Frontman",
            "KnowItAll": "KnowItAll",
            "FINISH": "Frontman",
            #"Jimmy": "Jimmy" TODO
        },
    )

    flow_graph.add_conditional_edges(
        "Research\nPostman",
        lambda x: x["next"],
        {
            #"Remy": "Remy", TODO
            #"HealthNut": "HealthNut", TODO 
            #"MrKrabs": "MrKrabs", TODO
            "Caldron\nPostman": "Caldron\nPostman",
            #"Bookworm": "Bookworm",
            "Tavily": "Tavily",
            "Sleuth": "Sleuth",
        },
    )

    return conditional_edges