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
from agent_tools import tavily_search_tool, get_recipe_info, generate_recipe, generate_ingredient, create_recipe_graph, get_recipe, add_node, get_foundational_recipe, get_graph, generate_mod, suggest_mod, get_mods_list, apply_mod, rank_mod, remove_mod, get_datetime

prompts_dict = {
    "Frontman": {
        "type": "agent",
        "prompt": "You are Cauldron, an intelligent assistant for recipe development. You are friendly and chipper in your responses. Your task is to summarize the entirety of the given message chain and deliver a concise explanation of changes to the user.",
        "tools": [get_datetime]
    },
    "Cauldron\nPostman": {
        "type": "supervisor",
        "prompt": "You are Cauldron\nPostman. You are tasked with managing user requests related to recipe development. Without notifying the user, you supervise the following agents: Research\nPostman, ModSquad, Spinnaret. When a user request is received, respond with a confirmation and specify which agent(s) will act next. Each agent will perform a task and respond with their results and status. Ensure that all changes are recorded by Spinnaret before completing.  When all tasks are complete, respond with FINISH.\n\nWhen a user request is first received with no prior history, you may assign the request to Research\nPostman to find an appropriate recipe to serve as the Foundational Recipe. Once a recipe is found (with appropriate name, ingredients, instructions, and optional tags and sourcing), you will assign the recipe to RecipeDevelopmentTracker to set the foundational recipe. You will then compile a comprehensive report summarizing the findings and status of the recipe development process and report this back to the user.",
        "members":["Research\nPostman", "ModSquad", "Spinnaret"] # TODO - "Critic" & "Jimmy"
    },
    "Research\nPostman": {
        "type": "supervisor",
        "prompt": "You are Research\nPostman, a supervisor agent focused on research for recipe development. You oversee the following nodes in the Cauldron application: Tavily Sleuth. Your task is to coordinate their efforts to ensure seamless recipe development. When a message is received, you may interpret it as you see fit and assign tasks to the appropriate agents based on their specializations. Collect and review the results from each agent, giving follow-up tasks as needed and resolving any detected looping issues or requests for additional input. Once all agents have completed their tasks, compile a comprehensive report summarizing their findings and the overall status of the recipe development process and report this back to the Cauldron\nPostman.",
        "members": ["Tavily", "Sleuth", "Cauldron\nPostman"] #TODO - "Bookworm", "Remy", "HealthNut", "MrKrabs" 
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
        "prompt": "You are Tavily. Your task is to search the internet for relevant recipes that match the user's request. You will use the Tavily search tool to fulfill these requests and find relevant recipes. Once you have found a recipe, forward it to the Research\nPostman.",
        "tools": [tavily_search_tool]
    },
    "Sleuth": {
        "type": "agent",
        "prompt": "You are Sleuth. Your task is to scrape recipe data from the internet. You will be given a scraper tool to fulfill these requests and find relevant recipe information. Once you have found information, complete your task by using the generate_recipe tool to summarize each recipe found. Pass your results to the Research\nPostman.",
        "tools": [get_recipe_info, generate_recipe, generate_ingredient],
        "tool_choice": {"type": "function", "function": {"name": "generate_recipe"}}
    },
    "ModSquad": {
        "type": "agent",
        "prompt": "You are ModSquad. Your task is to manage suggested modifications to the recipe based on inputs from other nodes. These modifications are stored in a variable called mod_list. Analyze suggestions from other agents that have been added to the mods_list and perform tasks as recommended by the User or Cauldron\nPostman. Forward the updated recipe to the Spinnaret. DO NOT ask if any more modifications are needed. If you are unsure about a modification, ask the Cauldron\nPostman for clarification.",
        "tools": [generate_mod, suggest_mod, get_mods_list, apply_mod, rank_mod, remove_mod],
    },
    "Spinnaret": {
        "type": "agent",
        "prompt": "You are Spinnaret. Your task is to plot and track the development process of the recipe, represented by the recipe_graph object, documenting all changes and decisions made by other nodes. You will recieve instruction from the Cauldron\nPostman or the ModSquad on how to develop the recipe_graph object appropriately. Prior to modifying the recipe_graph, always check its size and the foundational recipe. If recipe_graph has no nodes (like when the graph is first initialized), use context provided to generate a foundational recipe and add it to the recipe_graph. Ensure that the development path is clear and logical.",
        "tools": [generate_recipe, generate_ingredient, create_recipe_graph, get_recipe, get_foundational_recipe, get_graph],
    },
    #"Jimmy": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Peripheral Interpreter node for the Cauldron application. Your task is to interpret feedback from connected smart devices (e.g., smart ovens, kitchen scales) and provide actionable insights to other nodes. Ensure all outputs follow Pydantic standards and format them accordingly. Forward the insights to the relevant nodes (e.g., Feedback Interpreter, Recipe Modification Manager). Clearly address any looping issues or need for further input.",
    #    "tools": [suggest_mod],
    #},
    #"Glutton": {
    #    "type": "agent",
    #    "prompt": "You are Glutton. Your task is to discern YES or NO as a judgement of whether the following item is food or not:",
    #    "tools": [get_datetime]
    #}
}

direct_edges = [
    #("Research\nPostman", "Cauldron\nPostman"),
    ("ModSquad", "Cauldron\nPostman"),
    ("Spinnaret", "Cauldron\nPostman"),
    #("Critic", "Cauldron\nPostman"), TODO
    #("Jimmy", "Cauldron\nPostman"),
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
            if name == "Cauldron\nPostman":
                agent = createRouter(name, d["prompt"], llm, members=d["members"], exit=True)
            else:
                agent = createRouter(name, d["prompt"], llm, members=d["members"])

        elif d["type"] == "sql":
            logger.info(f"Creating SQL agent: {name}")
            agent = createBookworm(name, d["prompt"], llm_model, db_path, verbose=True)

        elif d["type"] == "agent":
            logger.info(f"Creating agent: {name}")
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
    ("Cauldron\nPostman", "Research\nPostman"),
    ("Cauldron\nPostman", "ModSquad"),
    ("Cauldron\nPostman", "Spinnaret"),
    ("Cauldron\nPostman", "Frontman"),
    ("Research\nPostman", "Tavily"),
    ("Research\nPostman", "Sleuth"),
]

def create_conditional_edges(flow_graph):
    
    flow_graph.add_conditional_edges(
        "Cauldron\nPostman",
        lambda x: x["next"],
        {
            "Research\nPostman": "Research\nPostman", 
            #"Critic": "Critic", TODO
            "ModSquad": "ModSquad", 
            "Spinnaret": "Spinnaret", 
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
            "Cauldron\nPostman": "Cauldron\nPostman",
            #"Bookworm": "Bookworm",
            "Tavily": "Tavily",
            "Sleuth": "Sleuth",
        },
    )

    return conditional_edges