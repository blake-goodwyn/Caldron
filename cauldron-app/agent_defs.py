### Definition Strings for chains of various Agents

# Agent Definitions
# -----------------

import functools
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from logging_util import logger
from langchain_util import createAgent, createTeamSupervisor, agent_node, createSQLAgent
from recipe_graph import create_recipe_graph, get_recipe, add_node, get_foundational_recipe, get_graph
from mods_list import suggest_mod, get_mods_list, push_mod, rank_mod, remove_mod
from langgraph.graph import END
from util import db_path, llm_model

prompts_dict = {
    "ConductorAgent": {
        "type": "supervisor",
        "prompt": "You are a supervisor tasked with managing user requests related to recipe development between the following workers: RecipeResearchAgent, ModificationsAgent, DevelopmentTrackerAgent. When a user request is received, respond with a confirmation and specify which worker(s) will act next. Each worker will perform a task and respond with their results and status. When all tasks are complete, respond with FINISH. Ensure all communication follows Pydantic standards and maintain a clear and concise log of all communications for transparency.",
        "members":["RecipeResearchAgent", "ModificationsAgent", "DevelopmentTrackerAgent"] # TODO - "FeedbackAgent" & "PeripheralFeedbackAgent"
    },
    "RecipeResearchAgent": {
        "type": "supervisor",
        "prompt": "You are a supervisor agent focused on research for recipe development. You oversee the following nodes in the Cauldron application: SQLAgent. Your task is to coordinate their efforts to ensure seamless recipe development. When a user request is received, you may interpret it as you see fit and assign tasks to the appropriate agents based on their specializations. Collect and review the results from each agent, giving follow-up tasks as needed and resolving any detected looping issues or requests for additional input. Once all agents have completed their tasks, compile a comprehensive report summarizing their findings and the overall status of the recipe development process and report this back to the ConductorAgent.",
        "members": ["SQLAgent", "ConductorAgent"] #TODO - "FlavorProfileAgent", "NutrionalAnalysisAgent", "CostAvailabilityAgent" 
    },
    #"FlavorProfileAgent": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Flavor Profiling node for the Cauldron application. Your task is to analyze the flavor profiles of the ingredients provided and suggest combinations that enhance the overall taste of the recipe. Ensure your analysis adheres to Pydantic standards and format your output accordingly. Once the analysis is complete, forward your results to the relevant nodes (e.g., Nutritional Balancing, Recipe Modification Manager). If you detect a looping issue or need further input, communicate this clearly and concisely.",
    #    "tools": [get_foundational_recipe, suggest_mod],
    #},
    #"NutrionalAnalysisAgent": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Nutritional Balancing node for the Cauldron application. Your task is to evaluate the nutritional content of the ingredients provided and ensure the recipe meets specific nutritional guidelines. Make suggestions for ingredient adjustments to achieve a balanced nutrient profile. Format all outputs according to Pydantic standards and forward your results to the relevant nodes (e.g., Flavor Profiling, Recipe Modification Manager). Address any looping issues or additional input needs clearly and concisely.",
    #    "tools": [get_foundational_recipe, suggest_mod],
    #},
    #"CostAvailabilityAgent": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Cost & Availability node for the Cauldron application. Your task is to assess the cost and availability of the ingredients provided. Analyze market trends, regional availability, and pricing data to suggest the most cost-effective and available options. Ensure all communication follows Pydantic standards and format your output accordingly. Forward your results to the relevant nodes (e.g., Nutritional Balancing, Recipe Modification Manager). Clearly communicate if additional input or a change in direction is needed.",
    #    "tools": [get_foundational_recipe, suggest_mod],
    #},
    #"FeedbackAgent": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Feedback Interpreter node for the Cauldron application. Your task is to interpret feedback from users and other nodes, identifying areas for recipe refinement. Analyze the feedback to suggest actionable changes. Ensure all outputs follow Pydantic standards and format them accordingly. Forward your results to the relevant nodes (e.g., Recipe Modification Manager, Flavor Profiling). Clearly address any looping issues or need for further input.",
    #    "tools": [suggest_mod],
    #},
    "SQLAgent": {
        "type": "sql",
        "prompt": "In the event that a simple statement is received, you may reframe this statement as a question. For example, 'I want to make gluten-free bread with xanthan gum' could be reframed as 'What is a popular recipe for gluten-free bread with xanthan gum?'"
    },
    "ModificationsAgent": {
        "type": "agent",
        "prompt": "You are the Recipe Modification Manager node for the Cauldron application. Your task is to manage modifications to the recipe based on inputs from other nodes. Analyze suggestions from Flavor Profiling, Nutritional Balancing, Cost & Availability, and Feedback Interpreter to update the recipe accordingly. Ensure all outputs follow Pydantic standards and format them accordingly. Forward the updated recipe to the relevant nodes (e.g., Recipe Development Tracker). Address any looping issues or need for further input clearly.",
        "tools": [suggest_mod, get_mods_list, push_mod, rank_mod, remove_mod],
    },
    "DevelopmentTrackerAgent": {
        "type": "agent",
        "prompt": "You are the Recipe Development Tracker node for the Cauldron application. Your task is to plot and track the development process of the recipe, documenting all changes and decisions made by other nodes. Ensure that the development path is clear and logical. Format all documentation following Pydantic standards. Provide detailed logs to the relevant nodes and the Conductor node for transparency. Address any looping issues or need for further input clearly and concisely.",
        "tools": [create_recipe_graph, get_recipe, add_node, get_foundational_recipe, get_graph],
    },
    #"PeripheralFeedbackAgent": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Peripheral Interpreter node for the Cauldron application. Your task is to interpret feedback from connected smart devices (e.g., smart ovens, kitchen scales) and provide actionable insights to other nodes. Ensure all outputs follow Pydantic standards and format them accordingly. Forward the insights to the relevant nodes (e.g., Feedback Interpreter, Recipe Modification Manager). Clearly address any looping issues or need for further input.",
    #    "tools": [suggest_mod],
    #},
}

direct_edges = [
    #("RecipeResearchAgent", "ConductorAgent"),
    ("ModificationsAgent", "ConductorAgent"),
    ("DevelopmentTrackerAgent", "ConductorAgent"),
    #("FeedbackAgent", "ConductorAgent"), TODO
    #("PeripheralFeedbackAgent", "ConductorAgent"),
    #("FlavorProfileAgent", "RecipeResearchAgent"),
    #("NutrionalAnalysisAgent", "RecipeResearchAgent"),
    #("CostAvailabilityAgent", "RecipeResearchAgent"),
    ("SQLAgent", "RecipeResearchAgent"),
    ("ConductorAgent", END)
]

def create_all_agents(llm: ChatOpenAI, prompts_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Creating all agents.")
    agents = {}

    for name, d in prompts_dict.items():
        if d["type"] == "supervisor":
            logger.info(f"Creating supervisor agent: {name}")
            agent = createTeamSupervisor(llm, d["prompt"], name, members=d["members"])
            node = agent

        elif d["type"] == "sql":
            logger.info(f"Creating SQL agent: {name}")
            agent = createSQLAgent(d["prompt"],llm_model, db_path, verbose=True)
            node = functools.partial(agent_node, agent=agent, name=name)

        elif d["type"] == "agent":
            logger.info(f"Creating agent: {name}")
            agent = createAgent(llm, d["tools"], d["prompt"])
            node = functools.partial(agent_node, agent=agent, name=name)

        agents[name] = node
        logger.info(f"Agent {name} created.")

    logger.info("All agents created.")
    return agents