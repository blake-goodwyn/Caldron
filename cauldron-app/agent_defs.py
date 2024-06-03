### Definition Strings for chains of various Agents

# Agent Definitions
# -----------------

import functools
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from logging_util import logger
from langchain_util import createAgent, createTeamSupervisor, agent_node, createSQLAgent
from langgraph.graph import END
from util import db_path, llm_model
from agent_tools import tavily_search_tool, get_recipe_info, generate_recipe, generate_ingredient, create_recipe_graph, get_recipe, add_node, get_foundational_recipe, get_graph, generate_mod, suggest_mod, get_mods_list, push_mod, rank_mod, remove_mod, get_datetime

prompts_dict = {
    "SummaryAgent": {
        "type": "agent",
        "prompt": "You are SummaryAgent. Your task is to summarize the given message chain and report back to the user. You will compile a comprehensive report summarizing the findings and status of the recipe development process.",
        "tools": [get_datetime]
    },
    "CauldronRouter": {
        "type": "supervisor",
        "prompt": "You are Cauldron, an intelligent assistant for recipe development. To other agents (not the user), you are the CauldronRouter. You are tasked with managing user requests related to recipe development. Without notifying the user, you supervise the following agents: RecipeResearchAgent, ModificationsAgent, DevelopmentTrackerAgent. When a user request is received, respond with a confirmation and specify which agent(s) will act next. Each agent will perform a task and respond with their results and status. When all tasks are complete, respond with FINISH.\n\nWhen a user request is first received with no prior history, you may assign the request to RecipeResearchAgent to find an appropriate recipe to serve as the Foundational Recipe. Once a recipe is found (with appropriate name, ingredients, instructions, and optional tags and sourcing), you will assign the recipe to RecipeDevelopmentTracker to set the foundational recipe. You will then compile a comprehensive report summarizing the findings and status of the recipe development process and report this back to the user.",
        "members":["RecipeResearchAgent", "ModificationsAgent", "DevelopmentTrackerAgent"] # TODO - "FeedbackAgent" & "PeripheralFeedbackAgent"
    },
    "RecipeResearchAgent": {
        "type": "supervisor",
        "prompt": "You are RecipeResearchAgent, a supervisor agent focused on research for recipe development. You oversee the following nodes in the Cauldron application: SearchAgent RecipeScraperAgent. Your task is to coordinate their efforts to ensure seamless recipe development. When a message is received, you may interpret it as you see fit and assign tasks to the appropriate agents based on their specializations. Collect and review the results from each agent, giving follow-up tasks as needed and resolving any detected looping issues or requests for additional input. Once all agents have completed their tasks, compile a comprehensive report summarizing their findings and the overall status of the recipe development process and report this back to the CauldronRouter.",
        "members": ["SearchAgent", "RecipeScraperAgent", "CauldronRouter"] #TODO - "SQLAgent", "FlavorProfileAgent", "NutrionalAnalysisAgent", "CostAvailabilityAgent" 
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
    #"SQLAgent": {
    #    "type": "sql",
    #    "prompt": "In the event that a simple statement is received, you may reframe this statement as a question. For example, 'I #want to make gluten-free bread with xanthan gum' could be reframed as 'What is a popular recipe for gluten-free bread with xanthan gum?'"
    #},
    "SearchAgent": {
        "type": "agent",
        "prompt": "You are SearchAgent. Your task is to search the internet for relevant recipes that match the user's request. You will use the Tavily search tool to fulfill these requests and find relevant recipes. Once you have found a recipe, forward it to the RecipeResearchAgent.",
        "tools": [tavily_search_tool]
    },
    "RecipeScraperAgent": {
        "type": "agent",
        "prompt": "You are RecipeScraperAgent. Your task is to scrape recipe data from the internet. You will be given a scraper tool to fulfill these requests and find relevant recipe information. Once you have found information, use the generate_recipe tool to summarize each recipe found. Pass your results to the RecipeResearchAgent.",
        "tools": [get_recipe_info, generate_recipe, generate_ingredient]
    },
    "ModificationsAgent": {
        "type": "agent",
        "prompt": "You are ModficationsAgent. Your task is to manage suggested modifications to the recipe based on inputs from other nodes. These modifications are stored in a variable called mod_list. Analyze suggestions from other agents that have been added to the mods_list and perform tasks as recommended by the User or CauldronRouter. Forward the updated recipe to the DevelopmentTrackerAgent. DO NOT ask if any more modifications are needed. If you are unsure about a modification, ask the CauldronRouter for clarification.",
        "tools": [generate_mod, suggest_mod, get_mods_list, push_mod, rank_mod, remove_mod],
    },
    "DevelopmentTrackerAgent": {
        "type": "agent",
        "prompt": "You are DevelopmentTrackerAgent. Your task is to plot and track the development process of the recipe, represented by the recipe_graph object, documenting all changes and decisions made by other nodes. You will recieve instruction from the CauldronRouter or the ModificationsAgent on how to develop the recipe_graph object appropriately. Prior to modifying the recipe_graph, always check its size and the foundational recipe. If recipe_graph has no nodes (like when the graph is first initialized), use context provided to generate a foundational recipe and add it to the recipe_graph. Ensure that the development path is clear and logical.",
        "tools": [generate_recipe, generate_ingredient, create_recipe_graph, get_recipe, add_node, get_foundational_recipe, get_graph],
    },
    #"PeripheralFeedbackAgent": { TODO
    #    "type": "agent",
    #    "prompt": "You are the Peripheral Interpreter node for the Cauldron application. Your task is to interpret feedback from connected smart devices (e.g., smart ovens, kitchen scales) and provide actionable insights to other nodes. Ensure all outputs follow Pydantic standards and format them accordingly. Forward the insights to the relevant nodes (e.g., Feedback Interpreter, Recipe Modification Manager). Clearly address any looping issues or need for further input.",
    #    "tools": [suggest_mod],
    #},
}

direct_edges = [
    #("RecipeResearchAgent", "CauldronRouter"),
    ("ModificationsAgent", "CauldronRouter"),
    ("DevelopmentTrackerAgent", "CauldronRouter"),
    #("FeedbackAgent", "CauldronRouter"), TODO
    #("PeripheralFeedbackAgent", "CauldronRouter"),
    #("FlavorProfileAgent", "RecipeResearchAgent"),
    #("NutrionalAnalysisAgent", "RecipeResearchAgent"),
    #("CostAvailabilityAgent", "RecipeResearchAgent"),
    #("SQLAgent", "RecipeResearchAgent"),
    ("SearchAgent", "RecipeResearchAgent"),
    ("RecipeScraperAgent", "RecipeResearchAgent"),
    ("SummaryAgent", END)
]

def create_all_agents(llm: ChatOpenAI, prompts_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Creating all agents.")
    agents = {}

    for name, d in prompts_dict.items():
        if d["type"] == "supervisor":
            logger.info(f"Creating supervisor agent: {name}")
            agent = createTeamSupervisor(name, d["prompt"], llm, members=d["members"])

        elif d["type"] == "sql":
            logger.info(f"Creating SQL agent: {name}")
            agent = createSQLAgent(name, d["prompt"], llm_model, db_path, verbose=True)

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
    ("CauldronRouter", "RecipeResearchAgent"),
    ("CauldronRouter", "ModificationsAgent"),
    ("CauldronRouter", "DevelopmentTrackerAgent"),
    ("CauldronRouter", "SummaryAgent"),
    ("RecipeResearchAgent", "SearchAgent"),
    ("RecipeResearchAgent", "RecipeScraperAgent"),
]

def create_conditional_edges(flow_graph):
    flow_graph.add_conditional_edges(
        "CauldronRouter",
        lambda x: x["next"],
        {
            "RecipeResearchAgent": "RecipeResearchAgent", 
            #"FeedbackAgent": "FeedbackAgent", TODO
            "ModificationsAgent": "ModificationsAgent", 
            "DevelopmentTrackerAgent": "DevelopmentTrackerAgent", 
            "FINISH": "SummaryAgent",
            #"PeripheralFeedbackAgent": "PeripheralFeedbackAgent" TODO
        },
    )

    flow_graph.add_conditional_edges(
        "RecipeResearchAgent",
        lambda x: x["next"],
        {
            #"FlavorProfileAgent": "FlavorProfileAgent", TODO
            #"NutrionalAnalysisAgent": "NutrionalAnalysisAgent", TODO 
            #"CostAvailabilityAgent": "CostAvailabilityAgent", TODO
            "CauldronRouter": "CauldronRouter",
            #"SQLAgent": "SQLAgent",
            "SearchAgent": "SearchAgent",
            "RecipeScraperAgent": "RecipeScraperAgent",
        },
    )

    return conditional_edges