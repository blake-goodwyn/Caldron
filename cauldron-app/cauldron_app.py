### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
from util import db_path, llm_model
import warnings
import functools
from logging_util import logger
from agent_defs import prompts_dict
from langchain_util import ChatOpenAI, OPENAI_API_KEY, createTeamSupervisor, createAgent, agentNode, CauldronState, enter_chain
from langgraph.graph import END, StateGraph
from agent_tools import datetime_tool

warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CauldronApp():
    
    def __init__(self, db_path, llm_model, verbose=False):
        
        logger.info("Initializing Cauldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = ChatOpenAI(model=llm_model)

        ##Determine Hierarchical Structure

        self.agents = []
        self.graph = StateGraph(CauldronState)

        for agent_name, prompt in prompts_dict.items():
            if agent_name != "ConductorAgent":
                logger.info(f"Creating agent for {agent_name}")
                self.agents.append(agent_name)
                curAgent = createAgent(self.llm, [datetime_tool], prompt)
                functools.partial(agentNode, agent=curAgent, name=agent_name)
                self.graph.add_node(agent_name, curAgent)
        
        logger.info(f"Creating agent for {agent_name}")
        self.Conductor = createTeamSupervisor(
            self.llm,
            prompts_dict["ConductorAgent"],
            self.agents,
        )

        self.graph.add_node("ConductorAgent", self.Conductor)

        # Define the control flow
        for agent_name in self.agents:
            self.graph.add_edge(agent_name, "ConductorAgent")

        self.graph.add_conditional_edges(
            "ConductorAgent",
            lambda x: x["next"],
            {"FlavorProfileAgent": "FlavorProfileAgent", 
             "NutrionalAnalysisAgent": "NutrionalAnalysisAgent",
             "CostAvailabilityAgent": "CostAvailabilityAgent",
             "FeedbackAgent": "FeedbackAgent",
            "SQLAgent": "SQLAgent",
            "ModificationsAgent": "ModificationsAgent",
            "DevelopmentTrackerAgent": "DevelopmentTrackerAgent",
            "PeripheralFeedbackAgent": "PeripheralFeedbackAgent",
            "FINISH": END},
        )
        
        self.graph.set_entry_point("ConductorAgent")
        self.chain = self.graph.compile()
        self.interface = enter_chain | self.chain


app = CauldronApp(db_path, llm_model)
for s in app.interface.stream(
    "I want to make gluten-free bread with xanthan gum", {"recursion_limit": 100}
):
    if "__end__" not in s:
        print(s)
        print("---")