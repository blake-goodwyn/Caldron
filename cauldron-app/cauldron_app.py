### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
import warnings
from logging_util import logger
from langchain_util import ChatOpenAI, workflow, enter_chain
from recipe_graph import fresh_graph, default_graph_file
from mods_list import fresh_mods_list, default_mods_list_file
from agent_defs import create_all_agents, prompts_dict, direct_edges
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CauldronApp():
    
    def __init__(self, db_path, llm_model, defs=prompts_dict, edges=direct_edges, gf=default_graph_file, mlist=default_mods_list_file, verbose=False):
        
        logger.info("Initializing Cauldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = ChatOpenAI(model=llm_model)

        #Central Data Structures
        self.recipe_graph = fresh_graph(gf)
        self.mods_list = fresh_mods_list(mlist)

        ##Determine Agent Structure
        self.agents = create_all_agents(self.llm,defs)

        # Define the control flow
        self.flow_graph = workflow()
        self.display_graph = nx.DiGraph()
        for node_name, node in self.agents.items():
            self.flow_graph.add_node(node_name, node)
            self.display_graph.add_node(node_name)

        for source, target in edges:
            self.flow_graph.add_edge(source, target)
            self.display_graph.add_edge(source, target)

        self.flow_graph.add_conditional_edges(
            "ConductorAgent",
            lambda x: x["next"],
            {
                "RecipeResearchAgent": "RecipeResearchAgent", 
                #"FeedbackAgent": "FeedbackAgent", TODO
                "ModificationsAgent": "ModificationsAgent", 
                "DevelopmentTrackerAgent": "DevelopmentTrackerAgent", 
                #"PeripheralFeedbackAgent": "PeripheralFeedbackAgent" TODO
            },
        )

        self.flow_graph.add_conditional_edges(
            "RecipeResearchAgent",
            lambda x: x["next"],
            {
                #"FlavorProfileAgent": "FlavorProfileAgent", TODO
                #"NutrionalAnalysisAgent": "NutrionalAnalysisAgent", TODO 
                #"CostAvailabilityAgent": "CostAvailabilityAgent", TODO
                "ConductorAgent": "ConductorAgent",
                "SQLAgent": "SQLAgent"
            },
        )

        self.flow_graph.set_entry_point("ConductorAgent")

        pos = nx.spring_layout(self.display_graph)  # positions for all nodes
        nx.draw(self.display_graph, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
        #plt.show()
        
        self.chain = self.flow_graph.compile()
        self.interface = enter_chain | self.chain