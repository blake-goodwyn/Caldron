### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
import warnings
from logging_util import logger
from langchain_util import ChatOpenAI, workflow, enter_chain
from recipe_graph import fresh_graph, default_graph_file, fresh_mods_list, default_mods_list_file
from agent_defs import create_all_agents, prompts_dict, form_edges, create_conditional_edges
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CauldronApp():
    
    def __init__(self, db_path, llm_model, defs=prompts_dict, gf=default_graph_file, mlist=default_mods_list_file, verbose=False):
        
        logger.info("Initializing Cauldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

        #Central Data Structures
        self.recipe_graph = fresh_graph(gf)
        self.mods_list = fresh_mods_list(mlist)

        ##Determine Agent Structure
        self.agents = create_all_agents(self.llm, defs)

        # Define the control flow
        self.flow_graph = workflow()
        self.display_graph = nx.DiGraph()
        for node_name, node in self.agents.items():
            self.flow_graph.add_node(node_name, node)
            self.display_graph.add_node(node_name)

        direct_edges = form_edges(self.flow_graph)
        for edge in direct_edges:
            self.display_graph.add_edge(*edge)

        conditional_edges = create_conditional_edges(self.flow_graph)

        self.flow_graph.set_entry_point("ConductorAgent")

        # Separate edges into solid and dotted line groups
        solid_edges = direct_edges
        dotted_edges = conditional_edges

        # Get positions for the nodes
        pos = nx.spring_layout(self.display_graph, k=0.5, iterations=50)

        # Centralize a specific node
        central_node = 'ConductorAgent'
        pos[central_node] = [0, 0]  # Position the central node at the center
        for node in pos:
            if node != central_node:
                pos[node][0] += 0.5  # Offset other nodes to maintain minimum distance

        nx.draw_networkx_nodes(self.display_graph, pos, node_size=3000, node_color="lightblue")
        nx.draw_networkx_edges(self.display_graph, pos, edgelist=solid_edges, style='solid', connectionstyle='arc3,rad=0.2', arrows=True)
        nx.draw_networkx_edges(self.display_graph, pos, edgelist=dotted_edges, style='dotted', connectionstyle='arc3,rad=0.2', arrows=True)
        nx.draw_networkx_labels(self.display_graph, pos, font_size=10, font_weight="bold")
        #plt.show()
        
        self.chain = self.flow_graph.compile()
        self.interface = enter_chain | self.chain