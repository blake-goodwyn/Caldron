### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
import warnings
from logging_util import logger
from langchain_util import ChatOpenAI, workflow, enter_chain, HumanMessage
from recipe_graph import fresh_graph, default_graph_file, fresh_mods_list, default_mods_list_file, load_graph_from_file
from agent_defs import create_all_agents, prompts_dict, form_edges, create_conditional_edges
from custom_print import printer
import matplotlib.pyplot as plt
import networkx as nx
import threading

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

        self.flow_graph.set_entry_point("Cauldron\nPostman")
        self.chain = self.flow_graph.compile()
        self.interface = enter_chain | self.chain

        def update_graph(self, node_colors=["lightblue" for n in self.display_graph.nodes()]):
            plt.clf()
            plt.legend(["Sender", "Receiver"], loc="upper left")
            nx.draw_networkx_nodes(self.display_graph, self.node_pos, node_size=3000, node_color=node_colors)
            nx.draw_networkx_edges(self.display_graph, self.node_pos, edgelist=direct_edges, style='solid', connectionstyle='arc3,rad=0.2', arrows=True)
            nx.draw_networkx_edges(self.display_graph, self.node_pos, edgelist=conditional_edges, style='dotted', connectionstyle='arc3,rad=0.2', arrows=True)
            nx.draw_networkx_labels(self.display_graph, self.node_pos, font_size=10, font_weight="bold")
            plt.draw()

        ## Visualization Thread
        def visualize_graph(self):
            plt.ion()

            # Get positions for the nodes
            self.node_pos = nx.spring_layout(self.display_graph, k=0.75, iterations=50)
            update_graph(self)

        ## Simple Interaction Thread
        def simple_interaction_loop(self):
            i = input("Enter a message: ")
            while i != "exit":
                for s in self.chain.stream(
                    {
                        "messages": [
                            HumanMessage(
                                content=i
                            )
                        ],
                        'sender': 'user',
                        'next': 'Cauldron\nPostman'
                    },
                    {"recursion_limit": 50}
                ):
                    print("--------------------")
                    if 'Frontman' in s.keys():
                        print(s['Frontman']['messages'][0].content)
                    else:
                        pass
                        #print(s)

                    # Change node color if its name matches a key in s
                    if 'next' in s[list(s.keys())[0]].keys():
                        update_graph(self)
                        plt.pause(0.1)
                        c = []
                        for n in self.display_graph.nodes():
                            if n == "Frontman":
                                n = 'FINISH' 
                            if n == s[list(s.keys())[0]]['sender']:
                                c.append("blue")
                            elif n == s[list(s.keys())[0]]['next']:
                                c.append("red")
                            else:
                                c.append("lightblue")
                        update_graph(self, node_colors=c)
                        plt.pause(0.1)

                    
                graph = load_graph_from_file(self.recipe_graph)
                printer.pprint(graph.get_foundational_recipe())
                i = input("Enter a message: ")

        self.visualize_thread = threading.Thread(target=visualize_graph(self))
        self.interface_thread = threading.Thread(target=simple_interaction_loop(self))
        self.visualize_thread.start()
        self.interface_thread.start()