### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
import warnings
from logging_util import logger
from langchain_util import ChatOpenAI, workflow, enter_chain, HumanMessage
from class_defs import fresh_graph, fresh_mods_list, load_graph_from_file, fresh_pot, default_pot_file, load_pot_from_file
from agent_defs import create_all_agents, prompts_dict, form_edges, create_conditional_edges
from custom_print import printer
import matplotlib.pyplot as plt
import networkx as nx
import threading

warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CaldronApp():
    
    def __init__(self, db_path, llm_model, defs=prompts_dict, verbose=False):
        
        logger.info("Initializing Caldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = ChatOpenAI(model=llm_model, temperature=0)

        #Central Data Structures
        self.recipe_pot_file = fresh_pot()
        self.recipe_graph_file = fresh_graph()
        self.mods_list_file = fresh_mods_list()

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

        self.flow_graph.set_entry_point("Caldron\nPostman")
        self.chain = self.flow_graph.compile()

        labeldict = {"__end__": "FINISH"}
        for node_name, node in prompts_dict.items():
            labeldict[node_name] = node['label']

        ## Update Visualization
        def update_graph(self, node_colors=["#457b9d" for n in self.display_graph.nodes()]):
            plt.clf()
            #plt.legend(["Sender", "Receiver"], loc="upper left")
            nx.draw_networkx_nodes(self.display_graph, self.node_pos, node_size=3000, node_color=node_colors)
            nx.draw_networkx_edges(self.display_graph, self.node_pos, edgelist=direct_edges, style='solid', connectionstyle='arc3,rad=0.2', arrows=True)
            nx.draw_networkx_edges(self.display_graph, self.node_pos, edgelist=conditional_edges, style='dotted', connectionstyle='arc3,rad=0.2', arrows=True)
            nx.draw_networkx_labels(self.display_graph, self.node_pos, labels=labeldict, font_size=10, font_weight="bold")
            plt.draw()

        ## Visualization Thread
        def visualize_graph(self):
            plt.ion()

            # Get positions for the nodes
            self.node_pos = nx.shell_layout(self.display_graph)
            update_graph(self)

        ## Simple Interaction Thread
        def simple_interaction_loop(self):
            i = input("Enter a message: ")
            msq_queue = []
            msq_queue.append(HumanMessage(content=i))
            while i != "exit":
                for s in self.chain.stream(
                    {
                        "messages": [msq_queue.pop()],
                        'sender': 'User',
                        'next': 'Caldron\nPostman'
                    },
                    {"recursion_limit": 50}
                ):
                    print("--------------------")
                    if 'Frontman' in s.keys():
                        print("\n")
                        print(s['Frontman']['messages'][0].content)
                        print("\n")
                    else:
                        print(s)
                        #pot = load_pot_from_file(self.recipe_pot_file)
                        #print(pot.get_all_recipes())

                    # Change node color if its name matches a key in s
                    if 'next' in s[list(s.keys())[0]].keys():
                        update_graph(self)
                        plt.pause(0.1)
                        c = []
                        for n in self.display_graph.nodes():
                            if n == "Frontman":
                                n = 'FINISH' 
                            if n == s[list(s.keys())[0]]['sender']:
                                c.append("#a8dadc")
                            elif n == s[list(s.keys())[0]]['next']:
                                c.append("#e63946")
                            else:
                                c.append("#457b9d")
                        update_graph(self, node_colors=c)
                        plt.pause(0.1)
                    
                graph = load_graph_from_file(self.recipe_graph_file)
                printer.pprint(graph.get_foundational_recipe())
                update_graph(self)
                i = input("Enter a message: ")
                msq_queue.append(HumanMessage(content=i))

        ## Start Threads
        self.visualize_thread = threading.Thread(target=visualize_graph(self))
        self.interface_thread = threading.Thread(target=simple_interaction_loop(self))
        self.visualize_thread.start()
        self.interface_thread.start()