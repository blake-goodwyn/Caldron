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

        self.direct_edges = form_edges(self.flow_graph)
        for edge in self.direct_edges:
            self.display_graph.add_edge(*edge)
        self.conditional_edges = create_conditional_edges(self.flow_graph)

        self.flow_graph.set_entry_point("Caldron\nPostman")
        self.chain = self.flow_graph.compile()

        self.labeldict = {"__end__": "USER"}
        for node_name, node in prompts_dict.items():
            self.labeldict[node_name] = node["label"]

        ## Start Threads
        self.visualize_thread = threading.Thread(target=self._visualize_graph)
        self.interface_thread = threading.Thread(target=self._simple_interaction_loop)
        self.visualize_thread.start()
        self.interface_thread.start()

    def _update_graph(self, node_colors=None):
        if node_colors is None:
            node_colors = ["#457b9d" for _ in self.display_graph.nodes()]
        plt.clf()
        nx.draw_networkx_nodes(self.display_graph, self.node_pos, node_size=3000, node_color=node_colors)
        nx.draw_networkx_edges(self.display_graph, self.node_pos, edgelist=self.direct_edges, style='solid', connectionstyle='arc3,rad=0.2', arrows=True)
        nx.draw_networkx_edges(self.display_graph, self.node_pos, edgelist=self.conditional_edges, style='dotted', connectionstyle='arc3,rad=0.2', arrows=True)
        nx.draw_networkx_labels(self.display_graph, self.node_pos, labels=self.labeldict, font_size=10, font_weight="bold")
        plt.draw()

    def _visualize_graph(self):
        plt.ion()
        self.node_pos = nx.shell_layout(self.display_graph)
        self._update_graph()

    def _simple_interaction_loop(self):
        i = input("Enter a message: ")
        msg_queue = []
        msg_queue.append(HumanMessage(content=i))
        while i != "exit":
            for s in self.chain.stream(
                {
                    "messages": [msg_queue.pop()],
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

                # Change node color if its name matches a key in s
                first_key = list(s.keys())[0]
                if 'next' in s[first_key].keys():
                    self._update_graph()
                    plt.pause(0.1)
                    c = []
                    for n in self.display_graph.nodes():
                        node_name = 'FINISH' if n == "Frontman" else n
                        if node_name == s[first_key]['sender']:
                            c.append("#a8dadc")
                        elif node_name == s[first_key]['next']:
                            c.append("#e63946")
                        else:
                            c.append("#457b9d")
                    self._update_graph(node_colors=c)
                    plt.pause(0.1)

            graph = load_graph_from_file(self.recipe_graph_file)
            printer.pprint(graph.get_foundational_recipe())
            self._update_graph()
            i = input("Enter a message: ")
            msg_queue.append(HumanMessage(content=i))
