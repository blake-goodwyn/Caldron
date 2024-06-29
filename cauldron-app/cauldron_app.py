### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
import warnings
from logging_util import logger
from langchain_util import ChatOpenAI, workflow, HumanMessage
from class_defs import fresh_pot, fresh_graph, fresh_mods_list
from agent_defs import create_all_agents, prompts_dict, form_edges
from neopixel_util import highlight_section

warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CaldronApp():
    
    def __init__(self, llm_model, db_path=None, defs=prompts_dict, verbose=False):
        
        logger.info("Initializing Caldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        #Central Data Structures
        self.recipe_pot_file = fresh_pot()
        self.recipe_graph_file = fresh_graph()
        self.mods_list_file = fresh_mods_list()
        self.printer_wait_flag = False

        ##Determine Agent Structure
        self.agents = create_all_agents(self.llm, defs)

        # Define the control flow
        self.flow_graph = workflow()
        for node_name, node in self.agents.items():
            self.flow_graph.add_node(node_name, node)

        form_edges(self.flow_graph)

        self.flow_graph.set_entry_point("Tavily")
        self.chain = self.flow_graph.compile()

    def clear_pot(self):
        self.recipe_pot_file = fresh_pot()

        ## Simple Interaction Thread
    def post(self, i: str):
        self.printer_wait_flag = True
        for s in self.chain.stream(
            {
                "messages": [HumanMessage(content=i)],
                'sender': 'User',
                'next': 'Caldron\nPostman'
            },
            {"recursion_limit": 50}
        ):
            logger.info(s)
            try:
                if 'Tavily' in s.keys():
                        highlight_section('Sleuth')
                elif 'Sleuth' in s.keys():
                        highlight_section('Spinnaret')
                elif 'Spinnaret' in s.keys():
                        highlight_section('Frontman')
            except Exception as e:
                logger.error(e)
                        
            if 'Frontman' in s.keys():
                self.printer_wait_flag = False
        
        self.printer_wait_flag = False
        return None
