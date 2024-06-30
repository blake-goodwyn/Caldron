### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
import warnings
from logging_util import logger
from langchain_util import ChatOpenAI, workflow, HumanMessage, finalCheck
from class_defs import fresh_pot, fresh_graph, fresh_mods_list, load_graph_from_file, load_pot_from_file
from agent_defs import create_all_agents, prompts_dict, form_edges, create_conditional_edges
from custom_print import printer as pretty
from custom_print import wrapper
#from neopixel_util import highlight_section
#from thermal_printer_util import printer

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
        create_conditional_edges(self.flow_graph)

        self.flow_graph.set_entry_point("Greeter")
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
            #logger.info(s)
            try:
                if 'Greeter' in s.keys():
                    print()
                    logger.info(f"Greeter: {s['Greeter']['messages'][0].content}")
                    print(s['Greeter']['messages'][0].content)
                    #printer.print(wrapper.fill(s['Greeter']['messages'][0].content))
                    print()
                elif 'Tavily' in s.keys():
                    highlight_section('Sleuth')
                elif 'Sleuth' in s.keys():
                    highlight_section('Spinnaret')
                elif 'Spinnaret' in s.keys():
                    highlight_section('Frontman')
            except Exception as e:
                logger.error(e)
                        
            if 'Frontman' in s.keys():

                recipe_graph = load_graph_from_file(self.recipe_graph_file)
                recipe = recipe_graph.get_foundational_recipe()
                if (recipe != None):
                    pot = load_pot_from_file(self.recipe_pot_file)
                    recipe = pot.pop_recipe()
                    if (recipe != None):
                        try:
                            out = finalCheck(pretty.pformat(recipe))
                            logger.info(f"Recipe Text:\n\n {out}")
                            print(wrapper.fill(out))
                            #printer.print(wrapper.fill(out))
                        except Exception as e:
                            logger.error(e)
                    else:
                        logger.info("Sorry, I couldn't find an appropriate recipe for what you were looking for.")
                        print(wrapper.fill("Sorry, I couldn't find an appropriate recipe for what you were looking for."))
                        #printer.print(wrapper.fill(out))
                else:
                    try:
                        out = finalCheck(pretty.pformat(recipe))
                        logger.info(f"Recipe Text:\n\n {out}")
                        print(wrapper.fill(out))
                        #printer.print(wrapper.fill(out))
                    except Exception as e:
                        logger.error(e)
                
                self.printer_wait_flag = False
        
        self.printer_wait_flag = False
        return None
