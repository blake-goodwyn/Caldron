### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
from util import SQLAgent, Conductor, db_path, llm_model
import warnings
from logging_util import logger
from agent_defs import prompts_dict


warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CauldronApp():
    
    def __init__(self, db_path, llm_model, verbose=False):
        
        logger.info("Initializing Cauldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = llm_model

        # UI Agent #
        ui_prompt = prompts_dict['ConductorAgent']
        ui_temp = 0.0
        self.UIAgent = Conductor(llm_model, ui_prompt, ui_temp)
        logger.info("UI Agent Initialized")

         # SQL Retrieval Agent (Boilerplate) #
        self.SQLAgent = SQLAgent(llm_model, db_path, verbose=verbose)
        logger.info("SQL Agent Initialized")

    def invoke(self, input):
        logger.info(f"Invoking Cauldron with input: {input}")
        return self.UIAgent.chat(input)

app = CauldronApp(db_path, llm_model)
print(app.invoke("I want to find a recipe for bread that uses xanthan gum."))