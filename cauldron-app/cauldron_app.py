### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### IMPORTS ###
from util import SQLAgent, ChatBot
import warnings
from logging_util import logger
import tkinter as tk
from interactive_demo import InteractiveDemo

warnings.filterwarnings("ignore", message="Parent run .* not found for run .* Treating as a root run.")

### DEFINITIONS ###

class CauldronApp():
    
    def __init__(self, db_path, llm_model, verbose=False):
        
        logger.info("Initializing Cauldron Application")

        #Pathways and Parameters
        self.db = db_path
        self.llm = llm_model

        ## Defining Agents ##
        
        # SQL Retrieval Agent (Boilerplate) #
        self.SQLAgent = SQLAgent(llm_model, db_path, verbose=verbose)
        logger.info("SQL Agent Initialized")

        # UI Agent #
        ui_prompt = "blake-goodwyn/cauldron-assistant-v0"
        ui_temp = 0.0
        self.UIAgent = ChatBot(llm_model, ui_prompt, ui_temp)
        logger.info("UI Agent Initialized")

        # Question Generation Agent #
        # TODO: Add question generation agent

        # Insight Consolidation Agent #
        # TODO: Add insight consolidation agent

        # Foundational Recipe Steward Agent #
        # TODO: Add foundational recipe steward agent

        # Peripheral Feedback Agent #
        # TODO: Add peripheral interpretation agent

        # "Sous Chef" Agent #
        # TODO: Add agent for in-media-res cooking feedback integration

        # TODO: Determine additional agents for fitness evaluation:
        # > flavor profile analysis
        # > nutritional analysis
        # > culinary trend analysis
        # > ingredient sourcing analysis
        # > recipe cost analysis
        # > sensory appeal analysis

### MAIN ###

#Parameter for chains & agents
db_path = "sqlite:///sql/recipes_0514_1821.db"
llm_model = "gpt-4"

def demo():
    logger.info(">> Cauldron Functional Demo <<")
    root = tk.Tk()
    CauldronDemo = InteractiveDemo(root, CauldronApp(db_path, llm_model, verbose=True))
    logger.info("Cauldron Functional Demo Initialized")
    root.protocol("WM_DELETE_WINDOW", CauldronDemo.on_close)  # Bind the close event
    root.mainloop()

if __name__ == "__main__":
    demo()