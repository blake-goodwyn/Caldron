### TEST FILE FOR CAULDRON PROOF-OF-CONCEPT ###

### Overview ###
#
# Cauldron is an AI-assisted recipe development platform that allows users to generate and quickly iterate on new recipes. 
# Cauldron leverages multi-agent generative artificial intelligence tools to generate a desired foundational recipe from a provided prompt and optionally provided context. Cauldron then provides channels for both human and machine sensory feedback to iteratively refine the recipe.

# The system works as follows:
# 1. User provides a prompt and optionally context (i.e. custom ingredients, dietary restrictions, recipe sources, etc.)
# 2. The system (Cauldron) parses the request and determines what type of SQL query to execute on a local database of recipes, ingredients, and processes
# 3. Cauldron executes the query and returns a list of relevant information
# 4. Cauldron formats the returned information into a context for recipe generation
# 5. Cauldron generates a foundational recipe based on the context and prompt.
# 6. Cauldron provides the foundational recipe to the user for feedback
# 7. The user provides feedback on the recipe and the foundational recipe is tweaked until the user is satisfied
# 8. Cauldron provides interaction points for human feedback (in the form of written and spoken language) and machine feedback (in the form of sensory data provided by IoT devices)
# 9. The user is free to cook/bake the recipe and provide feedback on the final product
# 10. Cauldron intelligently uses this feedback to further refine the recipe as well as save a "snapshot" of the recipe attempt for future reference

### IMPORTS ###
import os
from util import SQLAgent, Chain, ChatBot, find_SQL_prompt

### DEFINITIONS ###

class CauldronApp():
    
    def __init__(self, db_path, llm_model, assistant_prompt, assistant_temperature, verbose=False):
        self.db = db_path
        self.llm = llm_model
        self.SQLAgent = SQLAgent(llm_model, db_path)
        self.Assistant = ChatBot(llm_model, assistant_prompt, assistant_temperature)

### MAIN ###

db_path = "sqlite:///sql/recipes.db"
llm_model = "gpt-4"
assistant_prompt = "blake-goodwyn/cauldron-assistant-v0"
assistant_temperature = 0.0

app = CauldronApp(db_path, llm_model, assistant_prompt, assistant_temperature, verbose=True)

flag = True
os.system('cls')
while flag:
    user_input = input(">>> ")
    if user_input == "exit":
        flag = False
    else:
        response = app.Assistant.chat(user_input)
        sqlCommand, response = find_SQL_prompt(response)
        print(response)
        print("<SQL Command: ", sqlCommand, ">")
        if sqlCommand != "":
            print("<SQL Retrieval> ", end="")
            sqlResponse = app.SQLAgent.invoke(sqlCommand)
        