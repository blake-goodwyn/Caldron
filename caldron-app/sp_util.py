import spoonacular as sp
import os
from dotenv import load_dotenv

load_dotenv()
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")
spoon = sp.API(spoonacular_api_key)