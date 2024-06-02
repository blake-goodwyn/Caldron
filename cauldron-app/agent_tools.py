from langchain_core.tools import tool
from typing import Dict, List, Optional, Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
from recipe_scrapers import scrape_me
from logging_util import logger

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_search_tool = TavilySearchResults()

@tool
def get_recipe_info(
    url: Annotated[str, "The URL of the recipe to scrape."]
) -> Annotated[Dict[str, Optional[List[str]]], "A dictionary containing the recipe's name, ingredients, instructions, and tags."]:
    """
    Fetches recipe information from a given URL.

    Args:
        url (str): The URL of the recipe to scrape.

    Returns:
        dict: A dictionary containing the recipe's name, ingredients, instructions, and tags.
    """
    scraper = scrape_me(url, wild_mode=True)
    out: Dict[str, Optional[List[str]]] = {}

    try:
        ing: List[str] = scraper.ingredients()
        out["ingredients"] = ing
    except Exception as e:
        logger.error(f"Failed to get ingredients: {e}")

    try:
        inst: List[str] = scraper.instructions_list()
        out["instructions"] = inst
    except Exception as e:
        logger.error(f"Failed to get instructions: {e}")

    try:
        name: str = scraper.title()
        out["name"] = name
    except Exception as e:
        logger.error(f"Failed to get name: {e}")

    return out