from datetime import datetime
from langchain.tools import Tool
import random
import string
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define a new tool that returns the current datetime
datetime_tool = Tool(
    name="Datetime",
    func=lambda x: datetime.now().isoformat(),
    description="Returns the current datetime",
)

tavily_search_tool = TavilySearchResults()