from langchain_util import *
from pydantic_util import *

#Parameter for chains & agents
db_path = "sqlite:///sql/recipes_0514_1821.db"
llm_model = "gpt-3.5-turbo"