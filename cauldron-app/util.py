from langchain_util import *
from pydantic_util import *

#Parameter for chains & agents
db_path = "sqlite:///sql/recipes_0514_1658_views.db"
llm_model = "gpt-4"