from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase

def sqlTools(db_path, llm_model):
    db = SQLDatabase.from_uri(db_path)
    llm = ChatOpenAI(model=llm_model, temperature=0)
    toolkit = SQLDatabaseToolkit(llm=llm, db=db)
    return toolkit.get_tools()