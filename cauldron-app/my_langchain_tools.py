from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import create_tool_calling_agent, create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

from dotenv import load_dotenv
import os

load_dotenv()
LANGCHAIN_TRACING_V2=True
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY=os.getenv("TAVIL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def Parser():
    return StrOutputParser()

def CreateSQLAgent(llm_model, db, verbose=False):
    assert type(llm_model) == str, "Model must be a string"
    
    llm = ChatOpenAI(model=llm_model, temperature=0)
    return create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=verbose)

def createAgent(llm_model, hubPrompt, tools):

    assert type(llm_model) == str, "Model must be a string"
    assert type(hubPrompt) == str, "Hub prompt must be a string"
    assert type(tools) == list, "Tools must be a list"

    llm = ChatOpenAI(model=llm_model)
    prompt = hub.pull(hubPrompt)
    if tools != []:
        agent = create_tool_calling_agent(llm, tools, prompt)
    else:
        tools = [TavilySearchResults(max_results=1)]
        agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def createChatbot(llm_model, prompt, temperature):
        
    assert type(llm_model) == str, "Model must be a string"
    assert type(prompt) == str, "System instructions must be a string"
    assert type(temperature) == float, "Temperature must be a float"
    
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    prompt = hub.pull(prompt)
    chain = prompt | llm
    history = ChatMessageHistory()
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return chain_with_message_history

def createChain(llm_model, prompt):
    
    assert type(llm_model) == str, "Model must be a string"
    assert type(prompt) == str, "System instructions must be a string"
    
    llm = ChatOpenAI(model=llm_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    return prompt | llm | output_parser

class ChatBot:
    def __init__(self, model, prompt, temperature):
        self.bot = createChatbot(model, prompt, temperature)
        self.parser = Parser()
    
    def chat(self, input):
        return self.parser.invoke(self.bot.invoke({"input": input}, config={"configurable": {"session_id": "unused"}}))
    
class Chain:
    def __init__(self, model, prompt):
        self.chain = createChain(model, prompt)
    
    def invoke(self, input):
        return self.chain.invoke(input)
    
class Agent:
    def __init__(self, model, prompt, tools):
        self.agent = createAgent(model, prompt, tools)
    
    def invoke(self, input):
        return self.agent.invoke(input)

class SQLAgent:
    def __init__(self, model, db_path, verbose=False):
        db = SQLDatabase.from_uri(db_path)
        self.agent = CreateSQLAgent(model, db, verbose=verbose)
    
    def invoke(self, input):
        return self.agent.invoke(input)['output']
    
sqlBot = CreateSQLAgent("gpt-4", "sqlite:///sql/recipes.db")
sqlBot.invoke("How many recipes are there?")