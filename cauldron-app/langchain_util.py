# langchain_util.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
from logging_util import logger

load_dotenv()
LANGCHAIN_TRACING_V2=True
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY=os.getenv("TAVIL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def Parser():
    logger.debug("Creating StrOutputParser instance.")
    return StrOutputParser()

def CreateSQLAgent(llm_model, db, verbose=False):
    logger.info(f"Creating SQL Agent with model: {llm_model} and database: {db}")
    try:
        assert type(llm_model) == str, "Model must be a string"
        llm = ChatOpenAI(model=llm_model, temperature=0)
        agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=verbose)
        logger.info("SQL Agent created successfully.")
        return agent
    except AssertionError as e:
        logger.error(f"Error in CreateSQLAgent: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in CreateSQLAgent: {e}", exc_info=True)
        raise

def createAgent(llm_model, hubPrompt, tools):
    logger.info(f"Creating agent with model: {llm_model}, hubPrompt: {hubPrompt}, tools: {tools}")
    try:
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
        logger.info("Agent created successfully.")
        return AgentExecutor(agent=agent, tools=tools)
    except AssertionError as e:
        logger.error(f"Error in createAgent: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in createAgent: {e}", exc_info=True)
        raise

def createChatbot(llm_model, prompt, temperature):
    logger.info(f"Creating chatbot with model: {llm_model}, prompt: {prompt}, temperature: {temperature}")
    try:
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
        logger.info("Chatbot created successfully.")
        return chain_with_message_history
    except AssertionError as e:
        logger.error(f"Error in createChatbot: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in createChatbot: {e}", exc_info=True)
        raise

def createChain(llm_model, prompt):
    logger.info(f"Creating chain with model: {llm_model}, prompt: {prompt}")
    try:
        assert type(llm_model) == str, "Model must be a string"
        assert type(prompt) == str, "System instructions must be a string"
        
        llm = ChatOpenAI(model=llm_model)
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("user", "{input}")
        ])
        output_parser = StrOutputParser()
        logger.info("Chain created successfully.")
        return prompt | llm | output_parser
    except AssertionError as e:
        logger.error(f"Error in createChain: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in createChain: {e}", exc_info=True)
        raise

class ChatBot:
    def __init__(self, model, prompt, temperature):
        logger.info(f"Initializing ChatBot with model: {model}, prompt: {prompt}, temperature: {temperature}")
        self.bot = createChatbot(model, prompt, temperature)
        self.parser = Parser()
    
    def chat(self, input):
        logger.debug(f"ChatBot chat invoked with input: {input}")
        return self.parser.invoke(self.bot.invoke({"input": input}, config={"configurable": {"session_id": "unused"}}))
    
    def stream(self, input, callback, on_complete):
        logger.debug(f"ChatBot stream invoked with input: {input}")
        for chunk in self.bot.stream({"input": input}, config={"configurable": {"session_id": "unused"}}):
            callback(chunk.content)
        on_complete()  # Call the on_complete function after the stream is done
        return

class Chain:
    def __init__(self, model, prompt):
        logger.info(f"Initializing Chain with model: {model}, prompt: {prompt}")
        self.chain = createChain(model, prompt)
    
    def invoke(self, input):
        logger.debug(f"Chain invoke called with input: {input}")
        return self.chain.invoke(input)

class Agent:
    def __init__(self, model, prompt, tools):
        logger.info(f"Initializing Agent with model: {model}, prompt: {prompt}, tools: {tools}")
        self.agent = createAgent(model, prompt, tools)
    
    def invoke(self, input):
        logger.debug(f"Agent invoke called with input: {input}")
        return self.agent.invoke(input)

class SQLAgent:
    def __init__(self, model, db_path, verbose=False):
        logger.info(f"Initializing SQLAgent with model: {model}, db_path: {db_path}, verbose: {verbose}")
        try:
            db = SQLDatabase.from_uri(db_path)
            self.agent = CreateSQLAgent(model, db, verbose=verbose)
            logger.info("SQLAgent initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing SQLAgent: {e}", exc_info=True)
            raise
    
    def invoke(self, input):
        logger.debug(f"SQLAgent invoke called with input: {input}")
        return self.agent.invoke(input)['output']
    
    def stream(self, input, callback):
        logger.debug(f"SQLAgent stream called with input: {input}")
        # This method should use the real streaming functionality of your bot.
        # Assuming a streaming method `invoke_stream` exists similar to the given example.
        for chunk in self.agent.stream(input):
            print(chunk)
            print("------")
            callback(chunk)
        return
