"""Factory for compiling the LangGraph agent chain without CLI/matplotlib coupling."""

import sys
import os

# Add cauldron-app to path so we can import its modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))

from langchain_openai import ChatOpenAI
from langchain_util import workflow
from agent_defs import create_all_agents, prompts_dict, form_edges, create_conditional_edges
from logging_util import logger


def compile_chain(llm_model: str = "gpt-3.5-turbo"):
    """Compile the LangGraph agent chain.

    Extracts the compilation logic from CaldronApp.__init__ without
    threads, matplotlib, or display graph creation.

    Returns:
        A compiled LangGraph chain ready for .stream() or .invoke().
    """
    logger.info("Compiling agent chain.")
    llm = ChatOpenAI(model=llm_model, temperature=0)

    agents = create_all_agents(llm, prompts_dict)

    flow_graph = workflow()
    for node_name, node in agents.items():
        flow_graph.add_node(node_name, node)

    form_edges(flow_graph)
    create_conditional_edges(flow_graph)

    flow_graph.set_entry_point("Caldron\nPostman")
    chain = flow_graph.compile()

    logger.info("Agent chain compiled successfully.")
    return chain
