"""Integration tests for agent graph compilation."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_llm():
    """Create a mock ChatOpenAI that supports bind_functions."""
    llm = MagicMock()
    # bind_functions returns a chain-able mock
    bound = MagicMock()
    llm.bind_functions.return_value = bound
    bound.__or__ = MagicMock(return_value=bound)
    return llm


class TestAgentCreation:
    def test_create_all_agents_returns_all_names(self, mock_llm):
        from agent_defs import create_all_agents, prompts_dict
        agents = create_all_agents(mock_llm, prompts_dict)
        for name in prompts_dict:
            assert name in agents, f"Missing agent: {name}"

    def test_agent_nodes_are_callable(self, mock_llm):
        from agent_defs import create_all_agents, prompts_dict
        agents = create_all_agents(mock_llm, prompts_dict)
        for name, node_fn in agents.items():
            assert callable(node_fn), f"Agent {name} node is not callable"


class TestGraphStructure:
    def test_form_edges_adds_direct_edges(self, mock_llm):
        from agent_defs import create_all_agents, prompts_dict, form_edges, direct_edges
        from langchain_util import workflow
        agents = create_all_agents(mock_llm, prompts_dict)
        graph = workflow()
        for name, node_fn in agents.items():
            graph.add_node(name, node_fn)
        result = form_edges(graph)
        assert result == direct_edges

    def test_conditional_edges_added(self, mock_llm):
        from agent_defs import (
            create_all_agents, prompts_dict, form_edges,
            create_conditional_edges, conditional_edges
        )
        from langchain_util import workflow
        agents = create_all_agents(mock_llm, prompts_dict)
        graph = workflow()
        for name, node_fn in agents.items():
            graph.add_node(name, node_fn)
        form_edges(graph)
        result = create_conditional_edges(graph)
        assert result == conditional_edges

    def test_graph_compiles(self, mock_llm):
        """Full integration: create agents, add nodes, add edges, compile."""
        from agent_defs import (
            create_all_agents, prompts_dict, form_edges,
            create_conditional_edges
        )
        from langchain_util import workflow
        agents = create_all_agents(mock_llm, prompts_dict)
        graph = workflow()
        for name, node_fn in agents.items():
            graph.add_node(name, node_fn)
        form_edges(graph)
        create_conditional_edges(graph)
        graph.set_entry_point("Caldron\nPostman")
        compiled = graph.compile()
        assert compiled is not None


class TestAgentState:
    def test_enter_chain(self):
        from langchain_util import enter_chain
        result = enter_chain("Make me a cake")
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Make me a cake"

    def test_workflow_creates_state_graph(self):
        from langchain_util import workflow
        graph = workflow()
        assert graph is not None
