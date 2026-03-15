"""Tests for api/chain_factory.py — chain compilation without CLI coupling."""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Add api/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock ChatOpenAI to avoid needing real API keys."""
    mock = MagicMock()
    bound = MagicMock()
    mock.return_value = mock
    mock.bind_functions.return_value = bound
    mock.bind_tools.return_value = bound
    bound.__or__ = MagicMock(return_value=bound)

    import langchain_openai
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: mock)
    return mock


class TestCompileChain:
    def test_returns_compiled_chain(self, mock_llm):
        from chain_factory import compile_chain
        chain = compile_chain("gpt-3.5-turbo")
        assert chain is not None

    def test_chain_has_expected_nodes(self, mock_llm):
        from chain_factory import compile_chain
        from agent_defs import prompts_dict
        chain = compile_chain("gpt-3.5-turbo")
        # The compiled chain should have nodes for all agents in prompts_dict
        graph = chain.get_graph()
        node_ids = set()
        for node in graph.nodes:
            node_ids.add(node.id if hasattr(node, 'id') else str(node))
        for agent_name in prompts_dict:
            assert agent_name in node_ids, f"Missing agent node: {agent_name}"

    def test_no_threads_started(self, mock_llm):
        """compile_chain should not start any threads."""
        import threading
        thread_count_before = threading.active_count()
        from chain_factory import compile_chain
        compile_chain("gpt-3.5-turbo")
        thread_count_after = threading.active_count()
        assert thread_count_after == thread_count_before
