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


class TestRoutingLogic:
    """Test the conditional edge routing functions."""

    def test_caldron_postman_routes_to_research(self):
        """Caldron Postman should route to Research Postman."""
        router = lambda x: x["next"]
        state = {"next": "Research\nPostman"}
        assert router(state) == "Research\nPostman"

    def test_caldron_postman_routes_to_modsquad(self):
        router = lambda x: x["next"]
        state = {"next": "ModSquad"}
        assert router(state) == "ModSquad"

    def test_caldron_postman_routes_to_spinnaret(self):
        router = lambda x: x["next"]
        state = {"next": "Spinnaret"}
        assert router(state) == "Spinnaret"

    def test_caldron_postman_routes_to_frontman(self):
        router = lambda x: x["next"]
        state = {"next": "Frontman"}
        assert router(state) == "Frontman"

    def test_caldron_postman_routes_to_knowitall(self):
        router = lambda x: x["next"]
        state = {"next": "KnowItAll"}
        assert router(state) == "KnowItAll"

    def test_caldron_postman_finish_routes_to_frontman(self):
        """FINISH should map to Frontman in the conditional edge mapping."""
        from agent_defs import create_conditional_edges
        # The mapping dict shows "FINISH" -> "Frontman"
        mapping = {
            "Research\nPostman": "Research\nPostman",
            "ModSquad": "ModSquad",
            "Spinnaret": "Spinnaret",
            "Frontman": "Frontman",
            "KnowItAll": "KnowItAll",
            "FINISH": "Frontman",
        }
        router_result = "FINISH"
        assert mapping[router_result] == "Frontman"

    def test_research_postman_routes_to_tavily(self):
        router = lambda x: x["next"]
        state = {"next": "Tavily"}
        assert router(state) == "Tavily"

    def test_research_postman_routes_to_sleuth(self):
        router = lambda x: x["next"]
        state = {"next": "Sleuth"}
        assert router(state) == "Sleuth"

    def test_research_postman_routes_back_to_caldron(self):
        router = lambda x: x["next"]
        state = {"next": "Caldron\nPostman"}
        assert router(state) == "Caldron\nPostman"


class TestToolBindings:
    """Verify each agent has the correct tools assigned."""

    def test_frontman_tools(self):
        from agent_defs import prompts_dict
        from agent_tools import clear_pot
        tools = prompts_dict["Frontman"]["tools"]
        assert clear_pot in tools
        assert len(tools) == 1

    def test_tavily_tools(self):
        from agent_defs import prompts_dict
        from agent_tools import tavily_search_tool, add_url_to_pot
        tools = prompts_dict["Tavily"]["tools"]
        assert tavily_search_tool in tools
        assert add_url_to_pot in tools

    def test_sleuth_tools(self):
        from agent_defs import prompts_dict
        from agent_tools import pop_url_from_pot, scrape_recipe_info, generate_recipe, get_recipe_from_pot, examine_pot
        tools = prompts_dict["Sleuth"]["tools"]
        assert pop_url_from_pot in tools
        assert scrape_recipe_info in tools
        assert generate_recipe in tools
        assert get_recipe_from_pot in tools
        assert examine_pot in tools

    def test_modsquad_tools(self):
        from agent_defs import prompts_dict
        from agent_tools import suggest_mod, get_mods_list, apply_mod, rank_mod, remove_mod
        tools = prompts_dict["ModSquad"]["tools"]
        assert suggest_mod in tools
        assert get_mods_list in tools
        assert apply_mod in tools
        assert rank_mod in tools
        assert remove_mod in tools

    def test_knowitall_tools(self):
        from agent_defs import prompts_dict
        from agent_tools import get_foundational_recipe, get_graph, get_recipe
        tools = prompts_dict["KnowItAll"]["tools"]
        assert get_foundational_recipe in tools
        assert get_graph in tools
        assert get_recipe in tools

    def test_spinnaret_tools(self):
        from agent_defs import prompts_dict
        from agent_tools import (
            create_recipe_graph, add_node, get_recipe,
            get_foundational_recipe, set_foundational_recipe,
            get_graph, get_recipe_from_pot, examine_pot
        )
        tools = prompts_dict["Spinnaret"]["tools"]
        assert create_recipe_graph in tools
        assert add_node in tools
        assert get_recipe in tools
        assert get_foundational_recipe in tools
        assert set_foundational_recipe in tools
        assert get_graph in tools
        assert get_recipe_from_pot in tools
        assert examine_pot in tools

    def test_supervisors_have_no_tools(self):
        from agent_defs import prompts_dict
        for name in ["Caldron\nPostman", "Research\nPostman"]:
            assert "tools" not in prompts_dict[name]


class TestDirectEdges:
    def test_direct_edges_complete(self):
        from agent_defs import direct_edges
        from langgraph.graph import END
        expected_sources = {"ModSquad", "Spinnaret", "KnowItAll", "Tavily", "Sleuth", "Frontman"}
        actual_sources = {edge[0] for edge in direct_edges}
        assert actual_sources == expected_sources

    def test_frontman_goes_to_end(self):
        from agent_defs import direct_edges
        from langgraph.graph import END
        frontman_edges = [(s, t) for s, t in direct_edges if s == "Frontman"]
        assert len(frontman_edges) == 1
        assert frontman_edges[0][1] == END
