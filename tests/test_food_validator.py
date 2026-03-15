"""Unit tests for food_validator.py — food validation router."""

from unittest.mock import MagicMock


class TestCreateFoodValidator:
    def test_returns_chain(self):
        from food_validator import createFoodValidator
        llm = MagicMock()
        bound = MagicMock()
        llm.bind_tools.return_value = bound
        bound.__or__ = MagicMock(return_value=bound)

        result = createFoodValidator(
            "Glutton",
            "You determine if something is food.",
            llm
        )
        # Should return a chain (the result of prompt | llm | parser)
        assert result is not None
        # LLM should have been bound with the discernFood tool
        llm.bind_tools.assert_called_once()
        call_kwargs = llm.bind_tools.call_args
        tools = call_kwargs[1].get("tools") or call_kwargs[0][0]
        tool_names = [t["name"] for t in tools]
        assert "discernFood" in tool_names

    def test_conditional_edges_defined(self):
        from food_validator import food_validator_conditional_edges
        assert "TRUE" in food_validator_conditional_edges
        assert "FALSE" in food_validator_conditional_edges
        assert food_validator_conditional_edges["TRUE"] == "Caldron\nPostman"
        assert food_validator_conditional_edges["FALSE"] == "Frontman"
