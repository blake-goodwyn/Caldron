"""Tests for api/server.py — FastAPI endpoints."""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))

from fastapi.testclient import TestClient


@pytest.fixture
def mock_chain():
    """Mock the compiled chain to avoid LLM calls."""
    mock = MagicMock()
    mock.stream.return_value = iter([
        {
            "Caldron\nPostman": {
                "sender": "Caldron\nPostman",
                "next": "Research\nPostman",
            }
        },
        {
            "Frontman": {
                "messages": [MagicMock(content="Here is your recipe!")]
            }
        },
    ])
    return mock


@pytest.fixture
def client(mock_chain, tmp_path):
    """Create a test client with mocked chain and temp session dir."""
    with patch("server.compile_chain", return_value=mock_chain), \
         patch("server.SessionManager") as MockSessionMgr:
        # Set up mock session manager
        mgr_instance = MagicMock()
        MockSessionMgr.return_value = mgr_instance

        session_dir = str(tmp_path / "test-session")
        os.makedirs(session_dir, exist_ok=True)

        # Write fresh state files
        from class_defs import fresh_pot, fresh_graph, fresh_mods_list
        fresh_pot(os.path.join(session_dir, "recipe_pot.json"))
        fresh_graph(os.path.join(session_dir, "recipe_graph.json"))
        fresh_mods_list(os.path.join(session_dir, "mods_list.json"))

        mgr_instance.get_session_dir.return_value = session_dir
        mgr_instance.session_scope.return_value.__enter__ = MagicMock(return_value=session_dir)
        mgr_instance.session_scope.return_value.__exit__ = MagicMock(return_value=False)

        # Import server after patching
        import importlib
        import server
        importlib.reload(server)

        yield TestClient(server.app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestWebSocketEndpoint:
    def test_websocket_connect_and_send(self, client, mock_chain):
        with client.websocket_connect("/ws/test-session") as ws:
            ws.send_text(json.dumps({
                "type": "user_message",
                "content": "Make me a cake"
            }))
            # Should receive at least one message back
            response = ws.receive_text()
            data = json.loads(response)
            assert data["type"] in ("agent_event", "agent_response", "error")

    def test_websocket_invalid_json(self, client):
        with client.websocket_connect("/ws/test-session") as ws:
            ws.send_text("not valid json")
            response = ws.receive_text()
            data = json.loads(response)
            assert data["type"] == "error"
            assert "Invalid JSON" in data["detail"]

    def test_websocket_unknown_message_type(self, client):
        with client.websocket_connect("/ws/test-session") as ws:
            ws.send_text(json.dumps({"type": "unknown", "data": "test"}))
            response = ws.receive_text()
            data = json.loads(response)
            assert data["type"] == "error"
            assert "Unknown message type" in data["detail"]
