"""Tests for api/session.py — per-session state isolation."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))


class TestSessionManager:
    def test_create_session(self, tmp_path):
        from session import SessionManager
        mgr = SessionManager(sessions_dir=str(tmp_path))
        session_id = mgr.create_session()
        session_dir = mgr.get_session_dir(session_id)

        assert os.path.isdir(session_dir)
        assert os.path.exists(os.path.join(session_dir, "recipe_graph.json"))
        assert os.path.exists(os.path.join(session_dir, "mods_list.json"))
        assert os.path.exists(os.path.join(session_dir, "recipe_pot.json"))

    def test_create_session_with_id(self, tmp_path):
        from session import SessionManager
        mgr = SessionManager(sessions_dir=str(tmp_path))
        session_id = mgr.create_session("test-session-123")
        assert session_id == "test-session-123"
        assert os.path.isdir(os.path.join(str(tmp_path), "test-session-123"))

    def test_get_session_not_found(self, tmp_path):
        from session import SessionManager
        mgr = SessionManager(sessions_dir=str(tmp_path))
        with pytest.raises(KeyError, match="not found"):
            mgr.get_session_dir("nonexistent")

    def test_remove_session(self, tmp_path):
        from session import SessionManager
        mgr = SessionManager(sessions_dir=str(tmp_path))
        session_id = mgr.create_session()
        session_dir = mgr.get_session_dir(session_id)
        assert os.path.isdir(session_dir)

        mgr.remove_session(session_id)
        assert not os.path.isdir(session_dir)

    def test_session_scope_sets_contextvars(self, tmp_path):
        from session import SessionManager
        from agent_tools import _graph_file, _mods_file, _pot_file

        mgr = SessionManager(sessions_dir=str(tmp_path))
        session_id = mgr.create_session()
        session_dir = mgr.get_session_dir(session_id)

        with mgr.session_scope(session_id):
            assert _graph_file.get() == os.path.join(session_dir, "recipe_graph.json")
            assert _mods_file.get() == os.path.join(session_dir, "mods_list.json")
            assert _pot_file.get() == os.path.join(session_dir, "recipe_pot.json")

        # After exiting scope, ContextVars should be reset
        from class_defs import default_graph_file, default_mods_list_file, default_pot_file
        assert _graph_file.get() == default_graph_file
        assert _mods_file.get() == default_mods_list_file
        assert _pot_file.get() == default_pot_file

    def test_two_sessions_isolated(self, tmp_path):
        """Two sessions should have independent state directories."""
        from session import SessionManager
        from class_defs import load_pot_from_file, save_pot_to_file, Pot, Recipe, Ingredient

        mgr = SessionManager(sessions_dir=str(tmp_path))
        s1 = mgr.create_session()
        s2 = mgr.create_session()

        # Add a recipe to session 1's pot
        s1_pot_file = os.path.join(mgr.get_session_dir(s1), "recipe_pot.json")
        pot1 = load_pot_from_file(s1_pot_file)
        pot1.add_recipe(Recipe(
            name="Session 1 Recipe",
            ingredients=[Ingredient(name="water", quantity=1, unit="cup")],
            instructions=["Pour"],
            tags=[],
            sources=[]
        ))
        save_pot_to_file(pot1, s1_pot_file)

        # Session 2's pot should still be empty
        s2_pot_file = os.path.join(mgr.get_session_dir(s2), "recipe_pot.json")
        pot2 = load_pot_from_file(s2_pot_file)
        assert len(pot2.recipes) == 0
        assert len(pot1.recipes) == 1
