"""Per-session state isolation for the Caldron API."""

import os
import shutil
import uuid
from contextlib import contextmanager
from logging_util import logger


# Import after path setup (chain_factory handles sys.path)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))

from class_defs import fresh_graph, fresh_mods_list, fresh_pot
from agent_tools import _graph_file, _mods_file, _pot_file


SESSIONS_DIR = os.path.join(os.path.dirname(__file__), '..', 'sessions')


class SessionManager:
    """Manages per-session state directories and ContextVar tokens."""

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self._sessions: dict[str, str] = {}  # session_id -> dir path

    def create_session(self, session_id: str = None) -> str:
        """Create a new session with its own state directory."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        session_dir = os.path.join(self.sessions_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Initialize fresh state files in the session directory
        fresh_pot(os.path.join(session_dir, "recipe_pot.json"))
        fresh_graph(os.path.join(session_dir, "recipe_graph.json"))
        fresh_mods_list(os.path.join(session_dir, "mods_list.json"))

        self._sessions[session_id] = session_dir
        logger.info(f"Session {session_id} created at {session_dir}")
        return session_id

    def get_session_dir(self, session_id: str) -> str:
        """Get the state directory for a session."""
        if session_id not in self._sessions:
            # Check if directory exists on disk (server restart case)
            session_dir = os.path.join(self.sessions_dir, session_id)
            if os.path.isdir(session_dir):
                self._sessions[session_id] = session_dir
            else:
                raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

    @contextmanager
    def session_scope(self, session_id: str):
        """Context manager that sets ContextVars to this session's state files."""
        session_dir = self.get_session_dir(session_id)

        token_graph = _graph_file.set(os.path.join(session_dir, "recipe_graph.json"))
        token_mods = _mods_file.set(os.path.join(session_dir, "mods_list.json"))
        token_pot = _pot_file.set(os.path.join(session_dir, "recipe_pot.json"))

        try:
            yield session_dir
        finally:
            _graph_file.reset(token_graph)
            _mods_file.reset(token_mods)
            _pot_file.reset(token_pot)

    def remove_session(self, session_id: str) -> None:
        """Remove a session and its state directory."""
        if session_id in self._sessions:
            session_dir = self._sessions.pop(session_id)
            if os.path.isdir(session_dir):
                shutil.rmtree(session_dir)
            logger.info(f"Session {session_id} removed.")
