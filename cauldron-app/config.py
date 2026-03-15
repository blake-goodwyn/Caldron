"""Centralized configuration for the Caldron application.

All environment variables and hardcoded settings are managed here.
Other modules should import from this module instead of calling
load_dotenv() or os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangChain tracing (set as env var for LangChain to pick up)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# --- Application Settings ---
DB_PATH = os.getenv("CALDRON_DB_PATH", "sqlite:///sql/recipes_0514_1658_views.db")
LLM_MODEL = os.getenv("CALDRON_LLM_MODEL", "gpt-3.5-turbo")

# --- State Persistence ---
STATE_DIR = os.getenv("CALDRON_STATE_DIR", ".")
MODS_LIST_FILE = os.path.join(STATE_DIR, "mods_list.json")
RECIPE_GRAPH_FILE = os.path.join(STATE_DIR, "recipe_graph.json")
RECIPE_POT_FILE = os.path.join(STATE_DIR, "recipe_pot.json")


def validate_required_keys() -> None:
    """Validate that all required API keys are set. Call at app startup."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "See .env.example for setup instructions."
        )
