# Caldron

![Caldron banner](https://raw.githubusercontent.com/blake-goodwyn/Caldron/main/Caldron_banner.png)

Caldron is an AI-assisted recipe development platform that allows users to generate and quickly iterate on new recipes. Caldron leverages multi-agent generative artificial intelligence tools to quickly iterate a desired foundational recipe from a provided prompt and optionally provided context. Caldron then provides channels for both human and machine sensory feedback to iteratively refine the recipe.

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- (Optional) A [Tavily API key](https://tavily.com/) for web search

## Installation

```bash
git clone https://github.com/blake-goodwyn/Caldron.git
cd Caldron
pip install -r cauldron-app/requirements.txt
```

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
cd cauldron-app
python main.py
```

Caldron will start an interactive session where you can describe a recipe idea. The multi-agent system will research, generate, and refine the recipe through conversation.

## Architecture

Caldron uses a **LangChain / LangGraph** supervisor-agent pattern:

| Agent | Role |
|-------|------|
| **Caldron Postman** | Top-level supervisor — routes tasks to specialists |
| **Research Postman** | Coordinates web search (Tavily) and recipe scraping (Sleuth) |
| **Tavily** | Internet search for recipe URLs |
| **Sleuth** | Scrapes structured recipe data from URLs |
| **ModSquad** | Manages and applies recipe modifications |
| **Spinnaret** | Tracks recipe development in a versioned Recipe Graph |
| **KnowItAll** | Answers questions about the current recipe state |
| **Frontman** | User-facing interface — summarizes results |

Recipe state is persisted as JSON files:
- `recipe_graph.json` — versioned DAG of recipe iterations (NetworkX)
- `mods_list.json` — priority queue of pending modifications
- `recipe_pot.json` — short-term staging area for scraped recipes

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
cauldron-app/
  main.py              # Entry point
  cauldron_app.py      # Application class & graph setup
  agent_defs.py        # Agent definitions & graph wiring
  agent_tools.py       # LangChain @tool functions
  class_defs.py        # Data models (Recipe, Ingredient, etc.)
  langchain_util.py    # LangChain helpers (agent/router factories)
  config.py            # Centralized configuration
  logging_util.py      # Logging setup
tests/
  conftest.py          # Shared fixtures
  test_class_defs.py   # Unit tests — data models
  test_agent_tools.py  # Unit tests — tool functions
  test_agent_graph.py  # Integration tests — graph compilation
```

## License

See [LICENSE](LICENSE) for details.
