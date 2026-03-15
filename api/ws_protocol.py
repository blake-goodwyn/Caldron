"""WebSocket message protocol models for the Caldron API."""

from typing import Any, Optional
from pydantic import BaseModel


class UserMessage(BaseModel):
    type: str = "user_message"
    content: str


class AgentEvent(BaseModel):
    type: str = "agent_event"
    agent: str
    status: str  # "working" or "done"
    content: Optional[str] = None


class AgentResponse(BaseModel):
    type: str = "agent_response"
    content: str


class RecipeUpdate(BaseModel):
    type: str = "recipe_update"
    recipe: Optional[dict[str, Any]] = None


class GraphUpdate(BaseModel):
    type: str = "graph_update"
    graph: dict[str, Any]


class ErrorMessage(BaseModel):
    type: str = "error"
    detail: str
