"""FastAPI server for the Caldron conversational recipe development API."""

import sys
import os
import json

# Add cauldron-app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from chain_factory import compile_chain
from session import SessionManager
from ws_protocol import AgentEvent, AgentResponse, RecipeUpdate, GraphUpdate, ErrorMessage
from class_defs import load_graph_from_file
from agent_tools import _graph_file
from langchain_core.messages import HumanMessage
from logging_util import logger
from config import LLM_MODEL

app = FastAPI(title="Caldron API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile the chain once at startup
chain = compile_chain(LLM_MODEL)
session_manager = SessionManager()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Create session if it doesn't exist
    try:
        session_manager.get_session_dir(session_id)
    except KeyError:
        session_manager.create_session(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(
                    ErrorMessage(detail="Invalid JSON").model_dump_json()
                )
                continue

            if message.get("type") != "user_message":
                await websocket.send_text(
                    ErrorMessage(detail="Unknown message type").model_dump_json()
                )
                continue

            content = message.get("content", "")
            logger.info(f"Session {session_id}: received message: {content[:100]}")

            with session_manager.session_scope(session_id):
                try:
                    for event in chain.stream(
                        {
                            "messages": [HumanMessage(content=content)],
                            "sender": "User",
                            "next": "Caldron\nPostman",
                        },
                        {"recursion_limit": 50},
                    ):
                        # Each event is a dict with one key (the agent name)
                        agent_name = list(event.keys())[0]
                        agent_data = event[agent_name]

                        if agent_name == "Frontman":
                            # Final user-facing response
                            response_content = agent_data["messages"][0].content
                            await websocket.send_text(
                                AgentResponse(content=response_content).model_dump_json()
                            )
                        elif "next" in agent_data:
                            # Routing event — show which agent is working
                            await websocket.send_text(
                                AgentEvent(
                                    agent=agent_data.get("sender", agent_name),
                                    status="working",
                                    content=None,
                                ).model_dump_json()
                            )
                        else:
                            # Agent completed its work
                            msg_content = None
                            if "messages" in agent_data and agent_data["messages"]:
                                msg_content = str(agent_data["messages"][0].content)[:500]
                            await websocket.send_text(
                                AgentEvent(
                                    agent=agent_name,
                                    status="done",
                                    content=msg_content,
                                ).model_dump_json()
                            )

                    # After chain completes, send recipe and graph updates
                    graph_file = _graph_file.get()
                    try:
                        recipe_graph = load_graph_from_file(graph_file)
                        graph_dict = recipe_graph.to_dict()

                        # Send graph update
                        await websocket.send_text(
                            GraphUpdate(graph=graph_dict).model_dump_json()
                        )

                        # Send recipe update (foundational recipe)
                        foundational = recipe_graph.get_foundational_recipe()
                        recipe_dict = json.loads(foundational.model_dump_json()) if foundational else None
                        await websocket.send_text(
                            RecipeUpdate(recipe=recipe_dict).model_dump_json()
                        )
                    except (FileNotFoundError, Exception) as e:
                        logger.warning(f"Could not load graph for updates: {e}")

                except Exception as e:
                    logger.error(f"Chain error in session {session_id}: {e}")
                    await websocket.send_text(
                        ErrorMessage(detail=f"Agent error: {str(e)[:200]}").model_dump_json()
                    )

    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: client disconnected")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
