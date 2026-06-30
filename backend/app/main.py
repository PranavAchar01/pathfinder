from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .pipeline import NavigationPipeline
from .reasoning.conversation import ConversationAgent
from .schemas import ChatRequest, ChatResponse, FrameMessage

logging.basicConfig(level=logging.INFO)

settings = get_settings()
app = FastAPI(title="Pathfinder", version="0.1.0")

pipeline = NavigationPipeline()
agent = ConversationAgent()

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "perception_backend": settings.perception_backend,
        "reasoning_backend": settings.reasoning_backend,
        "rsi_enabled": settings.rsi_enabled,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return agent.respond(req)


@app.post("/api/rsi/cycle")
def rsi_cycle() -> dict:
    return pipeline.rsi.run_cycle()


@app.websocket("/ws/navigate")
async def navigate(ws: WebSocket) -> None:
    """Browser streams camera frames; server returns scene + haptic + narration."""
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_json()
            msg = FrameMessage(**raw)
            scene = pipeline.process(msg.image, frame_id=msg.frame_id)
            await ws.send_json(scene.model_dump())
    except WebSocketDisconnect:
        logging.info("navigation client disconnected")


if FRONTEND.is_dir():
    app.mount("/app", StaticFiles(directory=FRONTEND, html=True), name="frontend")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")
