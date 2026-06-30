from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .brightdata.client import BrightDataSearch
from .config import get_settings
from .rsi.loop import RSILoop
from .schemas import (
    EdgeConfig,
    IdentifyRequest,
    IdentifyResponse,
    IdentifyResult,
    TelemetryBatch,
)

logging.basicConfig(level=logging.INFO)

settings = get_settings()
app = FastAPI(title="Pathfinder", version="0.2.0")

search = BrightDataSearch()
rsi = RSILoop()

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
MODELS = ROOT / "backend" / "models"
MODELS.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "edge_llm": settings.llm_model,
        "brightdata": bool(settings.brightdata_api_key),
        "rsi_trainer": bool(settings.rsi_trainer_url),
        "rsi_enabled": settings.rsi_enabled,
    }


@app.get("/api/config", response_model=EdgeConfig)
def edge_config() -> EdgeConfig:
    return EdgeConfig(
        llm_model=settings.llm_model,
        collision_warn_m=settings.collision_warn_m,
        collision_stop_m=settings.collision_stop_m,
        rsi_enabled=settings.rsi_enabled,
    )


@app.post("/api/identify", response_model=IdentifyResponse)
def identify(req: IdentifyRequest) -> IdentifyResponse:
    """Web lookup for an object the on-device models can't name. The edge LLM grounds its
    spoken answer on these results."""
    query = req.query if not req.label_guess else f"{req.query} {req.label_guess}"
    results = search.search(query, limit=3)
    return IdentifyResponse(
        query=query,
        results=[IdentifyResult(title=r.title, url=r.url, snippet=r.snippet) for r in results],
        used_web=search.live,
    )


@app.post("/api/rsi/telemetry")
def rsi_telemetry(batch: TelemetryBatch) -> dict:
    return rsi.ingest(batch.items)


@app.post("/api/rsi/cycle")
def rsi_cycle() -> dict:
    return rsi.run_cycle()


# RSI-published detector weights (manifest + .onnx) live here.
app.mount("/models", StaticFiles(directory=MODELS), name="models")

# Frontend served at root (mounted last so /api/* and /models take precedence).
if FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
