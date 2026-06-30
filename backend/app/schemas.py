from __future__ import annotations

from pydantic import BaseModel, Field

# The real-time scene (detect -> depth -> fuse -> direction -> haptic) now lives entirely
# on the edge. The backend only sees these lightweight messages.


class IdentifyRequest(BaseModel):
    # What the user asked + the best detector guess (often low-confidence / "unknown").
    query: str
    label_guess: str = ""


class IdentifyResult(BaseModel):
    title: str
    url: str
    snippet: str


class IdentifyResponse(BaseModel):
    query: str
    results: list[IdentifyResult] = Field(default_factory=list)
    used_web: bool = False


class TelemetryItem(BaseModel):
    label: str
    confidence: float
    known: bool = True
    distance_m: float = 0.0


class TelemetryBatch(BaseModel):
    # Edge posts weak/unknown detections it saw; this is the RSI training signal.
    items: list[TelemetryItem] = Field(default_factory=list)


class EdgeConfig(BaseModel):
    llm_model: str
    collision_warn_m: float
    collision_stop_m: float
    rsi_enabled: bool
