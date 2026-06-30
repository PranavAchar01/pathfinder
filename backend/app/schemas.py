from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Side = Literal["left", "center", "right"]
Risk = Literal["clear", "caution", "stop"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: BBox


class SceneObject(BaseModel):
    label: str
    confidence: float
    bbox: BBox
    distance_m: float
    side: Side
    # 12 = straight ahead, 9 = hard left, 3 = hard right.
    clock: int
    known: bool = True


class HapticCommand(BaseModel):
    left: int = 0
    center: int = 0
    right: int = 0
    # 0 = steady, 1 = pulse, 2 = rapid (escalating urgency).
    pattern: int = 0

    def vibration_ms(self) -> list[int]:
        """Web Vibration API pattern, derived from intensity + urgency."""
        peak = max(self.left, self.center, self.right)
        if peak == 0:
            return []
        if self.pattern == 2:
            return [120, 60, 120, 60, 120]
        if self.pattern == 1:
            return [200, 120, 200]
        return [int(80 + peak)]


class Scene(BaseModel):
    objects: list[SceneObject] = Field(default_factory=list)
    risk: Risk = "clear"
    haptic: HapticCommand = Field(default_factory=HapticCommand)
    narration: str = ""
    frame_id: int = 0


class FrameMessage(BaseModel):
    type: Literal["frame"] = "frame"
    frame_id: int = 0
    # data URL or bare base64 JPEG from the browser camera.
    image: str


class ChatRequest(BaseModel):
    text: str
    # Latest scene snapshot so the agent can ground answers in what's nearby.
    scene: Optional[Scene] = None


class ChatResponse(BaseModel):
    reply: str
    used_web: bool = False
    sources: list[str] = Field(default_factory=list)
