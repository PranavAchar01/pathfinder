from __future__ import annotations

from ..schemas import SceneObject
from .qwen_client import QwenClient

_CLOCK_PHRASE = {
    9: "to your hard left", 10: "to your left", 11: "slightly left",
    12: "directly ahead", 1: "slightly right", 2: "to your right",
    3: "to your hard right",
}

SYSTEM = (
    "You are the spatial-reasoning core of a wearable navigation aid for a blind user. "
    "Given a list of detected objects with their distance and clock-face position, decide "
    "the single most important obstacle and produce one short spoken directional cue "
    "(max 12 words). Reply ONLY as JSON: "
    '{"guidance": "<spoken cue>", "priority_index": <int or null>}.'
)


class DirectionReasoner:
    """Qwen turns raw geometry into prioritized, human directional guidance."""

    def __init__(self, qwen: QwenClient | None = None) -> None:
        self.qwen = qwen or QwenClient()

    def guide(self, objects: list[SceneObject]) -> tuple[str, int | None]:
        if not objects:
            return "Path is clear ahead.", None

        payload = [
            {"index": i, "label": o.label, "distance_m": o.distance_m, "clock": o.clock, "side": o.side}
            for i, o in enumerate(objects[:6])
        ]
        result = self.qwen.chat_json(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Objects: {payload}\nReturn the guidance JSON."},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        guidance = result.get("guidance")
        idx = result.get("priority_index")
        if isinstance(guidance, str) and guidance.strip():
            return guidance.strip(), idx if isinstance(idx, int) else 0
        return self._fallback(objects), 0

    @staticmethod
    def _fallback(objects: list[SceneObject]) -> str:
        nearest = objects[0]
        where = _CLOCK_PHRASE.get(nearest.clock, "ahead")
        return f"{nearest.label} {nearest.distance_m:.0f} meters {where}."
