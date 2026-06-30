from __future__ import annotations

from ..brightdata.client import BrightDataSearch
from ..schemas import ChatRequest, ChatResponse, Scene
from .qwen_client import QwenClient

SYSTEM = (
    "You are Pathfinder, a calm, concise voice assistant for a blind user navigating the world. "
    "Answer in 1-3 spoken sentences. When given the current scene, ground answers in the objects "
    "listed (their distance and clock position). If the user asks about an object you cannot "
    "identify and web context is provided, use it to explain what the object is and whether it is "
    "a hazard."
)

# A user asking about something not in the scene, or an "unknown"/low-confidence object,
# signals we should look it up on the web via Bright Data.
_LOOKUP_HINTS = ("what is", "what's this", "identify", "unfamiliar", "never seen", "describe this")


class ConversationAgent:
    def __init__(self, qwen: QwenClient | None = None, search: BrightDataSearch | None = None) -> None:
        self.qwen = qwen or QwenClient()
        self.search = search or BrightDataSearch()

    def _scene_text(self, scene: Scene | None) -> str:
        if not scene or not scene.objects:
            return "Scene: nothing notable detected right now."
        items = [f"{o.label} ({o.confidence:.0%}) {o.distance_m:.1f}m {o.side}" for o in scene.objects[:8]]
        return "Scene right now: " + "; ".join(items)

    def _should_lookup(self, req: ChatRequest) -> bool:
        text = req.text.lower()
        if any(h in text for h in _LOOKUP_HINTS):
            return True
        if req.scene:
            return any(not o.known for o in req.scene.objects)
        return False

    def respond(self, req: ChatRequest) -> ChatResponse:
        used_web = False
        sources: list[str] = []
        web_context = ""

        if self._should_lookup(req):
            unknown = ""
            if req.scene:
                unknown = next((o.label for o in req.scene.objects if not o.known), "")
            query = req.text if unknown == "" else f"{req.text} {unknown}"
            results = self.search.search(query, limit=3)
            if results:
                used_web = True
                sources = [r.url for r in results]
                web_context = "\n".join(f"- {r.title}: {r.snippet}" for r in results)

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self._scene_text(req.scene)},
        ]
        if web_context:
            messages.append({"role": "user", "content": f"Web context:\n{web_context}"})
        messages.append({"role": "user", "content": req.text})

        reply = self.qwen.chat(messages, temperature=0.4, max_tokens=300)
        return ChatResponse(reply=reply, used_web=used_web, sources=sources)
