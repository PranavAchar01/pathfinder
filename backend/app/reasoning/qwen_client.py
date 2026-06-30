from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import get_settings


class QwenClient:
    """Wraps a RunPod-hosted Qwen served by vLLM (OpenAI-compatible API).

    In "mock" mode it answers heuristically so the whole system runs with no GPU.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def live(self) -> bool:
        return self.settings.reasoning_backend == "runpod" and bool(self.settings.runpod_api_key)

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.3, max_tokens: int = 512) -> str:
        if not self.live:
            return self._mock(messages)
        resp = httpx.post(
            f"{self.settings.runpod_qwen_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.runpod_api_key}"},
            json={
                "model": self.settings.runpod_qwen_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def chat_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        raw = self.chat(messages, **kw)
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {}

    def _mock(self, messages: list[dict[str, Any]]) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if isinstance(last, list):  # multimodal content blocks
            last = " ".join(b.get("text", "") for b in last if isinstance(b, dict))
        if "JSON" in last or "guidance" in last.lower():
            return '{"guidance": "Path is clear ahead.", "priority": null}'
        return "I can describe what's around you. Ask me about a specific object or direction."
