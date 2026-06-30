from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..config import get_settings

# Bright Data SERP API: requests are proxied through a zone and return live Google results.
_SERP_ENDPOINT = "https://api.brightdata.com/request"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class BrightDataSearch:
    """Live web search via Bright Data, used to identify objects the models don't know."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def live(self) -> bool:
        return bool(self.settings.brightdata_api_key)

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        if not self.live:
            return self._mock(query, limit)
        resp = httpx.post(
            _SERP_ENDPOINT,
            headers={"Authorization": f"Bearer {self.settings.brightdata_api_key}"},
            json={
                "zone": self.settings.brightdata_serp_zone,
                "url": f"https://www.google.com/search?q={query}&brd_json=1",
                "format": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])[:limit]
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("description", ""),
            )
            for r in organic
        ]

    def _mock(self, query: str, limit: int) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"About: {query}",
                url="https://example.com/result",
                snippet=(
                    f"A {query} is a common object. Set PATHFINDER_BRIGHTDATA_API_KEY "
                    "for live web identification."
                ),
            )
        ][:limit]
