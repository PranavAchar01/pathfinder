from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..config import get_settings

# Bright Data Web Unlocker / SERP API: requests are proxied through a zone and return live
# Google results (web + images). One endpoint, zone selects the product.
_ENDPOINT = "https://api.brightdata.com/request"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    image: str = ""  # direct image URL when this came from image search


class BrightDataSearch:
    """Live web search via Bright Data: identifies unknown objects (web) and harvests
    reference imagery (images) for RSI training."""

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def live(self) -> bool:
        return bool(self.settings.brightdata_api_key)

    def _request(self, url: str) -> dict:
        resp = httpx.post(
            _ENDPOINT,
            headers={"Authorization": f"Bearer {self.settings.brightdata_api_key}"},
            json={"zone": self.settings.brightdata_serp_zone, "url": url, "format": "json"},
            timeout=45,
        )
        resp.raise_for_status()
        return resp.json()

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        if not self.live:
            return self._mock(query, limit)
        try:
            data = self._request(f"https://www.google.com/search?q={query}&brd_json=1")
        except httpx.HTTPError:
            # Key present but zone not provisioned yet — degrade instead of erroring.
            return self._mock(query, limit)
        organic = data.get("organic", [])[:limit]
        return [
            SearchResult(title=r.get("title", ""), url=r.get("link", ""), snippet=r.get("description", ""))
            for r in organic
        ]

    def image_search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Google Images via Bright Data SERP — returns direct image URLs for harvesting."""
        if not self.live:
            return self._mock(query, limit)
        try:
            data = self._request(f"https://www.google.com/search?q={query}&tbm=isch&brd_json=1")
        except httpx.HTTPError:
            return []  # no zone yet -> no harvest this cycle (trainer returns no_labels, no GPU spend)
        images = data.get("images", []) or data.get("organic", [])
        out: list[SearchResult] = []
        for r in images[:limit]:
            src = r.get("image") or r.get("original") or r.get("link") or r.get("source", "")
            if not src:
                continue
            out.append(SearchResult(title=r.get("title", query), url=r.get("link", src),
                                    snippet=r.get("source", ""), image=src))
        return out

    def fetch_bytes(self, url: str) -> bytes:
        """Download a resource (image) through the Web Unlocker zone so bot-protected hosts
        still resolve."""
        if not self.live:
            return b""
        resp = httpx.post(
            _ENDPOINT,
            headers={"Authorization": f"Bearer {self.settings.brightdata_api_key}"},
            json={"zone": self.settings.brightdata_unlocker_zone, "url": url, "format": "raw"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content

    def _mock(self, query: str, limit: int) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"About: {query}",
                url="https://example.com/result",
                snippet=(f"A {query} is a common object. Set PATHFINDER_BRIGHTDATA_API_KEY "
                         "for live web identification."),
            )
        ][:limit]
