from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote_plus

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

    def _request(self, url: str, retries: int = 3) -> dict:
        # Bright Data wraps the response as {status_code, headers, body}; with brd_json=1 the
        # body is the parsed-SERP JSON encoded as a string. The SERP node occasionally returns
        # an empty/non-JSON body, so retry until it parses.
        last: Exception | None = None
        for _ in range(retries):
            try:
                resp = httpx.post(
                    _ENDPOINT,
                    headers={"Authorization": f"Bearer {self.settings.brightdata_api_key}"},
                    json={"zone": self.settings.brightdata_serp_zone, "url": url, "format": "json"},
                    timeout=90,
                )
                resp.raise_for_status()
                body = resp.json().get("body", "")
                if isinstance(body, dict):
                    return body
                if isinstance(body, str) and body.strip().startswith("{"):
                    return json.loads(body)
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
        if last:
            raise last
        return {}

    def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        if not self.live:
            return self._mock(query, limit)
        try:
            data = self._request(f"https://www.google.com/search?q={quote_plus(query)}&brd_json=1")
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
            data = self._request(f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&brd_json=1")
        except httpx.HTTPError:
            return []  # no zone yet -> no harvest this cycle (trainer returns no_labels, no GPU spend)
        images = data.get("images", [])
        out: list[SearchResult] = []
        for r in images[:limit]:
            src = r.get("original_image") or r.get("image") or ""
            if not src.startswith("http"):
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
