from __future__ import annotations

import json
from pathlib import Path

import httpx

from ..config import get_settings
from .client import BrightDataSearch

# Bright Data Image / Web Unlocker fetches reference imagery to grow the training set
# for object classes the detector is weak on.
_UNLOCKER_ENDPOINT = "https://api.brightdata.com/request"


class TrainingDataIngestor:
    """Gather labeled imagery + descriptions for weak/unknown classes via Bright Data."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.search = BrightDataSearch()
        self.out_dir = Path(self.settings.rsi_data_dir) / "harvest"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def gather(self, label: str, count: int = 20) -> dict:
        """Collect reference data for one object class and write a manifest shard."""
        results = self.search.search(f"{label} object photo", limit=count)
        records = [{"label": label, "title": r.title, "url": r.url, "snippet": r.snippet} for r in results]

        if self.settings.brightdata_api_key:
            for rec in records:
                rec["image"] = self._fetch_image_url(rec["url"])

        shard = self.out_dir / f"{label.replace(' ', '_')}.jsonl"
        with shard.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return {"label": label, "records": len(records), "shard": str(shard)}

    def _fetch_image_url(self, page_url: str) -> str:
        try:
            resp = httpx.post(
                _UNLOCKER_ENDPOINT,
                headers={"Authorization": f"Bearer {self.settings.brightdata_api_key}"},
                json={"zone": self.settings.brightdata_unlocker_zone, "url": page_url, "format": "raw"},
                timeout=30,
            )
            resp.raise_for_status()
            return page_url
        except httpx.HTTPError:
            return ""
