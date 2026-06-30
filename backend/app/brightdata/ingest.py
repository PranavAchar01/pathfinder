from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings
from .client import BrightDataSearch


class TrainingDataIngestor:
    """Gather labeled imagery for weak/unknown classes via Bright Data and write a real,
    trainable shard (downloaded image files + a manifest) that the RunPod trainer consumes."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.search = BrightDataSearch()
        self.out_dir = Path(self.settings.rsi_data_dir) / "harvest"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def gather(self, label: str, count: int = 20) -> dict:
        """Collect reference imagery for one class. Downloads real image bytes through Bright
        Data (SERP image search -> Web Unlocker fetch) into data/harvest/<label>/ and writes a
        shard manifest the trainer self-labels on."""
        slug = label.replace(" ", "_")
        img_dir = self.out_dir / slug
        img_dir.mkdir(parents=True, exist_ok=True)

        hits = self.search.image_search(f"{label}", limit=count)
        records = []
        for i, r in enumerate(hits):
            rec = {"label": label, "title": r.title, "url": r.url, "image_url": r.image, "file": ""}
            if r.image and self.settings.brightdata_api_key:
                try:
                    data = self.search.fetch_bytes(r.image)
                    if data and len(data) > 1024:
                        fp = img_dir / f"{slug}_{i:03d}.jpg"
                        fp.write_bytes(data)
                        rec["file"] = str(fp)
                except Exception:  # noqa: BLE001 - a dead image URL must not kill the harvest
                    pass
            records.append(rec)

        shard = self.out_dir / f"{slug}.jsonl"
        with shard.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

        downloaded = sum(1 for r in records if r["file"])
        return {"label": label, "records": len(records), "images": downloaded, "shard": str(shard)}
