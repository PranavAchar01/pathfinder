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

    def gather(self, label: str, count: int = 30) -> dict:
        """Collect reference image URLs for one class via Bright Data SERP image search and
        write a shard manifest. The GPU trainer downloads + self-labels these directly, so the
        backend keeps only the URLs (no slow per-image fetch here)."""
        slug = label.replace(" ", "_")
        hits = self.search.image_search(label, limit=count)
        records = [{"label": label, "title": r.title, "url": r.url, "image_url": r.image}
                   for r in hits if r.image]

        shard = self.out_dir / f"{slug}.jsonl"
        with shard.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return {"label": label, "records": len(records), "images": len(records), "shard": str(shard)}
