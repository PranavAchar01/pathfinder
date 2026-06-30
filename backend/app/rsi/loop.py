from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import httpx

from ..brightdata.ingest import TrainingDataIngestor
from ..config import get_settings
from ..schemas import TelemetryItem
from .evaluator import Evaluator, WeaknessReport

log = logging.getLogger("pathfinder.rsi")


class RSILoop:
    """Recursive self-improvement, driven by edge telemetry.

    The edge device runs detection in real time and posts the classes it was unsure about.
    This loop accumulates that signal, finds the weak classes, uses Bright Data to harvest
    training data for them, and submits a fine-tune job to the RunPod GPU trainer. The
    trainer exports new detector weights that the edge then pulls — closing the loop.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.evaluator = Evaluator()
        self.ingestor = TrainingDataIngestor()
        self._buffer: deque[TelemetryItem] = deque(maxlen=max(self.settings.rsi_review_every * 4, 200))
        self._seen = 0
        self.ledger = Path(self.settings.rsi_data_dir) / "rsi_ledger.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)

    def ingest(self, items: list[TelemetryItem]) -> dict:
        if not self.settings.rsi_enabled or not items:
            return {"ingested": 0, "triggered": False}
        self._buffer.extend(items)
        self._seen += len(items)
        triggered = False
        cycle = None
        if self._seen >= self.settings.rsi_review_every:
            self._seen = 0
            cycle = self.run_cycle()
            triggered = cycle.get("job") is not None
        return {"ingested": len(items), "triggered": triggered, "cycle": cycle}

    def run_cycle(self) -> dict:
        report = self.evaluator.evaluate(list(self._buffer))
        cycle = {"t": time.time(), "samples": report.samples, "weak_labels": report.weak_labels}

        if report.needs_improvement:
            harvested = [self.ingestor.gather(label) for label in report.weak_labels[:5]]
            cycle["harvested"] = harvested
            cycle["job"] = self._trigger_finetune(report, harvested)
        else:
            cycle["job"] = None

        with self.ledger.open("a") as fh:
            fh.write(json.dumps(cycle) + "\n")
        log.info("RSI cycle: weak=%s job=%s", report.weak_labels, cycle.get("job"))
        return cycle

    def _trigger_finetune(self, report: WeaknessReport, harvested: list[dict]) -> dict:
        """Submit a YOLO fine-tune job to the RunPod GPU trainer."""
        spec = {
            "task": "finetune",
            "target": "yolo",
            "classes": report.weak_labels,
            "shards": [h["shard"] for h in harvested],
        }
        if not (self.settings.rsi_trainer_url and self.settings.runpod_api_key):
            return {"status": "queued_local", "spec": spec}
        try:
            resp = httpx.post(
                self.settings.rsi_trainer_url,
                headers={"Authorization": f"Bearer {self.settings.runpod_api_key}"},
                json={"input": spec},
                timeout=30,
            )
            resp.raise_for_status()
            return {"status": "submitted", "runpod": resp.json()}
        except httpx.HTTPError as exc:
            return {"status": "error", "detail": str(exc)}
