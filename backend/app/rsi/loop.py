from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import httpx

from ..brightdata.ingest import TrainingDataIngestor
from ..config import get_settings
from ..schemas import Scene
from .evaluator import Evaluator, WeaknessReport

log = logging.getLogger("pathfinder.rsi")


class RSILoop:
    """Recursive self-improvement.

    Observes live perception, periodically evaluates where the models are weak,
    uses Bright Data to harvest training data for those classes, and triggers a
    retraining/threshold-adaptation job on RunPod. Each cycle is logged so the
    system's competence is auditable over time.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.evaluator = Evaluator()
        self.ingestor = TrainingDataIngestor()
        self._buffer: deque[Scene] = deque(maxlen=self.settings.rsi_review_every)
        self._seen = 0
        self.ledger = Path(self.settings.rsi_data_dir) / "rsi_ledger.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, scene: Scene) -> None:
        if not self.settings.rsi_enabled:
            return
        self._buffer.append(scene)
        self._seen += 1
        if self._seen % self.settings.rsi_review_every == 0:
            self.run_cycle()

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
        """Kick a RunPod training job on the freshly harvested data."""
        spec = {
            "task": "finetune",
            "target": "yolo",
            "classes": report.weak_labels,
            "shards": [h["shard"] for h in harvested],
        }
        if not (self.settings.runpod_perception_url and self.settings.runpod_api_key):
            return {"status": "queued_local", "spec": spec}
        try:
            resp = httpx.post(
                self.settings.runpod_perception_url,
                headers={"Authorization": f"Bearer {self.settings.runpod_api_key}"},
                json={"input": spec},
                timeout=30,
            )
            resp.raise_for_status()
            return {"status": "submitted", "runpod": resp.json()}
        except httpx.HTTPError as exc:
            return {"status": "error", "detail": str(exc)}
