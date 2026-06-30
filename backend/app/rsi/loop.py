from __future__ import annotations

import base64
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
    training data for them, submits a self-training fine-tune job to the RunPod GPU, polls it
    to completion, and publishes the new detector weights to /models — which the edge pulls.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.evaluator = Evaluator()
        self.ingestor = TrainingDataIngestor()
        self._buffer: deque[TelemetryItem] = deque(maxlen=max(self.settings.rsi_review_every * 4, 200))
        self._seen = 0
        data = Path(self.settings.rsi_data_dir)
        self.ledger = data / "rsi_ledger.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path(__file__).resolve().parents[3] / "backend" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _base_url(self) -> str:
        # Accept .../run, .../runsync, or the bare endpoint URL.
        u = self.settings.rsi_trainer_url.rstrip("/")
        for suffix in ("/runsync", "/run"):
            if u.endswith(suffix):
                return u[: -len(suffix)]
        return u

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

    def run_cycle(self, labels: list[str] | None = None) -> dict:
        report = self.evaluator.evaluate(list(self._buffer))
        weak = labels or report.weak_labels
        cycle = {"t": time.time(), "samples": report.samples, "weak_labels": weak}

        if weak:
            harvested = [self.ingestor.gather(label) for label in weak[:5]]
            cycle["harvested"] = harvested
            cycle["job"] = self._train(weak[:5], harvested)
        else:
            cycle["job"] = None

        with self.ledger.open("a") as fh:
            fh.write(json.dumps(cycle) + "\n")
        log.info("RSI cycle: weak=%s job=%s", weak, (cycle.get("job") or {}).get("status"))
        return cycle

    def _images_from_shards(self, harvested: list[dict]) -> dict[str, list[str]]:
        images: dict[str, list[str]] = {}
        for h in harvested:
            urls = []
            try:
                for line in Path(h["shard"]).read_text().splitlines():
                    rec = json.loads(line)
                    if rec.get("image_url"):
                        urls.append(rec["image_url"])
            except Exception:  # noqa: BLE001
                continue
            if urls:
                images[h["label"]] = urls
        return images

    def _train(self, classes: list[str], harvested: list[dict]) -> dict:
        spec = {
            "task": "finetune",
            "classes": classes,
            "images": self._images_from_shards(harvested),
            "epochs": self.settings.rsi_epochs,
        }
        if not (self.settings.rsi_trainer_url and self.settings.runpod_api_key):
            return {"status": "queued_local", "spec_classes": classes}
        try:
            job = self._submit(spec)
            job_id = job.get("id")
            if not job_id:
                return {"status": "error", "detail": f"no job id: {job}"}
            result = self._poll(job_id)
            output = result.get("output", {})
            published = self._publish_weights(output)
            return {"status": result.get("status"), "job_id": job_id,
                    "trainer": {k: v for k, v in output.items() if k != "weights_onnx_b64"},
                    "published": published}
        except httpx.HTTPError as exc:
            return {"status": "error", "detail": str(exc)}

    def _submit(self, spec: dict) -> dict:
        resp = httpx.post(f"{self._base_url}/run",
                          headers={"Authorization": f"Bearer {self.settings.runpod_api_key}"},
                          json={"input": spec}, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _poll(self, job_id: str, timeout_s: int = 1800) -> dict:
        headers = {"Authorization": f"Bearer {self.settings.runpod_api_key}"}
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = httpx.get(f"{self._base_url}/status/{job_id}", headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
                return data
            time.sleep(5)
        return {"status": "TIMEOUT", "id": job_id}

    def _publish_weights(self, output: dict) -> bool:
        b64 = output.get("weights_onnx_b64")
        if not b64:
            return False
        (self.models_dir / "detector.onnx").write_bytes(base64.b64decode(b64))
        manifest_path = self.models_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001
            manifest = {}
        manifest.update({
            "version": int(manifest.get("version", 0)) + 1,
            "format": "yolo11-onnx",
            "url": "/models/detector.onnx",
            "input_size": output.get("input_size", 640),
            "trained_by": "rsi-self-training",
            "labels": output.get("labels", manifest.get("labels", [])),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))
        log.info("RSI published detector v%s", manifest["version"])
        return True
