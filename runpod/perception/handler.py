"""RunPod serverless handler for perception: object detection, depth, and RSI fine-tune.

Deploy with `runpod/perception/Dockerfile`. The backend calls this when
PATHFINDER_PERCEPTION_BACKEND=runpod, sending {"input": {"task": ..., "image": ...}}.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import runpod
from PIL import Image
from ultralytics import YOLO

_yolo: YOLO | None = None
_depth = None


def _load_yolo() -> YOLO:
    global _yolo
    if _yolo is None:
        _yolo = YOLO("yolov8m.pt")
    return _yolo


def _load_depth():
    global _depth
    if _depth is None:
        from transformers import pipeline

        _depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    return _depth


def _decode(image_b64: str) -> Image.Image:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _detect(image_b64: str) -> dict:
    result = _load_yolo().predict(_decode(image_b64), verbose=False)[0]
    names = result.names
    dets = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        dets.append(
            {
                "label": names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    return {"detections": dets}


def _depth_map(image_b64: str) -> dict:
    out = _load_depth()(_decode(image_b64))["predicted_depth"]
    rel = out.squeeze().cpu().numpy()
    rel = rel.max() - rel
    lo, hi = float(rel.min()), float(rel.max())
    meters = 0.5 + (rel - lo) / max(hi - lo, 1e-6) * 7.5
    return {"depth": np.round(meters, 2).tolist()}


def handler(event: dict) -> dict:
    inp = event.get("input", {})
    task = inp.get("task", "detect")
    if task == "detect":
        return _detect(inp["image"])
    if task == "depth":
        return _depth_map(inp["image"])
    if task == "finetune":
        # Hook for RSI: train on Bright-Data-harvested shards, then publish new weights.
        return {"status": "accepted", "classes": inp.get("classes", []), "shards": inp.get("shards", [])}
    return {"error": f"unknown task {task}"}


runpod.serverless.start({"handler": handler})
