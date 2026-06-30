from __future__ import annotations

import base64
import io
import random

import httpx
from PIL import Image

from ..config import get_settings
from ..schemas import BBox, Detection

_COCO_COMMON = [
    "person", "chair", "door", "table", "couch", "potted plant",
    "bicycle", "car", "dog", "stairs", "backpack", "bottle",
]


def _decode(image_b64: str) -> Image.Image:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


class Detector:
    """Object detection. Backend is selected at runtime: mock | local | runpod."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._yolo = None

    def _load_local(self):
        if self._yolo is None:
            from ultralytics import YOLO  # imported lazily; heavy + optional

            self._yolo = YOLO(self.settings.yolo_weights)
        return self._yolo

    def detect(self, image_b64: str) -> list[Detection]:
        backend = self.settings.perception_backend
        if backend == "local":
            return self._detect_local(image_b64)
        if backend == "runpod":
            return self._detect_runpod(image_b64)
        return self._detect_mock(image_b64)

    def _detect_local(self, image_b64: str) -> list[Detection]:
        img = _decode(image_b64)
        model = self._load_local()
        result = model.predict(img, verbose=False)[0]
        names = result.names
        out: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(
                Detection(
                    label=names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return out

    def _detect_runpod(self, image_b64: str) -> list[Detection]:
        resp = httpx.post(
            self.settings.runpod_perception_url,
            headers={"Authorization": f"Bearer {self.settings.runpod_api_key}"},
            json={"input": {"task": "detect", "image": image_b64}},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json().get("output", {})
        return [Detection(**d) for d in payload.get("detections", [])]

    def _detect_mock(self, image_b64: str) -> list[Detection]:
        img = _decode(image_b64)
        w, h = img.size
        rng = random.Random(len(image_b64))
        out: list[Detection] = []
        for _ in range(rng.randint(1, 3)):
            bw = rng.uniform(0.12, 0.4) * w
            bh = rng.uniform(0.2, 0.6) * h
            x1 = rng.uniform(0, w - bw)
            y1 = rng.uniform(h * 0.2, h - bh)
            out.append(
                Detection(
                    label=rng.choice(_COCO_COMMON),
                    confidence=round(rng.uniform(0.4, 0.95), 2),
                    bbox=BBox(x1=x1, y1=y1, x2=x1 + bw, y2=y1 + bh),
                )
            )
        return out
