from __future__ import annotations

import base64
import io

import httpx
import numpy as np
from PIL import Image

from ..config import get_settings


def _decode(image_b64: str) -> Image.Image:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


class DepthEstimator:
    """Monocular depth. Returns a HxW array of metric-ish meters (smaller = closer)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pipe = None

    def _load_local(self):
        if self._pipe is None:
            from transformers import pipeline  # heavy + optional

            self._pipe = pipeline("depth-estimation", model=self.settings.depth_model)
        return self._pipe

    def estimate(self, image_b64: str) -> np.ndarray:
        backend = self.settings.perception_backend
        if backend == "local":
            return self._estimate_local(image_b64)
        if backend == "runpod":
            return self._estimate_runpod(image_b64)
        return self._estimate_mock(image_b64)

    def _estimate_local(self, image_b64: str) -> np.ndarray:
        img = _decode(image_b64)
        depth = self._load_local()(img)["predicted_depth"]
        rel = depth.squeeze().cpu().numpy()
        # Model outputs inverse-depth; invert + scale to a usable meter range.
        rel = rel.max() - rel
        return self._to_meters(rel)

    def _estimate_runpod(self, image_b64: str) -> np.ndarray:
        resp = httpx.post(
            self.settings.runpod_perception_url,
            headers={"Authorization": f"Bearer {self.settings.runpod_api_key}"},
            json={"input": {"task": "depth", "image": image_b64}},
            timeout=30,
        )
        resp.raise_for_status()
        out = resp.json()["output"]
        return np.array(out["depth"], dtype=np.float32)

    def _estimate_mock(self, image_b64: str) -> np.ndarray:
        img = _decode(image_b64)
        w, h = img.size
        # Floor-plane heuristic: bottom of frame is nearer than the top.
        rows = np.linspace(0.6, 6.0, h, dtype=np.float32)[:, None]
        return np.repeat(rows, w, axis=1)

    @staticmethod
    def _to_meters(rel: np.ndarray) -> np.ndarray:
        lo, hi = float(rel.min()), float(rel.max())
        if hi - lo < 1e-6:
            return np.full_like(rel, 3.0)
        norm = (rel - lo) / (hi - lo)
        return 0.5 + norm * 7.5
