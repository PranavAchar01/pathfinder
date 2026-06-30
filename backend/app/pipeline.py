from __future__ import annotations

import base64
import io

from PIL import Image

from . import collision
from .perception.depth import DepthEstimator
from .perception.detector import Detector
from .perception.fusion import fuse
from .reasoning.direction import DirectionReasoner
from .rsi.loop import RSILoop
from .schemas import Scene


def _frame_width(image_b64: str) -> int:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    return img.size[0]


class NavigationPipeline:
    """frame -> detect -> depth -> fuse -> direction (Qwen) -> collision -> haptic + narration."""

    def __init__(self) -> None:
        self.detector = Detector()
        self.depth = DepthEstimator()
        self.direction = DirectionReasoner()
        self.rsi = RSILoop()

    def process(self, image_b64: str, frame_id: int = 0) -> Scene:
        detections = self.detector.detect(image_b64)
        depth_map = self.depth.estimate(image_b64)
        objects = fuse(detections, depth_map, _frame_width(image_b64))

        risk, haptic = collision.assess(objects)
        narration, _priority = self.direction.guide(objects)

        scene = Scene(
            objects=objects,
            risk=risk,
            haptic=haptic,
            narration=narration,
            frame_id=frame_id,
        )
        self.rsi.observe(scene)
        return scene
