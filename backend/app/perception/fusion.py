from __future__ import annotations

import numpy as np

from ..schemas import Detection, SceneObject, Side


def _side_and_clock(cx: float, width: float) -> tuple[Side, int]:
    frac = cx / max(width, 1.0)
    if frac < 0.4:
        side: Side = "left"
    elif frac > 0.6:
        side = "right"
    else:
        side = "center"
    # Map horizontal fraction to a clock face: 9 (hard left) .. 12 .. 3 (hard right).
    if frac <= 0.5:
        clock = round(9 + (frac / 0.5) * 3)  # 9 (hard left) .. 12 (center)
    else:
        clock = round(((frac - 0.5) / 0.5) * 3)  # 12 .. 3 (hard right), 0 -> 12
    clock = 12 if clock == 0 else clock
    return side, clock


def _sample_distance(depth: np.ndarray, det: Detection) -> float:
    h, w = depth.shape
    x1 = max(int(det.bbox.x1), 0)
    x2 = min(int(det.bbox.x2), w)
    y1 = max(int(det.bbox.y1), 0)
    y2 = min(int(det.bbox.y2), h)
    if x2 <= x1 or y2 <= y1:
        return float(np.median(depth))
    patch = depth[y1:y2, x1:x2]
    # Nearest decile: the closest part of the object is what can hit you.
    return float(np.percentile(patch, 15))


def fuse(detections: list[Detection], depth: np.ndarray, frame_w: int) -> list[SceneObject]:
    objects: list[SceneObject] = []
    for det in detections:
        side, clock = _side_and_clock(det.bbox.cx, frame_w)
        objects.append(
            SceneObject(
                label=det.label,
                confidence=det.confidence,
                bbox=det.bbox,
                distance_m=round(_sample_distance(depth, det), 2),
                side=side,
                clock=clock,
                known=det.confidence >= 0.5,
            )
        )
    objects.sort(key=lambda o: o.distance_m)
    return objects
