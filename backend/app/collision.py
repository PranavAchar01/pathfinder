from __future__ import annotations

from .config import get_settings
from .schemas import HapticCommand, Risk, SceneObject


def _intensity(distance_m: float, warn: float, stop: float) -> int:
    if distance_m >= warn:
        return 0
    # Linear ramp from 90 at the warning edge to 255 at the stop threshold.
    span = max(warn - stop, 0.1)
    t = min(max((warn - distance_m) / span, 0.0), 1.0)
    return int(90 + t * 165)


def assess(objects: list[SceneObject]) -> tuple[Risk, HapticCommand]:
    """Map the nearest obstacle on each side to directional haptic intensity."""
    s = get_settings()
    haptic = HapticCommand()
    nearest = float("inf")

    for o in objects:
        level = _intensity(o.distance_m, s.collision_warn_m, s.collision_stop_m)
        if level == 0:
            continue
        nearest = min(nearest, o.distance_m)
        if o.side == "left":
            haptic.left = max(haptic.left, level)
        elif o.side == "right":
            haptic.right = max(haptic.right, level)
        else:
            haptic.center = max(haptic.center, level)

    if nearest <= s.collision_stop_m:
        risk: Risk = "stop"
        haptic.pattern = 2
    elif nearest < s.collision_warn_m:
        risk = "caution"
        haptic.pattern = 1
    else:
        risk = "clear"
        haptic.pattern = 0
    return risk, haptic
