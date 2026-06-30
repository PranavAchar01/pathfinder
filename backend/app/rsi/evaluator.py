from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..config import get_settings
from ..schemas import Scene


@dataclass
class WeaknessReport:
    weak_labels: list[str] = field(default_factory=list)
    unknown_count: int = 0
    low_conf_count: int = 0
    samples: int = 0

    @property
    def needs_improvement(self) -> bool:
        return bool(self.weak_labels) or self.unknown_count > 0


class Evaluator:
    """Scores recent perception to find classes the system is weak on."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(self, scenes: list[Scene]) -> WeaknessReport:
        report = WeaknessReport(samples=len(scenes))
        weak: Counter[str] = Counter()
        for scene in scenes:
            for obj in scene.objects:
                if not obj.known:
                    report.unknown_count += 1
                    weak[obj.label] += 1
                elif obj.confidence < self.settings.rsi_min_confidence:
                    report.low_conf_count += 1
                    weak[obj.label] += 1
        report.weak_labels = [label for label, _ in weak.most_common(10)]
        return report
