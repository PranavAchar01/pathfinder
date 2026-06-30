#!/usr/bin/env python3
"""Export a YOLOv8 detector to ONNX for the edge (browser) to run via onnxruntime-web.

    pip install ultralytics
    python scripts/export_detector.py            # base yolov8n
    python scripts/export_detector.py best.pt    # an RSI-trained checkpoint

Writes backend/models/detector.onnx and bumps backend/models/manifest.json. If this file
is absent, the edge falls back to mock detection, so this is optional for a first run.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "backend" / "models"
MANIFEST = MODELS / "manifest.json"


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    from ultralytics import YOLO

    model = YOLO(weights)
    onnx_path = Path(model.export(format="onnx", opset=12, imgsz=640, simplify=True))
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, MODELS / "detector.onnx")

    manifest = json.loads(MANIFEST.read_text())
    manifest["version"] = int(manifest.get("version", 0)) + 1
    manifest["trained_by"] = weights
    manifest["labels"] = list(model.names.values())
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MODELS / 'detector.onnx'} (version {manifest['version']}, {len(manifest['labels'])} classes)")


if __name__ == "__main__":
    main()
