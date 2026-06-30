"""RunPod serverless handler — RSI training for the on-device detector.

The edge runs detection/depth/LLM in real time. This GPU endpoint exists for the heavy,
periodic job: fine-tuning the YOLO detector on Bright-Data-harvested data for the classes
the edge was weak on, then exporting fresh ONNX weights for the edge to pull.

Invoked by the backend RSI loop with {"input": {"task": "finetune", "classes": [...],
"shards": [...]}}.  Deploy with runpod/rsi/Dockerfile.
"""
from __future__ import annotations

import runpod


def _finetune(inp: dict) -> dict:
    classes = inp.get("classes", [])
    shards = inp.get("shards", [])

    from ultralytics import YOLO

    # Start from the current best weights; in production these come from a network volume /
    # object store so each cycle builds on the last (true recursive self-improvement).
    model = YOLO(inp.get("base_weights", "yolov8n.pt"))

    # A real run builds a dataset YAML from the harvested shards and trains:
    #   data_yaml = build_dataset(shards, classes)
    #   model.train(data=data_yaml, epochs=inp.get("epochs", 30), imgsz=640)
    # Kept as a guarded stub so the endpoint is callable before the data pipeline is wired.
    trained = False

    onnx_path = None
    if trained:
        onnx_path = model.export(format="onnx", opset=12, dynamic=False, simplify=True)

    return {
        "status": "completed" if trained else "accepted",
        "classes": classes,
        "shards": shards,
        "weights_onnx": onnx_path,
        "note": "Wire shards->dataset + publish ONNX to the backend /models manifest.",
    }


def handler(event: dict) -> dict:
    inp = event.get("input", {})
    task = inp.get("task", "finetune")
    if task == "finetune":
        return _finetune(inp)
    return {"error": f"unknown task {task}"}


runpod.serverless.start({"handler": handler})
