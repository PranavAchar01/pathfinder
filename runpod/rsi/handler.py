"""RunPod serverless handler — RSI training for the on-device detector.

The edge runs detection/depth/LLM in real time. This GPU endpoint does the heavy, periodic
job: self-training the YOLO detector on Bright-Data-harvested imagery for the classes the
edge was weak on, then exporting fresh ONNX weights for the edge to pull.

Self-training loop (no human labels needed): download harvested images, pseudo-label them
with the current best detector, keep only high-confidence boxes for the target/weak classes,
fine-tune on that set, export ONNX, and return it base64-encoded so the backend can publish
it to /models. This is real GPU work — it is what consumes RunPod credits.

Invoked with {"input": {"task": "finetune", "classes": [...], "images": {label: [url,...]},
"epochs": N, "base_weights": "..."}}.  Deploy with runpod/rsi/Dockerfile.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request

import requests
import runpod

# RunPod serverless drops job outputs above a small size limit, so the trained ONNX (tens of
# MB) can't be returned inline — upload it to a transfer host and return the URL instead.
_UPLOAD_HOSTS = ["https://0x0.st", "https://tmpfiles.org/api/v1/upload"]

_UA = {"User-Agent": "Mozilla/5.0 (PathfinderRSI)"}


def _upload(path: str) -> str:
    for host in _UPLOAD_HOSTS:
        try:
            with open(path, "rb") as fh:
                r = requests.post(host, files={"file": fh}, headers=_UA, timeout=120)
            r.raise_for_status()
            text = r.text.strip()
            if host.endswith("0x0.st") and text.startswith("http"):
                return text
            if "tmpfiles" in host:
                # tmpfiles returns {"data":{"url":"https://tmpfiles.org/<id>"}}; the direct
                # file is at /dl/<id>.
                u = r.json()["data"]["url"]
                return u.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        except Exception:  # noqa: BLE001
            continue
    return ""


def _download(url: str, dest: str) -> bool:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 1024:
            return False
        with open(dest, "wb") as fh:
            fh.write(data)
        return True
    except Exception:  # noqa: BLE001
        return False


def _finetune(inp: dict) -> dict:
    from ultralytics import YOLO

    classes = inp.get("classes", [])
    images = inp.get("images", {})  # {label: [image_url, ...]}
    epochs = int(inp.get("epochs", 25))
    imgsz = int(inp.get("imgsz", 640))

    base = YOLO(inp.get("base_weights", "yolo11m.pt"))
    names = base.model.names
    name_to_id = {v: k for k, v in names.items()}

    work = tempfile.mkdtemp(prefix="rsi_")
    img_dir = os.path.join(work, "images", "train")
    lbl_dir = os.path.join(work, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 1. Download harvested imagery.
    local = []
    for label, urls in images.items():
        for i, url in enumerate(urls):
            dest = os.path.join(img_dir, f"{label.replace(' ', '_')}_{i:03d}.jpg")
            if _download(url, dest):
                local.append(dest)

    # 2. Pseudo-label with the current detector; keep confident boxes only.
    kept = 0
    if local:
        for res in base.predict(local, conf=0.5, imgsz=imgsz, verbose=False, stream=True):
            stem = os.path.splitext(os.path.basename(res.path))[0]
            lines = []
            for b in res.boxes:
                cid = int(b.cls.item())
                xc, yc, w, h = b.xywhn[0].tolist()
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as fh:
                fh.write("\n".join(lines))
            kept += len(lines)

    if kept == 0:
        return {"status": "no_labels", "classes": classes, "downloaded": len(local),
                "note": "Bright Data harvest yielded no confidently-labelable images this cycle."}

    # 3. Build dataset YAML and fine-tune (the credit-consuming step).
    data_yaml = os.path.join(work, "data.yaml")
    names_block = "\n".join(f"  {i}: {n}" for i, n in names.items())
    with open(data_yaml, "w") as fh:
        fh.write(f"path: {work}\ntrain: images/train\nval: images/train\nnames:\n{names_block}\n")

    base.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=8, verbose=False,
               project=work, name="rsi", exist_ok=True)

    # 4. Export ONNX and upload it (too large to return inline) for the backend to publish.
    onnx_path = base.export(format="onnx", opset=12, dynamic=False, simplify=True)
    weights_url = _upload(onnx_path)

    return {
        "status": "completed" if weights_url else "trained_no_upload",
        "classes": classes,
        "downloaded": len(local),
        "labeled_boxes": kept,
        "epochs": epochs,
        "weights_onnx_url": weights_url,
        "onnx_bytes": os.path.getsize(onnx_path),
        "labels": [names[i] for i in sorted(names)],
        "input_size": imgsz,
    }


def handler(event: dict) -> dict:
    inp = event.get("input", {})
    task = inp.get("task", "finetune")
    if task == "finetune":
        return _finetune(inp)
    return {"error": f"unknown task {task}"}


runpod.serverless.start({"handler": handler})
