# Architecture

## Principle: edge-first, GPU-for-training-only

Everything in the real-time loop runs **in the phone's browser** over WebGPU, so there is
no per-frame network latency. The cloud GPU (RunPod) is reserved for the one genuinely
heavy, infrequent job: **retraining the detector (RSI)**.

| Layer | Where | Responsibility |
|---|---|---|
| Edge | Phone browser (WebGPU) | Camera, YOLO detection, depth, fusion, collision, haptics, LLM (direction + chat), voice |
| Backend | Anywhere hostable | Serve the app, Bright Data lookups, RSI orchestration, publish detector weights |
| GPU | RunPod serverless | RSI fine-tune jobs only — no LLM, no per-frame inference |
| Web data | Bright Data | Identify unknown objects + harvest training data |

## Edge real-time loop (`frontend/src/app.js`, ~4 fps)

1. Draw the camera frame to a 480×360 processing canvas.
2. In parallel: `Detector.detect` (YOLO11 ONNX via onnxruntime-web/WebGPU) and
   `Depth.estimate` (Depth Anything V2 via transformers.js/WebGPU).
3. `fuse()` samples the nearest-15th-percentile depth inside each detection box →
   `distance_m`, and maps box center-x → `side` (left/center/right) + `clock` (9=hard left,
   12=ahead, 3=hard right).
4. `assess()` → `risk` (clear/caution/stop) + a `HapticCommand`; intensity ramps from the
   warn threshold (2 m) to the stop threshold (1 m), pattern escalates. `vibrate()` fires
   `navigator.vibrate(...)`.
5. Throttled (~1.5 s) `LLM.guide()` — **Qwen via WebLLM** turns the object geometry into one
   short spoken cue; falls back to a deterministic phrase if WebGPU/LLM is unavailable.

No backend call is on this path. Detection/depth/LLM each degrade to a mock implementation
if their model or WebGPU is missing, so the loop always produces output.

## Conversation + unknown-object identification

Hold-to-talk → Web Speech API transcript → `app.ask()`:

- If the user asks "what is this / identify…", or the scene contains a low-confidence
  ("unknown") object, the edge calls the backend **`/api/identify`**, which queries
  **Bright Data** (the API key must stay server-side) and returns snippets.
- `LLM.chat()` (WebLLM) answers in 1–3 sentences, grounded in the live scene and any web
  snippets, then the browser speaks it.

## Recursive Self-Improvement (RSI)

1. The edge `Telemetry` buffer records every weak/unknown detection and POSTs batches to
   **`/api/rsi/telemetry`**.
2. Backend `RSILoop` accumulates them; every `RSI_REVIEW_EVERY` items `Evaluator` finds the
   weak classes.
3. `TrainingDataIngestor` uses **Bright Data** to harvest reference data for those classes
   into `data/harvest/<label>.jsonl`.
4. `_trigger_finetune` submits `{"task":"finetune", classes, shards}` to the **RunPod** RSI
   endpoint (`runpod/rsi/handler.py`), which fine-tunes YOLO11 on the GPU and exports ONNX.
5. The new weights + `manifest.json` are published under the backend's **`/models`**; the
   edge `Detector` reads the manifest on load and runs the improved model. Every cycle is
   appended to `data/rsi_ledger.jsonl`.

This closes the loop: the device's field blind-spots drive what the GPU trains next.

## Why this split

- **Latency:** obstacle→haptic must be tens of ms; a cloud round-trip per frame can't meet
  that. On-device WebGPU can.
- **Cost/availability:** GPUs are expensive and contended. Using them only for periodic RSI
  training (not per-frame) keeps spend near zero and removes the real-time dependency on GPU
  availability.
- **Privacy:** the camera stream never leaves the device; only anonymized weak-class
  telemetry and explicit "what is this?" queries hit the network.

## Safety notes

- Mock depth is a floor-plane heuristic — **not** real distance. Never rely on mock mode
  for actual navigation.
- Collision thresholds are conservative defaults served from the backend
  (`/api/config`); tune `COLLISION_WARN_M` / `STOP_M` per user gait and mount height.
