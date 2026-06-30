# Pathfinder

**Live (open in Chrome, allow camera + mic):** https://pathfinder-ten-delta.vercel.app

A wearable navigation aid for blind and low-vision users — **web only, edge-first**.
You mount a phone on your chest, open the web app, and **everything in the real-time loop
runs on the device, in the browser**, for the lowest possible latency:

- **camera** → object detection: **RF-DETR** (medium, transformers.js/WebGPU, NMS-free, SOTA accuracy) with **YOLO11m** (onnxruntime-web) as automatic fallback
- → **monocular depth** (Depth Anything V2, transformers.js, WebGPU) → per-object distance
- → **direction reasoning + conversation** by **Qwen** running fully in-browser (**WebLLM**, WebGPU)
- → **haptics**: the phone buzzes via the **Web Vibration API** as you near collision
  (left/center/right intensity + escalating pattern encode where the obstacle is)
- → **voice**: ask "what's around me?" and it answers from the live scene (Web Speech API)

The **GPU cloud (RunPod) is used only for the heavy lifting: Recursive Self-Improvement
(RSI) training of the detector** — not for any per-frame inference. The edge reports the
classes it was unsure about; the backend harvests training data for them via **Bright
Data** and submits a fine-tune job to RunPod; the trainer publishes new detector weights
that the edge pulls. Bright Data also powers live **web identification** of unfamiliar
objects.

```
            ┌──────────────────────── PHONE BROWSER (edge, real-time) ───────────────────────┐
 camera ──▶ │  YOLO (WebGPU) ─┐                                                                │
            │  depth (WebGPU) ─┴▶ fuse ─▶ collision ─▶ haptics                                 │
            │  Qwen / WebLLM (WebGPU) ─▶ direction + chat (voice)                              │
            └───────┬───────────────────────────────────────────────────┬────────────────────┘
                    │ weak/unknown telemetry                              │ "what is this?"
                    ▼                                                     ▼
              backend (hostable) ── Bright Data harvest ──▶ RunPod GPU (RSI training only)
                    ▲                                                     │
                    └──────────────── new detector weights (/models) ◀────┘
```

Design principle: **runs with zero keys/GPU**. No model published yet or no WebGPU? The
edge falls back to mock detection and a mock LLM so the whole flow still demos.

## Quick start

```bash
cd pathfinder
./scripts/run_local.sh
```

Open `http://localhost:8000` on a desktop with WebGPU (Chrome), or
`http://<your-laptop-ip>:8000` on an Android Chrome phone on the same Wi-Fi. Tap **Start
navigation**. The first load downloads the in-browser models (LLM ~0.9 GB, cached after).

> **Requirements:** WebGPU (Chrome/Edge desktop, Chrome on Android) for real models; other
> browsers get the mock fallback. Camera + `navigator.vibrate` need a secure context —
> `localhost` is fine; over a LAN IP use an HTTPS tunnel (cloudflared/ngrok). iOS Safari
> has no Vibration API (narration still works).

Tests: `source backend/.venv/bin/activate && pytest tests/ -q`

## Going live

| Capability | Where | How to enable |
|---|---|---|
| LLM (direction + chat) | **Edge** (WebLLM) | on by default; needs WebGPU |
| Detection + depth | **Edge** (WebGPU) | depth works out-of-box; detection uses `backend/models/detector.onnx` — run `scripts/export_detector.py` (else mock) |
| Web identification | Backend → Bright Data | set `PATHFINDER_BRIGHTDATA_API_KEY` |
| RSI training | RunPod GPU | set `PATHFINDER_RUNPOD_API_KEY` + `PATHFINDER_RSI_TRAINER_URL` (see [RUNPOD.md](RUNPOD.md)) |

## Host the backend

```bash
docker compose up --build      # serves the web app + APIs on :8000
```

Stateless except the `data/` volume (RSI ledger + harvested shards) and `backend/models/`
(published detector weights).

## Layout

```
frontend/src/   detector.js (YOLO/WebGPU)  depth.js (WebGPU)  llm.js (WebLLM)
                fusion.js  collision.js  rsi.js (telemetry+identify)  app.js (loop, voice, haptics)
backend/app/    main.py (serve + APIs)  brightdata/ (search+harvest)  rsi/ (loop+evaluator)  config.py
runpod/rsi/     handler.py + Dockerfile  (GPU fine-tune job, published to ghcr via CI)
backend/models/ manifest.json (+ detector.onnx, RSI-published)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow.
