# Pathfinder

A wearable navigation aid for blind and low-vision users — **web only, no hardware**.
You mount a phone on your chest, open the web app in the browser, and Pathfinder:

- **sees** the world through the phone camera,
- **detects** objects (YOLO) and **estimates distance** (monocular depth model),
- **reasons** about *which direction* each obstacle is, using **Qwen**,
- **buzzes** the phone via the **Web Vibration API** as you approach collision distance
  (left / center / right intensity maps to where the obstacle is),
- **talks** with you — ask "what's around me?" and it answers from the live scene,
- **looks things up** — ask about an unfamiliar object and it searches the web via
  **Bright Data**, then explains what it is and whether it's a hazard,
- **improves itself** — a **Recursive Self-Improvement (RSI)** loop finds the classes it's
  weak on, harvests training data with Bright Data, and retrains on RunPod.

All heavy compute (YOLO, depth, Qwen) runs on **RunPod**. The backend is a single
hostable FastAPI service that also serves the web app. It runs end-to-end **with no GPU
and no API keys** in mock mode, so you can demo the full flow on a laptop.

```
phone browser ──frames(WS)──▶ backend ──▶ YOLO + depth (RunPod) ──▶ fusion (distance, side)
   ▲  ▲                          │                                      │
   │  └──haptic + narration──────┘            Qwen (RunPod) ◀──direction reasoning
   │                                              │
   └──voice Q&A──/api/chat──▶ conversation ◀──────┘──▶ Bright Data (web lookup)
                                                  │
                                       RSI loop ──┴──▶ Bright Data (harvest) ──▶ RunPod (retrain)
```

## Quick start (mock mode, no keys)

```bash
cd pathfinder
./scripts/run_local.sh
```

Open `http://localhost:8000` on your laptop, or `http://<your-laptop-ip>:8000` on a phone
on the same Wi-Fi (camera + vibration need a real device). Tap **Start navigation**.

> Camera and `navigator.vibrate` require a **secure context**. `localhost` is fine; over a
> LAN IP use an HTTPS tunnel (e.g. `cloudflared`/`ngrok`). iOS Safari does not implement
> the Vibration API — narration still works; haptics require Android/Chromium.

Run the tests:

```bash
source backend/.venv/bin/activate && pytest tests/ -q
```

## Going live

Set these in `backend/.env` (copied from [`.env.example`](.env.example)) and the matching
backend flips from mock to real:

| Capability | Env | Provider |
|---|---|---|
| Direction + chat | `PATHFINDER_REASONING_BACKEND=runpod` + `RUNPOD_*` | RunPod (Qwen/vLLM) |
| Detection + depth | `PATHFINDER_PERCEPTION_BACKEND=runpod` + `RUNPOD_PERCEPTION_URL` | RunPod |
| Web lookup + RSI harvest | `PATHFINDER_BRIGHTDATA_API_KEY` | Bright Data |

Deployment details: [`runpod/README.md`](runpod/README.md). You can also run perception
locally on a GPU box with `PATHFINDER_PERCEPTION_BACKEND=local` (needs `ultralytics`,
`torch`, `transformers`).

## Host the backend

```bash
docker compose up --build      # serves API + web app on :8000
```

The container is the whole deployable: stateless except the `data/` volume (RSI ledger +
harvested training shards).

## Layout

```
backend/app/
  perception/   detector.py (YOLO)  depth.py (depth)  fusion.py (distance + side + clock)
  reasoning/    qwen_client.py  direction.py (Qwen direction)  conversation.py (voice Q&A)
  brightdata/   client.py (web search)  ingest.py (training-data harvest)
  rsi/          loop.py (self-improvement)  evaluator.py (weakness detection)
  collision.py  pipeline.py  main.py (FastAPI + WS + static)
frontend/       index.html  app.js (camera, WS, vibrate, voice)  styles.css
runpod/         qwen/  perception/   (Dockerfiles + serverless handlers)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full data flow and design decisions.
