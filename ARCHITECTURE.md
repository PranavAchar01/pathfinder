# Architecture

## Components

| Layer | Where it runs | Responsibility |
|---|---|---|
| Web app (edge) | Phone browser | Camera capture, frame streaming, haptics, voice I/O, overlay |
| Backend | Anywhere hostable (laptop, VM, container) | Orchestration, collision logic, RSI, API/WS |
| Perception | RunPod GPU (or local) | YOLO detection + monocular depth |
| Reasoning | RunPod GPU (Qwen via vLLM) | Direction guidance + conversation |
| Web data | Bright Data | Object identification + training-data harvest |

There is **no native app and no microcontroller**. The "edge device" is a phone running a
web page; haptics use the Web Vibration API, so directionality is conveyed through pulse
*intensity and pattern* rather than separate left/right motors. The backend still computes
per-side intensities (`HapticCommand.left/center/right`) so a directional belt could be
added later without backend changes.

## Real-time navigation loop

1. `frontend/app.js` grabs a 320×240 JPEG from the rear camera ~4×/sec and sends it over
   `WS /ws/navigate`.
2. `pipeline.NavigationPipeline.process`:
   - `Detector.detect` → list of `Detection` (label, confidence, bbox) — YOLO.
   - `DepthEstimator.estimate` → per-pixel meters — depth model.
   - `fusion.fuse` → `SceneObject`s: samples the nearest decile of each bbox's depth for
     `distance_m`, and maps bbox center-x to `side` (left/center/right) and a `clock`
     position (9 = hard left, 12 = ahead, 3 = hard right).
   - `collision.assess` → `risk` (clear/caution/stop) + `HapticCommand`. Intensity ramps
     linearly from the warn threshold (2 m) to the stop threshold (1 m); pattern escalates.
   - `DirectionReasoner.guide` → **Qwen** receives the object geometry and returns one
     short spoken cue plus which object is the priority. Falls back to a deterministic
     phrase if Qwen is unavailable.
3. The `Scene` (objects + risk + haptic + narration) is returned; the browser draws the
   overlay, calls `navigator.vibrate(...)`, and speaks the narration when not "clear".

## Conversation + unknown-object lookup

`POST /api/chat` carries the user's utterance plus the latest `Scene`.
`ConversationAgent`:

- grounds answers in the live scene ("there's a chair 1.5 m to your right"),
- decides a **web lookup** is needed when the user asks "what is this / identify…" or when
  the scene contains a low-confidence ("unknown") object,
- calls `BrightDataSearch` (SERP API) and feeds the snippets to Qwen for a grounded,
  hazard-aware explanation.

The browser does speech-to-text (Web Speech API) and text-to-speech locally; only text
crosses the wire.

## Recursive Self-Improvement (RSI)

`rsi.RSILoop` observes every `Scene`. Every `RSI_REVIEW_EVERY` frames it runs a cycle:

1. `Evaluator` scans the buffer for **weak classes** — unknown labels and detections below
   `RSI_MIN_CONFIDENCE`.
2. For each weak class, `TrainingDataIngestor` uses **Bright Data** to harvest reference
   imagery + descriptions into `data/harvest/<label>.jsonl`.
3. `_trigger_finetune` submits a `{"task": "finetune", ...}` job to the **RunPod**
   perception endpoint — the same GPU that serves detection retrains on the new shards and
   publishes updated weights.
4. Every cycle is appended to `data/rsi_ledger.jsonl` for auditability.

This closes the loop: the system's blind spots in the field directly drive what it learns
next, without manual labeling.

## Backend selection

`config.Settings` has `perception_backend` and `reasoning_backend`, each `mock | local |
runpod`. Every client (`Detector`, `DepthEstimator`, `QwenClient`, `BrightDataSearch`)
checks its backend at call time, so a single env flip moves a capability from mock → cloud
with no code change. This is what lets the whole system run on a laptop for development and
demo, then run entirely on RunPod in production.

## Safety notes

- Mock depth is a floor-plane heuristic — **not** a real distance. Never rely on mock mode
  for actual navigation.
- Collision thresholds are conservative defaults; tune `COLLISION_WARN_M` / `STOP_M` per
  user gait and phone mounting height.
