# RunPod — RSI training only

The LLM, detection, and depth all run on the **edge** (browser/WebGPU). RunPod's only job
is the heavy, periodic **Recursive Self-Improvement** training of the detector. The
endpoint is serverless with `workersMin=0`, so it **costs nothing when idle** — billing
accrues only while a fine-tune job runs.

## Live resource

| Resource | ID | Invoke URL |
|---|---|---|
| `pathfinder-rsi` (serverless GPU trainer) | `f1jndtovwo2rjs` | `https://api.runpod.ai/v2/f1jndtovwo2rjs/runsync` |
| Template `pathfinder-rsi` (`ghcr.io/pranavachar01/pathfinder-rsi:latest`) | `gim5kgbqcu` | — |

GPU pool (priority): **RTX 4090** → L40S → RTX A5000, 1 GPU, flashboot on.

`backend/.env` is wired:

```
PATHFINDER_RUNPOD_API_KEY=...
PATHFINDER_RSI_TRAINER_URL=https://api.runpod.ai/v2/f1jndtovwo2rjs/runsync
```

## How RSI uses it

The backend `RSILoop` submits `{"input":{"task":"finetune","classes":[...],"shards":[...]}}`
to the trainer when edge telemetry shows weak classes. `runpod/rsi/handler.py` fine-tunes
YOLO11 on the Bright-Data-harvested shards and exports ONNX, which is published back to the
backend `/models` manifest for the edge to pull.

## One step to activate the trainer image

The endpoint pulls `ghcr.io/pranavachar01/pathfinder-rsi:latest`, built by GitHub Actions
([`.github/workflows/rsi-image.yml`](.github/workflows/rsi-image.yml)) on push to `main`.
After the first build:

1. Make the ghcr package **public** (GitHub → Packages → `pathfinder-rsi` → settings →
   visibility), so RunPod can pull it without registry auth. *(Or add a RunPod
   container-registry credential.)*

Until the image is published the endpoint stays idle (no cost); the RSI loop logs
`queued_local` and still harvests data, so nothing breaks.

## Managing it

Resources were created via the RunPod REST API (`https://rest.runpod.io/v1`). The `runpod`
MCP server is registered (user scope) for conversational management in new Claude Code
sessions.
