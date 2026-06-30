# RunPod deployment

All heavy compute runs on RunPod. Two endpoints:

| Endpoint | What it serves | Folder |
|---|---|---|
| **Qwen** | Direction reasoning + conversation (vLLM, OpenAI-compatible) | [`qwen/`](qwen/) |
| **Perception** | YOLO detection, monocular depth, and RSI fine-tune jobs | [`perception/`](perception/) |

## Perception endpoint

```bash
cd runpod/perception
docker build -t <registry>/pathfinder-perception .
docker push <registry>/pathfinder-perception
```

Create a RunPod Serverless endpoint from that image. Then in the backend `.env`:

```
PATHFINDER_PERCEPTION_BACKEND=runpod
PATHFINDER_RUNPOD_API_KEY=...
PATHFINDER_RUNPOD_PERCEPTION_URL=https://api.runpod.ai/v2/<endpoint-id>/runsync
```

The same endpoint accepts `{"task": "finetune", ...}` calls from the RSI loop, so the
self-improvement cycle retrains on the GPU that already hosts the detector.

## Qwen endpoint

See [`qwen/README.md`](qwen/README.md) — the fastest path is RunPod's vLLM quick-deploy.
