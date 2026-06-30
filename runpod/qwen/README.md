# Qwen on RunPod (vLLM, OpenAI-compatible)

Pathfinder uses Qwen for two things: turning detected-object geometry into spoken
directional guidance, and the conversational agent. Both go through an OpenAI-compatible
endpoint, so the simplest deploy is RunPod's official **vLLM Serverless** worker.

## Option A — RunPod vLLM Serverless (recommended)

1. RunPod console → Serverless → **New Endpoint** → *vLLM* quick-deploy.
2. Model: `Qwen/Qwen2.5-0.5B-Instruct` (~0.9 GB — chosen for fast cold start and low
   compute time; bump to `Qwen/Qwen2.5-7B-Instruct` for higher quality if latency allows).
3. GPU: any 24 GB card (A5000/L4); the 0.5B model loads in seconds.
4. Copy the endpoint id and set in the backend `.env`:

   ```
   PATHFINDER_REASONING_BACKEND=runpod
   PATHFINDER_RUNPOD_API_KEY=...
   PATHFINDER_RUNPOD_QWEN_BASE_URL=https://api.runpod.ai/v2/<endpoint-id>/openai/v1
   PATHFINDER_RUNPOD_QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct
   ```

The backend's `QwenClient` calls `<base_url>/chat/completions` directly.

## Option B — Custom GPU Pod

Use the `Dockerfile` here to run vLLM on a dedicated pod, then expose the HTTP port and
point `PATHFINDER_RUNPOD_QWEN_BASE_URL` at `http://<pod-ip>:8000/v1`.
