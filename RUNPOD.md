# RunPod hosting

All heavy compute runs on RunPod serverless. Both endpoints use `workersMin=0`, so they
**scale to zero and cost nothing when idle** — billing only accrues while a request is
being served.

## Live resources

| Resource | Type | ID | OpenAI / invoke URL |
|---|---|---|---|
| `pathfinder-qwen` | Serverless (vLLM) | `7gyxvsvtfeiqkm` | `https://api.runpod.ai/v2/7gyxvsvtfeiqkm/openai/v1` |
| `pathfinder-perception` | Serverless (custom) | `f1jndtovwo2rjs` | `https://api.runpod.ai/v2/f1jndtovwo2rjs/runsync` |

| Template | ID |
|---|---|
| `pathfinder-qwen` (`runpod/worker-v1-vllm:v2.22.5`, `Qwen/Qwen2.5-0.5B-Instruct`) | `u830yqh4ih` |
| `pathfinder-perception` (`ghcr.io/<owner>/pathfinder-perception:latest`) | `gim5kgbqcu` |

GPU pool for both (priority order): **RTX 5090** (High stock → fast scheduling) → RTX 4090
→ L40S, 1 GPU, flashboot on. 5090 was chosen because every other fast card was Low stock,
which is what stalled worker scheduling on the first attempt.

## Qwen — ready now

`backend/.env` already points at it:

```
PATHFINDER_REASONING_BACKEND=runpod
PATHFINDER_RUNPOD_QWEN_BASE_URL=https://api.runpod.ai/v2/7gyxvsvtfeiqkm/openai/v1
PATHFINDER_RUNPOD_QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

Model is `Qwen2.5-0.5B-Instruct` (~0.9 GB), chosen so cold start and inference stay fast.
First request downloads weights (~30-60 s), then warm.

## Perception — one step to activate

The endpoint exists but pulls `ghcr.io/<owner>/pathfinder-perception:latest`, which is
published by GitHub Actions ([`.github/workflows/perception-image.yml`](.github/workflows/perception-image.yml))
on push to `main`. To go live:

1. Push to GitHub → Actions builds and pushes the image (native amd64).
2. Make the ghcr package public (once, so RunPod can pull without registry auth):
   GitHub → your profile → Packages → `pathfinder-perception` → Package settings →
   Change visibility → Public.  *(Or keep it private and add a RunPod container-registry
   credential.)*
3. Flip the backend:
   ```
   PATHFINDER_PERCEPTION_BACKEND=runpod
   ```
   (`PATHFINDER_RUNPOD_PERCEPTION_URL` is already set.)

The same endpoint also serves the RSI `finetune` task, so self-improvement retrains on the
GPU that hosts detection.

## Recreate / manage

Endpoints and templates were created via the RunPod REST API (`https://rest.runpod.io/v1`).
The `runpod` MCP server is also registered (user scope) and will be available in new
Claude Code sessions for managing these resources conversationally.
