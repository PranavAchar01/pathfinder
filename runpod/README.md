# RunPod — RSI trainer

A single serverless GPU endpoint that fine-tunes the on-device YOLO detector. No LLM or
per-frame inference runs here (those are on the edge).

```bash
cd runpod/rsi
docker build -t ghcr.io/pranavachar01/pathfinder-rsi:latest .   # or let CI build it
docker push ghcr.io/pranavachar01/pathfinder-rsi:latest
```

CI ([`.github/workflows/rsi-image.yml`](../.github/workflows/rsi-image.yml)) builds and
pushes this image on every change to `runpod/rsi/**`. The RunPod endpoint
(`pathfinder-rsi`) pulls it; make the ghcr package public so RunPod can pull without auth.

Input contract (sent by the backend RSI loop):

```json
{ "input": { "task": "finetune", "classes": ["scooter", ...], "shards": ["data/harvest/scooter.jsonl", ...] } }
```

Output: status + the path of the exported ONNX weights, which the backend publishes to its
`/models` manifest for the edge to pull. See [../RUNPOD.md](../RUNPOD.md) for live IDs.
