import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2";

env.allowLocalModels = false;

// On-device monocular depth via WebGPU (Depth Anything V2 small). Returns a metric-ish
// depth grid (meters, smaller = closer). Falls back to a floor-plane heuristic.
export class Depth {
  constructor() {
    this.pipe = null;
    this.mock = true;
  }

  async load() {
    try {
      this.pipe = await pipeline("depth-estimation", "onnx-community/depth-anything-v2-small", {
        device: "webgpu",
      });
      this.mock = false;
    } catch (e) {
      console.warn("[depth] mock mode:", e.message);
      this.mock = true;
    }
    return !this.mock;
  }

  async estimate(canvas) {
    if (this.mock) return this._mock(canvas);
    const url = canvas.convertToBlob
      ? URL.createObjectURL(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 }))
      : canvas.toDataURL("image/jpeg", 0.7);
    const out = await this.pipe(url);
    if (url.startsWith("blob:")) URL.revokeObjectURL(url);

    const t = out.predicted_depth;
    const dims = t.dims;
    const h = dims[dims.length - 2];
    const w = dims[dims.length - 1];
    const raw = t.data;

    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < raw.length; i++) {
      if (raw[i] < lo) lo = raw[i];
      if (raw[i] > hi) hi = raw[i];
    }
    const span = hi - lo || 1;
    // Depth Anything outputs larger = closer; invert and scale to ~0.5-8 m.
    const meters = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      meters[i] = 0.5 + (1 - (raw[i] - lo) / span) * 7.5;
    }
    return { data: meters, width: w, height: h };
  }

  _mock(canvas) {
    const w = 64;
    const h = 64;
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const m = 0.6 + (1 - y / h) * 5.4; // bottom of frame is nearer
      for (let x = 0; x < w; x++) data[y * w + x] = m;
    }
    return { data, width: w, height: h };
  }
}
