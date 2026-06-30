import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;

// On-device monocular depth via WebGPU — Depth-Anything-V2-Small, the EXACT model
// SixthSense uses. Returns the raw INVERSE-RELATIVE depth grid (LARGER = CLOSER),
// unnormalized; scene.js (port of SixthSense DepthDecoder) does the zone math.
// Falls back to a floor-plane heuristic that is also larger-at-bottom (= nearer).
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
    // Depth-Anything-V2 output is inverse relative depth: larger = closer. SixthSense's
    // DepthDecoder consumes exactly this, so pass it through raw.
    return { data: t.data, width: w, height: h };
  }

  _mock(canvas) {
    const w = 64;
    const h = 64;
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const near = 0.2 + (y / h) * 0.8; // inverse depth: bottom rows larger = nearer
      for (let x = 0; x < w; x++) data[y * w + x] = near;
    }
    return { data, width: w, height: h };
  }
}
