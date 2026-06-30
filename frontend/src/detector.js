import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.webgpu.bundle.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";

const COCO_FALLBACK = [
  "person", "chair", "door", "table", "couch", "potted plant",
  "bicycle", "car", "dog", "bottle", "backpack", "tv",
];

// On-device YOLOv8 object detection via WebGPU. Weights come from /models/manifest.json,
// which the RSI trainer republishes as the detector improves. Falls back to mock boxes if
// no model is published yet or WebGPU is unavailable, so the app always runs.
export class Detector {
  constructor() {
    this.session = null;
    this.labels = COCO_FALLBACK;
    this.size = 640;
    this.version = 0;
    this.mock = true;
    this.confThreshold = 0.35;
    this.iouThreshold = 0.45;
  }

  async load() {
    try {
      const m = await fetch("/models/manifest.json").then((r) => r.json());
      this.labels = m.labels?.length ? m.labels : COCO_FALLBACK;
      this.size = m.input_size || 640;
      this.version = m.version || 0;
      const head = await fetch(m.url, { method: "HEAD" });
      if (!head.ok) throw new Error("detector weights not published yet");
      this.session = await ort.InferenceSession.create(m.url, {
        executionProviders: ["webgpu", "wasm"],
      });
      this.mock = false;
    } catch (e) {
      console.warn("[detector] mock mode:", e.message);
      this.mock = true;
    }
    return !this.mock;
  }

  async detect(canvas) {
    if (this.mock) return this._mock(canvas);
    const { tensor, scale, padX, padY } = this._preprocess(canvas);
    const feeds = {};
    feeds[this.session.inputNames[0]] = tensor;
    const out = await this.session.run(feeds);
    const t = out[this.session.outputNames[0]];
    return this._postprocess(t, scale, padX, padY);
  }

  _preprocess(canvas) {
    const s = this.size;
    const scale = Math.min(s / canvas.width, s / canvas.height);
    const nw = Math.round(canvas.width * scale);
    const nh = Math.round(canvas.height * scale);
    const padX = (s - nw) / 2;
    const padY = (s - nh) / 2;

    const off = new OffscreenCanvas(s, s);
    const ctx = off.getContext("2d");
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, s, s);
    ctx.drawImage(canvas, padX, padY, nw, nh);
    const { data } = ctx.getImageData(0, 0, s, s);

    const chw = new Float32Array(3 * s * s);
    const plane = s * s;
    for (let i = 0; i < plane; i++) {
      chw[i] = data[i * 4] / 255;
      chw[plane + i] = data[i * 4 + 1] / 255;
      chw[2 * plane + i] = data[i * 4 + 2] / 255;
    }
    return { tensor: new ort.Tensor("float32", chw, [1, 3, s, s]), scale, padX, padY };
  }

  _postprocess(t, scale, padX, padY) {
    const d = t.data;
    const [, ch, n] = t.dims; // [1, 4+numClasses, 8400]
    const numClasses = ch - 4;
    const boxes = [];
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestC = 0;
      for (let c = 0; c < numClasses; c++) {
        const v = d[(4 + c) * n + i];
        if (v > best) { best = v; bestC = c; }
      }
      if (best < this.confThreshold) continue;
      const cx = d[i];
      const cy = d[n + i];
      const w = d[2 * n + i];
      const h = d[3 * n + i];
      boxes.push({
        label: this.labels[bestC] ?? String(bestC),
        confidence: best,
        bbox: {
          x1: (cx - w / 2 - padX) / scale,
          y1: (cy - h / 2 - padY) / scale,
          x2: (cx + w / 2 - padX) / scale,
          y2: (cy + h / 2 - padY) / scale,
        },
      });
    }
    return this._nms(boxes);
  }

  _nms(boxes) {
    boxes.sort((a, b) => b.confidence - a.confidence);
    const keep = [];
    for (const b of boxes) {
      if (keep.every((k) => this._iou(k.bbox, b.bbox) < this.iouThreshold)) keep.push(b);
    }
    return keep;
  }

  _iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
  }

  _mock(canvas) {
    const w = canvas.width;
    const h = canvas.height;
    const out = [];
    const count = 1 + Math.floor(Math.abs(Math.sin(performance.now() / 1000)) * 2);
    for (let i = 0; i < count; i++) {
      const bw = (0.15 + 0.2 * ((i + 1) / count)) * w;
      const bh = 0.4 * h;
      const x1 = (0.1 + 0.3 * i) * w;
      const y1 = 0.3 * h;
      out.push({
        label: COCO_FALLBACK[(i + Math.floor(performance.now() / 1500)) % COCO_FALLBACK.length],
        confidence: 0.5 + 0.3 * Math.random(),
        bbox: { x1, y1, x2: x1 + bw, y2: y1 + bh },
      });
    }
    return out;
  }
}
