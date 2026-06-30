import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.webgpu.bundle.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";

const COCO_FALLBACK = [
  "person", "chair", "door", "table", "couch", "potted plant",
  "bicycle", "car", "dog", "bottle", "backpack", "tv",
];

// On-device object detection. Three tiers, chosen at load time:
//   1. "onnx"         — RSI-trained YOLOv8 from /models/manifest.json (onnxruntime-web/WebGPU)
//   2. "transformers" — Xenova/yolos-tiny from CDN (transformers.js/WebGPU); real detection
//                       out-of-the-box, no weights to host. Used until RSI publishes an ONNX.
//   3. "mock"         — random boxes, only if WebGPU is unavailable, so the app still runs.
export class Detector {
  constructor() {
    this.mode = "mock";
    this.session = null;
    this.tjs = null;
    this.labels = COCO_FALLBACK;
    this.size = 640;
    this.version = 0;
    this.confThreshold = 0.35;
    this.iouThreshold = 0.45;
  }

  async load() {
    if (await this._tryOnnx()) this.mode = "onnx";
    else if (await this._tryTransformers()) this.mode = "transformers";
    else this.mode = "mock";
    return this.mode;
  }

  async _tryOnnx() {
    try {
      const m = await fetch("/models/manifest.json").then((r) => r.json());
      const head = await fetch(m.url, { method: "HEAD" });
      if (!head.ok) return false; // no RSI-published weights yet
      this.labels = m.labels?.length ? m.labels : COCO_FALLBACK;
      this.size = m.input_size || 640;
      this.version = m.version || 0;
      this.session = await ort.InferenceSession.create(m.url, { executionProviders: ["webgpu", "wasm"] });
      return true;
    } catch (e) {
      console.warn("[detector] no onnx:", e.message);
      return false;
    }
  }

  async _tryTransformers() {
    try {
      const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2");
      env.allowLocalModels = false;
      this.tjs = await pipeline("object-detection", "Xenova/yolos-tiny", { device: "webgpu" });
      return true;
    } catch (e) {
      console.warn("[detector] no transformers.js:", e.message);
      return false;
    }
  }

  async detect(canvas) {
    if (this.mode === "onnx") return this._detectOnnx(canvas);
    if (this.mode === "transformers") return this._detectTransformers(canvas);
    return this._mock(canvas);
  }

  async _detectTransformers(canvas) {
    const url = canvas.convertToBlob
      ? URL.createObjectURL(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 }))
      : canvas.toDataURL("image/jpeg", 0.7);
    const res = await this.tjs(url, { threshold: 0.3 });
    if (url.startsWith("blob:")) URL.revokeObjectURL(url);
    return res.map((r) => ({
      label: r.label,
      confidence: r.score,
      bbox: { x1: r.box.xmin, y1: r.box.ymin, x2: r.box.xmax, y2: r.box.ymax },
    }));
  }

  async _detectOnnx(canvas) {
    const { tensor, scale, padX, padY } = this._preprocess(canvas);
    const feeds = {};
    feeds[this.session.inputNames[0]] = tensor;
    const out = await this.session.run(feeds);
    return this._postprocess(out[this.session.outputNames[0]], scale, padX, padY);
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
    const [, ch, n] = t.dims;
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
