// Direct JS port of SixthSense's perception math (Decoders.kt DepthDecoder + SceneAssembler,
// BeltMapper.kt, and VoiceAgent.describeScene). Same algorithm, same constants → same behavior.
//
// Depth grid is INVERSE relative depth (larger = closer). Detection boxes arrive in the
// processing-canvas pixel space; zones are thirds of that width.

// --- BeltMapper constants (exact) ---
const NEAR_THRESHOLD = 0.55;        // depth nearness below this => no buzz
const OBJECT_NEAR_THRESHOLD = 0.45; // object nearness below this => no buzz
const OBJECT_MIN = 90;              // floor buzz for a detected object past threshold
const LOW_CONF = 0.4;
const CURB_CENTER_MIN = 180;
const CAUTION_CENTER = 80;
const CLEAR_HUM = 30;
const PCT = 0.9;                    // 90th percentile per zone band

function zoneForCenterX(cx, w) {
  if (cx < w / 3) return "left";
  if (cx < (2 * w) / 3) return "center";
  return "right";
}

function bandPercentile(depth, w, r0, r1, c0, c1, pct) {
  const vals = [];
  for (let r = r0; r < r1; r++) {
    const base = r * w;
    for (let c = c0; c < c1; c++) vals.push(depth[base + c]);
  }
  if (!vals.length) return 0;
  vals.sort((a, b) => a - b);
  const idx = Math.min(Math.max(Math.floor((vals.length - 1) * pct), 0), vals.length - 1);
  return vals[idx];
}

function frameRange(depth, w, h) {
  let lo = Infinity, hi = -Infinity;
  const r0 = Math.floor(h / 3); // walking space = lower two-thirds
  for (let r = r0; r < h; r++) {
    const base = r * w;
    for (let c = 0; c < w; c++) {
      const v = depth[base + c];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo > hi ? [0, 1] : [lo, hi];
}

function detectCurbAhead(depth, w, h, gap = 6, spikeRatio = 0.18) {
  const c0 = Math.floor(w / 3), c1 = Math.floor((2 * w) / 3);
  const r0 = Math.floor((h * 2) / 3), r1 = h - gap;
  if (r1 <= r0) return false;
  let maxGrad = 0, scaleSum = 0, scaleN = 0;
  for (let r = r0; r < r1; r++) {
    let rowGrad = 0, n = 0;
    const baseA = r * w, baseB = (r + gap) * w;
    for (let c = c0; c < c1; c++) {
      rowGrad += Math.abs(depth[baseA + c] - depth[baseB + c]);
      scaleSum += depth[baseA + c]; scaleN++; n++;
    }
    if (n > 0) maxGrad = Math.max(maxGrad, rowGrad / n);
  }
  const scale = scaleN > 0 ? Math.abs(scaleSum / scaleN) : 1;
  return maxGrad / (scale < 1e-6 ? 1 : scale) > spikeRatio;
}

function toZones(depth, w, h) {
  const rowStart = Math.floor(h / 3);
  const third = Math.floor(w / 3);
  const leftP = bandPercentile(depth, w, rowStart, h, 0, third, PCT);
  const centerP = bandPercentile(depth, w, rowStart, h, third, 2 * third, PCT);
  const rightP = bandPercentile(depth, w, rowStart, h, 2 * third, w, PCT);
  const lo = Math.min(leftP, centerP, rightP);
  const hi = Math.max(leftP, centerP, rightP);
  const span = hi - lo < 1e-6 ? 1 : hi - lo;
  const norm = (v) => Math.min(Math.max((v - lo) / span, 0), 1);
  return {
    left: norm(leftP),
    center: norm(centerP),
    right: norm(rightP),
    curbAhead: detectCurbAhead(depth, w, h),
  };
}

function nearnessInBox(depth, w, h, bx1, by1, bx2, by2, lo, hi) {
  const c0 = Math.min(Math.max(Math.floor(bx1), 0), w - 1);
  const c1 = Math.min(Math.max(Math.floor(bx2), c0 + 1), w);
  const r0 = Math.min(Math.max(Math.floor(by1), 0), h - 1);
  const r1 = Math.min(Math.max(Math.floor(by2), r0 + 1), h);
  const p = bandPercentile(depth, w, r0, r1, c0, c1, PCT);
  const span = hi - lo < 1e-6 ? 1 : hi - lo;
  return Math.min(Math.max((p - lo) / span, 0), 1);
}

function intensity(v) {
  if (v < NEAR_THRESHOLD) return 0;
  return Math.min(Math.max(Math.round(((v - NEAR_THRESHOLD) / (1 - NEAR_THRESHOLD)) * 255), 0), 255);
}
function objectIntensity(n) {
  if (n < OBJECT_NEAR_THRESHOLD) return 0;
  const scaled = Math.round(((n - OBJECT_NEAR_THRESHOLD) / (1 - OBJECT_NEAR_THRESHOLD)) * 255);
  return Math.min(Math.max(scaled, OBJECT_MIN), 255);
}

// detections: [{label, confidence, bbox:{x1,y1,x2,y2}}] in proc-canvas pixels.
// depth: {data, width, height} raw inverse depth. procW/procH = detection coord space.
export function analyze(detections, depth, procW, procH) {
  const dw = depth.width, dh = depth.height;
  const zones = toZones(depth.data, dw, dh);
  const [lo, hi] = frameRange(depth.data, dw, dh);
  const mx = dw / procW, my = dh / procH;

  const objects = detections.map((d) => {
    const cx = (d.bbox.x1 + d.bbox.x2) / 2;
    const nearness = nearnessInBox(
      depth.data, dw, dh,
      d.bbox.x1 * mx, d.bbox.y1 * my, d.bbox.x2 * mx, d.bbox.y2 * my, lo, hi,
    );
    return { label: d.label, zone: zoneForCenterX(cx, procW), nearness, conf: d.confidence, bbox: d.bbox };
  });
  objects.sort((a, b) => b.nearness - a.nearness);

  const conf = objects.length ? Math.max(...objects.map((o) => o.conf)) : 0.6;

  // BeltMapper.packetAsInts
  let l = intensity(zones.left), c = intensity(zones.center), r = intensity(zones.right);
  for (const o of objects) {
    const oi = objectIntensity(o.nearness);
    if (!oi) continue;
    if (o.zone === "left") l = Math.max(l, oi);
    else if (o.zone === "right") r = Math.max(r, oi);
    else c = Math.max(c, oi);
  }

  // Danger (red) = any confirmed obstacle past threshold, computed from the raw thresholds.
  const danger = zones.curbAhead ||
    zones.left >= NEAR_THRESHOLD || zones.center >= NEAR_THRESHOLD || zones.right >= NEAR_THRESHOLD ||
    objects.some((o) => o.nearness >= OBJECT_NEAR_THRESHOLD);

  let pattern = 0;
  if (zones.curbAhead) { pattern = 2; c = Math.max(c, CURB_CENTER_MIN); }
  if (conf < LOW_CONF) { pattern = 1; c = Math.max(c, CAUTION_CENTER); l = 0; r = 0; }

  const pathClear = !danger;
  if (l === 0 && c === 0 && r === 0 && pattern === 0 && pathClear) {
    l = CLEAR_HUM; c = CLEAR_HUM; r = CLEAR_HUM;
  }

  const dominant = l >= c && l >= r ? "left" : r >= c ? "right" : "center";
  return {
    zones, objects, conf, danger, pathClear,
    packet: { left: l, center: c, right: r, pattern },
    dominant,
    announce: describe(zones, objects),
  };
}

// Port of VoiceAgent.describeScene, naming the actual detected object(s) "in front".
function describe(zones, objects) {
  if (zones.curbAhead) return "Curb or step ahead, slow down.";
  const inZone = (z) => objects.find((o) => o.zone === z && o.nearness >= OBJECT_NEAR_THRESHOLD);
  const name = (z, depthV) => (inZone(z)?.label) || (depthV >= NEAR_THRESHOLD ? "obstacle" : null);

  const center = name("center", zones.center);
  const left = name("left", zones.left);
  const right = name("right", zones.right);

  if (center) return `${cap(center)} ahead, move left or right.`;
  if (left && right) return "Obstacles on both sides, go straight slowly.";
  if (left) return `${cap(left)} on your left, stay right.`;
  if (right) return `${cap(right)} on your right, stay left.`;
  return "Obstacle nearby, proceed carefully.";
}

function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
