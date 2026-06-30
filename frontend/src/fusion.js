// Fuse detections (source-pixel boxes) with the depth grid into scene objects that carry
// distance, side, and clock position. Pure geometry, runs on the edge each frame.

function sideAndClock(cx, frameW) {
  const frac = cx / Math.max(frameW, 1);
  let side;
  if (frac < 0.4) side = "left";
  else if (frac > 0.6) side = "right";
  else side = "center";

  let clock;
  if (frac <= 0.5) clock = Math.round(9 + (frac / 0.5) * 3); // 9..12
  else clock = Math.round(((frac - 0.5) / 0.5) * 3); // 12..3
  if (clock === 0) clock = 12;
  return { side, clock };
}

function sampleDistance(depth, bbox, frameW, frameH) {
  const sx = depth.width / frameW;
  const sy = depth.height / frameH;
  const x1 = Math.max(0, Math.floor(bbox.x1 * sx));
  const x2 = Math.min(depth.width, Math.ceil(bbox.x2 * sx));
  const y1 = Math.max(0, Math.floor(bbox.y1 * sy));
  const y2 = Math.min(depth.height, Math.ceil(bbox.y2 * sy));
  const vals = [];
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) vals.push(depth.data[y * depth.width + x]);
  }
  if (!vals.length) return 5;
  vals.sort((a, b) => a - b);
  // Nearest 15th percentile — the closest part of the object is what can hit you.
  return vals[Math.floor(vals.length * 0.15)];
}

export function fuse(detections, depth, frameW, frameH) {
  const objects = detections.map((d) => {
    const cx = (d.bbox.x1 + d.bbox.x2) / 2;
    const { side, clock } = sideAndClock(cx, frameW);
    return {
      label: d.label,
      confidence: d.confidence,
      bbox: d.bbox,
      distance_m: Math.round(sampleDistance(depth, d.bbox, frameW, frameH) * 100) / 100,
      side,
      clock,
      known: d.confidence >= 0.5,
    };
  });
  objects.sort((a, b) => a.distance_m - b.distance_m);
  return objects;
}
