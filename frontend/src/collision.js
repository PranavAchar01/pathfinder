// Collision assessment -> directional haptic command + Web Vibration pattern. Runs on edge.

function intensity(distance, warn, stop) {
  if (distance >= warn) return 0;
  const span = Math.max(warn - stop, 0.1);
  const t = Math.min(Math.max((warn - distance) / span, 0), 1);
  return Math.round(90 + t * 165);
}

export function assess(objects, warn, stop) {
  const haptic = { left: 0, center: 0, right: 0, pattern: 0 };
  let nearest = Infinity;

  for (const o of objects) {
    const level = intensity(o.distance_m, warn, stop);
    if (level === 0) continue;
    nearest = Math.min(nearest, o.distance_m);
    haptic[o.side] = Math.max(haptic[o.side], level);
  }

  let risk;
  if (nearest <= stop) { risk = "stop"; haptic.pattern = 2; }
  else if (nearest < warn) { risk = "caution"; haptic.pattern = 1; }
  else { risk = "clear"; haptic.pattern = 0; }
  return { risk, haptic };
}

export function vibrate(haptic) {
  if (!navigator.vibrate) return;
  const peak = Math.max(haptic.left, haptic.center, haptic.right);
  if (peak === 0) return;
  if (haptic.pattern === 2) navigator.vibrate([120, 60, 120, 60, 120]);
  else if (haptic.pattern === 1) navigator.vibrate([200, 120, 200]);
  else navigator.vibrate(80 + peak);
}
