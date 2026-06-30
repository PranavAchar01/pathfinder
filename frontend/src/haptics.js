// Port of SixthSense DirectionalEncoding to the Web Vibration API. A phone has one motor,
// so direction is encoded by PULSE COUNT: LEFT=1, CENTER=2, RIGHT=3; CURB=2 hard buzzes;
// CAUTION=1 soft pulse. (Web vibrate has no amplitude control, so only the timing/rhythm
// carries over — exactly SixthSense's no-amplitude fallback path.)
const PULSE = 90, PULSE_GAP = 90, CURB_ON = 200, CURB_GAP = 120, CAUTION_MS = 150;

export function encodePattern(packet) {
  const l = packet.left | 0, c = packet.center | 0, r = packet.right | 0;
  const pat = packet.pattern | 0;
  if (l === 0 && c === 0 && r === 0) return null;
  if (pat === 1) return [CAUTION_MS]; // low-confidence caution, no direction
  if (pat === 2) return [CURB_ON, CURB_GAP, CURB_ON]; // curb / step

  const max = Math.max(l, c, r);
  let count;
  if (max === l) count = 1;       // LEFT
  else if (max === r) count = 3;  // RIGHT
  else count = 2;                 // CENTER

  const pattern = [];
  for (let i = 0; i < count; i++) {
    pattern.push(PULSE);
    if (i < count - 1) pattern.push(PULSE_GAP);
  }
  return pattern;
}

export function fire(packet) {
  if (!navigator.vibrate) return null;
  const p = encodePattern(packet);
  if (p) navigator.vibrate(p);
  return p;
}
