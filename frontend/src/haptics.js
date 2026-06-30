// Haptic output, in priority order:
//   1. Capacitor native Haptics plugin (real Taptic Engine) — works on iOS AND Android when
//      the app runs inside a Capacitor wrapper. This is the ONLY way to get iPhone haptics.
//   2. navigator.vibrate (Android web).
//   3. none -> fire() returns false, and the caller plays a directional audio cue (iOS web).
//
// Direction is a PULSE COUNT (SixthSense scheme): LEFT=1, CENTER=2, RIGHT=3; curb=2 hard.
const PULSE = 90, PULSE_GAP = 90, CURB_ON = 200, CURB_GAP = 120, CAUTION_MS = 150;

export function encodePattern(packet) {
  const l = packet.left | 0, c = packet.center | 0, r = packet.right | 0;
  const pat = packet.pattern | 0;
  if (l === 0 && c === 0 && r === 0) return null;
  if (pat === 1) return [CAUTION_MS];
  if (pat === 2) return [CURB_ON, CURB_GAP, CURB_ON];
  const max = Math.max(l, c, r);
  const count = max === l ? 1 : max === r ? 3 : 2;
  const out = [];
  for (let i = 0; i < count; i++) { out.push(PULSE); if (i < count - 1) out.push(PULSE_GAP); }
  return out;
}

function pulseCount(packet) {
  if (packet.pattern === 2) return 2;
  const max = Math.max(packet.left, packet.center, packet.right);
  return max === packet.left ? 1 : max === packet.right ? 3 : 2;
}

// Capacitor wrapper: window.Capacitor.Plugins.Haptics is injected natively (no build-time
// import needed). impact(style) hits the Taptic Engine; we repeat it for the pulse count.
function capacitorHaptics(packet) {
  const cap = window.Capacitor;
  const H = cap?.Plugins?.Haptics;
  if (!cap?.isNativePlatform?.() || !H) return false;
  const max = Math.max(packet.left, packet.center, packet.right);
  const style = max > 200 ? "HEAVY" : max > 120 ? "MEDIUM" : "LIGHT";
  const count = pulseCount(packet);
  for (let i = 0; i < count; i++) {
    setTimeout(() => { try { H.impact({ style }); } catch (_) {} }, i * 130);
  }
  return true;
}

// Returns true if real haptic output was produced; false if the caller should use audio.
export function fire(packet) {
  if (!packet) return false;
  const pattern = encodePattern(packet);
  if (!pattern) return false;
  if (capacitorHaptics(packet)) return true;            // native (iOS/Android wrapper)
  if (typeof navigator.vibrate === "function") { navigator.vibrate(pattern); return true; } // Android web
  return false;                                          // iOS web -> caller plays audio cue
}
