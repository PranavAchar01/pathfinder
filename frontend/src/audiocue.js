// Directional audio cue — the haptic substitute for iOS, where the browser cannot vibrate.
// Encodes the SAME belt packet as sound: stereo pan = direction (L/C/R), pitch + loudness =
// closeness, pulse count = zone (L=1, C=2, R=3), matching the haptic pulse-count scheme.
// Web Audio IS supported on iOS Safari (unlike the Vibration API).

let actx = null;

// Must be called inside a user gesture (START tap) — iOS starts AudioContext suspended.
export function unlockAudio() {
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === "suspended") actx.resume();
  } catch (_) {}
}

export function cue(packet) {
  if (!actx || actx.state !== "running") return false;
  const max = Math.max(packet.left, packet.center, packet.right);
  if (max === 0) return false;

  const pan = max === packet.left ? -1 : max === packet.right ? 1 : 0;
  const urgency = Math.min(max / 255, 1);
  const count = packet.pattern === 2 ? 2 : max === packet.left ? 1 : max === packet.right ? 3 : 2;
  const freq = 320 + urgency * 540;      // closer obstacle -> higher pitch
  const dur = packet.pattern === 2 ? 0.13 : 0.07;

  for (let i = 0; i < count; i++) beep(freq, pan, dur, i * 0.13, urgency);
  return true;
}

function beep(freq, pan, dur, delay, gainScale) {
  const t = actx.currentTime + delay;
  const osc = actx.createOscillator();
  osc.type = "sine";
  osc.frequency.value = freq;
  const gain = actx.createGain();
  gain.gain.setValueAtTime(0.0001, t);
  gain.gain.linearRampToValueAtTime(0.12 + 0.5 * gainScale, t + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);

  let tail = gain;
  if (actx.createStereoPanner) {
    const panner = actx.createStereoPanner();
    panner.pan.value = pan;
    gain.connect(panner);
    tail = panner;
  }
  osc.connect(gain);
  tail.connect(actx.destination);
  osc.start(t);
  osc.stop(t + dur + 0.02);
}
