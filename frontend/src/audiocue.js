// Directional audio "buzz" for iPhone. The Web Audio API is muted by iOS's ring/silent
// switch and is fragile to unlock, so this drives the beep through an HTMLAudioElement
// playing an encoded WAV — iOS treats that as media playback (routes to the speaker and
// unlocks reliably from one gesture). Direction is baked into the stereo channels:
// pan = zone (L/C/R), pitch = closeness, beep count = zone (L=1, C=2, R=3), curb = 2 tones.
// Same API as before: unlockAudio() (call in the START tap) + cue(packet).

const SR = 22050;
let audioEl = null;

function encodeWav(L, R) {
  const len = L.length, numCh = 2, dataSize = len * numCh * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + dataSize, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, numCh, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * numCh * 2, true);
  v.setUint16(32, numCh * 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, dataSize, true);
  let off = 44;
  for (let i = 0; i < len; i++) {
    for (const ch of [L, R]) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
  }
  let bin = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return "data:audio/wav;base64," + btoa(bin);
}

function buildTone(freq, pan, beeps) {
  const beepDur = 0.08, gap = 0.06;
  const dn = Math.floor(beepDur * SR), gn = Math.floor(gap * SR);
  const total = beeps * dn + (beeps - 1) * gn;
  const L = new Float32Array(total), R = new Float32Array(total);
  const lg = pan <= 0 ? 1 : 1 - pan;   // pan -1=left .. +1=right
  const rg = pan >= 0 ? 1 : 1 + pan;
  let idx = 0;
  for (let b = 0; b < beeps; b++) {
    for (let i = 0; i < dn; i++) {
      const t = i / SR;
      const env = Math.min(1, t / 0.005) * Math.min(1, (beepDur - t) / 0.02);
      const s = Math.sin(2 * Math.PI * freq * t) * 0.7 * env;
      L[idx] = s * lg; R[idx] = s * rg; idx++;
    }
    idx += gn;
  }
  return [L, R];
}

const SILENT = encodeWav(new Float32Array(Math.floor(0.05 * SR)), new Float32Array(Math.floor(0.05 * SR)));

// Call inside the START tap. iOS unlocks HTMLAudio from one user-gesture play().
export function unlockAudio() {
  if (!audioEl) {
    audioEl = new Audio();
    audioEl.setAttribute("playsinline", "");
    audioEl.preload = "auto";
    audioEl.volume = 1;
  }
  try { audioEl.src = SILENT; audioEl.play().catch(() => {}); } catch (_) {}
}

export function cue(packet) {
  if (!audioEl) return false;
  const max = Math.max(packet.left, packet.center, packet.right);
  if (max === 0) return false;
  const pan = max === packet.left ? -1 : max === packet.right ? 1 : 0;
  const urgency = Math.min(max / 255, 1);
  const beeps = packet.pattern === 2 ? 2 : max === packet.left ? 1 : max === packet.right ? 3 : 2;
  const freq = 320 + urgency * 540;
  const [L, R] = buildTone(freq, pan, beeps);
  try {
    audioEl.src = encodeWav(L, R);
    audioEl.play().catch(() => {});
    return true;
  } catch (_) {
    return false;
  }
}
