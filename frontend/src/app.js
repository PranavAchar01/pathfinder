import { Detector } from "./detector.js";
import { Depth } from "./depth.js";
import { LLM } from "./llm.js";
import { analyze } from "./scene.js";
import { fire as fireHaptics } from "./haptics.js";
import { cue as audioCue, unlockAudio } from "./audiocue.js";
import { identify, Telemetry } from "./rsi.js";
import { probeWebGPU } from "./webgpu.js";

const $ = (id) => document.getElementById(id);
const els = {
  cam: $("cam"), overlay: $("overlay"), state: $("state"), status: $("status"),
  narration: $("narration"), objects: $("objects"), objectsEmpty: $("objects-empty"),
  start: $("start"), talk: $("talk"), speak: $("speak"), reply: $("reply"),
};
const zoneEls = {};
document.querySelectorAll(".zone").forEach((z) => {
  zoneEls[z.dataset.zone] = { root: z, fill: z.querySelector(".zone__fill"), val: z.querySelector(".zone__val") };
});

const ctx = els.overlay.getContext("2d");
const proc = document.createElement("canvas");
proc.width = 480; proc.height = 360;
const pctx = proc.getContext("2d", { willReadFrequently: true });

const PROC_FPS = 4;
const ALERT_REPEAT_MS = 2500;   // don't re-speak the same alert faster than this
const HAPTIC_REPEAT_MS = 700;   // re-buzz cadence while danger persists

let cfg = { llm_model: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC" };
let detector, depth, llm;
const telemetry = new Telemetry();
let running = false, lastScene = null;
let lastAlertText = "", lastAlertAt = 0, lastHapticAt = 0;
let wakeLock = null;

const IS_PHONE = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
  (navigator.maxTouchPoints > 1 && matchMedia("(pointer:coarse)").matches);
const HAS_VIBRATE = typeof navigator.vibrate === "function";

function setStatus(m) { els.status.textContent = m; }

// Mobile browsers block speech + vibration until they fire inside a user gesture. Call this
// synchronously from the START tap (before any await) to unlock both for the session.
function unlockOutputs() {
  try {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance("Pathfinder ready");
    u.rate = 1.1;
    speechSynthesis.speak(u);            // primes/unlocks TTS within the gesture
    lastAlertText = "Pathfinder ready"; lastAlertAt = performance.now();
  } catch (_) {}
  try { if (HAS_VIBRATE) navigator.vibrate(30); } catch (_) {} // unlocks + confirms haptics
  unlockAudio(); // iOS: start the AudioContext for the directional audio cue (haptic substitute)
}

// Keep the screen awake during navigation (re-acquire when tab returns to foreground).
async function keepAwake() {
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (_) {}
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && running) keepAwake();
});

function speak(text, { force = false } = {}) {
  if (!els.speak.checked || !text) return;
  const now = performance.now();
  if (!force && text === lastAlertText && now - lastAlertAt < ALERT_REPEAT_MS) return;
  lastAlertText = text; lastAlertAt = now;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.12;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function speak(text, { force = false } = {}) {
  if (!els.speak.checked || !text) return;
  const now = performance.now();
  if (!force && text === lastAlertText && now - lastAlertAt < ALERT_REPEAT_MS) return;
  lastAlertText = text; lastAlertAt = now;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.12;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

async function boot() {
  try { cfg = await fetch("/api/config", { cache: "no-store" }).then((r) => r.json()); } catch (e) {}
  const gpu = await probeWebGPU();
  detector = new Detector(); depth = new Depth(); llm = new LLM(cfg.llm_model);

  if (!gpu.ok) {
    setStatus(`⚠ ${gpu.reason} — mock`);
    els.start.disabled = false; els.start.textContent = "Start (mock)";
    return;
  }
  setStatus(`WebGPU ✓ ${gpu.name} — loading…`);
  const [det, dep] = await Promise.all([detector.load(), depth.load()]);
  const detTag = det === "onnx" ? `${detector.modelName}(v${detector.version})`
    : det === "rfdetr" ? "rf-detr·m"
    : det === "transformers" ? "yolos-tiny"
    : det;
  const native = !!window.Capacitor?.isNativePlatform?.();
  const out = native ? "native·haptics+audio"
    : IS_PHONE ? (HAS_VIBRATE ? "phone·haptics+audio" : "iOS·directional-audio+voice")
    : "desktop·audio";
  setStatus(`${gpu.name} · detect:${detTag} · depth:${dep ? "v2" : "mock"} · ${out}`);
  llm.load().then(() => {}); // background, for HOLD TO SPEAK Q&A only
  els.start.disabled = false; els.start.textContent = IS_PHONE ? "START (enables haptics + audio)" : "START";
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: 1280, height: 720 }, audio: false,
  });
  els.cam.srcObject = stream;
  await els.cam.play();
  els.overlay.width = els.cam.videoWidth || 1280;
  els.overlay.height = els.cam.videoHeight || 720;
}

function drawOverlay(scene) {
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  const sx = els.overlay.width / proc.width;
  const sy = els.overlay.height / proc.height;
  ctx.lineWidth = 3;
  ctx.font = "bold 15px system-ui";
  ctx.textBaseline = "top";
  for (const o of scene.objects) {
    const near = o.nearness >= 0.45;
    const color = near ? "#c01525" : "#15803d";
    const x = o.bbox.x1 * sx, y = o.bbox.y1 * sy;
    const w = (o.bbox.x2 - o.bbox.x1) * sx, h = (o.bbox.y2 - o.bbox.y1) * sy;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const tag = `${o.label} ${(o.conf * 100).toFixed(0)}%`;
    const tw = ctx.measureText(tag).width + 10;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 20), tw, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, x + 5, Math.max(2, y - 18));
  }
}

function renderZones(zones) {
  for (const key of ["left", "center", "right"]) {
    const z = zoneEls[key]; if (!z) continue;
    const v = zones[key];
    z.fill.style.height = `${Math.round(v * 100)}%`;
    z.val.textContent = v.toFixed(2);
    z.root.classList.toggle("is-near", v >= 0.55);
  }
}

function renderObjects(objects) {
  if (!objects.length) { els.objects.innerHTML = ""; els.objectsEmpty.style.display = "block"; return; }
  els.objectsEmpty.style.display = "none";
  els.objects.innerHTML = objects.slice(0, 6).map((o) =>
    `<tr class="${o.nearness >= 0.45 ? "is-near" : ""}"><td>${o.label}</td><td>${o.zone}</td><td>${o.nearness.toFixed(2)}</td><td>${(o.conf * 100).toFixed(0)}%</td></tr>`
  ).join("");
}

function renderState(scene) {
  if (scene.danger) {
    els.state.className = "state state--red";
    els.state.textContent = "OBSTACLE";
  } else {
    els.state.className = "state state--green";
    els.state.textContent = "CLEAR";
  }
}

async function tick() {
  if (!running) return;
  const t0 = performance.now();
  pctx.drawImage(els.cam, 0, 0, proc.width, proc.height);

  const [dets, dmap] = await Promise.all([detector.detect(proc), depth.estimate(proc)]);
  const scene = analyze(dets, dmap, proc.width, proc.height);
  lastScene = scene;

  drawOverlay(scene);
  renderZones(scene.zones);
  renderObjects(scene.objects);
  renderState(scene);

  if (scene.danger) {
    els.narration.textContent = scene.announce;
    speak(scene.announce);                                  // RED -> read what's in front
    if (performance.now() - lastHapticAt > HAPTIC_REPEAT_MS) {
      const buzzed = fireHaptics(scene.packet);             // RED -> native/Android haptics
      if (!buzzed) audioCue(scene.packet);                  // iOS web -> directional audio cue
      lastHapticAt = performance.now();
    }
  } else {
    els.narration.textContent = "";
  }

  telemetry.observe(scene.objects.map((o) => ({
    label: o.label, confidence: o.conf, known: o.conf >= 0.5, distance_m: 0,
  })));

  const elapsed = performance.now() - t0;
  setTimeout(tick, Math.max(0, 1000 / PROC_FPS - elapsed));
}

async function start() {
  unlockOutputs();        // MUST run synchronously in the tap gesture to enable phone audio + haptics
  els.start.textContent = "RUNNING…";
  els.start.disabled = true;
  await startCamera();
  await keepAwake();
  running = true;
  tick();
}

async function ask(text) {
  els.reply.textContent = "…";
  let web = "";
  const unknown = lastScene?.objects?.find((o) => o.conf < 0.5);
  if (/what is|what's this|identify|describe this|unfamiliar/i.test(text) || unknown) {
    const res = await identify(text, unknown?.label);
    if (res.results?.length) web = res.results.map((r) => `- ${r.title}: ${r.snippet}`).join("\n");
  }
  const reply = await llm.chat(text, lastScene, web);
  els.reply.textContent = reply + (web ? "  (web)" : "");
  speak(reply, { force: true });
}

function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { els.talk.addEventListener("click", () => ask("What is in front of me?")); return; }
  const rec = new SR();
  rec.lang = "en-US";
  rec.onresult = (e) => ask(e.results[0][0].transcript);
  rec.onstart = () => els.talk.classList.add("listening");
  rec.onend = () => els.talk.classList.remove("listening");
  const s = (e) => { e.preventDefault(); try { rec.start(); } catch (_) {} };
  const t = (e) => { e.preventDefault(); try { rec.stop(); } catch (_) {} };
  els.talk.addEventListener("pointerdown", s);
  els.talk.addEventListener("pointerup", t);
  els.talk.addEventListener("pointercancel", t);
}

els.start.disabled = true;
els.start.addEventListener("click", start);
setupVoice();
boot();
