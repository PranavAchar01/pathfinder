import { Detector } from "./detector.js";
import { Depth } from "./depth.js";
import { LLM } from "./llm.js";
import { fuse } from "./fusion.js";
import { assess, vibrate } from "./collision.js";
import { identify, Telemetry } from "./rsi.js";

const els = {
  cam: document.getElementById("cam"),
  overlay: document.getElementById("overlay"),
  risk: document.getElementById("risk"),
  narration: document.getElementById("narration"),
  reply: document.getElementById("reply"),
  start: document.getElementById("start"),
  talk: document.getElementById("talk"),
  speak: document.getElementById("speak"),
  status: document.getElementById("status"),
};

const ctx = els.overlay.getContext("2d");
const proc = document.createElement("canvas");
proc.width = 480;
proc.height = 360;
const pctx = proc.getContext("2d", { willReadFrequently: true });

const PROC_FPS = 4;
const GUIDE_MS = 1500;

let cfg = { collision_warn_m: 2, collision_stop_m: 1, llm_model: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC" };
let detector, depth, llm;
const telemetry = new Telemetry();

let running = false;
let lastScene = null;
let lastGuideAt = 0;
let lastSpoken = "";

function setStatus(msg) { els.status.textContent = msg; }

function speak(text) {
  if (!els.speak.checked || !text || text === lastSpoken) return;
  lastSpoken = text;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

async function boot() {
  try { cfg = await fetch("/api/config").then((r) => r.json()); } catch (e) { /* defaults */ }
  detector = new Detector();
  depth = new Depth();
  llm = new LLM(cfg.llm_model);

  setStatus("Loading on-device models…");
  const [det, dep] = await Promise.all([detector.load(), depth.load()]);
  setStatus(`detector:${det} · depth:${dep ? "webgpu" : "mock"} · loading LLM…`);
  // LLM is heaviest; load in background so navigation can start immediately.
  llm.load((p) => setStatus(`LLM ${(p.progress * 100 || 0).toFixed(0)}% ${p.text || ""}`.slice(0, 60)))
    .then((ok) => setStatus(ok ? "ready (LLM on edge)" : "ready (LLM mock — no WebGPU)"));
  els.start.disabled = false;
  els.start.textContent = "Start navigation";
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: 1280, height: 720 },
    audio: false,
  });
  els.cam.srcObject = stream;
  await els.cam.play();
  els.overlay.width = els.cam.videoWidth;
  els.overlay.height = els.cam.videoHeight;
}

function drawOverlay(objects) {
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  const sx = els.overlay.width / proc.width;
  const sy = els.overlay.height / proc.height;
  ctx.lineWidth = 3;
  ctx.font = "16px system-ui";
  for (const o of objects) {
    const color = o.distance_m < 1 ? "#b91c1c" : o.distance_m < 2 ? "#f59e0b" : "#22d3ee";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    const b = o.bbox;
    ctx.strokeRect(b.x1 * sx, b.y1 * sy, (b.x2 - b.x1) * sx, (b.y2 - b.y1) * sy);
    ctx.fillText(`${o.label} ${o.distance_m}m`, b.x1 * sx + 4, b.y1 * sy + 18);
  }
}

async function tick() {
  if (!running) return;
  const t0 = performance.now();
  pctx.drawImage(els.cam, 0, 0, proc.width, proc.height);

  const [dets, dmap] = await Promise.all([detector.detect(proc), depth.estimate(proc)]);
  const objects = fuse(dets, dmap, proc.width, proc.height);
  const { risk, haptic } = assess(objects, cfg.collision_warn_m, cfg.collision_stop_m);

  lastScene = { objects, risk };
  els.risk.textContent = risk.toUpperCase();
  els.risk.className = risk;
  drawOverlay(objects);
  vibrate(haptic);
  telemetry.observe(objects);

  if (performance.now() - lastGuideAt > GUIDE_MS) {
    lastGuideAt = performance.now();
    llm.guide(objects).then((g) => {
      els.narration.textContent = g;
      if (risk !== "clear") speak(g);
    });
  }

  const elapsed = performance.now() - t0;
  setTimeout(tick, Math.max(0, 1000 / PROC_FPS - elapsed));
}

async function start() {
  await startCamera();
  running = true;
  els.start.textContent = "Navigating…";
  els.start.disabled = true;
  tick();
}

async function ask(text) {
  els.reply.textContent = "…";
  let web = "";
  const unknown = lastScene?.objects?.find((o) => !o.known);
  if (/what is|what's this|identify|describe this|unfamiliar/i.test(text) || unknown) {
    const res = await identify(text, unknown?.label);
    if (res.results?.length) web = res.results.map((r) => `- ${r.title}: ${r.snippet}`).join("\n");
  }
  const reply = await llm.chat(text, lastScene, web);
  els.reply.textContent = reply + (web ? "  (web)" : "");
  lastSpoken = "";
  speak(reply);
}

function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    els.talk.addEventListener("click", () => ask("What is around me?"));
    return;
  }
  const rec = new SR();
  rec.lang = "en-US";
  rec.onresult = (e) => ask(e.results[0][0].transcript);
  rec.onstart = () => els.talk.classList.add("listening");
  rec.onend = () => els.talk.classList.remove("listening");
  const start = (e) => { e.preventDefault(); try { rec.start(); } catch (_) {} };
  const stop = (e) => { e.preventDefault(); try { rec.stop(); } catch (_) {} };
  els.talk.addEventListener("pointerdown", start);
  els.talk.addEventListener("pointerup", stop);
  els.talk.addEventListener("pointercancel", stop);
}

els.start.disabled = true;
els.start.addEventListener("click", start);
setupVoice();
boot();
