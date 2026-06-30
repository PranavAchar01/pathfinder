const cam = document.getElementById("cam");
const overlay = document.getElementById("overlay");
const riskEl = document.getElementById("risk");
const narrationEl = document.getElementById("narration");
const replyEl = document.getElementById("reply");
const startBtn = document.getElementById("start");
const talkBtn = document.getElementById("talk");
const speakToggle = document.getElementById("speak");

const ctx = overlay.getContext("2d");
const grab = document.createElement("canvas");
const FPS = 4;

let ws = null;
let running = false;
let frameId = 0;
let lastScene = null;
let lastSpoken = "";

function wsUrl(path) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" }, width: 640, height: 480 },
    audio: false,
  });
  cam.srcObject = stream;
  await cam.play();
  overlay.width = cam.videoWidth;
  overlay.height = cam.videoHeight;
  grab.width = 320;
  grab.height = 240;
}

function captureFrame() {
  const g = grab.getContext("2d");
  g.drawImage(cam, 0, 0, grab.width, grab.height);
  return grab.toDataURL("image/jpeg", 0.6);
}

function speak(text) {
  if (!speakToggle.checked || !text || text === lastSpoken) return;
  lastSpoken = text;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.1;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function applyHaptic(haptic) {
  if (!navigator.vibrate) return;
  const peak = Math.max(haptic.left, haptic.center, haptic.right);
  if (peak === 0) return;
  if (haptic.pattern === 2) navigator.vibrate([120, 60, 120, 60, 120]);
  else if (haptic.pattern === 1) navigator.vibrate([200, 120, 200]);
  else navigator.vibrate(80 + peak);
}

function drawScene(scene) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const sx = overlay.width / 320;
  const sy = overlay.height / 240;
  ctx.lineWidth = 3;
  ctx.font = "16px system-ui";
  for (const o of scene.objects) {
    const color = o.distance_m < 1 ? "#b91c1c" : o.distance_m < 2 ? "#f59e0b" : "#22d3ee";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    const b = o.bbox;
    ctx.strokeRect(b.x1 * sx, b.y1 * sy, (b.x2 - b.x1) * sx, (b.y2 - b.y1) * sy);
    ctx.fillText(`${o.label} ${o.distance_m}m`, b.x1 * sx + 4, b.y1 * sy + 18);
  }
}

function renderScene(scene) {
  lastScene = scene;
  riskEl.textContent = scene.risk.toUpperCase();
  riskEl.className = scene.risk;
  narrationEl.textContent = scene.narration;
  drawScene(scene);
  applyHaptic(scene.haptic);
  if (scene.risk !== "clear") speak(scene.narration);
}

function loop() {
  if (!running || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "frame", frame_id: frameId++, image: captureFrame() }));
  setTimeout(loop, 1000 / FPS);
}

async function start() {
  await startCamera();
  ws = new WebSocket(wsUrl("/ws/navigate"));
  ws.onmessage = (e) => renderScene(JSON.parse(e.data));
  ws.onopen = () => { running = true; loop(); };
  startBtn.textContent = "Navigating…";
  startBtn.disabled = true;
}

async function ask(text) {
  replyEl.textContent = "…";
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, scene: lastScene }),
  });
  const data = await res.json();
  replyEl.textContent = data.reply + (data.used_web ? "  (web)" : "");
  lastSpoken = "";
  speak(data.reply);
}

function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    talkBtn.addEventListener("click", () => ask("What is around me?"));
    return;
  }
  const rec = new SR();
  rec.lang = "en-US";
  rec.onresult = (e) => ask(e.results[0][0].transcript);
  talkBtn.addEventListener("pointerdown", () => rec.start());
  talkBtn.addEventListener("pointerup", () => rec.stop());
}

startBtn.addEventListener("click", start);
setupVoice();
