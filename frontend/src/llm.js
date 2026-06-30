import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

const CLOCK_PHRASE = {
  9: "to your hard left", 10: "to your left", 11: "slightly left",
  12: "directly ahead", 1: "slightly right", 2: "to your right", 3: "to your hard right",
};

const GUIDE_SYS =
  "You are the spatial-reasoning core of a wearable navigation aid for a blind user. " +
  "Given detected objects with distance and clock position, pick the single most important " +
  "obstacle and give one short spoken cue (max 12 words). " +
  'Reply ONLY as JSON: {"guidance":"<cue>","priority_index":<int or null>}.';

const CHAT_SYS =
  "You are Pathfinder, a calm, concise voice assistant for a blind user. Answer in 1-3 " +
  "spoken sentences, grounded in the listed scene objects (distance + clock position). If web " +
  "context is provided about an unfamiliar object, use it to explain what it is and whether " +
  "it is a hazard.";

// The LLM runs fully in the browser (WebLLM / WebGPU). No network round-trip for inference.
export class LLM {
  constructor(model) {
    this.model = model;
    this.engine = null;
    this.mock = true;
  }

  async load(onProgress) {
    try {
      this.engine = await CreateMLCEngine(this.model, { initProgressCallback: onProgress });
      this.mock = false;
    } catch (e) {
      console.warn("[llm] mock mode:", e.message);
      this.mock = true;
    }
    return !this.mock;
  }

  async guide(objects) {
    if (!objects.length) return "Path is clear ahead.";
    if (this.mock) return this._fallbackGuide(objects);
    const payload = objects.slice(0, 6).map((o, i) => ({
      index: i, label: o.label, distance_m: o.distance_m, clock: o.clock, side: o.side,
    }));
    try {
      const r = await this.engine.chat.completions.create({
        messages: [
          { role: "system", content: GUIDE_SYS },
          { role: "user", content: `Objects: ${JSON.stringify(payload)}\nReturn the guidance JSON.` },
        ],
        temperature: 0.2,
        max_tokens: 80,
      });
      const txt = r.choices[0].message.content;
      const j = JSON.parse(txt.slice(txt.indexOf("{"), txt.lastIndexOf("}") + 1));
      return (j.guidance || "").trim() || this._fallbackGuide(objects);
    } catch (e) {
      return this._fallbackGuide(objects);
    }
  }

  async chat(text, scene, webContext) {
    if (this.mock) return this._fallbackChat(scene);
    const band = (n) => (n >= 0.7 ? "very near" : n >= 0.45 ? "near" : "far");
    const sceneText = scene?.objects?.length
      ? "Scene: " + scene.objects.slice(0, 8).map((o) =>
          `${o.label} (${band(o.nearness)}, ${o.zone})`).join("; ")
      : "Scene: nothing notable detected.";
    const messages = [
      { role: "system", content: CHAT_SYS },
      { role: "user", content: sceneText },
    ];
    if (webContext) messages.push({ role: "user", content: `Web context:\n${webContext}` });
    messages.push({ role: "user", content: text });
    try {
      const r = await this.engine.chat.completions.create({ messages, temperature: 0.4, max_tokens: 200 });
      return r.choices[0].message.content.trim();
    } catch (e) {
      return this._fallbackChat(scene);
    }
  }

  _fallbackGuide(objects) {
    const n = objects[0];
    return `${n.label} ${Math.round(n.distance_m)} meters ${CLOCK_PHRASE[n.clock] || "ahead"}.`;
  }

  _fallbackChat(scene) {
    if (!scene?.objects?.length) return "I don't detect anything notable right now.";
    const n = scene.objects[0];
    return `Nearest is a ${n.label} on your ${n.zone}.`;
  }
}
