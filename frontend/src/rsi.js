// Edge -> backend signals: telemetry for RSI, and web identification of unknown objects.

export async function identify(query, labelGuess) {
  try {
    const r = await fetch("/api/identify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, label_guess: labelGuess || "" }),
    });
    return await r.json();
  } catch (e) {
    return { results: [], used_web: false };
  }
}

// Buffers weak/unknown detections and flushes them to the RSI loop in batches.
export class Telemetry {
  constructor(flushEvery = 40) {
    this.buffer = [];
    this.flushEvery = flushEvery;
  }

  observe(objects) {
    for (const o of objects) {
      if (!o.known || o.confidence < 0.5) {
        this.buffer.push({
          label: o.label,
          confidence: o.confidence,
          known: o.known,
          distance_m: o.distance_m,
        });
      }
    }
    if (this.buffer.length >= this.flushEvery) this.flush();
  }

  async flush() {
    if (!this.buffer.length) return;
    const items = this.buffer.splice(0, this.buffer.length);
    try {
      await fetch("/api/rsi/telemetry", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      });
    } catch (e) {
      /* offline: drop this batch, more will come */
    }
  }
}
