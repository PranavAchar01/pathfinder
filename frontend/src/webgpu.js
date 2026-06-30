// WebGPU preflight. Probes the adapter once at boot so the rest of the app knows whether
// real GPU inference is available, and surfaces the actual device to the user.

export async function probeWebGPU() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { ok: false, reason: "WebGPU unavailable — use Chrome/Edge (desktop or Android)" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { ok: false, reason: "no WebGPU adapter (GPU blocked or disabled)" };

    // adapter.info is the current API; requestAdapterInfo() is the older one.
    let info = adapter.info;
    if (!info && adapter.requestAdapterInfo) info = await adapter.requestAdapterInfo();
    info = info || {};
    const name = info.description || [info.vendor, info.architecture].filter(Boolean).join(" ") || "GPU";

    // Hold a device so the GPU stays warm and we confirm it actually initializes.
    const device = await adapter.requestDevice();
    return {
      ok: true,
      adapter,
      device,
      name,
      fp16: adapter.features.has("shader-f16"),
    };
  } catch (e) {
    return { ok: false, reason: `WebGPU init failed: ${e.message}` };
  }
}
