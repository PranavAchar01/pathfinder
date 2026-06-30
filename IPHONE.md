# iPhone haptics

iOS browsers (all WebKit) can't vibrate from the web — the Vibration API is unsupported and
the old `<input switch>` hack was patched in iOS 26.5. Pathfinder handles this two ways:

1. **Directional audio cue (automatic, web).** On iOS the red-alert feedback becomes a
   stereo-panned beep (`frontend/src/audiocue.js`): pan = direction (L/C/R), pitch/loudness =
   closeness, pulse count = zone (L=1, C=2, R=3) — the audio analog of the belt buzz. Plus the
   spoken alert. Nothing to build; works in Safari now.

2. **Real Taptic-Engine haptics via a Capacitor wrapper.** `frontend/src/haptics.js` already
   detects `window.Capacitor` and routes to the native `@capacitor/haptics` plugin — so the
   moment Pathfinder runs inside a Capacitor iOS app, obstacle alerts fire real impacts. No web
   code change needed.

## Build the iOS wrapper (needs macOS + Xcode + an Apple Developer account)

The simplest wrapper is a thin native shell that loads the live deployment and injects the
Haptics plugin:

```bash
mkdir pathfinder-ios && cd pathfinder-ios
npm init -y
npm i @capacitor/core @capacitor/cli @capacitor/ios @capacitor/haptics
npx cap init Pathfinder app.pathfinder.aid --web-dir=www
mkdir www && echo "redirecting…" > www/index.html   # placeholder; we load the live site
```

Edit `capacitor.config.json` to point at the deployed app:

```json
{
  "appId": "app.pathfinder.aid",
  "appName": "Pathfinder",
  "webDir": "www",
  "server": { "url": "https://pathfinder-ten-delta.vercel.app", "cleartext": false }
}
```

Then:

```bash
npx cap add ios
npx cap sync
npx cap open ios     # opens Xcode → set your Team/signing → Run on your iPhone
```

That's it — `haptics.js` sees `window.Capacitor.isNativePlatform()` and fires
`Haptics.impact({style})` (HEAVY/MEDIUM/LIGHT by closeness, repeated for the L/C/R pulse
count). The status line shows `native·haptics+audio`.

Notes:
- Camera + mic need `NSCameraUsageDescription` / `NSMicrophoneUsageDescription` in the iOS
  `Info.plist` (Xcode → Signing & Capabilities / Info).
- WebGPU works in the iOS WKWebView on recent iOS; if a device lacks it, the app falls back to
  mock detection + the audio cue, same as the web.
- To ship without the App Store, use a free provisioning profile (7-day) or TestFlight.
