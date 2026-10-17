# Viewer QA notes (first pass)

Checked 2026-10-17 with headless Chromium (Playwright, SwiftShader WebGL) against the local PyAT API (`hackring.state/1`), not live Osprey.

- **Readings vs `/state`:** `scripts/qa_state_match.py` loads the viewer at 1440×900 and 390×844 and compares all 24 PV values rendered in the DOM (`data-testid="pv-<PV name>"`) to `GET /state` (tolerance 5e-5 display rounding): 0 mismatches. No horizontal overflow at either width.
- **Live:** status tag "Live (fixture)", source label says local PyAT fixture, not live Osprey.
- **Stale / disconnected:** API killed while open → "Stale" with dimmed readings after ~2 s (last sample age shown), "Disconnected" after ~8 s; fresh load with API down → "Disconnected", no data, no fabricated values.
- **Replay:** `/?replay=/replay/demo.json` plays 20 recorded, unmodified `/state` frames from a seeded (7) Xopt Nelder–Mead run; badge "Replay", run ID, seed and frame counter visible; no interpolation.
- **Unavailable by design:** agent intent, Osprey action status, target/goal show an explicit unavailable state (no backend contract).
- **Read-only:** outside `TestDriver.tsx` (dev-only, `?driver=1`), `/write` and `/scenario` appear nowhere in `frontend/src`, and `frontend/dist` contains neither. The viewer itself only issues `GET /state`.
- **Test driver:** 7 presets applied via the drop-down, rendered values matched `/state` each time (0 mismatches); the out-of-limit preset returned 422 and left `/state` unchanged; the panel is absent without `?driver=1`.
- Not covered: EPICS-backend mode through the browser (covered at API level by `tests/test_fixtures.py`), device click-inspection (manual), real-GPU WebGL rendering.

- **Live beam view:** per-BPM transverse x–y spot panels (shared scale, reference crosshair; latest dot plus the last 8 samples as unjoined faint dots, `SPOT_DOTS` in `BeamView.tsx`) and an RMS-vs-time strip. Connecting lines and 3D trails were tried and removed: lines between 0.5 s samples imply a path that was not measured, and Xopt's exploratory jumps made them unreadable. The 3D scene maps measured deviation into the translucent tube: its cross-section is a selectable display window (±0.5/1/2/5 mm, default ±2), with the equivalent magnification printed in the legend. Dots beyond the window are clamped to the tube wall and turn yellow; the tube is not the physical pipe. Verified with the large-disturbance preset: all dots inside at ±2 mm, three yellow off-scale at ±0.5 mm. Tables stay in true units.

- **Poll ordering (PR #1 review):** `/state` is now polled serially (next request scheduled after the previous settles, in-flight request aborted on unmount) with a sequence guard. `scripts/qa_poll_order.py` delays replies in the browser (slow first, then fast) and checks that at most one request is in flight and the displayed snapshot never goes backwards: old code showed 3–4 backwards steps and 2 in flight, new code 0 and 1.
