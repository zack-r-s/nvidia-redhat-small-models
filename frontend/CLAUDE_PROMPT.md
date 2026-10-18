# Frontend handoff prompt for Claude

Continue the existing `frontend/` application for the shared local PyAT demo. The ring, PV names, and units are fixed by `data/pv_contract.json` and `data/facility/decks/RING.json`. The default `scripts/bootstrap_demo.sh` serves v004, the local PyAT fixture API, chat bridge, and viewer; chat and viewer share that **one fixture**. The viewer reads `/state`; the chat bridge handles model/tool interaction and returns measured turn evidence. Osprey/EPICS is a separate integration path under `scripts/bootstrap_demo.sh --osprey`. Do not let model prose directly change the display.

The demonstration is orbit correction on a 20-cell FODO storage ring: four BPMs measure X/Y positions and four horizontal plus four vertical correctors steer within ±0.35 mrad. The chat supports measured BPM reports, bounded exact setpoint requests through a labeled harness read → write → read action, and v004 model selection of `orbit_correct(plane, effort)`. The latter delegates numerical correction to a deterministic bounded solver; the model does not calculate kicks. Keep model intent, harness action, confirmed write, and measured result distinct in UI copy.

Read `README.md`, `data/pv_contract.json`, `src/hackring/api.py`, `src/hackring/chat_bridge.py`, `src/hackring/machine.py`, `frontend/src/types.ts`, and the current UI before changing it. Preserve `hackring.state/1` and the 24 PV names. Use `scripts/bootstrap_demo.sh` for the live shared fixture; `scripts/dev_up.sh` runs the scripted bridge without inference for frontend-only work. The `?driver=1` controls remain visibly development-only. `scripts/demo.sh --epics` is a caproto fixture on port 6064, not the Osprey accelerator.

## Product and data boundaries

- Blueprint handles status, numerical telemetry, orbit plots, and clear labels. React Three Fiber handles lattice/device geometry and the schematic orbit. Make the display understandable without EPICS vocabulary while retaining exact PV names and units in an expert view.
- Treat `/state` as authoritative. It provides circumference, element positions, PV values, and RMS summaries; it does **not** provide a continuous measured closed-orbit curve, a target threshold, agent events, or benchmark results. Show the four measured BPM samples as samples. A circular ring layout is schematic, not a physical floor plan.
- Separate the reference trajectory from measured orbit deviation. If you magnify a visual deviation, label the display scale; keep tables in true mm and mrad. Do not animate an inferred physical response between measurements.
- The Agent & goal panel uses the bridge's `turn` evidence: requested action, actual write outcome, measured before/after RMS, and whether the requested measured goal was met. “Unavailable” is still correct before any chat turn or after reset. A tool call is not itself proof that a setpoint changed.
- Keep the visualization read-only: its polling path issues `GET /state`, while the explicit chat component uses `/chat` and `/chat/reset`. It must not call `/write` or `/scenario`; those POST endpoints belong to the development driver.
- Preserve live, stale/disconnected, and replay states. Replay uses recorded immutable snapshots through the same display components and disables live chat and driver controls. The Osprey URL-panel snippet in `frontend/osprey-panel.yml` is optional and is not applied to `profile.yml`.

## Useful next work

Refine inspection and responsive behavior, especially the 3D ring scene on a real GPU. Keep all 24 displayed values synchronized with `/state` and preserve serialized polling. Do not add an animated particle or extrapolated 6D trajectory as part of this handoff; the current plant publishes discrete BPM measurements. Evaluation-result visuals should consume shared result records rather than inventing scores.

Run `npm run build`, `scripts/qa_state_match.py`, and `scripts/qa_shared_fixture_ui.py` before handing back UI changes. Pass `--chrome /path/to/Chrome` if Playwright Chromium is not installed. Check desktop and narrow layouts, disconnected and replay states, and the shared live fixture. `frontend/QA.md` records the current checks. Do not start training or data generation as part of this lane.
