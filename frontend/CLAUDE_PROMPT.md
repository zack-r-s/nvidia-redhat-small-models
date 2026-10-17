# Frontend handoff prompt for Claude

Continue the existing `frontend/` application as the **read-only visual layer** for the Osprey/PyAT accelerator demo. The ring, PV names, and units are fixed by `data/pv_contract.json` and `data/facility/decks/RING.json`. LFM2.5-230M uses the custom Osprey web-terminal agent to propose controls MCP calls; the served PyAT virtual accelerator supplies the actual readings. The viewer gets those readings from the EPICS-backed FastAPI `/state` endpoint. The chat and viewer are separate tabs connected to the same served machine. Do not give the production viewer a write path or let model text directly change the display.

The longer-term demonstration is orbit correction on a 20-cell FODO storage ring: four BPMs measure X/Y positions and four horizontal plus four vertical correctors steer within ±0.35 mrad. The current chat supports explicit setpoint commands and measured BPM reports. It does **not** turn a vague target-orbit request into an optimization plan. Keep that distinction visible in any demo copy or UI. A later comparison may present stock, specialized, and compressed model runs against the same scenario; this frontend lane owns presentation, not training, inference, scoring, or machine control.

Read `README.md`, `data/pv_contract.json`, `src/hackring/api.py`, `src/hackring/machine.py`, `frontend/src/types.ts`, and the current UI before changing it. Preserve `hackring.state/1` and the 24 PV names. For live integration, start the served Osprey accelerator and `scripts/shared_demo.sh` as documented in the README. For frontend-only work, use `scripts/demo.sh --manual` or plain `scripts/demo.sh` for the local PyAT/Xopt fixture; that fixture is a separate machine and its driver controls must stay visibly development-only. `scripts/demo.sh --epics` is a caproto fixture on port 6064, not the Osprey accelerator.

## Product and data boundaries

- Blueprint handles status, numerical telemetry, orbit plots, and clear labels. React Three Fiber handles lattice/device geometry and the schematic orbit. Make the display understandable without EPICS vocabulary while retaining exact PV names and units in an expert view.
- Treat `/state` as authoritative. It provides circumference, element positions, PV values, and RMS summaries; it does **not** provide a continuous measured closed-orbit curve, a target threshold, agent events, or benchmark results. Show the four measured BPM samples as samples. A circular ring layout is schematic, not a physical floor plan.
- Separate the reference trajectory from measured orbit deviation. If you magnify a visual deviation, label the display scale; keep tables in true mm and mrad. Do not animate an inferred physical response between measurements.
- The current agent intent, action status, and target panels intentionally show “Unavailable.” Add them only after an agreed event or target contract exists. If a future unsafe-request demo is added, distinguish a model proposal, Osprey's reported outcome, and the measured machine state.
- Keep the viewer read-only: it may issue `GET /state` or consume a future one-way event stream, but it must not call `/write` or `/scenario`. Those POST endpoints support the local test driver, not the live panel.
- Preserve live, stale/disconnected, and replay states. Replay uses recorded immutable snapshots through the same display components and must remain clearly labeled. The current Osprey URL-panel snippet in `frontend/osprey-panel.yml` is optional and is not applied to `profile.yml`; the chat and viewer work in separate tabs.

## Useful next work

Refine inspection and responsive behavior, especially the 3D ring scene on a real GPU. Keep all 24 displayed values synchronized with `/state` and preserve serialized polling. If the team adds a target or agent event stream, agree on its schema with the backend teammate before rendering it; absent data should still say “Unavailable.” Evaluation-result visuals should consume shared result records rather than inventing scores.

Run `npm run build` and `scripts/qa_state_match.py` before handing back UI changes. Check desktop and narrow layouts, disconnected state, and the live Osprey-backed viewer when the shared demo is available. `frontend/QA.md` records the existing fixture and live checks. Do not start training or data generation as part of this lane.
