# Prompt for the frontend lane

Build out the existing Blueprint + React Three Fiber viewer in `frontend/` as a read-only Osprey panel. Start by reading `README.md`, `data/pv_contract.json`, `src/hackring/api.py`, and the current frontend code. Keep `hackring.state/1` as the shared state contract. The local API is the immediate development source; Osprey's EPICS-backed API mode should provide the same shape later.

Priorities: show four BPMs in both planes, corrector setpoint/readback values, goal thresholds, connection status, and recent state changes; improve the lattice geometry and camera controls; ensure small screens work. Build an Osprey URL-panel manifest and document how to mount it. The rendered orbit must come from measured BPM values or a documented simulator output; do not invent physical evolution in Three.js. Keep all controls outside this viewer. Run `npm run build` and include a screenshot or concise visual QA notes with the work.
