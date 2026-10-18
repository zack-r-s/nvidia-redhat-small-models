# v004 measured skill-routing conversations

This version tests a different control boundary on the unchanged 24-PV PyAT ring. The model learns to map an operator's orbit-centering request to `orbit_correct(plane, effort)`. The skill reads BPMs and corrector setpoints, uses the fixed v002 response matrix for a bounded least-squares solve, writes the selected correctors, and reads the result. The model does not generate kick values. `gentle`, `balanced`, and `aggressive` change the residual-versus-kick penalty; they are not new plant physics.

`build_v004.py` generated 432 train, 72 validation, and 72 test conversations from 192 fresh seeded disturbances. The effort labels are balanced across planes, prompt openings are selected independently of effort, and some balanced requests deliberately omit an effort preference. Each row contains a pre-action operator prompt, a structured skill call, the measured PyAT tool result, and a short measured response. `sources.jsonl` keeps the complete before/after PV states. `index.jsonl` maps SFT rows back to source IDs. All splits are seed-disjoint from each other and from the frozen `evals/mark-i/pyat-v002` cases. No GPT annotation was used. The manifest records source and output hashes.

The source response teacher was fitted from earlier v001 EPICS data and is checked against the fixed PV contract and source attempts at runtime. The local PyAT generator rejects any action that fails to reduce the requested plane's measured RMS. This is simulated data, not a live Osprey trace.

Rebuild into an empty directory:

```sh
MPLCONFIGDIR=/private/tmp/hackring-mpl PYTHONPATH=src .venv/bin/python scripts/build_v004.py \
  --output /private/tmp/hackring-v004-rebuild
```
