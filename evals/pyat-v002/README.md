# Expanded frozen local-PyAT evaluation

The 56 cases comprise the eight v001 cases plus 48 newly frozen cases on seeds `20267000`–`20267047`. They cover BPM reports, plane comparisons, exact writes, and X, Y, and two-plane orbit corrections. All seeds are disjoint from v002/v003 training and validation data. The case file was frozen before the v002 fine-tune started.

The evaluator uses the checked-in lattice with seeded error kicks, a verified pre-read, literal/bounded write validation, measured after-state, and greedy Transformers CPU inference. The write tool name is forced for write tasks. Results therefore compare model behavior under this local harness, not full Osprey agent behavior. Model-specific reports and traces live in sibling directories; each run uses the same `cases.jsonl` checksum.

A bounded response-teacher check on the fixed ring reduced requested RMS on all 28 correction cases, so the correction prompts are physically feasible under this local plant.

The expanded v001 baseline passed 21/56: 5/9 BPM reports, 0/9 plane comparisons, 10/10 exact writes, and 6/28 corrections. The v002 mixed-data checkpoint passed 20/56: 7/9 BPM reports, 0/9 comparisons, 10/10 exact writes, and 3/28 corrections. All 28 correction proposals from each checkpoint were accepted as valid bounded writes. `comparison.json` records paired case outcomes, accepted writes, and measured aggregate RMS deltas; the plane-specific pass rule is stricter for two-plane requests.
