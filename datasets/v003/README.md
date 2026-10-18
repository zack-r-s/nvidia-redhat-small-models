# Mixed PyAT control conversations v003

This pass preserves the fixed 24-PV ring and response teacher. It combines all 640 canonical Osprey v002 conversations with 480 new local-PyAT conversations from 80 fresh seeds: 320 measured correction/exact-write actions and 160 read/compare examples. The split is 864 train, 128 validation, and 128 test conversations. No GPT request was made for this version; the new prompt variants are deterministic and the old canonical prompts are unchanged.

The local generator resets the checked-in ring with seeded error kicks, reads its PVs, uses the v002 response matrix to propose bounded setpoints, writes them through `LocalMachine`, and records the measured after-state. It rejects any correction that does not reduce target RMS. Train seeds are `20264000`–`20264063`; validation seeds `20265000`–`20265007`; test seeds `20266000`–`20266007`. The v001 fixed evaluation seeds and v002 expanded evaluation seeds are disjoint. The v002 Osprey rows retain their original seed-level splits.

`train.jsonl`, `validation.jsonl`, and `test.jsonl` are Liquid Finetune `messages` conversations. `local_actions.jsonl` holds the new measured source actions. `index.jsonl` binds every conversation to its source and split; `manifest.json` records source and output checksums. The local error kicks are never inserted in model-visible messages. This is simulated data; it does not demonstrate live plant control.

Public release: [dataset tag `v003`](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v003). The matching checkpoint is [model tag `v002`](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v002).

Rebuild only into an empty output directory:

```sh
MPLCONFIGDIR=/private/tmp/hackring-mpl PYTHONPATH=src .venv/bin/python scripts/build_v003.py --output /private/tmp/hackring-v003-rebuild
PYTHONPATH=src .venv/bin/python scripts/validate_control_conversations.py datasets/v003
```
