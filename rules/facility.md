---
summary: Hackathon FODO ring context
description: Simulator identity, units, and public channel contract
---

# Hackathon FODO ring

This is a simulated 20-cell, 1 GeV storage ring. Its PyAT deck and channel mapping are in `data/facility/`. The public controls are four horizontal and four vertical correctors with kick setpoints in mrad. Four BPMs report X and Y orbit positions in mm. The exact 24 addresses are in `data/pv_contract.json`.

Only the simulator is configured. Corrector setpoints are bounded by the facility limits file. Disturbance kicks are hidden model elements, not public PVs.
