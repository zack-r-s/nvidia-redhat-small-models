"""Generate the ring and Osprey facility records from one device definition."""

from __future__ import annotations

import json
import math
from pathlib import Path

import at
import yaml

ROOT = Path(__file__).resolve().parents[2]
FACILITY = ROOT / "data" / "facility"
INSTRUMENTED_CELLS = (1, 6, 11, 16)
KICK_RAD_PER_MRAD = 1e-3
MAX_KICK_MRAD = 0.35


def make_ring() -> at.Lattice:
    elements = []
    for cell in range(1, 21):
        tag = f"{cell:02d}"
        elements.extend(
            [
                at.Drift(f"D{tag}A", 0.25),
                at.Dipole(f"B{tag}A", 1.0, math.pi / 20),
                at.Drift(f"D{tag}B", 0.5),
                at.Quadrupole(f"QF{tag}", 0.5, 1.2),
                at.Drift(f"D{tag}C", 0.25),
            ]
        )
        if cell in INSTRUMENTED_CELLS:
            device = INSTRUMENTED_CELLS.index(cell) + 1
            elements.extend(
                [
                    at.Corrector(f"H{device:02d}", 0.0, [0.0, 0.0]),
                    at.Corrector(f"V{device:02d}", 0.0, [0.0, 0.0]),
                    at.Corrector(f"ERR{device:02d}", 0.0, [0.0, 0.0]),
                    at.Monitor(f"BPM{device:02d}"),
                ]
            )
        elements.extend(
            [
                at.Drift(f"D{tag}D", 0.25),
                at.Dipole(f"B{tag}B", 1.0, math.pi / 20),
                at.Drift(f"D{tag}E", 0.5),
                at.Quadrupole(f"QD{tag}", 0.5, -1.2),
                at.Drift(f"D{tag}F", 0.25),
            ]
        )
    return at.Lattice(elements, name="Hackathon FODO storage ring", energy=1e9)


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def write_facility() -> None:
    ring = make_ring()
    deck = FACILITY / "decks" / "RING.json"
    deck.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(ring, str(deck))

    channels = []
    devices = []
    wiring = []
    limits = []
    for number in range(1, 5):
        tag = f"{number:02d}"
        bpm = f"BPM{tag}"
        devices.append({"id": f"RING/{bpm}", "class": "BeamPositionMonitor", "place": "RING"})
        for axis in ("X", "Y"):
            address = f"RING:DIAG:{bpm}:POSITION:{axis}"
            channels.append({"id": address, "on": {"device": f"RING/{bpm}"}, "unit": "mm"})
            wiring.append(
                {
                    "address": address,
                    "element": bpm,
                    "engine": {"axis": axis.lower()},
                    "calibration": {"curve": {"linear": {"gain": 0.001, "offset": 0.0}}, "energy_scaling": "none"},
                }
            )
        for plane in ("H", "V"):
            device = f"{plane}{tag}"
            devices.append({"id": f"RING/{device}", "class": "Corrector", "place": "RING"})
            base = f"RING:MAG:{device}:KICK"
            rb, sp = f"{base}:RB", f"{base}:SP"
            channels.extend(
                [
                    {"id": rb, "on": {"device": f"RING/{device}"}, "unit": "mrad"},
                    {"id": sp, "role": "setpoint", "pair": rb, "on": {"device": f"RING/{device}"}, "unit": "mrad"},
                ]
            )
            for address in (rb, sp):
                wiring.append(
                    {
                        "address": address,
                        "element": device,
                        "engine": {"attribute": "KickAngle", "index": 0 if plane == "H" else 1},
                        "calibration": {
                            "curve": {"linear": {"gain": KICK_RAD_PER_MRAD, "offset": 0.0}},
                            "energy_scaling": "none",
                        },
                    }
                )
            limits.append({"address": sp, "min_value": -MAX_KICK_MRAD, "max_value": MAX_KICK_MRAD})

    _write_yaml(FACILITY / "identity.yaml", {"code": "hack", "name": "Hackathon FODO Ring"})
    _write_yaml(FACILITY / "records" / "places.yaml", [{"id": "RING", "level": "machine"}])
    _write_yaml(FACILITY / "records" / "devices.yaml", devices)
    _write_yaml(FACILITY / "records" / "channels.yaml", channels)
    _write_yaml(FACILITY / "models.yaml", [{"name": "RING", "engine": "pyat", "deck": "decks/RING.json", "settings": {"pyat": {"solve": "periodic"}}, "wiring": wiring}])
    _write_yaml(FACILITY / "limits.yaml", {"records": limits})
    (ROOT / "data" / "pv_contract.json").write_text(
        json.dumps({"schema": 1, "units": {"orbit": "mm", "kick": "mrad"}, "bpm": [f"RING:DIAG:BPM{i:02d}:POSITION:{axis}" for i in range(1, 5) for axis in ("X", "Y")], "corrector_setpoints": [f"RING:MAG:{plane}{i:02d}:KICK:SP" for i in range(1, 5) for plane in ("H", "V")], "kick_limit_mrad": MAX_KICK_MRAD}, indent=2) + "\n"
    )
    print(f"Wrote ring deck and {len(channels)} PV definitions to {FACILITY}")


if __name__ == "__main__":
    write_facility()
