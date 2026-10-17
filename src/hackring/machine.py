"""Shared snapshot contract for the local PyAT plant and the EPICS IOC."""

from __future__ import annotations

import json
import math
from pathlib import Path

import at
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / "data" / "pv_contract.json").read_text())
SETPOINTS = CONTRACT["corrector_setpoints"]
BPMS = CONTRACT["bpm"]
LIMIT = CONTRACT["kick_limit_mrad"]


def _summary(pvs: dict[str, float]) -> dict[str, float]:
    return {
        "x_rms_mm": math.sqrt(sum(pvs[name] ** 2 for name in BPMS if name.endswith(":X")) / 4),
        "y_rms_mm": math.sqrt(sum(pvs[name] ** 2 for name in BPMS if name.endswith(":Y")) / 4),
    }


def _snapshot(pvs: dict[str, float], lattice: at.Lattice | None = None) -> dict:
    devices = []
    if lattice is not None:
        spos = lattice.get_s_pos(range(len(lattice)))
        for s, element in zip(spos, lattice):
            name = element.FamName
            if name.startswith(("BPM", "H", "V", "QF", "QD", "B")) and not name.startswith("ERR"):
                devices.append({"name": name, "kind": "bpm" if name.startswith("BPM") else "corrector" if name.startswith(("H", "V")) else "quadrupole" if name.startswith("Q") else "dipole", "s_m": float(s)})
    return {"schema": "hackring.state/1", "pvs": pvs, "summary": _summary(pvs), "lattice": {"circumference_m": float(lattice.get_s_pos(len(lattice))[0]) if lattice is not None else None, "devices": devices}}


class LocalMachine:
    def __init__(self) -> None:
        self.lattice = at.load_lattice(str(ROOT / "data" / "facility" / "decks" / "RING.json"))
        self.elements = {element.FamName: element for element in self.lattice}
        self.setpoints = {name: 0.0 for name in SETPOINTS}
        self.seed = 0

    def reset(self, seed: int = 1, disturbance_mrad: float = 0.08) -> dict:
        if not 0 <= disturbance_mrad <= 0.2:
            raise ValueError("disturbance_mrad must be within [0, 0.2]")
        rng = np.random.default_rng(seed)
        self.seed = seed
        for name in SETPOINTS:
            self._put(name, 0.0)
        for i in range(1, 5):
            self.elements[f"ERR{i:02d}"].KickAngle[:] = rng.uniform(-disturbance_mrad, disturbance_mrad, 2) * 1e-3
        return self.read()

    def _put(self, name: str, value: float) -> None:
        if name not in SETPOINTS:
            raise KeyError(f"unknown setpoint: {name}")
        if not math.isfinite(value) or abs(value) > LIMIT:
            raise ValueError(f"{name} must be within ±{LIMIT} mrad")
        device = name.split(":")[2]
        axis = 0 if device.startswith("H") else 1
        self.elements[device].KickAngle[axis] = value * 1e-3
        self.setpoints[name] = value

    def write(self, settings: dict[str, float]) -> dict:
        for name, value in settings.items():
            if name not in SETPOINTS or not math.isfinite(value) or abs(value) > LIMIT:
                raise ValueError(f"invalid setpoint {name}={value}")
        before = dict(self.setpoints)
        try:
            for name, value in settings.items():
                self._put(name, value)
            state = self.read()  # A lost closed orbit fails the write.
        except Exception:
            for name, value in before.items():
                self._put(name, value)
            raise
        return {"status": "completed", "readbacks": {name.replace(":SP", ":RB"): state["pvs"][name.replace(":SP", ":RB")] for name in settings}, "state": state}

    def read(self) -> dict:
        indices = [next(i for i, e in enumerate(self.lattice) if e.FamName == f"BPM{n:02d}") for n in range(1, 5)]
        _, orbit = at.find_orbit4(self.lattice, refpts=indices)
        pvs = {}
        for n, point in enumerate(orbit, 1):
            pvs[f"RING:DIAG:BPM{n:02d}:POSITION:X"] = float(point[0] * 1e3)
            pvs[f"RING:DIAG:BPM{n:02d}:POSITION:Y"] = float(point[2] * 1e3)
        for sp, value in self.setpoints.items():
            pvs[sp] = value
            pvs[sp.replace(":SP", ":RB")] = value
        return _snapshot(pvs, self.lattice)


class EpicsMachine:
    """Read/write the Osprey virtual accelerator's Channel Access PVs."""

    def __init__(self) -> None:
        import epics

        self.epics = epics

    def read(self) -> dict:
        names = BPMS + SETPOINTS + [name.replace(":SP", ":RB") for name in SETPOINTS]
        values = self.epics.caget_many(names)
        if any(value is None for value in values):
            raise RuntimeError("one or more EPICS PVs did not answer")
        return _snapshot({name: float(value) for name, value in zip(names, values)})

    def write(self, settings: dict[str, float]) -> dict:
        for name, value in settings.items():
            if name not in SETPOINTS or not math.isfinite(value) or abs(value) > LIMIT:
                raise ValueError(f"invalid setpoint {name}={value}")
        for name, value in settings.items():
            if self.epics.caput(name, value, wait=True, timeout=5) != 1:
                raise RuntimeError(f"EPICS put failed: {name}")
        state = self.read()
        for name, value in settings.items():
            if abs(state["pvs"][name.replace(":SP", ":RB")] - value) > 1e-6:
                raise RuntimeError(f"readback did not match: {name}")
        return {"status": "completed", "readbacks": {name.replace(":SP", ":RB"): state["pvs"][name.replace(":SP", ":RB")] for name in settings}, "state": state}
