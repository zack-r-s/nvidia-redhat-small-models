"""Deterministic checks: local PyAT plant vs. caproto IOC driven through PyEPICS."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hackring.machine import BPMS, LIMIT, SETPOINTS, EpicsMachine, LocalMachine

ROOT = Path(__file__).resolve().parents[1]
CA_ENV = {
    "HACKRING_EPICS_SOURCE": "caproto-epics",
    "EPICS_CA_SERVER_PORT": "6064",
    "EPICS_CA_REPEATER_PORT": "6065",
    "EPICS_CA_ADDR_LIST": "127.0.0.1",
    "EPICS_CA_AUTO_ADDR_LIST": "NO",
    "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1",
    "EPICS_CAS_AUTO_BEACON_ADDR_LIST": "NO",
    "EPICS_CAS_BEACON_ADDR_LIST": "127.0.0.1",
}
KICKS = {name: 0.05 * ((-1) ** i) * (i + 1) / 8 for i, name in enumerate(SETPOINTS)}


def test_local_scenario_is_deterministic():
    a, b = LocalMachine(), LocalMachine()
    assert a.reset(seed=7) == b.reset(seed=7)
    assert a.write(KICKS)["state"] == b.write(KICKS)["state"]


def test_closed_orbit_profile_follows_pyat_without_changing_pvs():
    machine = LocalMachine()
    before = machine.reset(seed=7)
    profile = before["orbit_profile"]
    samples = profile["samples"]
    assert profile["kind"] == "pyat-closed-orbit-4d"
    assert len(samples) > 400
    assert samples[0]["s_m"] == 0
    assert samples[-1]["s_m"] == pytest.approx(before["lattice"]["circumference_m"])
    assert all(a["s_m"] <= b["s_m"] for a, b in zip(samples, samples[1:]))
    for number in range(1, 5):
        name = f"BPM{number:02d}"
        s = next(device["s_m"] for device in before["lattice"]["devices"] if device["name"] == name)
        point = next(p for p in samples if p["s_m"] == pytest.approx(s, abs=1e-9))
        for axis in ("X", "Y"):
            assert point[f"{axis.lower()}_mm"] == pytest.approx(before["pvs"][f"RING:DIAG:{name}:POSITION:{axis}"], abs=3e-5)
    after = machine.write({SETPOINTS[0]: 0.1})["state"]
    assert after["orbit_profile"]["samples"] != samples
    assert after["pvs"][SETPOINTS[0]] == 0.1


def test_limit_rejected_and_state_unchanged():
    m = LocalMachine()
    before = m.reset(seed=7)["pvs"]
    with pytest.raises(ValueError):
        m.write({SETPOINTS[0]: LIMIT + 0.01})
    assert m.read()["pvs"] == before


@pytest.fixture
def ioc():
    env = {**os.environ, **CA_ENV, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "ca_fixture.py"), "--seed", "7"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.environ.update(CA_ENV)
    try:
        time.sleep(1)
        assert proc.poll() is None, "fixture IOC did not start on isolated port 6064"
        yield
    finally:
        proc.terminate()
        proc.wait(5)


def test_epics_matches_local(ioc):
    import epics

    expected = LocalMachine()
    expected.reset(seed=7)
    want = expected.write(KICKS)["state"]["pvs"]
    deadline = time.time() + 15
    while epics.caget(BPMS[0], timeout=1) is None and time.time() < deadline:
        pass
    for name, value in KICKS.items():
        assert epics.caput(name, value, wait=True, timeout=5) == 1
    time.sleep(0.5)
    for name in BPMS + [n.replace(":SP", ":RB") for n in SETPOINTS]:
        assert epics.caget(name, timeout=5) == pytest.approx(want[name], abs=1e-6)

    state = EpicsMachine().read()
    assert state["backend"] == "caproto-epics"
    assert "orbit_profile" not in state
    assert len(state["pvs"]) == 24
    assert state["lattice"]["circumference_m"] > 0
    assert [d["name"] for d in state["lattice"]["devices"] if d["kind"] == "bpm"] == [f"BPM{i:02d}" for i in range(1, 5)]
    for name in BPMS:
        assert state["pvs"][name] == pytest.approx(want[name], abs=1e-6)
