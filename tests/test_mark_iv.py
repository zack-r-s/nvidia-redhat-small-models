"""Check Mark IV scoring against actual seeded PyAT states."""

from __future__ import annotations

from hackring.machine import LocalMachine
from mark_iv_common import DISTURBANCE_MRAD, cases, expected_action, pv, score


def test_frozen_suite_and_both_conditional_branches():
    rows = cases()
    assert len(rows) == 36
    assert {row["id"][0] for row in rows} == {"R", "A", "B"}
    by_id = {row["id"]: row for row in rows}
    assert expected_action(by_id["A01"], LocalMachine().reset(by_id["A01"]["seed"], DISTURBANCE_MRAD)) == "exact"
    assert expected_action(by_id["A05"], LocalMachine().reset(by_id["A05"]["seed"], DISTURBANCE_MRAD)) == "none"


def test_exact_write_requires_measured_follow_up():
    case = next(row for row in cases() if row["id"] == "A08")
    machine = LocalMachine()
    before = machine.reset(case["seed"], DISTURBANCE_MRAD)
    target = pv("V04")
    after = machine.write({target: -0.03})["state"]
    writes = [{"channel": target, "outcome": "confirmed", "observed_value": after["pvs"][target]}]
    required = [pv(code) for code in case["channels"]]
    without_follow_up = score(case, before, after, [], writes)
    assert without_follow_up["plant_success"] and not without_follow_up["success"]
    with_follow_up = score(case, before, after, [{"channels": required, "ok": True, "after_write": True}], writes)
    assert with_follow_up["success"]


def test_read_only_case_rejects_a_confirmed_write():
    case = next(row for row in cases() if row["id"] == "R01")
    before = LocalMachine().reset(case["seed"], DISTURBANCE_MRAD)
    required = [pv(code) for code in case["channels"]]
    reads = [{"channels": required, "ok": True, "after_write": False}]
    assert score(case, before, before, reads, [])["success"]
    assert not score(case, before, before, reads, [{"channel": pv("H01"), "outcome": "confirmed"}])["success"]
