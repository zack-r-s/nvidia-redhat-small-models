"""The PyAT evaluation accepts only bounded literal model proposals."""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_local_pyat import CALL_PREFIX, evaluate_case, parse_write, score_read, validate_write
from generate_control_dataset import target
from hackring.machine import BPMS, LocalMachine


def test_literal_write_is_parsed_and_scored_against_exact_request():
    case = {"kind": "exact_write", "channel": "RING:MAG:H02:KICK:SP", "value_mrad": 0.075}
    text = "<|tool_call_start|>[channel_write(operations=[{'channel': 'RING:MAG:H02:KICK:SP', 'value': 0.075}], confirm=True)]<|tool_call_end|><|im_end|>"
    operations = parse_write(text)
    assert validate_write(operations, [case["channel"]], case) == [
        {"channel": case["channel"], "value": 0.075}]
    with pytest.raises(ValueError, match="exact write differs"):
        validate_write([{"channel": case["channel"], "value": 0.08}], [case["channel"]], case)


def test_eval_rejects_unsafe_or_executable_proposals():
    case = {"kind": "correct", "plane": "x"}
    allowed = ["RING:MAG:H01:KICK:SP"]
    with pytest.raises(ValueError, match="bounds"):
        validate_write([{"channel": allowed[0], "value": 0.5}], allowed, case)
    with pytest.raises(ValueError, match="disallowed"):
        validate_write([{"channel": "RING:MAG:V01:KICK:SP", "value": 0.05}], allowed, case)
    with pytest.raises(ValueError):
        parse_write("<|tool_call_start|>[channel_write(operations=__import__('os').system('true'))]<|tool_call_end|>")


def test_read_scoring_uses_measured_bpm_order():
    values = [0.12, -0.34, 0.56, -0.78, 0.21, -0.43, 0.65, -0.87]
    pvs = dict(zip(BPMS, values))
    assert score_read({"kind": "read_bpms"}, " ".join(str(x) for x in values), pvs)
    assert not score_read({"kind": "read_bpms"}, "No BPM values are available.", pvs)


def test_valid_exact_write_reaches_pyat_and_records_readback():
    class FakeModel:
        def generate(self, messages, *, prefix="", max_new_tokens=192):
            if prefix:
                assert prefix == CALL_PREFIX
                return prefix + "operations=[{'channel': 'RING:MAG:H02:KICK:SP', 'value': 0.075}], confirm=True)]<|tool_call_end|>"
            return "Measured BPM follow-up."

    case = {"id": "exact-smoke", "seed": 20263002, "kind": "exact_write",
            "prompt": "Set RING:MAG:H02:KICK:SP to 0.075 mrad, then read RING:DIAG:BPM01:POSITION:X.",
            "channel": "RING:MAG:H02:KICK:SP", "value_mrad": 0.075,
            "follow_up": "RING:DIAG:BPM01:POSITION:X"}
    trace = evaluate_case(case, FakeModel(), 0.08)
    assert trace["passed"] and trace["write_attempted"]
    assert trace["before"][case["channel"]] == 0.0
    assert trace["after"][case["channel"]] == 0.075
    assert trace["after"][case["channel"].replace(":SP", ":RB")] == 0.075


def test_heldout_correction_cases_are_reachable_with_bounded_teacher():
    from eval_local_pyat import rms

    matrix = np.asarray(json.loads((ROOT / "datasets/v002/response_teacher.json").read_text())["matrix_mm_per_mrad"])
    cases = [json.loads(line) for line in (ROOT / "evals/mark-i/pyat-v001/cases.jsonl").read_text().splitlines()]
    for case in cases:
        if case["kind"] != "correct":
            continue
        machine = LocalMachine()
        before = machine.reset(case["seed"], 0.08)["pvs"]
        settings = target(case["plane"], before, matrix, np.random.default_rng(case["seed"]), case["seed"])
        after = machine.write(settings)["state"]["pvs"]
        planes = ("x", "y") if case["plane"] == "both" else (case["plane"],)
        assert all(rms(after, plane) < rms(before, plane) for plane in planes)
