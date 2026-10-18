"""Replay saved skill proposals with the live harness's balanced-default normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hackring.control import EFFORT_PENALTY, requested_effort, rms, solve
from hackring.machine import ROOT, LocalMachine


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(cases: list[dict], traces: list[dict]) -> dict:
    by_id = {trace["id"]: trace for trace in traces}
    expected_plane = {case["id"]: case["plane"] for case in cases}
    if len(by_id) != len(traces) or set(by_id) != {case["id"] for case in cases}:
        raise ValueError("saved proposals do not match frozen cases")
    outcomes = []
    for case in cases:
        trace = by_id[case["id"]]
        if trace["seed"] != case["seed"] or trace["prompt"] != case["prompt"]:
            raise ValueError("saved proposal is attached to a different case")
        plane, effort = trace.get("selected_plane"), trace.get("selected_effort")
        wanted = requested_effort(case["prompt"])
        row = {"id": case["id"], "selected_plane": plane, "selected_effort": effort,
               "requested_effort": wanted, "normalized": False, "passed": False}
        if plane == case["plane"] and effort in EFFORT_PENALTY and (effort == wanted or wanted == "balanced"):
            if effort != wanted:
                effort = wanted
                row["normalized"] = True
            machine = LocalMachine()
            before = machine.reset(case["seed"], 0.08)["pvs"]
            settings = solve(before, plane, effort)
            after = machine.write(settings)["state"]["pvs"]
            planes = ("x", "y") if plane == "both" else (plane,)
            row.update({"executed_effort": effort,
                        "before_target_rms_mm": rms(before, plane),
                        "after_target_rms_mm": rms(after, plane),
                        "passed": all(rms(after, item) < rms(before, item) for item in planes)})
        outcomes.append(row)
    return {"total": len(outcomes), "passed": sum(x["passed"] for x in outcomes),
            "normalized": sum(x["normalized"] for x in outcomes),
            "raw_intent_correct": sum(x["selected_plane"] == expected_plane[x["id"]]
                                      and x["selected_effort"] == x["requested_effort"] for x in outcomes),
            "outcomes": outcomes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/mark-i/pyat-v002/cases.jsonl")
    parser.add_argument("--baseline", type=Path, default=ROOT / "evals/mark-ii/pyat-v004/base-230m/traces.jsonl")
    parser.add_argument("--candidate", type=Path, default=ROOT / "evals/mark-ii/pyat-v004/liquid-v004/traces.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/mark-ii/pyat-v004/default-replay.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    cases = [case for case in cases if case["kind"] == "correct"]
    results = {}
    for label, path in (("base_230m", args.baseline), ("liquid_v004", args.candidate)):
        traces = [json.loads(line) for line in path.read_text().splitlines()]
        results[label] = {"source_traces_sha256": digest(path), **replay(cases, traces)}
    report = {"schema": "hackring.skill_default_replay/1", "evaluation_series": "mark-ii",
              "method": "same frozen model proposals; balanced-default effort normalized before deterministic PyAT solve; no model reinference",
              "cases_sha256": digest(args.cases),
              "response_teacher_sha256": digest(ROOT / "datasets/v002/response_teacher.json"),
              "replay_script_sha256": digest(Path(__file__)), **results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({label: {k: results[label][k] for k in ("passed", "total", "normalized", "raw_intent_correct")}
                      for label in results}, indent=2))
