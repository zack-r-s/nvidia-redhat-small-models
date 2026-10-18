"""Summarize paired model results on the same frozen PyAT cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads((folder / "report.json").read_text())
    traces_path = folder / "traces.jsonl"
    if sha(traces_path) != report["traces_sha256"]:
        raise ValueError(f"trace checksum mismatch: {folder}")
    traces = {row["id"]: row for row in map(json.loads, traces_path.read_text().splitlines())}
    if len(traces) != report["total"]:
        raise ValueError(f"duplicate or missing cases: {folder}")
    return report, traces


def summary(traces: dict[str, dict]) -> dict:
    kinds = Counter((trace["kind"], bool(trace["passed"])) for trace in traces.values())
    corrections = [trace for trace in traces.values() if trace["kind"] == "correct"]
    accepted = [trace for trace in corrections if trace.get("write_attempted")]
    deltas = {trace["id"]: trace["before_target_rms_mm"] - trace["after_target_rms_mm"]
              for trace in accepted}
    return {"passed": sum(t["passed"] for t in traces.values()), "total": len(traces),
            "by_kind": {kind: {"passed": kinds[(kind, True)], "total": kinds[(kind, True)] + kinds[(kind, False)]}
                        for kind in sorted({t["kind"] for t in traces.values()})},
            "correction_writes_accepted": len(accepted),
            "correction_writes_reducing_target_aggregate_rms": sum(delta > 0 for delta in deltas.values()),
            "correction_rms_reduction_mm": deltas}


def compare(base: Path, candidate: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    a_report, a = load(base)
    b_report, b = load(candidate)
    if a_report["cases_sha256"] != b_report["cases_sha256"] or set(a) != set(b):
        raise ValueError("evaluations do not use identical frozen cases")
    if a_report["evaluation_series"] != b_report["evaluation_series"]:
        raise ValueError("evaluations belong to different series")
    result = {"schema": "hackring.pyat_eval_comparison/1", "evaluation_series": a_report["evaluation_series"],
              "cases_sha256": a_report["cases_sha256"],
              "baseline": {"model": a_report["model"], "report_sha256": sha(base / "report.json"), **summary(a)},
              "candidate": {"model": b_report["model"], "report_sha256": sha(candidate / "report.json"), **summary(b)},
              "paired": {key: {"baseline_passed": a[key]["passed"], "candidate_passed": b[key]["passed"]}
                         for key in a}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "evals/mark-ii/pyat-v002/liquid-v001")
    parser.add_argument("--candidate", type=Path, default=ROOT / "evals/mark-ii/pyat-v002/liquid-v002")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/mark-ii/pyat-v002/comparison.json")
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.output), indent=2))
