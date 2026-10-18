"""Validate complete Mark IV local/Gym matrices and publish an apples-to-apples index."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from mark_iv_common import CASES, ROOT, cases, sha

MODELS = json.loads((ROOT / "evals/mark-iv/models.json").read_text())
GYM_ROOT = ROOT / "var/gym/mark-iv"
OUT = ROOT / "evals/mark-iv/results.json"
TASK = ROOT / "gym/resources_servers/hackring_mark_iv/data/validation.jsonl"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def local(label: str, model: dict) -> dict:
    folder = ROOT / "evals/mark-iv/local" / label
    report = json.loads((folder / "report.json").read_text())
    traces = [json.loads(line) for line in (folder / "traces.jsonl").open()]
    assert report["cases_sha256"] == sha(CASES)
    assert report["model_source"] == model["source"] and report["model_revision"] == model["revision"]
    assert report["model"] == model["served_name"] and len(traces) == report["total"] == 36
    assert report["traces_sha256"] == sha(folder / "traces.jsonl")
    assert [row["case"]["id"] for row in traces] == [case["id"] for case in cases()]
    assert sum(bool(row["score"]["success"]) and not row["error"] for row in traces) == report["passed"]
    return {"report": relative(folder / "report.json"), "report_sha256": sha(folder / "report.json"),
            "traces": relative(folder / "traces.jsonl"), "traces_sha256": sha(folder / "traces.jsonl"),
            "passed": report["passed"], "plant_passed": report["plant_passed"],
            "evidence_passed": report["evidence_passed"],
            "answer_present": sum(bool(row["reply"].strip()) and row["reply"] != "(no reply)"
                                  for row in traces), "total": 36, "by_kind": report["by_kind"],
            "case_results": {row["case"]["id"]: bool(row["score"]["success"] and not row["error"])
                             for row in traces}}


def gym_answer_present(row: dict) -> bool:
    for item in row["response"]["output"]:
        if item.get("type") != "message":
            continue
        content = item.get("content") or []
        if any(isinstance(part, dict) and str(part.get("text") or "").strip() for part in content):
            return True
    return False


def gym(label: str, model: dict) -> dict:
    model_dir = model["served_name"].replace("/", "_")
    candidates = list((GYM_ROOT / model_dir).glob("validation-*/rollouts_aggregate_metrics.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one full Gym run for {label}; found {len(candidates)}")
    folder = candidates[0].parent
    meta = json.loads((folder / "run.json").read_text())
    assert meta["evaluation_series"] == "mark-iv" and meta["cases_sha256"] == sha(CASES)
    assert meta["task_sha256"] == sha(TASK)
    assert meta["model_source"] == model["source"] and meta["model_revision"] == model["revision"]
    assert meta["served_model_name"] == model["served_name"]
    prepared = [json.loads(line) for line in (folder / "preprocessed_datasets/validation.jsonl").open()]
    assert len(prepared) == 36
    names = {"channel_read", "channel_write", "channel_limits", "orbit_correct"}
    assert all({tool["name"] for tool in row["responses_create_params"]["tools"]} == names for row in prepared)
    assert not (folder / "rollouts_failures.jsonl").read_text().strip()
    rollouts = [json.loads(line) for line in (folder / "rollouts.jsonl").open()]
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rollouts:
        by_id[row["episode_id"]].append(row)
    assert len(rollouts) == 144 and set(by_id) == {case["id"] for case in cases()}
    assert all(len(group) == 4 for group in by_id.values())
    kinds = {case["id"]: case["kind"] for case in cases()}
    by_kind = {kind: {"passed": sum(bool(row["goal_success"]) for row in rollouts if kinds[row["episode_id"]] == kind),
                      "total": sum(kinds[row["episode_id"]] == kind for row in rollouts)}
               for kind in sorted(set(kinds.values()))}
    return {"folder": relative(folder), "run_sha256": sha(folder / "run.json"),
            "rollouts_sha256": sha(folder / "rollouts.jsonl"), "passed": sum(bool(row["goal_success"]) for row in rollouts),
            "plant_passed": sum(bool(row["plant_success"]) for row in rollouts),
            "evidence_passed": sum(bool(row["evidence_success"]) for row in rollouts),
            "answer_present": sum(gym_answer_present(row) for row in rollouts),
            "total": 144, "by_kind": by_kind,
            "case_results": {case_id: sum(bool(row["goal_success"]) for row in group)
                             for case_id, group in sorted(by_id.items())}}


def main() -> None:
    novelty = json.loads((ROOT / "evals/mark-iv/novelty-audit.json").read_text())
    assert novelty["cases_sha256"] == sha(CASES) and novelty["exact_matches"] == 0
    result = {"schema": "hackring.mark_iv_results/1", "cases_sha256": sha(CASES),
              "task_sha256": sha(TASK), "model_manifest_sha256": sha(ROOT / "evals/mark-iv/models.json"),
              "novelty_audit_sha256": sha(ROOT / "evals/mark-iv/novelty-audit.json"),
              "scoring": "confirmed plant action plus requested PV reads; natural-language answer quality not automatically scored",
              "local": {}, "gym": {}}
    for label, model in MODELS.items():
        result["local"][label] = local(label, model)
        result["gym"][label] = gym(label, model)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Validated {len(MODELS)} local and {len(MODELS)} Gym Mark IV runs: {OUT}")


if __name__ == "__main__":
    main()
