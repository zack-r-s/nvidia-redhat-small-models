"""Validate complete Mark III runs and summarize distinct evaluation interfaces."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals/mark-iii/results.json"
GYM = ROOT / "var/gym/mark-iii"
CASES = ROOT / "evals/mark-i/pyat-v002/cases.jsonl"
MODELS = {
    "direct": {"base-230m": "LFM2.5-230M-base", "base-1p2b": "LFM2.5-1.2B-Instruct",
               "v002": "hackring-v002", "v003": "hackring-v003"},
    "skill": {"base-230m": "LFM2.5-230M-base", "base-1p2b": "LFM2.5-1.2B-Instruct",
              "v004": "LFM2.5-230M"},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def gym_run(track: str, label: str, model_dir: str) -> tuple[dict, dict[str, dict]]:
    candidates = list((GYM / track / model_dir).glob("validation-*/rollouts_aggregate_metrics.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one full {track}/{label} run, found {len(candidates)}")
    folder = candidates[0].parent
    meta = read_json(folder / "run.json")
    task = ROOT / f"gym/resources_servers/hackring_mark_iii/data/{track}.jsonl"
    if meta["task_sha256"] != sha(task) or meta["cases_sha256"] != sha(CASES):
        raise ValueError(f"Source hash mismatch: {folder}")
    if meta["model_revision_source"] != "VLLM_MODEL_REVISION" or not meta.get("model_revision"):
        raise ValueError(f"Unpinned model: {folder}")
    prepared = [json.loads(line) for line in (folder / "preprocessed_datasets/validation.jsonl").open()]
    expected_tool = "channel_write" if track == "direct" else "orbit_correct"
    if len(prepared) != 28 or any([tool["name"] for tool in row["responses_create_params"]["tools"]] != [expected_tool]
                                 for row in prepared):
        raise ValueError(f"Gym prepared the wrong task: {folder}")
    rollouts = [json.loads(line) for line in (folder / "rollouts.jsonl").open()]
    if (folder / "rollouts_failures.jsonl").read_text().strip():
        raise ValueError(f"Gym recorded failed rollouts: {folder}")
    by_id: dict[str, list[dict]] = {}
    for row in rollouts:
        by_id.setdefault(row["episode_id"], []).append(row)
    if len(rollouts) != 112 or len(by_id) != 28 or any(len(rows) != 4 for rows in by_id.values()):
        raise ValueError(f"Incomplete four-repeat run: {folder}")
    metrics = read_json(candidates[0])[0]["agent_metrics"]
    result = {
        "model_source": meta["model_source"], "model_revision": meta["model_revision"],
        "repo_commit_at_run": meta["repo_commit"], "folder": relative(folder),
        "rollouts_sha256": sha(folder / "rollouts.jsonl"), "tasks": 28, "rollouts": 112,
        "successes": sum(bool(row["goal_success"]) for row in rollouts),
        "accepted_writes": sum(row["accepted_writes"] for row in rollouts),
        "tool_attempts": sum(row["tool_attempts"] for row in rollouts),
        "mean_reward": metrics["mean/reward"],
    }
    if track == "skill":
        result["raw_intent_correct"] = sum(row["raw_intent_correct"] is True for row in rollouts)
    if result["successes"] != round(metrics["mean/goal_success"] * 112):
        raise ValueError(f"Aggregate success mismatch: {folder}")
    return result, by_id


def paired_interval(base: dict[str, dict], candidate: dict[str, dict], field: str) -> dict:
    ids = sorted(base)
    if ids != sorted(candidate):
        raise ValueError("Skill runs have different case IDs")
    differences = [sum(bool(row[field]) for row in candidate[key]) / 4 -
                   sum(bool(row[field]) for row in base[key]) / 4 for key in ids]
    rng = random.Random(42)
    estimates = sorted(sum(differences[rng.randrange(28)] for _ in range(28)) / 28 for _ in range(20000))
    return {"candidate_minus_base": sum(differences) / 28,
            "task_bootstrap_95_percent_interval": [estimates[500], estimates[19500]],
            "resampling_unit": "28 frozen tasks; four rollouts per task; 20,000 seeded paired resamples"}


def local_report(track: str, label: str, revision: str) -> dict:
    name = "base" if label == "base-230m" else label
    path = ROOT / f"evals/mark-iii/{track}/{name}/report.json"
    report = read_json(path)
    if report["evaluation_series"] != "mark-iii" or report["cases_sha256"] != sha(CASES):
        raise ValueError(f"Wrong local evaluation source: {path}")
    evaluator = ROOT / ("scripts/eval_local_pyat.py" if track == "direct" else "scripts/eval_v004_skill.py")
    if not str(report["model"]).endswith(revision) or report["evaluator_sha256"] != sha(evaluator):
        raise ValueError(f"Local model revision or evaluator hash differs: {path}")
    expected_total = 56 if track == "direct" else 28
    if report["total"] != expected_total:
        raise ValueError(f"Incomplete local evaluation: {path}")
    return {"report": relative(path), "report_sha256": sha(path),
            "passed": report["passed"], "total": report["total"],
            "inference": report["inference"],
            **({"passed_by_kind": report["passed_by_kind"]} if track == "direct" else
               {"route_correct": report["route_correct"], "intent_correct": report["intent_correct"]})}


def main() -> None:
    gym = {}
    skill_rows = {}
    for track, models in MODELS.items():
        gym[track] = {}
        for label, model_dir in models.items():
            result, rows = gym_run(track, label, model_dir)
            gym[track][label] = result
            if track == "skill":
                skill_rows[label] = rows
    local = {track: {label: local_report(track, label, gym[track][label]["model_revision"]) for label in models}
             for track, models in MODELS.items()}
    comparison = {
        "goal_success": paired_interval(skill_rows["base-230m"], skill_rows["v004"], "goal_success"),
        "raw_intent_correct": paired_interval(skill_rows["base-230m"], skill_rows["v004"], "raw_intent_correct"),
    }
    result = {"schema": "hackring.mark_iii_results/1", "cases_sha256": sha(CASES),
              "task_sha256": {track: sha(ROOT / f"gym/resources_servers/hackring_mark_iii/data/{track}.jsonl")
                              for track in MODELS},
              "local": local, "gym": gym, "paired_skill_comparison": comparison,
              "dataset_validation": "evals/mark-iii/dataset-validation.json",
              "live_smoke": "evals/mark-iii/live-fixture-v004.json",
              "ui_smoke": "evals/mark-iii/ui-smoke-v004.json"}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Validated {sum(len(v) for v in MODELS.values())} Gym and local model runs; saved {OUT}")


if __name__ == "__main__":
    main()
