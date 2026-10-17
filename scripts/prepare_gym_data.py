"""Write NeMo Gym task rows for the hackring_orbit resources server from a dataset split.

One row per episode. The input reproduces the small agent's goal turn: its goal system prompt, an
operator goal, the harness's pre-write `channel_read` of all BPMs and correctors, and a single
offered `channel_write` tool with `tool_choice: required`. The read result is the episode's
recorded initial public PVs. Hidden calibration offsets go only to the resources server under
`episode`; Gym sends the model only `responses_create_params`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_agent import GOAL_SYSTEM, compact_read, goal_channels, goal_request

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "gym" / "resources_servers" / "hackring_orbit"
PROMPT = "Correct the beam orbit."
CALL_ID = "prewrite-read"
PORTIONS = {"finetune": "train", "eval": "validation"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def channel_write_tool() -> dict:
    spec = json.loads((SERVER / "osprey_tools.json").read_text())["tools"]["channel_write"]
    return {"type": "function", "name": "channel_write", "description": spec["description"],
            "parameters": spec["parameters"], "strict": False}


def task_row(episode: dict, attempts: list[dict], tool: dict) -> dict:
    plane = goal_request(PROMPT)
    bpm, correctors = goal_channels(plane)
    needed = bpm + correctors
    initial = episode["initial_pvs"]
    read_result = json.dumps({"status": "success",
                              "summary": {"readings": {pv: {"value": initial[pv]} for pv in needed}}})
    costs = [attempt["after_metrics"]["orbit_cost"] for attempt in attempts]
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": GOAL_SYSTEM},
                {"role": "user", "content": PROMPT},
                {"type": "function_call", "call_id": CALL_ID, "name": "channel_read",
                 "arguments": json.dumps({"channels": needed})},
                {"type": "function_call_output", "call_id": CALL_ID, "output": compact_read(read_result, needed)},
            ],
            "tools": [tool],
            "tool_choice": "required",
        },
        "episode": {
            "episode_id": episode["episode_id"],
            "hidden_calibration_offsets_mrad": episode["hidden_calibration_offsets_mrad"],
            "initial_setpoints_mrad": {sp: initial[sp] for sp in correctors},
            "allowed_correctors": correctors,
            "xopt_best_orbit_cost": min([episode["initial_metrics"]["orbit_cost"], *costs]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, help="split.json from scripts/split_dataset.py")
    parser.add_argument("--output", type=Path, default=SERVER / "data")
    parser.add_argument("--examples", type=int, default=3, help="Evaluation rows copied to example.jsonl")
    args = parser.parse_args()
    if goal_request(PROMPT) != "both":
        raise SystemExit(f"{PROMPT!r} must request both planes in the small agent's goal parser")
    record = json.loads(args.split.read_text())
    folder = args.split.parents[2]
    episodes = {row["episode_id"]: row for row in load_jsonl(folder / "episodes.jsonl")}
    attempts: dict[str, list[dict]] = {}
    for row in load_jsonl(folder / "attempts.jsonl"):
        attempts.setdefault(row["episode_id"], []).append(row)
    tool = channel_write_tool()
    args.output.mkdir(parents=True, exist_ok=True)
    rows_by_portion = {}
    for portion, name in PORTIONS.items():
        rows = [task_row(episodes[eid], attempts[eid], tool) for eid in record["episodes"][portion]]
        rows_by_portion[portion] = rows
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"{name}.jsonl: {len(rows)} tasks from {portion} ({rows[0]['episode']['episode_id']}..)")
    examples = rows_by_portion["eval"][: args.examples]
    (args.output / "example.jsonl").write_text("".join(json.dumps(row) + "\n" for row in examples))


if __name__ == "__main__":
    main()
