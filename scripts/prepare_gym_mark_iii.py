"""Freeze/check Mark III Gym tasks from the seed-disjoint local PyAT case set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hackring.control import SKILL_SYSTEM, SKILL_TOOL
from hackring.machine import LocalMachine
from small_agent import GOAL_SYSTEM, compact_read, goal_channels, goal_request

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals/mark-i/pyat-v002/cases.jsonl"
DATA = ROOT / "gym/resources_servers/hackring_mark_iii/data"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": parameters, "strict": False}


def rows() -> dict[str, list[dict]]:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    cases = [case for case in cases if case["kind"] == "correct"]
    if len(cases) != 28 or len({case["seed"] for case in cases}) != 28:
        raise ValueError("frozen correction set changed")
    direct_tool = tool("channel_write", "Write one bounded batch of corrector setpoints.", {
        "type": "object", "properties": {"operations": {"type": "array", "items": {"type": "object"}},
                                          "confirm": {"type": "boolean"}}, "required": ["operations"],
    })
    skill_tool = tool(SKILL_TOOL["function"]["name"], SKILL_TOOL["function"]["description"],
                      SKILL_TOOL["function"]["parameters"])
    output: dict[str, list[dict]] = {"direct": [], "skill": []}
    for case in cases:
        plane = case["plane"]
        if goal_request(case["prompt"]) != plane:
            raise ValueError(f"goal parser disagrees with frozen case {case['id']}")
        machine = LocalMachine()
        before = machine.reset(case["seed"], 0.08)["pvs"]
        bpm, correctors = goal_channels(plane)
        needed = bpm + correctors
        read_result = json.dumps({"status": "success", "summary": {"readings": {
            pv: {"value": before[pv]} for pv in needed}}})
        episode = {"episode_id": case["id"], "seed": case["seed"], "disturbance_mrad": 0.08,
                   "plane": plane, "prompt": case["prompt"]}
        output["direct"].append({
            "responses_create_params": {
                "input": [{"role": "system", "content": GOAL_SYSTEM},
                          {"role": "user", "content": case["prompt"]},
                          {"type": "function_call", "call_id": "prewrite-read", "name": "channel_read",
                           "arguments": json.dumps({"channels": needed})},
                          {"type": "function_call_output", "call_id": "prewrite-read",
                           "output": compact_read(read_result, needed)}],
                "tools": [direct_tool], "tool_choice": "required",
            }, "episode": episode | {"interface": "channel_write"},
        })
        output["skill"].append({
            "responses_create_params": {
                "input": [{"role": "system", "content": SKILL_SYSTEM},
                          {"role": "user", "content": case["prompt"]}],
                "tools": [skill_tool], "tool_choice": "required",
            }, "episode": episode | {"interface": "orbit_correct"},
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the task files once; default verifies frozen files")
    args = parser.parse_args()
    generated = rows()
    if args.write:
        DATA.mkdir(parents=True, exist_ok=True)
    for track, items in generated.items():
        path = DATA / f"{track}.jsonl"
        payload = "".join(json.dumps(item, allow_nan=False, sort_keys=True) + "\n" for item in items)
        if args.write:
            if path.exists():
                raise FileExistsError(path)
            path.write_text(payload)
        elif not path.is_file() or path.read_text() != payload:
            raise ValueError(f"frozen Mark III tasks differ from {path}")
        print(f"{track}: {len(items)} tasks, {hashlib.sha256(payload.encode()).hexdigest()}")
    print(f"source cases: {sha(CASES)}")


if __name__ == "__main__":
    main()
