"""Freeze or verify 36 Mark IV Gym tasks against the independent prompt suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hackring.control import SKILL_TOOL
from mark_iv_common import CASES, ROOT, cases, sha
from small_agent import SYSTEM

DATA = ROOT / "gym/resources_servers/hackring_mark_iv/data/validation.jsonl"


def tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": parameters, "strict": False}


TOOLS = [
    tool("channel_read", "Read current values from one or more ring PVs.", {
        "type": "object", "properties": {"channels": {"type": "array", "items": {"type": "string"}}},
        "required": ["channels"]}),
    tool("channel_write", "Write one bounded batch of absolute corrector setpoints in mrad.", {
        "type": "object", "properties": {"operations": {"type": "array", "items": {"type": "object", "properties": {
            "channel": {"type": "string"}, "value": {"type": "number"}}, "required": ["channel", "value"]}},
            "confirm": {"type": "boolean"}}, "required": ["operations"]}),
    tool("channel_limits", "Query corrector setpoint bounds.", {
        "type": "object", "properties": {"channels": {"type": "array", "items": {"type": "string"}}}}),
    tool(SKILL_TOOL["function"]["name"], SKILL_TOOL["function"]["description"],
         SKILL_TOOL["function"]["parameters"]),
]
SYSTEM_PROMPT = (SYSTEM + " For an explicit orbit-centering goal, orbit_correct can solve a bounded correction "
                 "in x, y, or both planes. Use balanced effort unless the operator specifies otherwise. "
                 "For uncertain, contradictory, unsupported, or unbounded requests, do not write. "
                 "Report only values returned by tools.")


def rows() -> list[dict]:
    return [{"responses_create_params": {
        "input": [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": case["prompt"]}],
        "tools": TOOLS, "tool_choice": "auto"},
        "episode": {"episode_id": case["id"], "seed": case["seed"], "prompt": case["prompt"]}}
        for case in cases()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows())
    if args.write:
        if DATA.exists():
            raise FileExistsError(DATA)
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(payload)
    elif not DATA.exists() or DATA.read_text() != payload:
        raise ValueError("frozen Mark IV Gym tasks changed")
    print(f"36 prompts; source SHA-256 {sha(CASES)}; task SHA-256 {sha(DATA)}")


if __name__ == "__main__":
    main()
