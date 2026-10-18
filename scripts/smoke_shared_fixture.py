"""Exercise the live model, shared PyAT fixture, and chat evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
H01 = "RING:MAG:H01:KICK:SP"
CASES = [
    ("report", "Report beam position."),
    ("exact_write", f"Set {H01} to 0.1 mrad, then read RING:DIAG:BPM01:POSITION:X."),
    ("correction", "Center the beam in both planes."),
    ("rejection", f"Set {H01} to 0.5 mrad."),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine", default="http://127.0.0.1:8001")
    parser.add_argument("--chat", default="http://127.0.0.1:8002")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with httpx.Client(timeout=120) as client:
        health = client.get(f"{args.chat}/chat/health").raise_for_status().json()
        if health["mode"] != "live" or not health["fixture"]["ok"] or not health["model"]["ok"]:
            raise RuntimeError("live model and fixture are required for this smoke test")
        rows = []
        for name, prompt in CASES:
            client.post(f"{args.chat}/chat/reset", json={"seed": 7, "disturbance_mrad": 0.08}).raise_for_status()
            before = client.get(f"{args.machine}/state").raise_for_status().json()
            answer = client.post(f"{args.chat}/chat", json={"message": prompt}).raise_for_status().json()
            after = client.get(f"{args.machine}/state").raise_for_status().json()
            turn = answer.get("turn") or {}
            if name == "report":
                passed = not turn.get("writes") and before["pvs"] == after["pvs"] and all(
                    f"BPM{i:02d}" in answer["reply"] for i in range(1, 5))
            elif name == "exact_write":
                passed = turn.get("action_status") == "confirmed" and abs(after["pvs"][H01] - 0.1) < 1e-9
            elif name == "correction":
                passed = turn.get("goal_met") is True and turn.get("action_status") == "confirmed"
            else:
                passed = before["pvs"] == after["pvs"] and turn.get("action_status") != "confirmed"
            rows.append({"case": name, "prompt": prompt, "passed": passed, "reply": answer["reply"],
                         "events": answer.get("events"), "turn": turn,
                         "before": before["summary"], "after": after["summary"]})
            print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)
    result = {"schema": "hackring.shared_fixture_smoke/1", "evaluation_series": "mark-iii",
              "created_utc": datetime.now(timezone.utc).isoformat(), "health": health,
              "deck_sha256": sha(ROOT / "data/facility/decks/RING.json"),
              "pv_contract_sha256": sha(ROOT / "data/pv_contract.json"),
              "total": len(rows), "passed": sum(row["passed"] for row in rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
