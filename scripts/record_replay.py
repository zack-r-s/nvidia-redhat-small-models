"""Record /state snapshots (unmodified) while another process drives the machine, for viewer replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--out", default="frontend/public/replay/demo.json")
    parser.add_argument("--seed", type=int, required=True, help="Scenario seed used by the driver (metadata only)")
    parser.add_argument("--run-id", default="xopt-neldermead")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    frames, t0, last = [], time.time(), None
    with httpx.Client(base_url=args.url, timeout=10) as client:
        while time.time() - t0 < args.seconds:
            state = client.get("/state").json()
            if state["pvs"] != last:
                frames.append({"t_s": round(time.time() - t0, 3), "state": state})
                last = state["pvs"]
            time.sleep(args.interval)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_id": args.run_id, "seed": args.seed, "source": f"{args.url}/state (local PyAT fixture) driven by scripts/xopt_demo.py", "frames": frames}))
    print(f"recorded {len(frames)} distinct frames -> {out}")


if __name__ == "__main__":
    main()
