"""Move the ring for frontend development; no model inference or dataset output."""

from __future__ import annotations

import argparse
import time

import httpx
from xopt import Evaluator, VOCS, Xopt
from xopt.generators import get_generator

from hackring.machine import LIMIT, SETPOINTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.35, help="Seconds between visible steps")
    args = parser.parse_args()
    if args.steps < 1 or args.delay < 0:
        parser.error("steps must be positive and delay nonnegative")

    with httpx.Client(base_url=args.url, timeout=10) as client:
        reset = client.post("/scenario", json={"seed": args.seed})
        if reset.status_code == 501:
            print("EPICS backend: use the IOC's model RPC to inject a disturbance first.")
        else:
            reset.raise_for_status()

        def evaluate(settings: dict) -> dict:
            response = client.post("/write", json={"settings": settings})
            response.raise_for_status()
            summary = response.json()["state"]["summary"]
            # The small regularizer prefers corrections that use less authority.
            objective = summary["x_rms_mm"] ** 2 + summary["y_rms_mm"] ** 2
            objective += 0.01 * sum(value * value for value in settings.values())
            print(f"x={summary['x_rms_mm']:.4f} mm  y={summary['y_rms_mm']:.4f} mm  objective={objective:.6f}")
            time.sleep(args.delay)
            return {"orbit_cost": objective}

        vocs = VOCS(variables={name: [-LIMIT, LIMIT] for name in SETPOINTS}, objectives={"orbit_cost": "MINIMIZE"})
        # A visible 0.06 mrad simplex avoids Xopt's tiny default zero-centered simplex.
        simplex = {name: [0.06 if vertex == index + 1 else 0.0 for vertex in range(len(SETPOINTS) + 1)] for index, name in enumerate(sorted(SETPOINTS))}
        generator = get_generator("neldermead")(vocs=vocs, initial_point={name: 0.0 for name in SETPOINTS}, initial_simplex=simplex)
        optimizer = Xopt(generator=generator, evaluator=Evaluator(function=evaluate))
        for _ in range(args.steps):
            optimizer.step()
        print(f"Evaluated {len(optimizer.data)} candidates. Frontend state remains at the final candidate.")


if __name__ == "__main__":
    main()
