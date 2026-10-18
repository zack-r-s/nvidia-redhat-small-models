"""Run every frozen Mark IV prompt through the deployed fixture chat harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from mark_iv_common import CASES, DISTURBANCE_MRAD, ROOT, cases, score, sha


async def run(args: argparse.Namespace) -> None:
    # Import after endpoint variables have been set; chat_bridge reads them at import time.
    os.environ["HACKRING_API_URL"] = args.fixture_url
    os.environ["VLLM_BASE_URL"] = args.model_url
    os.environ["VLLM_MODEL_NAME"] = args.model
    from hackring.chat_bridge import FixtureControls, Session, live_turn

    class AuditedControls(FixtureControls):
        def __init__(self, client: httpx.AsyncClient) -> None:
            super().__init__(client)
            self.reads: list[dict] = []

        async def call_tool(self, name: str, arguments: dict):
            result = await super().call_tool(name, arguments)
            if name == "channel_read":
                try:
                    payload = json.loads(result.content[0].text)
                except (ValueError, IndexError):
                    payload = {}
                self.reads.append({"channels": arguments.get("channels") or [],
                                   "ok": payload.get("status") == "success",
                                   "after_write": any(op.get("outcome") == "confirmed" for op in self.writes)})
            return result

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = cases()[:args.limit] if args.limit else cases()
    traces = []
    async with httpx.AsyncClient(base_url=args.fixture_url, timeout=45) as fixture:
        async with httpx.AsyncClient(base_url=args.model_url, timeout=5) as model:
            response = await model.get("/models")
            response.raise_for_status()
            if args.model not in [item["id"] for item in response.json()["data"]]:
                raise RuntimeError(f"{args.model} is not the served model")
        for case in rows:
            reset = await fixture.post("/scenario", json={"seed": case["seed"],
                                                        "disturbance_mrad": DISTURBANCE_MRAD})
            reset.raise_for_status()
            before = (await fixture.get("/state")).json()
            controls = AuditedControls(fixture)
            try:
                reply, events = await live_turn(case["prompt"], Session(), controls)
                error = None
            except Exception as exc:
                reply, events, error = "", [], f"{type(exc).__name__}: {exc}"
            after = (await fixture.get("/state")).json()
            result = score(case, before, after, controls.reads, controls.writes)
            traces.append({"case": case, "reply": reply, "events": events, "error": error,
                           "reads": controls.reads, "writes": controls.writes,
                           "before": before, "after": after, "score": result})
            print(f"{case['id']}: {'PASS' if result['success'] and not error else 'FAIL'}"
                  f" plant={result['plant_success']} evidence={result['evidence_success']}", flush=True)
    output.mkdir(parents=True)
    trace_path = output / "traces.jsonl"
    trace_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in traces))
    by_kind = {kind: {"passed": sum(row["score"]["success"] and not row["error"] for row in traces if row["case"]["kind"] == kind),
                      "total": sum(row["case"]["kind"] == kind for row in traces)}
               for kind in sorted({row["case"]["kind"] for row in traces})}
    report = {"schema": "hackring.mark_iv_local/1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "model_source": args.model_source, "model_revision": args.model_revision,
              "inference": "deployed fixture chat harness over vLLM, temperature 0.2",
              "cases_sha256": sha(CASES), "deck_sha256": sha(ROOT / "data/facility/decks/RING.json"),
              "traces_sha256": sha(trace_path), "disturbance_mrad": DISTURBANCE_MRAD,
              "passed": sum(row["score"]["success"] and not row["error"] for row in traces),
              "plant_passed": sum(row["score"]["plant_success"] and not row["error"] for row in traces),
              "evidence_passed": sum(row["score"]["evidence_success"] and not row["error"] for row in traces),
              "total": len(traces), "by_kind": by_kind,
              "answer_quality_scored": False}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--fixture-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke only; omit for a full 36-case run")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
