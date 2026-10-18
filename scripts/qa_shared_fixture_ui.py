"""Check that live chat, Agent & goal, and PV readings share one fixture."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

SETPOINT = "RING:MAG:H01:KICK:SP"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", default="http://127.0.0.1:5173")
    parser.add_argument("--machine", default="http://127.0.0.1:8001")
    parser.add_argument("--chat", default="http://127.0.0.1:8002")
    parser.add_argument("--chrome", help="Chrome/Chromium executable if Playwright's browser is unavailable")
    parser.add_argument("--output", type=Path, default=Path("evals/mark-iii/ui-smoke-v004.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with httpx.Client(timeout=20) as client:
        health = client.get(f"{args.chat}/chat/health").raise_for_status().json()
        assert health["mode"] == "live" and health["fixture"]["ok"] and health["model"]["ok"]
        client.post(f"{args.chat}/chat/reset", json={"seed": 7, "disturbance_mrad": 0.08}).raise_for_status()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=args.chrome, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(args.viewer)
            page.get_by_test_id("chat-status").get_by_text("Live model", exact=False).wait_for()
            page.get_by_test_id("chat-input").fill(f"Set {SETPOINT} to 0.1 mrad, then read RING:DIAG:BPM01:POSITION:X.")
            page.get_by_test_id("chat-send").click()
            page.get_by_test_id("msg-agent").last.wait_for(timeout=30000)
            page.get_by_role("tab", name="Agent & goal").click()
            page.get_by_test_id("agent-goal").get_by_text("confirmed", exact=True).wait_for()
            state = client.get(f"{args.machine}/state").raise_for_status().json()
            assert abs(state["pvs"][SETPOINT] - 0.1) < 1e-9
            page.wait_for_function("() => document.querySelector('[data-testid=\"pv-RING:MAG:H01:KICK:SP\"]')?.textContent?.trim() === '0.1000'")
            panel = page.get_by_test_id("agent-goal").inner_text()
            assert "Measured goal: met" in panel and "0.1 mrad" in panel
            page.get_by_test_id("fixture-reset").click()
            page.get_by_test_id("unavailable-Agent & goal").wait_for()
            reset = client.get(f"{args.machine}/state").raise_for_status().json()
            assert abs(reset["pvs"][SETPOINT]) < 1e-9
            browser.close()
    result = {"schema": "hackring.ui_smoke/1", "evaluation_series": "mark-iii",
              "created_utc": datetime.now(timezone.utc).isoformat(), "model_mode": health["mode"],
              "setpoint": SETPOINT, "written_mrad": state["pvs"][SETPOINT],
              "reset_mrad": reset["pvs"][SETPOINT], "agent_panel": panel, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"UI shared fixture: PASS ({args.output})")


if __name__ == "__main__":
    main()
