"""Check that numbers rendered in the viewer DOM match /state. Needs: playwright (not a project dependency)."""

from __future__ import annotations

import argparse
import sys

import httpx
from playwright.sync_api import sync_playwright


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--viewer", default="http://127.0.0.1:5173")
    p.add_argument("--api", default="http://127.0.0.1:8001")
    p.add_argument("--shot", help="Optional screenshot path prefix")
    p.add_argument("--chrome", help="Chrome/Chromium executable if Playwright's bundled browser is unavailable")
    a = p.parse_args()
    state = httpx.get(f"{a.api}/state").json()
    bad = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=a.chrome, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        for label, size in {"desktop": (1440, 900), "mobile": (390, 844)}.items():
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            page.goto(a.viewer)
            page.wait_for_selector("[data-testid=bpm-table]", state="attached")  # tables live in side-panel tabs
            page.wait_for_timeout(800)
            checked = 0
            for name, value in state["pvs"].items():
                text = page.locator(f"[data-testid='pv-{name}']").first.text_content()
                checked += 1
                if abs(float(text) - value) > 5e-5:
                    bad += 1
                    print(f"MISMATCH {label} {name}: dom={text} state={value}")
            overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth")
            print(f"{label}: {checked} PVs compared, horizontal overflow={overflow}, status={page.locator('[data-testid=status-tag]').text_content()}")
            if a.shot:
                # Snapshots show the central workspace (scene, beam views, side panel), not the browser chrome or top bar.
                page.locator(".workspace").screenshot(path=f"{a.shot}-{label}.png")
        browser.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
