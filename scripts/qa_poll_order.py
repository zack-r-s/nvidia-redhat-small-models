"""Check the viewer never has two /state requests in flight and never shows an older snapshot after a newer one.

Intercepts /state in the browser with delayed, numbered replies (a slow first reply, then fast ones) and
watches which reply the page displays. Needs: playwright (not a project dependency) and the viewer
running (npm run dev) with the machine API up (only used as a template snapshot).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
from playwright.async_api import async_playwright

DELAYS = [1.2, 0.05, 0.05, 0.05, 0.6, 0.05]  # first reply is slow; later ones are fast


async def run(viewer: str, api: str, chrome: str | None) -> int:
    base = httpx.get(f"{api}/state").json()
    in_flight = max_in_flight = count = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=chrome, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        page = await browser.new_page()

        async def handle(route):
            nonlocal count
            n, count = count, count + 1
            await asyncio.sleep(DELAYS[n % len(DELAYS)])
            state = json.loads(json.dumps(base))
            # A strictly increasing marker reveals which reply the page is displaying.
            state["pvs"]["RING:DIAG:BPM01:POSITION:X"] = float(n)
            await route.fulfill(json=state, headers={"access-control-allow-origin": "*"})

        def opened(request):
            nonlocal in_flight, max_in_flight
            if request.url.endswith("/state"):
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)

        def closed(request):
            nonlocal in_flight
            if request.url.endswith("/state"):
                in_flight -= 1

        # Browser-side counts: an aborted request frees its slot, a pending one does not.
        page.on("request", opened)
        page.on("requestfinished", closed)
        page.on("requestfailed", closed)
        await page.route("**/state", handle)
        await page.goto(viewer)
        await page.wait_for_selector("[data-testid=bpm-table]", state="attached", timeout=20000)
        max_in_flight = in_flight  # ignore the dev-mode (StrictMode) startup handoff; measure steady state
        seen: list[float] = []
        for _ in range(60):
            await page.wait_for_timeout(100)
            seen.append(float(await page.locator("[data-testid='pv-RING:DIAG:BPM01:POSITION:X']").first.text_content()))
        await browser.close()

    backwards = [(x, y) for x, y in zip(seen, seen[1:]) if y < x]
    print(f"requests={count} max_in_flight={max_in_flight} backwards_steps={len(backwards)} last_seen={seen[-1]}")
    return 1 if max_in_flight > 1 or backwards else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", default="http://127.0.0.1:5173")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--chrome", help="Chrome/Chromium executable if Playwright's bundled browser is unavailable")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.viewer, args.api, args.chrome)))
