"""Dev-mode chat bridge: HTTP chat endpoints over the local fixture machine.

DEV FIXTURE ONLY. This is not the Osprey path. It lets chat-UI work proceed without the served
Osprey accelerator:

* ``FixtureControls`` is a thin stand-in for Osprey's ``channel_read`` / ``channel_write`` /
  ``channel_limits`` MCP tools (same tool names, argument shapes and result envelopes), implemented
  over the fixture API (``GET /state``, ``POST /write``). Using that API, not a private machine,
  means the agent's writes show up in the viewer.
* ``live`` mode runs the merged small-model agent's own turn logic (``scripts/small_agent.py``,
  imported unmodified) against that stand-in and the local vLLM model.
* ``scripted`` mode is a deterministic, model-free responder for building the UI.

Merge note: when the real agent gets an HTTP surface, this module should be replaced by it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import math
import os
import re
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hackring.machine import BPMS, CONTRACT, LIMIT, SETPOINTS

ROOT = Path(__file__).resolve().parents[2]
API_URL = os.getenv("HACKRING_API_URL", "http://127.0.0.1:8001")
VLLM_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODE = os.getenv("CHAT_MODE", "auto")  # auto | live | scripted
READBACK = {sp: sp.replace(":SP", ":RB") for sp in SETPOINTS}
UNITS = {**{pv: CONTRACT["units"]["orbit"] for pv in BPMS}, **{pv: CONTRACT["units"]["kick"] for pv in [*SETPOINTS, *READBACK.values()]}}

SCHEMAS = {
    "channel_read": {"type": "object", "properties": {"channels": {"type": "array", "items": {"type": "string"}}, "include_metadata": {"type": "boolean"}}, "required": ["channels"]},
    "channel_write": {"type": "object", "properties": {"operations": {"type": "array", "items": {"type": "object"}}, "confirm": {"type": "boolean"}}, "required": ["operations"]},
    "channel_limits": {"type": "object", "properties": {"channels": {"type": "array", "items": {"type": "string"}}}},
}
DESCRIPTIONS = {
    "channel_read": "Read current values from one or more control-system channels.",
    "channel_write": "Write values to one or more control-system channels. Args: operations: list of {channel, value}.",
    "channel_limits": "Query the channel write-safety limits.",
}


@dataclass
class _Part:
    text: str


@dataclass
class _Result:
    content: list[_Part]


@dataclass
class _Tool:
    name: str
    description: str
    inputSchema: dict


def _result(payload: dict) -> _Result:
    return _Result([_Part(json.dumps(payload))])


class FixtureControls:
    """Stand-in for Osprey's controls tools over the fixture API. Mirrors the tools' result envelopes."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    def tools(self) -> list[_Tool]:
        return [_Tool(name, DESCRIPTIONS[name], SCHEMAS[name]) for name in SCHEMAS]

    async def _pvs(self) -> dict[str, float]:
        response = await self.client.get("/state")
        response.raise_for_status()
        return response.json()["pvs"]

    async def call_tool(self, name: str, args: dict) -> _Result:
        if name == "channel_read":
            return await self._read(args)
        if name == "channel_write":
            return await self._write(args)
        if name == "channel_limits":
            channels = args.get("channels") or SETPOINTS
            return _result({"status": "success", "summary": {"limits": {pv: {"min_value": -LIMIT, "max_value": LIMIT, "unit": "mrad"} for pv in channels if pv in SETPOINTS}}})
        return _result({"status": "error", "message": f"unknown tool {name!r}"})

    async def _read(self, args: dict) -> _Result:
        names = args.get("channels") or []
        pvs = await self._pvs()
        unknown = [pv for pv in names if pv not in pvs]
        if unknown:
            return _result({"status": "error", "message": f"unknown channels: {unknown}"})
        return _result({"status": "success", "summary": {"readings": {pv: {"value": pvs[pv], "unit": UNITS.get(pv)} for pv in names}}})

    async def _write(self, args: dict) -> _Result:
        results: list[dict] = []
        accepted: dict[str, float] = {}
        for op in args.get("operations") or []:
            channel, value = (op.get("channel"), op.get("value")) if isinstance(op, dict) else (None, None)
            if channel not in SETPOINTS:
                results.append({"channel": channel, "outcome": "refused", "refusal_reason": "CHANNEL_NOT_FOUND"})
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > LIMIT:
                results.append({"channel": channel, "outcome": "refused", "refusal_reason": "LIMIT_VIOLATION", "requested_value": value})
            else:
                accepted[channel] = float(value)
        if accepted:
            response = await self.client.post("/write", json={"settings": accepted})
            if response.is_error:
                results.extend({"channel": pv, "outcome": "failed", "message": response.text[:200]} for pv in accepted)
            else:
                pvs = await self._pvs()
                for pv, value in accepted.items():
                    observed = pvs.get(pv)
                    outcome = "confirmed" if observed is not None and math.isclose(observed, value, abs_tol=1e-9) else "mismatch"
                    results.append({"channel": pv, "outcome": outcome, "observed_value": observed})
        if results and all(entry["outcome"] in ("refused", "failed") for entry in results):
            return _result({"status": "error", "message": "no value reached any channel: " + "; ".join(f"{e['channel']}: {e.get('refusal_reason', e['outcome'])}" for e in results)})
        outcomes: dict[str, int] = {}
        for entry in results:
            outcomes[entry["outcome"]] = outcomes.get(entry["outcome"], 0) + 1
        return _result({"status": "success", "summary": {"outcomes": outcomes, "results": results}})


# ---- turn results -----------------------------------------------------------------------------


@dataclass
class TurnResult:
    reply: str
    events: list[dict] = field(default_factory=list)
    mode: str = "scripted"
    model: str | None = None
    elapsed_s: float = 0.0


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)


def _event(text: str) -> dict:
    text = text.strip()
    match = re.match(r"\[(?P<tool>[a-z_]+)(?: · (?P<phase>[^\]]+))?(?P<rej> rejected)?\]\s*(?P<detail>.*)", text)
    if not match:
        return {"kind": "note", "text": text}
    return {"kind": "tool", "tool": match["tool"], "phase": match["phase"], "rejected": bool(match["rej"]), "detail": match["detail"]}


# ---- scripted (model-free) responder ----------------------------------------------------------

_SET = re.compile(r"(RING:MAG:[HV]0[1-4]:KICK:SP)[^\n]*?\bto\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
_GOAL = re.compile(r"\b(correct|center|centre|reduce|minimi[sz]e|flatten|restore)\b", re.IGNORECASE)
_BEAM = re.compile(r"\b(beam|orbit|bpm)\b", re.IGNORECASE)
_READ = re.compile(r"\b(read|show|report|measure|get|status|position)\b", re.IGNORECASE)
HELP = ("Scripted dev responder (no model). Try: “report the beam position”, “set RING:MAG:H01:KICK:SP to 0.1”, or “correct the orbit”. "
        "Start vLLM and use live mode for the real small model.")
# Rough orbit response of BPM n to corrector n, mm per mrad (from the fixture; only used to size a scripted step).
_MM_PER_MRAD = 5.0


async def scripted_turn(prompt: str, controls: FixtureControls) -> tuple[str, list[dict]]:
    events: list[dict] = []

    async def call(name: str, args: dict, phase: str | None = None) -> dict:
        events.append({"kind": "tool", "tool": name, "phase": phase, "rejected": False, "detail": json.dumps(args)})
        return json.loads((await controls.call_tool(name, args)).content[0].text)

    def beam_report(readings: dict) -> str:
        lines = ["Measured beam position (mm):"]
        for i in range(0, len(BPMS), 2):
            x, y = BPMS[i : i + 2]
            lines.append(f"  {x.split(':')[2]}: X={readings[x]['value']:+.4f}, Y={readings[y]['value']:+.4f}")
        return "\n".join(lines)

    def rms(readings: dict, axis: str) -> float:
        values = [readings[pv]["value"] for pv in BPMS if pv.endswith(f":{axis}")]
        return math.sqrt(sum(v * v for v in values) / len(values))

    set_match = _SET.search(prompt)
    if set_match and re.search(r"\b(set|write|change|adjust)\b", prompt, re.IGNORECASE):
        channel, value = set_match.group(1).upper(), float(set_match.group(2))
        await call("channel_read", {"channels": [channel]}, "pre-write")
        wrote = await call("channel_write", {"operations": [{"channel": channel, "value": value}]})
        if wrote["status"] != "success":
            return f"No write was made. {wrote['message']}", events
        entry = wrote["summary"]["results"][0]
        after = await call("channel_read", {"channels": [READBACK[channel]]}, "post-write")
        rb = after["summary"]["readings"][READBACK[channel]]["value"]
        return f"{channel}: dev fixture reported {entry['outcome']} for {value:g} mrad; readback {rb} mrad.", events
    if _GOAL.search(prompt) and _BEAM.search(prompt):
        before = await call("channel_read", {"channels": BPMS}, "pre-write")
        readings = before["summary"]["readings"]
        ops = []
        for n in range(1, 5):
            for plane, axis in (("H", "X"), ("V", "Y")):
                kick = max(-LIMIT, min(LIMIT, -readings[f"RING:DIAG:BPM0{n}:POSITION:{axis}"]["value"] / _MM_PER_MRAD))
                ops.append({"channel": f"RING:MAG:{plane}0{n}:KICK:SP", "value": round(kick, 4)})
        wrote = await call("channel_write", {"operations": ops})
        if wrote["status"] != "success":
            return f"No write was made. {wrote['message']}", events
        after = (await call("channel_read", {"channels": BPMS}, "post-write"))["summary"]["readings"]
        return "\n".join([
            f"Applied {len(ops)} corrector setpoints (scripted proportional step, not the model).",
            f"X RMS: {rms(readings, 'X'):.4f} → {rms(after, 'X'):.4f} mm (measured).",
            f"Y RMS: {rms(readings, 'Y'):.4f} → {rms(after, 'Y'):.4f} mm (measured).",
        ]), events
    if _BEAM.search(prompt) and _READ.search(prompt):
        data = await call("channel_read", {"channels": BPMS})
        return beam_report(data["summary"]["readings"]), events
    return HELP, events


# ---- live (small model) turns -----------------------------------------------------------------

_small_agent: Any = None


def load_small_agent() -> Any:
    """Import scripts/small_agent.py unmodified, and route its console output to the bridge."""
    global _small_agent
    if _small_agent is None:
        spec = importlib.util.spec_from_file_location("small_agent", ROOT / "scripts" / "small_agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        _small_agent = module
    return _small_agent


def model_name() -> str:
    agent = load_small_agent()
    return os.getenv("VLLM_MODEL_NAME") or agent.PROVIDER["default_model"]


async def live_turn(prompt: str, session: Session, controls: FixtureControls) -> tuple[str, list[dict]]:
    agent = load_small_agent()
    events: list[str] = []
    agent.dim = lambda text: events.append(text)  # the agent's trace lines; its answers use print()
    tools = [agent.to_openai(tool) for tool in controls.tools()]
    args = SimpleNamespace(model=model_name(), temperature=0.2, max_tokens=512, max_steps=5, history=8)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        async with httpx.AsyncClient(base_url=VLLM_URL, timeout=300) as http:
            try:
                clarification = agent.move_clarification(prompt)
                if clarification:
                    print(clarification)
                else:
                    await agent.turn(prompt, session.history, controls, http, tools, dict(SCHEMAS), args)
            except (httpx.HTTPError, KeyError) as exc:
                print(f"model request failed: {exc}")
    # The agent's wording credits "Osprey"; on this stand-in that would be false, so say what answered.
    relabel = lambda text: re.sub(r"\bOsprey\b", "dev fixture", text)
    return relabel(buffer.getvalue().strip() or "(no reply)"), [_event(relabel(line)) for line in events]


# ---- app --------------------------------------------------------------------------------------

app = FastAPI(title="Hackring dev chat bridge", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])
_sessions: dict[str, Session] = {}
_lock = asyncio.Lock()  # live turns redirect stdout, so run one turn at a time


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ClearRequest(BaseModel):
    session_id: str = "default"


async def _model_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as http:
            response = await http.get(f"{VLLM_URL}/models")
        return response.is_success and any(m.get("id") == model_name() for m in response.json().get("data", []))
    except (httpx.HTTPError, ValueError):
        return False


async def _fixture_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as http:
            return (await http.get(f"{API_URL}/state")).is_success
    except httpx.HTTPError:
        return False


async def resolve_mode() -> str:
    if MODE in ("live", "scripted"):
        return MODE
    return "live" if await _model_ready() else "scripted"


@app.get("/chat/health")
async def health() -> dict:
    model_ok = await _model_ready()
    return {
        "mode": await resolve_mode(),
        "requested_mode": MODE,
        "backend": "dev fixture (local PyAT API); not Osprey",
        "fixture": {"url": API_URL, "ok": await _fixture_ready()},
        "model": {"url": VLLM_URL, "ok": model_ok},
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    started = time.monotonic()
    mode = await resolve_mode()
    session = _sessions.setdefault(request.session_id, Session())
    message = request.message.strip()
    if not message:
        return {"reply": "", "events": [], "mode": mode, "elapsed_s": 0.0}
    if not await _fixture_ready():
        return {"reply": f"The fixture machine API is not reachable at {API_URL}; start it (scripts/dev_up.sh) and retry.", "events": [], "mode": mode, "error": "fixture_unavailable", "elapsed_s": 0.0}
    if mode == "live" and not await _model_ready():
        return {"reply": f"No model server at {VLLM_URL}. Start vLLM, or run the bridge with CHAT_MODE=scripted.", "events": [], "mode": mode, "error": "model_unavailable", "elapsed_s": 0.0}
    async with _lock:
        async with httpx.AsyncClient(base_url=API_URL, timeout=30) as http:
            controls = FixtureControls(http)
            if mode == "live":
                reply, events = await live_turn(message, session, controls)
            else:
                reply, events = await scripted_turn(message, controls)
                session.history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
    return {"reply": reply, "events": events, "mode": mode, "model": model_name() if mode == "live" else None, "elapsed_s": round(time.monotonic() - started, 2)}


@app.post("/chat/clear")
async def clear(request: ClearRequest) -> dict:
    _sessions.pop(request.session_id, None)
    return {"cleared": True}
