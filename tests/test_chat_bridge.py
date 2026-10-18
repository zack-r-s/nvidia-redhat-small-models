"""Dev chat bridge: the Osprey-shaped tool stand-in and the scripted responder, on the real fixture API."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from hackring import api
from hackring.chat_bridge import FixtureControls, load_small_agent, scripted_turn
from hackring.machine import BPMS, LIMIT, SETPOINTS


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def controls():
    api.machine.reset(seed=7)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://fixture")
    yield FixtureControls(client)
    run(client.aclose())


def payload(result) -> dict:
    return json.loads(result.content[0].text)


def test_read_envelope_parses_with_the_agents_own_reader(controls):
    text = run(controls.call_tool("channel_read", {"channels": BPMS})).content[0].text
    readings = load_small_agent().readings_from(text)
    assert set(readings) == set(BPMS) and all(isinstance(r["value"], float) for r in readings.values())


def test_write_is_confirmed_and_visible_in_state(controls):
    out = payload(run(controls.call_tool("channel_write", {"operations": [{"channel": SETPOINTS[0], "value": 0.1}]})))
    assert out["status"] == "success" and out["summary"]["results"][0]["outcome"] == "confirmed"
    assert api.machine.read()["pvs"][SETPOINTS[0]] == pytest.approx(0.1)


def test_out_of_limit_write_is_refused_and_state_unchanged(controls):
    before = api.machine.read()["pvs"]
    out = payload(run(controls.call_tool("channel_write", {"operations": [{"channel": SETPOINTS[0], "value": LIMIT + 0.1}]})))
    assert out["status"] == "error"
    assert api.machine.read()["pvs"] == before


def test_unknown_channel_is_refused(controls):
    out = payload(run(controls.call_tool("channel_write", {"operations": [{"channel": "RING:NOPE", "value": 0.0}]})))
    assert out["status"] == "error" and "CHANNEL_NOT_FOUND" in out["message"]


def test_scripted_beam_report(controls):
    reply, events = run(scripted_turn("report the beam position", controls))
    assert "BPM01" in reply and events[0]["tool"] == "channel_read"


def test_scripted_explicit_write(controls):
    reply, events = run(scripted_turn(f"set {SETPOINTS[0]} to 0.05", controls))
    assert "confirmed" in reply and [e["phase"] for e in events if e["tool"] == "channel_read"] == ["pre-write", "post-write"]
    assert api.machine.read()["pvs"][SETPOINTS[0]] == pytest.approx(0.05)


def test_scripted_write_over_limit_makes_no_write(controls):
    before = api.machine.read()["pvs"]
    reply, _ = run(scripted_turn(f"set {SETPOINTS[0]} to 0.9", controls))
    assert reply.startswith("No write was made")
    assert api.machine.read()["pvs"] == before


def test_scripted_goal_measures_before_and_after(controls):
    reply, events = run(scripted_turn("correct the orbit", controls))
    assert "X RMS:" in reply and "Y RMS:" in reply
    assert [e["tool"] for e in events] == ["channel_read", "channel_write", "channel_read"]


def test_live_replies_never_credit_osprey(controls, monkeypatch):
    """The agent's wording says "Osprey"; the bridge must not pass that off as the real thing."""
    import hackring.chat_bridge as bridge

    agent = load_small_agent()

    async def fake_turn(prompt, history, mcp, http, tools, schemas, args):
        agent.dim("  [channel_write] {}")
        print("RING:MAG:H01:KICK:SP: Osprey reported confirmed for 0.1 mrad.")

    monkeypatch.setattr(agent, "turn", fake_turn)
    reply, events = run(bridge.live_turn("Set RING:MAG:H01:KICK:SP to 0.1", bridge.Session(), controls))
    assert "Osprey" not in reply and "dev fixture reported confirmed" in reply
    assert events[0]["tool"] == "channel_write"


def test_live_v004_skill_changes_the_viewers_fixture(controls, monkeypatch):
    """The real agent turn must route through the skill and write the shared fixture API."""
    import hackring.chat_bridge as bridge

    monkeypatch.setenv("VLLM_MODEL_NAME", "candidate-v004")

    class ModelHTTP:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, path, json):
            assert path == "/chat/completions"
            assert json["model"] == "candidate-v004"
            assert json["tools"][0]["function"]["name"] == "orbit_correct"
            return SimpleNamespace(is_error=False, json=lambda: {"choices": [{"message": {
                "tool_calls": [{"function": {"name": "orbit_correct",
                                              "arguments": '{"plane":"both","effort":"balanced"}'}}]
            }}]})

    monkeypatch.setattr(bridge.httpx, "AsyncClient", ModelHTTP)
    before = api.machine.read()["pvs"]
    reply, events = run(bridge.live_turn("Correct the beam orbit.", bridge.Session(), controls))
    after = api.machine.read()["pvs"]
    assert after != before
    assert "X RMS:" in reply and "Y RMS:" in reply and "Osprey" not in reply
    assert [event["tool"] for event in events if event["kind"] == "tool"] == [
        "orbit_correct", "channel_read", "channel_write", "channel_read"
    ]
