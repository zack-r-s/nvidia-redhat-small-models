"""Measured skill execution and model-routing guards on the fixed local ring."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hackring.control import requested_effort, rms, solve
from hackring.machine import LocalMachine
import small_agent


def test_bounded_solve_improves_measured_orbit():
    for plane in ("x", "y", "both"):
        machine = LocalMachine()
        before = machine.reset(20269991, 0.08)["pvs"]
        settings = solve(before, plane, "gentle")
        assert settings and all(abs(value) <= 0.35 for value in settings.values())
        after = machine.write(settings)["state"]["pvs"]
        for target in (("x", "y") if plane == "both" else (plane,)):
            assert rms(after, target) < rms(before, target)


def test_effort_intent_defaults_to_balanced():
    assert requested_effort("Center the beam") == "balanced"
    assert requested_effort("Center the beam gently with small kicks") == "gentle"
    assert requested_effort("Center the beam, prioritizing the lowest residual") == "aggressive"


def test_skill_turn_uses_measured_pvs_and_rejects_wrong_plane(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    class FakeHTTP:
        def __init__(self, plane, effort="balanced"):
            self.planes = plane if isinstance(plane, list) else [plane]
            self.effort = effort
            self.calls = 0

        async def post(self, path, json):
            assert path == "/chat/completions"
            assert [tool["function"]["name"] for tool in json["tools"]] == ["orbit_correct"]
            plane = self.planes[min(self.calls, len(self.planes) - 1)]
            self.calls += 1
            return SimpleNamespace(is_error=False, json=lambda: {"choices": [{"message": {
                "tool_calls": [{"function": {"name": "orbit_correct", "arguments":
                                           __import__("json").dumps({"plane": plane, "effort": self.effort})}}]}}]})

    class FakeMCP:
        def __init__(self):
            self.machine = LocalMachine()
            self.machine.reset(20269992, 0.08)
            self.writes = 0

        async def call_tool(self, name, args):
            if name == "channel_read":
                pvs = self.machine.read()["pvs"]
                result = {"status": "success", "summary": {"readings": {
                    pv: {"value": pvs[pv]} for pv in args["channels"]}}}
            else:
                assert name == "channel_write" and args["confirm"] is True
                self.writes += 1
                settings = {op["channel"]: op["value"] for op in args["operations"]}
                self.machine.write(settings)
                result = {"status": "success", "summary": {"results": [
                    {"channel": pv, "outcome": "confirmed"} for pv in settings]}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(result))])

    args = SimpleNamespace(model="test", temperature=0, max_tokens=64, history=8)
    rejected = FakeMCP()
    asyncio.run(small_agent.skill_goal_turn("Center the horizontal orbit", "x", [], rejected,
                                            FakeHTTP("y"), args))
    assert rejected.writes == 0
    assert "No write was made" in capsys.readouterr().out

    accepted = FakeMCP()
    before = accepted.machine.read()["pvs"]
    asyncio.run(small_agent.skill_goal_turn("Center the horizontal orbit", "x", [], accepted,
                                            FakeHTTP("x"), args))
    after = accepted.machine.read()["pvs"]
    assert accepted.writes == 1
    assert rms(after, "x") < rms(before, "x")
    assert "(measured)" in capsys.readouterr().out

    retried = FakeMCP()
    model = FakeHTTP(["y", "x"])
    asyncio.run(small_agent.skill_goal_turn("Center the horizontal orbit", "x", [], retried,
                                            model, args))
    assert model.calls == 2
    assert retried.writes == 1
    assert "model_choice · retry" in capsys.readouterr().out

    normalized = FakeMCP()
    asyncio.run(small_agent.skill_goal_turn("Center the horizontal orbit", "x", [], normalized,
                                            FakeHTTP("x", "aggressive"), args))
    assert normalized.writes == 1
    assert "effort normalized" in capsys.readouterr().out
