"""A read-only operator request must never reach the control-system write tool."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import small_agent


def test_read_request_rejects_model_write(capsys):
    class FakeResponse:
        is_error = False

        def json(self):
            return {
                "choices": [{"message": {"role": "assistant", "tool_calls": [{
                    "id": "bad-write",
                    "type": "function",
                    "function": {"name": "channel_write", "arguments": '{"operations":[{"channel":"RING:MAG:H01:KICK:SP","value":0.05}]}'},
                }]}}]
            }

    class FakeHTTP:
        async def post(self, path, json):
            assert path == "/chat/completions"
            assert {tool["function"]["name"] for tool in json["tools"]} == {"channel_read"}
            return FakeResponse()

    class FakeMCP:
        async def call_tool(self, name, args):
            raise AssertionError(f"unexpected MCP call: {name} {args}")

    tools = [{"type": "function", "function": {"name": name}} for name in ("channel_read", "channel_write")]
    asyncio.run(small_agent.turn(
        "report beam position", [], FakeMCP(), FakeHTTP(), tools,
        {"channel_read": {}, "channel_write": {}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, max_steps=1, history=8),
    ))
    assert "[channel_write rejected]" in capsys.readouterr().out


def test_known_setpoint_uses_contract_units_but_rejects_wrong_unit():
    assert small_agent.requested_write("Set RING:MAG:H01:KICK:SP to 0.2") == (
        "RING:MAG:H01:KICK:SP", 0.2,
    )
    response = small_agent.move_clarification("Set RING:MAG:H01:KICK:SP to 0.2 mm")
    assert response.startswith("No write was made.")
    assert "mrad" in response


def test_model_final_cannot_claim_an_unexecuted_write(capsys):
    prompt = "Set RING:MAG:H01:KICK:SP to 0.2, then read RING:DIAG:BPM01:POSITION:X"

    class FakeResponse:
        is_error = False

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "Set to 0.2 mm."}}]}

    class FakeHTTP:
        async def post(self, path, json):
            assert {tool["function"]["name"] for tool in json["tools"]} == {"channel_read", "channel_write"}
            return FakeResponse()

    class FakeMCP:
        async def call_tool(self, name, args):
            raise AssertionError("model returned no tool call")

    asyncio.run(small_agent.turn(
        prompt, [], FakeMCP(), FakeHTTP(),
        [{"type": "function", "function": {"name": name}} for name in ("channel_read", "channel_write")],
        {"channel_read": {}, "channel_write": {}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, max_steps=1, history=8),
    ))
    output = capsys.readouterr().out
    assert "0.2 mm" not in output
    assert "No write was made" in output
    assert "no verified reading returned" in output


def test_model_write_uses_verified_read_write_read_sequence(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    setpoint = "RING:MAG:H01:KICK:SP"
    readback = "RING:MAG:H01:KICK:RB"
    bpm = "RING:DIAG:BPM01:POSITION:X"
    prompt = f"Set {setpoint} to 0.2, then read {bpm}"

    class FakeResponse:
        is_error = False

        def __init__(self, message):
            self.message = message

        def json(self):
            return {"choices": [{"message": self.message}]}

    class FakeHTTP:
        count = 0

        async def post(self, path, json):
            self.count += 1
            if self.count == 1:
                return FakeResponse({"role": "assistant", "tool_calls": [
                    {"id": "write", "type": "function", "function": {"name": "channel_write",
                        "arguments": '{"operations":[{"name":"RING:MAG:H01:KICK:SP","value":0.2}]}'}},
                    {"id": "read", "type": "function", "function": {"name": "channel_read",
                        "arguments": '{"channels":["RING:DIAG:BPM01:POSITION:X"]}'}},
                ]})
            return FakeResponse({"role": "assistant", "content": "Set to 0.2 mm."})

    class FakeMCP:
        calls = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "channel_write":
                data = {"status": "success", "summary": {"results": [{"channel": setpoint,
                    "outcome": "confirmed", "observed_value": 0.2}]}}
            else:
                values = {setpoint: 0.0, readback: 0.2, bpm: 0.1}
                data = {"status": "success", "summary": {"readings": {
                    pv: {"value": values[pv]} for pv in args["channels"]}}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))], isError=False)

    mcp = FakeMCP()
    history = []
    asyncio.run(small_agent.turn(
        prompt, history, mcp, FakeHTTP(),
        [{"type": "function", "function": {"name": name}} for name in ("channel_read", "channel_write")],
        {"channel_read": {"properties": {"channels": {"type": "array"}}},
         "channel_write": {"properties": {"operations": {"type": "array"}}}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, max_steps=2, history=8),
    ))
    assert [(name, args.get("channels", [None])[0]) for name, args in mcp.calls] == [
        ("channel_read", setpoint), ("channel_write", None),
        ("channel_read", readback), ("channel_read", bpm),
    ]
    output = capsys.readouterr().out
    assert "confirmed for 0.2 mrad" in output
    assert f"{bpm}: 0.1 mm" in output
    assert "Set to 0.2 mm." not in output
    assert [entry["role"] for entry in history] == ["user", "assistant"]
    assert "Post-write readback" not in history[1]["content"]


def test_write_alone_triggers_requested_post_write_read(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    setpoint = "RING:MAG:H01:KICK:SP"
    readback = "RING:MAG:H01:KICK:RB"
    bpm = "RING:DIAG:BPM01:POSITION:X"
    prompt = f"Set {setpoint} to 0.2, then read {bpm}"

    class FakeResponse:
        is_error = False

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "write", "type": "function", "function": {"name": "channel_write",
                    "arguments": json.dumps({"operations": [{"channel": setpoint, "value": 0.2}]})},
            }]}}]}

    class FakeHTTP:
        calls = 0

        async def post(self, path, json):
            self.calls += 1
            assert self.calls == 1, "a second model request should not be needed after a verified write"
            return FakeResponse()

    class FakeMCP:
        calls = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "channel_write":
                data = {"status": "success", "summary": {"results": [{
                    "channel": setpoint, "outcome": "confirmed", "observed_value": 0.2,
                }]}}
            else:
                values = {setpoint: 0.0, readback: 0.2, bpm: 0.7669}
                data = {"status": "success", "summary": {"readings": {
                    pv: {"value": values[pv]} for pv in args["channels"]}}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))], isError=False)

    history = []
    mcp = FakeMCP()
    asyncio.run(small_agent.turn(
        prompt, history, mcp, FakeHTTP(),
        [{"type": "function", "function": {"name": name}} for name in ("channel_read", "channel_write")],
        {"channel_read": {"properties": {"channels": {"type": "array"}}},
         "channel_write": {"properties": {"operations": {"type": "array"}}}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, max_steps=3, history=8),
    ))
    assert [name for name, _ in mcp.calls] == [
        "channel_read", "channel_write", "channel_read", "channel_read",
    ]
    assert mcp.calls[-1][1] == {"channels": [bpm]}
    assert f"{bpm}: 0.7669 mm" in capsys.readouterr().out
    assert [entry["role"] for entry in history] == ["user", "assistant"]


def test_failed_post_write_read_does_not_report_stale_bpm(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    setpoint = "RING:MAG:H01:KICK:SP"
    readback = "RING:MAG:H01:KICK:RB"
    bpm = "RING:DIAG:BPM01:POSITION:X"
    prompt = f"Set {setpoint} to 0.2, then read {bpm}"

    class FakeHTTP:
        async def post(self, path, json):
            return SimpleNamespace(is_error=False, json=lambda: {"choices": [{"message": {
                "role": "assistant", "tool_calls": [
                    {"id": "early", "type": "function", "function": {"name": "channel_read",
                        "arguments": json_module.dumps({"channels": [bpm]})}},
                    {"id": "write", "type": "function", "function": {"name": "channel_write",
                        "arguments": json_module.dumps({"operations": [{"channel": setpoint, "value": 0.2}]})}},
                ],
            }}]})

    class FakeMCP:
        bpm_reads = 0

        async def call_tool(self, name, args):
            if name == "channel_write":
                data = {"status": "success", "summary": {"results": [{
                    "channel": setpoint, "outcome": "confirmed", "observed_value": 0.2,
                }]}}
            elif args["channels"] == [bpm]:
                self.bpm_reads += 1
                data = ({"status": "success", "summary": {"readings": {bpm: {"value": 0.0}}}}
                        if self.bpm_reads == 1 else {"status": "error"})
            else:
                pv = args["channels"][0]
                data = {"status": "success", "summary": {"readings": {
                    pv: {"value": 0.0 if pv == setpoint else 0.2}}}}
            return SimpleNamespace(content=[SimpleNamespace(text=json_module.dumps(data))], isError=False)

    json_module = json
    asyncio.run(small_agent.turn(
        prompt, [], FakeMCP(), FakeHTTP(),
        [{"type": "function", "function": {"name": name}} for name in ("channel_read", "channel_write")],
        {"channel_read": {"properties": {"channels": {"type": "array"}}},
         "channel_write": {"properties": {"operations": {"type": "array"}}}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, max_steps=2, history=8),
    ))
    output = capsys.readouterr().out
    assert f"{bpm}: no verified reading returned." in output
    assert f"{bpm}: 0.0 mm" not in output


def test_goal_request_and_batch_validation():
    assert small_agent.goal_request("Center the horizontal orbit") == "x"
    assert small_agent.goal_request("Correct the vertical beam") == "y"
    assert small_agent.goal_request("Reduce the beam orbit") == "both"
    assert small_agent.goal_request("The vertical orbit needs centering") == "y"
    assert small_agent.goal_request("Steer both transverse planes toward the axis") == "both"
    assert small_agent.goal_request("Steer the horizontal beam nearer the axis") == "x"
    assert small_agent.goal_request("Report the horizontal orbit") is None
    assert small_agent.goal_request("Do not correct the orbit") is None
    assert small_agent.goal_request("Set RING:MAG:H01:KICK:SP to 0.2, then center the orbit") is None
    bpm, allowed = small_agent.goal_channels("x")
    assert len(bpm) == len(allowed) == 4
    assert all(pv.endswith(":X") for pv in bpm)
    assert all(":MAG:H" in pv for pv in allowed)
    assert small_agent.valid_goal_operations([{"channel": allowed[0], "value": "0.05"}], allowed) == [
        {"channel": allowed[0], "value": 0.05}]
    import pytest
    for operations in ([{"channel": "RING:MAG:V01:KICK:SP", "value": 0.05}],
                       [{"channel": allowed[0], "value": 0.4}],
                       [{"channel": allowed[0], "value": float("nan")}],
                       [{"channel": allowed[0], "value": 0.1}, {"channel": allowed[0], "value": 0.2}]):
        with pytest.raises(ValueError):
            small_agent.valid_goal_operations(operations, allowed)


def test_legacy_goal_correction_reads_before_and_after_write(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    bpm, correctors = small_agent.goal_channels("x")
    command = correctors[0]
    rb = small_agent.READBACKS[command]

    class FakeHTTP:
        async def post(self, path, json):
            assert path == "/chat/completions"
            assert [tool["function"]["name"] for tool in json["tools"]] == ["channel_write"]
            assert json["messages"][-1]["role"] == "tool"
            assert json["messages"][-1]["content"].startswith('{"status": "success"')
            return SimpleNamespace(is_error=False, json=lambda: {"choices": [{"message": {
                "role": "assistant", "tool_calls": [{"id": "write", "function": {
                    "name": "channel_write", "arguments": '{"operations":[{"channel":"RING:MAG:H01:KICK:SP","value":0.05}]}'}}]
            }}]})

    class FakeMCP:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, args):
            self.calls.append((name, args))
            if name == "channel_write":
                data = {"status": "success", "summary": {"results": [{
                    "channel": command, "outcome": "confirmed", "observed_value": 0.05}]}}
            else:
                post = len(self.calls) > 2
                data = {"status": "success", "summary": {"readings": {
                    pv: {"value": (0.2 if not post else 0.1) if pv in bpm else
                         0.05 if pv == rb and post else 0.0} for pv in args["channels"]}}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))], isError=False)

    mcp = FakeMCP()
    history = []
    asyncio.run(small_agent.goal_turn("Center the horizontal orbit", "x", history, mcp, FakeHTTP(),
        [{"type": "function", "function": {"name": "channel_read"}},
         {"type": "function", "function": {"name": "channel_write"}}],
        {"channel_write": {"properties": {"operations": {"type": "array"}}}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, history=8)))
    assert [name for name, _ in mcp.calls] == ["channel_read", "channel_write", "channel_read"]
    assert mcp.calls[0][1]["channels"] == bpm + correctors
    assert mcp.calls[1][1]["confirm"] is True
    assert mcp.calls[2][1]["channels"] == bpm + [rb]
    output = capsys.readouterr().out
    assert "X RMS: 0.2000 → 0.1000 mm (measured)." in output
    assert "readback 0.05 mrad" in output
    assert history[-1]["role"] == "assistant"


def test_goal_rejects_wrong_plane_without_write(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    bpm, correctors = small_agent.goal_channels("x")

    class FakeMCP:
        async def call_tool(self, name, args):
            assert name == "channel_read"
            data = {"status": "success", "summary": {"readings": {
                pv: {"value": 0.1} for pv in bpm + correctors}}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))], isError=False)

    class FakeHTTP:
        async def post(self, path, json):
            return SimpleNamespace(is_error=False, json=lambda: {"choices": [{"message": {
                "role": "assistant", "tool_calls": [{"id": "bad", "function": {
                    "name": "channel_write", "arguments": '{"operations":[{"channel":"RING:MAG:V01:KICK:SP","value":0.1}]}'}}]
            }}]})

    asyncio.run(small_agent.goal_turn("Correct the horizontal orbit", "x", [], FakeMCP(), FakeHTTP(),
        [{"type": "function", "function": {"name": "channel_write"}}],
        {"channel_write": {}}, SimpleNamespace(model="test", temperature=0, max_tokens=32, history=8)))
    assert "No write was made" in capsys.readouterr().out


def test_read_only_plane_comparison_uses_measured_bpms(capsys, monkeypatch):
    monkeypatch.setattr(small_agent, "dim", print)
    bpm = small_agent.CONTRACT["bpm"]

    class FakeHTTP:
        async def post(self, path, json):
            assert all(tool["function"]["name"] != "channel_write" for tool in json["tools"])
            return SimpleNamespace(is_error=False, json=lambda: {"choices": [{"message": {
                "role": "assistant", "tool_calls": [{"id": "read", "function": {
                    "name": "channel_read", "arguments": '{"channels":["RING:DIAG:BPM01:POSITION:X"]}'}}]
            }}]})

    class FakeMCP:
        async def call_tool(self, name, args):
            assert name == "channel_read"
            assert args["channels"] == bpm
            values = {pv: (0.2 if pv.endswith(":X") else 0.1) for pv in bpm}
            data = {"status": "success", "summary": {"readings": {
                pv: {"value": value} for pv, value in values.items()}}}
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))], isError=False)

    asyncio.run(small_agent.turn("Which orbit plane has the larger measured RMS?", [], FakeMCP(), FakeHTTP(),
        [{"type": "function", "function": {"name": "channel_read"}},
         {"type": "function", "function": {"name": "channel_write"}}],
        {"channel_read": {"properties": {"channels": {"type": "array"}}}},
        SimpleNamespace(model="test", temperature=0, max_tokens=32, max_steps=2, history=8)))
    output = capsys.readouterr().out
    assert "Measured RMS: X=0.2000 mm, Y=0.1000 mm. The X plane is larger." in output
