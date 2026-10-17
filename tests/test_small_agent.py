"""A read-only operator request must never reach the control-system write tool."""

import asyncio
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
