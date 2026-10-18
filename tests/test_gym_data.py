import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_gym_data
from small_agent import GOAL_SYSTEM

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "datasets" / "v001" / "splits" / "ft75-eval25" / "split.json"


def test_tool_snapshot_matches_osprey_controls_server():
    server = pytest.importorskip("osprey.mcp_server.control_system.server")
    import osprey.mcp_server.control_system.tools.channel_write  # noqa: F401

    async def listed():
        return {tool.name: tool for tool in (t.to_mcp_tool() for t in await server.mcp.list_tools())}

    live = asyncio.run(listed())["channel_write"]
    snapshot = json.loads((prepare_gym_data.SERVER / "osprey_tools.json").read_text())["tools"]["channel_write"]
    assert snapshot == {"description": live.description, "parameters": live.inputSchema}


def test_rows_replay_goal_turn_without_hidden_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prepare_gym_data.py", str(SPLIT), "--output", str(tmp_path)])
    prepare_gym_data.main()
    record = json.loads(SPLIT.read_text())
    episodes = {json.loads(line)["episode_id"]: json.loads(line)
                for line in (SPLIT.parents[2] / "episodes.jsonl").open()}
    rows = [json.loads(line) for line in (tmp_path / "validation.jsonl").open()]
    assert [row["episode"]["episode_id"] for row in rows] == record["episodes"]["eval"]
    for row in rows:
        params = row["responses_create_params"]
        assert params["input"][0] == {"role": "system", "content": GOAL_SYSTEM}
        assert [tool["name"] for tool in params["tools"]] == ["channel_write"] and params["tool_choice"] == "required"
        readings = json.loads(params["input"][3]["output"])["summary"]["readings"]
        initial = episodes[row["episode"]["episode_id"]]["initial_pvs"]
        assert {pv: reading["value"] for pv, reading in readings.items()} == {pv: initial[pv] for pv in readings}
        # The model-facing request must not leak the hidden disturbance.
        assert "hidden_calibration_offsets_mrad" not in json.dumps(params)
