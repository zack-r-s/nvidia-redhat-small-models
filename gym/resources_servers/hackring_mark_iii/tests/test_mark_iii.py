import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from nemo_gym.server_utils import ServerClient

from resources_servers.hackring_mark_iii.app import HackringMarkIIIResourcesServer
from nemo_gym.base_resources_server import BaseResourcesServerConfig

DATA = Path(__file__).resolve().parents[1] / "data"
RESPONSE = {"id": "r", "created_at": 0, "model": "m", "object": "response", "output": [],
            "parallel_tool_calls": False, "tool_choice": "required", "tools": []}


@pytest.fixture(scope="module")
def client():
    config = BaseResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    server = HackringMarkIIIResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return TestClient(server.setup_webserver())


def row(track: str):
    return json.loads((DATA / f"{track}.jsonl").read_text().splitlines()[0])


def test_seeded_state_matches_frozen_direct_preread(client):
    task = row("direct")
    cookies = client.post("/seed_session", json=task).cookies
    read = client.post("/channel_read", json={"channels": ["RING:DIAG:BPM01:POSITION:X"]}, cookies=cookies).json()
    expected = json.loads(task["responses_create_params"]["input"][3]["output"])["summary"]["readings"]
    assert read["summary"]["readings"]["RING:DIAG:BPM01:POSITION:X"]["value"] == pytest.approx(
        expected["RING:DIAG:BPM01:POSITION:X"]["value"], abs=1e-12)


def test_skill_corrects_once_and_extra_call_invalidates_score(client):
    task = row("skill")
    cookies = client.post("/seed_session", json=task).cookies
    args = {"plane": task["episode"]["plane"], "effort": "balanced"}
    first = client.post("/orbit_correct", json=args, cookies=cookies).json()
    assert first["status"] == "success"
    second = client.post("/orbit_correct", json=args, cookies=cookies).json()
    assert second["status"] == "error"
    report = client.post("/verify", json=task | {"response": RESPONSE}, cookies=cookies).json()
    assert report["tool_attempts"] == 2 and report["accepted_writes"] == 1
    assert report["reward"] == 0 and not report["goal_success"]


def test_valid_skill_has_measured_success(client):
    task = row("skill")
    cookies = client.post("/seed_session", json=task).cookies
    answer = client.post("/orbit_correct", json={"plane": task["episode"]["plane"], "effort": "balanced"}, cookies=cookies).json()
    assert answer["status"] == "success"
    report = client.post("/verify", json=task | {"response": RESPONSE}, cookies=cookies).json()
    assert report["goal_success"] and report["reward"] > 0 and report["raw_intent_correct"]


def test_wrong_plane_and_over_limit_direct_write_do_not_mutate(client):
    skill = row("skill")
    cookies = client.post("/seed_session", json=skill).cookies
    wrong = "y" if skill["episode"]["plane"] == "x" else "x"
    assert client.post("/orbit_correct", json={"plane": wrong, "effort": "balanced"}, cookies=cookies).json()["status"] == "error"
    report = client.post("/verify", json=skill | {"response": RESPONSE}, cookies=cookies).json()
    assert report["target_cost_initial"] == report["target_cost_final"] and report["reward"] == 0

    direct = row("direct")
    cookies = client.post("/seed_session", json=direct).cookies
    operation = {"channel": "RING:MAG:H01:KICK:SP", "value": 0.5}
    assert client.post("/channel_write", json={"operations": [operation]}, cookies=cookies).json()["status"] == "error"
    report = client.post("/verify", json=direct | {"response": RESPONSE}, cookies=cookies).json()
    assert report["target_cost_initial"] == report["target_cost_final"] and report["accepted_writes"] == 0
