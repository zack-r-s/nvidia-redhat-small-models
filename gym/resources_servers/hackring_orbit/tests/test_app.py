import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.hackring_orbit.app import (
    REPO,
    SETPOINTS,
    HackringOrbitResourcesServer,
    HackringOrbitResourcesServerConfig,
    objective,
    validate_operations,
)

DATASET = REPO / "datasets" / "v001"
EXAMPLES = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"


@pytest.fixture(scope="module")
def server() -> HackringOrbitResourcesServer:
    config = HackringOrbitResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return HackringOrbitResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture()
def client(server: HackringOrbitResourcesServer) -> TestClient:
    return TestClient(server.setup_webserver())


def example_row() -> dict:
    return json.loads(EXAMPLES.read_text().splitlines()[0])


def test_replay_reproduces_recorded_v001_costs(server: HackringOrbitResourcesServer) -> None:
    episodes = {json.loads(line)["episode_id"]: json.loads(line) for line in (DATASET / "episodes.jsonl").open()}
    for index, line in enumerate((DATASET / "attempts.jsonl").open()):
        if index % 32:
            continue
        attempt = json.loads(line)
        offsets = episodes[attempt["episode_id"]]["hidden_calibration_offsets_mrad"]
        before = {sp: attempt["read_before"][sp] for sp in SETPOINTS}
        after = {sp: attempt["read_after"][sp] for sp in SETPOINTS}
        for setpoints, recorded in ((before, attempt["before_metrics"]), (after, attempt["after_metrics"])):
            assert objective(server.ring.pvs(setpoints, offsets))["orbit_cost"] == pytest.approx(
                recorded["orbit_cost"], rel=1e-9, abs=1e-12
            )


def test_validate_operations_matches_small_agent_rules() -> None:
    allowed = ["RING:MAG:H01:KICK:SP", "RING:MAG:V01:KICK:SP"]
    assert validate_operations([{"name": allowed[0], "value": "0.1"}], allowed) == [
        {"channel": allowed[0], "value": 0.1}
    ]
    for bad in (
        [],
        [{"channel": allowed[0], "value": True}],
        [{"channel": allowed[0], "value": 0.36}],
        [{"channel": "RING:MAG:H09:KICK:SP", "value": 0.1}],
        [{"channel": allowed[0], "value": 0.1}, {"channel": allowed[0], "value": 0.2}],
        [{"channel": allowed[0], "value": "nan"}],
        "not a list",
    ):
        with pytest.raises(ValueError):
            validate_operations(bad, allowed)


def test_seeded_write_is_measured_and_rewarded(client: TestClient) -> None:
    row = example_row()
    seeded = client.post("/seed_session", json=row)
    cookies = seeded.cookies
    read = client.post("/channel_read", json={"channels": ["RING:DIAG:BPM01:POSITION:X"]}, cookies=cookies).json()
    first_input = row["responses_create_params"]["input"][3]["output"]
    expected = json.loads(first_input)["summary"]["readings"]["RING:DIAG:BPM01:POSITION:X"]["value"]
    assert read["summary"]["readings"]["RING:DIAG:BPM01:POSITION:X"]["value"] == pytest.approx(expected, abs=1e-12)

    rejected = client.post(
        "/channel_write", json={"operations": [{"channel": "RING:MAG:H01:KICK:SP", "value": 1.0}]}, cookies=cookies
    ).json()
    assert rejected["status"] == "error"

    # The offsets are a known disturbance, so cancelling them with setpoints flattens the orbit.
    offsets = row["episode"]["hidden_calibration_offsets_mrad"]
    operations = [{"channel": sp, "value": -offsets[sp]} for sp in row["episode"]["allowed_correctors"]]
    written = client.post("/channel_write", json={"operations": operations}, cookies=cookies).json()
    assert written["status"] == "success"
    assert {result["outcome"] for result in written["summary"]["results"]} == {"confirmed"}

    response = {"id": "r", "created_at": 0, "model": "m", "object": "response", "output": [],
                "parallel_tool_calls": False, "tool_choice": "required", "tools": []}
    verified = client.post("/verify", json=row | {"response": response}, cookies=cookies).json()
    assert verified["session_found"] and verified["accepted_writes"] == 1 and verified["rejected_writes"] == 1
    assert verified["orbit_cost_final"] < verified["orbit_cost_initial"]
    assert verified["reward"] == pytest.approx(verified["cost_reduction"])
    assert verified["reward"] > 0.9


def test_verify_without_write_scores_zero(client: TestClient) -> None:
    row = example_row()
    cookies = client.post("/seed_session", json=row).cookies
    response = {"id": "r", "created_at": 0, "model": "m", "object": "response", "output": [],
                "parallel_tool_calls": False, "tool_choice": "required", "tools": []}
    verified = client.post("/verify", json=row | {"response": response}, cookies=cookies).json()
    assert verified["reward"] == 0.0 and verified["accepted_writes"] == 0
