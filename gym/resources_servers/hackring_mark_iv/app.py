"""Mark IV Gym resource server: one seeded PyAT plant per operator prompt."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig, BaseSeedSessionRequest, BaseSeedSessionResponse,
    BaseVerifyRequest, BaseVerifyResponse, SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "gym"))
from hackring.control import solve  # noqa: E402
from hackring.machine import LIMIT, SETPOINTS, LocalMachine  # noqa: E402
from mark_iv_common import DISTURBANCE_MRAD, cases, score  # noqa: E402
from resources_servers.hackring_orbit.app import validate_operations  # noqa: E402

CASES = {case["id"]: case for case in cases()}


class Episode(BaseModel):
    episode_id: str
    seed: int
    prompt: str


class SeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="ignore")
    episode: Episode


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="ignore")
    episode: Episode


class VerifyResponse(BaseVerifyResponse):
    episode_id: str
    kind: str
    session_found: bool
    plant_success: bool
    evidence_success: bool
    goal_success: bool
    accepted_writes: int
    rejected_writes: int
    read_calls: int
    before_x_rms_mm: float
    before_y_rms_mm: float
    after_x_rms_mm: float
    after_y_rms_mm: float


class ReadRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    channels: Any = None


class WriteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operations: Any = None
    confirm: Any = None


class SkillRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    plane: Any = None
    effort: Any = None


class Session(BaseModel):
    machine: Any
    before: dict
    reads: list[dict] = Field(default_factory=list)
    writes: list[dict] = Field(default_factory=list)
    write_calls: int = 0


class MarkIVResourcesServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig
    sessions: Dict[str, Session] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/channel_read")(self.channel_read)
        app.post("/channel_write")(self.channel_write)
        app.post("/channel_limits")(self.channel_limits)
        app.post("/orbit_correct")(self.orbit_correct)
        return app

    async def seed_session(self, request: Request, body: SeedRequest) -> BaseSeedSessionResponse:
        episode = body.episode
        case = CASES.get(episode.episode_id)
        if case is None or case["seed"] != episode.seed or case["prompt"] != episode.prompt:
            raise ValueError("Mark IV episode differs from the frozen case")
        machine = LocalMachine()
        before = machine.reset(episode.seed, DISTURBANCE_MRAD)
        self.sessions[request.session[SESSION_ID_KEY]] = Session(machine=machine, before=before)
        return BaseSeedSessionResponse()

    async def channel_read(self, request: Request, body: ReadRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode"}
        channels = body.channels
        pvs = session.machine.read()["pvs"]
        ok = (isinstance(channels, list) and bool(channels) and
              all(isinstance(name, str) and name in pvs for name in channels))
        session.reads.append({"channels": channels if isinstance(channels, list) else [],
                              "ok": ok, "after_write": any(op["outcome"] == "confirmed" for op in session.writes)})
        if not ok:
            return {"status": "error", "message": "unknown or empty channel list"}
        return {"status": "success", "summary": {"readings": {name: {"value": pvs[name]} for name in channels}}}

    async def channel_limits(self, request: Request, body: ReadRequest) -> dict:
        if request.session[SESSION_ID_KEY] not in self.sessions:
            return {"status": "error", "message": "no seeded episode"}
        channels = body.channels if isinstance(body.channels, list) else SETPOINTS
        return {"status": "success", "summary": {"limits": {
            name: {"min_value": -LIMIT, "max_value": LIMIT, "unit": "mrad"}
            for name in channels if name in SETPOINTS}}}

    def _write(self, session: Session, operations: Any, *, skill_reads_plant: bool = False) -> dict:
        session.write_calls += 1
        if session.write_calls != 1:
            session.writes.append({"channel": None, "outcome": "refused", "reason": "more than one write batch"})
            return {"status": "error", "message": "more than one write batch"}
        if not skill_reads_plant and not any(entry["ok"] for entry in session.reads):
            session.writes.append({"channel": None, "outcome": "refused", "reason": "no pre-write PV read"})
            return {"status": "error", "message": "Read the plant before writing a setpoint"}
        try:
            clean = validate_operations(operations, SETPOINTS)
            state = session.machine.write({op["channel"]: op["value"] for op in clean})["state"]
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            session.writes.append({"channel": None, "outcome": "refused", "reason": str(exc)})
            return {"status": "error", "message": f"No write was made: {exc}"}
        results = [{"channel": op["channel"], "outcome": "confirmed",
                    "observed_value": state["pvs"][op["channel"]]} for op in clean]
        session.writes.extend(results)
        return {"status": "success", "summary": {"results": results}}

    async def channel_write(self, request: Request, body: WriteRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode"}
        return self._write(session, body.operations)

    async def orbit_correct(self, request: Request, body: SkillRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode"}
        if body.plane not in ("x", "y", "both") or body.effort not in ("gentle", "balanced", "aggressive"):
            session.writes.append({"channel": None, "outcome": "refused", "reason": "invalid skill arguments"})
            return {"status": "error", "message": "invalid plane or effort"}
        try:
            settings = solve(session.machine.read()["pvs"], body.plane, body.effort)
        except (ValueError, KeyError, RuntimeError) as exc:
            session.writes.append({"channel": None, "outcome": "refused", "reason": str(exc)})
            return {"status": "error", "message": str(exc)}
        return self._write(session, [{"channel": name, "value": value} for name, value in settings.items()],
                           skill_reads_plant=True)

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        case = CASES[body.episode.episode_id]
        session = self.sessions.pop(request.session[SESSION_ID_KEY], None)
        if session is None:
            machine = LocalMachine()
            before = machine.reset(case["seed"], DISTURBANCE_MRAD)
            after = before
            reads, writes = [], []
        else:
            before, after = session.before, session.machine.read()
            reads, writes = session.reads, session.writes
        result = score(case, before, after, reads, writes)
        return VerifyResponse(**body.model_dump(), reward=float(result["success"]),
                              episode_id=case["id"], kind=case["kind"], session_found=session is not None,
                              plant_success=result["plant_success"], evidence_success=result["evidence_success"],
                              goal_success=result["success"], accepted_writes=result["accepted_writes"],
                              rejected_writes=result["rejected_writes"], read_calls=result["read_calls"],
                              before_x_rms_mm=result["before_x_rms_mm"], before_y_rms_mm=result["before_y_rms_mm"],
                              after_x_rms_mm=result["after_x_rms_mm"], after_y_rms_mm=result["after_y_rms_mm"])


if __name__ == "__main__":
    MarkIVResourcesServer.run_webserver()
