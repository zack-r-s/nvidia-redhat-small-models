"""Mark III Gym task on the seed-disjoint local PyAT fixture.

The model receives either the deployed direct-write pre-read or the v004 skill
selection prompt. Each session owns a fresh LocalMachine and permits one tool
attempt. Scoring uses only its measured requested-plane BPM values.
"""

from __future__ import annotations

import math
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
sys.path.insert(0, str(REPO / "gym"))
from hackring.control import requested_effort, solve  # noqa: E402
from hackring.machine import BPMS, SETPOINTS, LocalMachine  # noqa: E402
from resources_servers.hackring_orbit.app import validate_operations  # noqa: E402


def axes(plane: str) -> tuple[str, ...]:
    return ("x", "y") if plane == "both" else (plane,)


def target_cost(state: dict, plane: str) -> float:
    return sum(state["summary"][f"{axis}_rms_mm"] ** 2 for axis in axes(plane))


class Episode(BaseModel):
    episode_id: str
    seed: int
    disturbance_mrad: float = 0.08
    plane: str
    prompt: str
    interface: str


class SeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="ignore")
    episode: Episode


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="ignore")
    episode: Episode


class VerifyResponse(BaseVerifyResponse):
    episode_id: str
    interface: str
    session_found: bool
    tool_attempts: int
    accepted_writes: int
    rejected_calls: int
    selected_plane: str | None
    selected_effort: str | None
    raw_intent_correct: bool | None
    goal_success: bool
    x_rms_initial_mm: float
    y_rms_initial_mm: float
    x_rms_final_mm: float
    y_rms_final_mm: float
    target_cost_initial: float
    target_cost_final: float
    cost_reduction: float


class WriteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operations: Any = None


class ReadRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    channels: Any = None


class SkillRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    plane: Any = None
    effort: Any = None


class Session(BaseModel):
    machine: Any
    episode: Episode
    before: dict
    tool_attempts: int = 0
    accepted_writes: int = 0
    rejected_calls: int = 0
    selected_plane: str | None = None
    selected_effort: str | None = None


class HackringMarkIIIResourcesServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig
    sessions: Dict[str, Session] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/channel_read")(self.channel_read)
        app.post("/channel_write")(self.channel_write)
        app.post("/orbit_correct")(self.orbit_correct)
        return app

    async def seed_session(self, request: Request, body: SeedRequest) -> BaseSeedSessionResponse:
        episode = body.episode
        if episode.interface not in ("channel_write", "orbit_correct") or episode.plane not in ("x", "y", "both"):
            raise ValueError("invalid Mark III episode interface or plane")
        machine = LocalMachine()
        before = machine.reset(episode.seed, episode.disturbance_mrad)
        self.sessions[request.session[SESSION_ID_KEY]] = Session(machine=machine, episode=episode, before=before)
        return BaseSeedSessionResponse()

    async def channel_read(self, request: Request, body: ReadRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode"}
        pvs = session.machine.read()["pvs"]
        channels = body.channels
        if not isinstance(channels, list) or not channels or any(pv not in pvs for pv in channels):
            return {"status": "error", "message": "unknown or empty channel list"}
        return {"status": "success", "summary": {"readings": {pv: {"value": pvs[pv]} for pv in channels}}}

    def _write(self, session: Session, operations: Any) -> dict:
        allowed = [pv for pv in SETPOINTS if session.episode.plane == "both" or
                   (":MAG:H" in pv if session.episode.plane == "x" else ":MAG:V" in pv)]
        try:
            clean = validate_operations(operations, allowed)
            result = session.machine.write({op["channel"]: op["value"] for op in clean})
        except (ValueError, KeyError, RuntimeError) as exc:
            session.rejected_calls += 1
            return {"status": "error", "message": f"No write was made: {exc}"}
        session.accepted_writes += 1
        return {"status": "success", "summary": {"results": [
            {"channel": op["channel"], "outcome": "confirmed",
             "observed_value": result["state"]["pvs"][op["channel"].replace(":SP", ":RB")]}
            for op in clean]}}

    async def channel_write(self, request: Request, body: WriteRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode"}
        session.tool_attempts += 1
        if session.episode.interface != "channel_write" or session.tool_attempts != 1:
            session.rejected_calls += 1
            return {"status": "error", "message": "wrong interface or more than one tool attempt"}
        return self._write(session, body.operations)

    async def orbit_correct(self, request: Request, body: SkillRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode"}
        session.tool_attempts += 1
        session.selected_plane = body.plane if isinstance(body.plane, str) else None
        session.selected_effort = body.effort if isinstance(body.effort, str) else None
        if session.episode.interface != "orbit_correct" or session.tool_attempts != 1:
            session.rejected_calls += 1
            return {"status": "error", "message": "wrong interface or more than one tool attempt"}
        wanted = requested_effort(session.episode.prompt)
        if body.plane != session.episode.plane or body.effort not in ("gentle", "balanced", "aggressive") or (
            wanted != "balanced" and body.effort != wanted
        ):
            session.rejected_calls += 1
            return {"status": "error", "message": "skill plane or explicit effort differs from operator request"}
        effort = wanted if wanted == "balanced" else body.effort
        try:
            operations = [{"channel": pv, "value": value} for pv, value in
                          solve(session.machine.read()["pvs"], body.plane, effort).items()]
        except (ValueError, RuntimeError, KeyError) as exc:
            session.rejected_calls += 1
            return {"status": "error", "message": f"No write was made: {exc}"}
        return self._write(session, operations)

    async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
        session = self.sessions.pop(request.session[SESSION_ID_KEY], None)
        if session is None:
            machine = LocalMachine()
            before = machine.reset(body.episode.seed, body.episode.disturbance_mrad)
            after = before
        else:
            before = session.before
            after = session.machine.read()
        initial = target_cost(before, body.episode.plane)
        final = target_cost(after, body.episode.plane)
        reduction = (initial - final) / initial if initial > 0 else 0.0
        compliant = bool(session and session.tool_attempts == 1 and session.accepted_writes == 1 and session.rejected_calls == 0)
        success = compliant and all(after["summary"][f"{axis}_rms_mm"] < before["summary"][f"{axis}_rms_mm"]
                                    for axis in axes(body.episode.plane))
        raw_intent = (session.selected_plane == body.episode.plane and
                      session.selected_effort == requested_effort(body.episode.prompt)) if session and body.episode.interface == "orbit_correct" else None
        return VerifyResponse(
            **body.model_dump(), reward=max(-1.0, min(1.0, reduction)) if compliant else 0.0,
            episode_id=body.episode.episode_id, interface=body.episode.interface,
            session_found=session is not None,
            tool_attempts=session.tool_attempts if session else 0,
            accepted_writes=session.accepted_writes if session else 0,
            rejected_calls=session.rejected_calls if session else 0,
            selected_plane=session.selected_plane if session else None,
            selected_effort=session.selected_effort if session else None,
            raw_intent_correct=raw_intent, goal_success=success,
            x_rms_initial_mm=before["summary"]["x_rms_mm"], y_rms_initial_mm=before["summary"]["y_rms_mm"],
            x_rms_final_mm=after["summary"]["x_rms_mm"], y_rms_final_mm=after["summary"]["y_rms_mm"],
            target_cost_initial=initial, target_cost_final=final, cost_reduction=reduction,
        )


if __name__ == "__main__":
    HackringMarkIIIResourcesServer.run_webserver()
