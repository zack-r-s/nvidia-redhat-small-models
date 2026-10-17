"""NeMo Gym resources server: correct the hackring orbit, rewarded by the measured cost reduction.

Each task replays one recorded episode offline. The episode's hidden corrector calibration offsets
are applied to the checked-in PyAT deck, so a corrector's kick is its setpoint plus its offset; this
reproduces the v001 BPM readings to rounding error without Osprey or EPICS. The model sees only the
public PVs through `channel_read` and `channel_write`, which return the same JSON shapes as the
Osprey controls tools the deployed small agent uses. `verify` scores the session's final state with
the dataset objective.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import at
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

REPO = Path(__file__).resolve().parents[3]
CONTRACT = json.loads((REPO / "data" / "pv_contract.json").read_text())
BPMS: list[str] = CONTRACT["bpm"]
SETPOINTS: list[str] = CONTRACT["corrector_setpoints"]
LIMIT = float(CONTRACT["kick_limit_mrad"])


def readback(setpoint: str) -> str:
    return setpoint.removesuffix(":SP") + ":RB"


class Ring:
    """The checked-in deck with per-call corrector kicks; calls are synchronous, so one lattice is shared."""

    def __init__(self, deck: Path) -> None:
        self.lattice = at.load_lattice(str(deck))
        names = [element.FamName for element in self.lattice]
        self.correctors = {sp: self.lattice[names.index(sp.split(":")[2])] for sp in SETPOINTS}
        self.bpm_indices = [names.index(f"BPM{n:02d}") for n in range(1, len(BPMS) // 2 + 1)]

    def pvs(self, setpoints: Dict[str, float], offsets: Dict[str, float]) -> Dict[str, float]:
        for sp, element in self.correctors.items():
            axis = 0 if ":MAG:H" in sp else 1
            element.KickAngle[axis] = (setpoints[sp] + offsets.get(sp, 0.0)) * 1e-3
        _, orbit = at.find_orbit4(self.lattice, refpts=self.bpm_indices)
        pvs = {}
        for n, point in enumerate(orbit, 1):
            pvs[f"RING:DIAG:BPM{n:02d}:POSITION:X"] = float(point[0] * 1e3)
            pvs[f"RING:DIAG:BPM{n:02d}:POSITION:Y"] = float(point[2] * 1e3)
        for sp, value in setpoints.items():
            pvs[sp] = value
            pvs[readback(sp)] = value
        return pvs


def objective(pvs: Dict[str, float]) -> Dict[str, float]:
    """The v001 Xopt objective: x_rms_mm^2 + y_rms_mm^2 + 0.01*sum(sp_mrad^2)."""
    x = math.sqrt(sum(pvs[name] ** 2 for name in BPMS if name.endswith(":X")) / 4)
    y = math.sqrt(sum(pvs[name] ** 2 for name in BPMS if name.endswith(":Y")) / 4)
    penalty = 0.01 * sum(pvs[name] ** 2 for name in SETPOINTS)
    return {"x_rms_mm": x, "y_rms_mm": y, "orbit_cost": x * x + y * y + penalty}


def validate_operations(operations: object, allowed: list[str]) -> list[dict]:
    """The small agent's goal-write rules: one batch of distinct allowed correctors within the kick limit."""
    if not isinstance(operations, list) or not 1 <= len(operations) <= len(allowed):
        raise ValueError("expected one bounded write batch for the selected plane")
    normalized = []
    seen = set()
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("every write operation must be an object")
        channel = op.get("channel", op.get("name"))
        value = op.get("value")
        if channel not in allowed or channel in seen:
            raise ValueError(f"duplicate or disallowed corrector: {channel}")
        if isinstance(value, bool):
            raise ValueError("kick must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("kick must be a finite number") from None
        if not math.isfinite(value) or abs(value) > LIMIT:
            raise ValueError(f"kick for {channel} exceeds ±{LIMIT:g} mrad")
        seen.add(channel)
        normalized.append({"channel": channel, "value": value})
    return normalized


class HackringOrbitResourcesServerConfig(BaseResourcesServerConfig):
    deck_path: Optional[str] = None


class Episode(BaseModel):
    episode_id: str
    hidden_calibration_offsets_mrad: Dict[str, float]
    initial_setpoints_mrad: Dict[str, float]
    allowed_correctors: list[str]
    xopt_best_orbit_cost: float


class HackringSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="ignore")
    episode: Episode


class ChannelReadRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    channels: Any = None


class ChannelWriteRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operations: Any = None


class HackringVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="ignore")
    episode: Episode


class HackringVerifyResponse(BaseVerifyResponse):
    episode_id: str
    session_found: bool
    orbit_cost_initial: float
    orbit_cost_final: float
    cost_reduction: float
    x_rms_final_mm: float
    y_rms_final_mm: float
    accepted_writes: int
    rejected_writes: int
    xopt_best_orbit_cost: float
    beats_xopt_best: float


class Session(BaseModel):
    episode: Episode
    setpoints: Dict[str, float]
    accepted_writes: int = 0
    rejected_writes: int = 0


class HackringOrbitResourcesServer(SimpleResourcesServer):
    config: HackringOrbitResourcesServerConfig
    sessions: Dict[str, Session] = Field(default_factory=dict)
    ring: Any = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        deck = Path(self.config.deck_path) if self.config.deck_path else REPO / "data/facility/decks/RING.json"
        self.ring = Ring(deck)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/channel_read")(self.channel_read)
        app.post("/channel_write")(self.channel_write)
        return app

    def state(self, session: Session) -> Dict[str, float]:
        return self.ring.pvs(session.setpoints, session.episode.hidden_calibration_offsets_mrad)

    async def seed_session(self, request: Request, body: HackringSeedSessionRequest) -> BaseSeedSessionResponse:
        setpoints = {sp: float(body.episode.initial_setpoints_mrad.get(sp, 0.0)) for sp in SETPOINTS}
        self.sessions[request.session[SESSION_ID_KEY]] = Session(episode=body.episode, setpoints=setpoints)
        return BaseSeedSessionResponse()

    async def channel_read(self, request: Request, body: ChannelReadRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode for this session"}
        channels = body.channels
        if isinstance(channels, str):
            channels = [channels]
        if not isinstance(channels, list) or not channels:
            return {"status": "error", "message": "channels must be a non-empty list of channel names"}
        pvs = self.state(session)
        unknown = [name for name in channels if name not in pvs]
        readings = {name: {"value": pvs[name]} for name in channels if name in pvs}
        result = {"status": "error" if unknown else "success", "summary": {"readings": readings}}
        if unknown:
            result["message"] = f"unknown channels: {', '.join(map(str, unknown))}"
        return result

    async def channel_write(self, request: Request, body: ChannelWriteRequest) -> dict:
        session = self.sessions.get(request.session[SESSION_ID_KEY])
        if session is None:
            return {"status": "error", "message": "no seeded episode for this session"}
        try:
            operations = validate_operations(body.operations, session.episode.allowed_correctors)
        except ValueError as exc:
            session.rejected_writes += 1
            return {"status": "error", "message": f"No write was made: {exc}."}
        before = dict(session.setpoints)
        session.setpoints.update({op["channel"]: op["value"] for op in operations})
        pvs = self.state(session)
        if not all(math.isfinite(pvs[name]) for name in BPMS):
            # A lost closed orbit fails the whole batch, as the local machine does.
            session.setpoints = before
            session.rejected_writes += 1
            return {"status": "error", "message": "No write was made: the closed orbit was lost."}
        session.accepted_writes += 1
        results = [{"channel": op["channel"], "outcome": "confirmed", "observed_value": op["value"]} for op in operations]
        return {"status": "success", "summary": {"results": results, "outcomes": {"confirmed": len(results)}}}

    async def verify(self, request: Request, body: HackringVerifyRequest) -> HackringVerifyResponse:
        episode = body.episode
        initial_setpoints = {sp: float(episode.initial_setpoints_mrad.get(sp, 0.0)) for sp in SETPOINTS}
        initial = objective(self.ring.pvs(initial_setpoints, episode.hidden_calibration_offsets_mrad))
        session = self.sessions.pop(request.session[SESSION_ID_KEY], None)
        final = objective(self.state(session)) if session is not None else initial
        reduction = (initial["orbit_cost"] - final["orbit_cost"]) / initial["orbit_cost"]
        return HackringVerifyResponse(
            **body.model_dump(),
            # Fractional cost reduction; a worsening correction is negative, floored at -1.
            reward=max(-1.0, min(1.0, reduction)),
            episode_id=episode.episode_id,
            session_found=session is not None,
            orbit_cost_initial=initial["orbit_cost"],
            orbit_cost_final=final["orbit_cost"],
            cost_reduction=reduction,
            x_rms_final_mm=final["x_rms_mm"],
            y_rms_final_mm=final["y_rms_mm"],
            accepted_writes=session.accepted_writes if session else 0,
            rejected_writes=session.rejected_writes if session else 0,
            xopt_best_orbit_cost=episode.xopt_best_orbit_cost,
            beats_xopt_best=float(final["orbit_cost"] < episode.xopt_best_orbit_cost),
        )


if __name__ == "__main__":
    HackringOrbitResourcesServer.run_webserver()
