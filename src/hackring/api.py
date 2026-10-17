"""Small frontend and test-driver bridge over the same machine instance."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hackring.machine import EpicsMachine, LocalMachine

machine = EpicsMachine() if os.getenv("HACKRING_BACKEND") == "epics" else LocalMachine()
app = FastAPI(title="Hackathon ring state", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])


class WriteRequest(BaseModel):
    settings: dict[str, float]


class ScenarioRequest(BaseModel):
    seed: int = 1
    disturbance_mrad: float = 0.08


@app.get("/state")
def state() -> dict:
    try:
        return machine.read()
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/write")
def write(request: WriteRequest) -> dict:
    try:
        return machine.write(request.settings)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/scenario")
def scenario(request: ScenarioRequest) -> dict:
    if not isinstance(machine, LocalMachine):
        raise HTTPException(501, "EPICS scenario injection needs Osprey model RPC")
    try:
        return machine.reset(request.seed, request.disturbance_mrad)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
