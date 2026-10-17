"""Collect versioned Xopt read/write/read episodes from Osprey's served PyAT ring."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xopt import Evaluator, VOCS, Xopt
from xopt.generators import get_generator

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "data/pv_contract.json").read_text())
BPMS: list[str] = CONTRACT["bpm"]
SETPOINTS: list[str] = CONTRACT["corrector_setpoints"]
READBACKS = [name.replace(":SP", ":RB") for name in SETPOINTS]
PVS = BPMS + SETPOINTS + READBACKS
LIMIT = float(CONTRACT["kick_limit_mrad"])
SCHEMA = "hackring.xopt_attempt/1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def objective(pvs: dict[str, float]) -> dict[str, float]:
    x = math.sqrt(sum(pvs[name] ** 2 for name in BPMS if name.endswith(":X")) / 4)
    y = math.sqrt(sum(pvs[name] ** 2 for name in BPMS if name.endswith(":Y")) / 4)
    penalty = 0.01 * sum(pvs[name] ** 2 for name in SETPOINTS)
    return {"x_rms_mm": x, "y_rms_mm": y, "orbit_cost": x * x + y * y + penalty}


def read_token(deployment: Path) -> str:
    for line in (deployment / ".env").read_text().splitlines():
        if line.startswith("VA_MODEL_WRITE_TOKEN="):
            token = line.partition("=")[2].strip()
            if token:
                return token
    raise RuntimeError("temporary deployment has no VA_MODEL_WRITE_TOKEN")


class ServedRing:
    def __init__(self, deployment: Path, ca_port: int, pva_port: int, tick_s: float):
        # Both libraries read discovery settings when their first context is made.
        os.environ.update({
            "EPICS_CA_NAME_SERVERS": f"127.0.0.1:{ca_port}",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_CA_ADDR_LIST": "",
            "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{pva_port}",
            "EPICS_PVA_AUTO_ADDR_LIST": "NO",
            "EPICS_PVA_ADDR_LIST": "",
        })
        import epics
        from p4p.client.thread import Context
        from osprey.services.virtual_accelerator.serving.model_rpc import (
            RPC_PV, RPC_TIMEOUT_S, build_request, parse_reply,
        )

        self.epics = epics
        self.rpc_context = Context("pva")
        self.rpc_pv = RPC_PV
        self.rpc_timeout = RPC_TIMEOUT_S
        self.build_request = build_request
        self.parse_reply = parse_reply
        self.token = read_token(deployment)
        self.tick_s = tick_s
        self.fault_names: dict[str, str] = {}

    def close(self) -> None:
        self.rpc_context.close()

    def rpc(self, verb: str, *, values: dict[str, float] | None = None) -> object:
        token = self.token if verb in {"set", "reset"} else ""
        request = self.build_request(verb, values=values, token=token)
        return self.parse_reply(self.rpc_context.rpc(self.rpc_pv, request, timeout=self.rpc_timeout))

    def preflight(self) -> dict:
        status = self.rpc("status")
        if status["health"]["state"] != "serving":
            raise RuntimeError(f"Osprey model is not serving: {status['health']['state']}")
        variables = self.rpc("info")["variables"]
        for sp in SETPOINTS:
            matches = [item["name"] for item in variables
                       if item["surface"] == "model" and item["name"].endswith(f"/{sp}/cal_offset")]
            if len(matches) != 1:
                raise RuntimeError(f"expected one model-only calibration offset for {sp}, got {matches}")
            self.fault_names[sp] = matches[0]
        self.read()
        return {"instance": status["instance"], "fault_names": self.fault_names}

    def read(self) -> dict[str, float]:
        values = self.epics.caget_many(PVS, timeout=5)
        if len(values) != len(PVS) or any(value is None for value in values):
            missing = [name for name, value in zip(PVS, values) if value is None]
            raise RuntimeError(f"EPICS read failed: {missing}")
        reading = {name: float(value) for name, value in zip(PVS, values)}
        if not all(math.isfinite(value) for value in reading.values()):
            raise RuntimeError("EPICS returned non-finite PV value")
        return reading

    def write(self, settings: dict[str, float], *, before: dict[str, float] | None = None,
              force_all: bool = False) -> tuple[dict[str, float], list[str]]:
        if set(settings) != set(SETPOINTS):
            raise ValueError("an attempt must command exactly the eight corrector setpoints")
        if before is None and not force_all:
            raise ValueError("before snapshot is required when selecting changed PVs")
        for name, value in settings.items():
            if not math.isfinite(value) or abs(value) > LIMIT + 1e-12:
                raise ValueError(f"out-of-bounds command {name}={value}")
        changed = list(SETPOINTS) if force_all else [sp for sp in SETPOINTS if abs(settings[sp] - before[sp]) > 1e-12]
        if not changed:
            changed = [SETPOINTS[0]]
        for name in changed:
            # Normal Xopt candidates often change every knob. Submit those CA
            # puts together, then wait on the public readbacks and BPM tick.
            # Episode reset keeps synchronous puts so its hidden faults are
            # certainly applied even though all visible SPs are already zero.
            if self.epics.caput(name, settings[name], wait=force_all, timeout=5) != 1:
                raise RuntimeError(f"EPICS write failed: {name}")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            time.sleep(max(0.2, 2 * self.tick_s))
            reading = self.read()
            if all(abs(reading[sp.replace(":SP", ":RB")] - settings[sp]) <= 1e-6
                   for sp in SETPOINTS):
                return reading, changed
        raise RuntimeError("corrector readbacks did not settle to commanded values")

    def reset_episode(self, offsets: dict[str, float]) -> dict[str, float]:
        self.rpc("reset")
        self.rpc("set", values={self.fault_names[sp]: value for sp, value in offsets.items()})
        after, _ = self.write({sp: 0.0 for sp in SETPOINTS}, force_all=True)
        if max(abs(after[name]) for name in BPMS) < 1e-5:
            raise RuntimeError("hidden disturbance did not move any BPM")
        return after


def split_map(count: int, seed: int) -> dict[int, str]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    train = round(count * 0.8)
    validation = round(count * 0.1)
    return {index: "train" if position < train else "validation" if position < train + validation else "test"
            for position, index in enumerate(indices)}


def run(args: argparse.Namespace) -> None:
    deployment = args.deployment.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"dataset version already exists: {output}")
    provenance = json.loads((deployment / "dataset_deployment.json").read_text())
    deck = ROOT / "data/facility/decks/RING.json"
    if sha256(deck) != provenance["source_deck_sha256"] or sha256(deployment / "data/facility/decks/RING.json") != provenance["staged_deck_sha256"]:
        raise RuntimeError("source or temporary deployment lattice changed after staging")
    if set(provenance["temporary_bpm_class_conversion"]) != {f"BPM{i:02d}" for i in range(1, 5)}:
        raise RuntimeError("temporary deck has an unexpected compatibility conversion")
    if sha256(ROOT / "data/pv_contract.json") != provenance["source_pv_contract_sha256"]:
        raise RuntimeError("PV contract changed after deployment staging")
    if args.episodes < 1 or args.steps < 1 or not 0 < args.disturbance_mrad <= 0.2:
        raise ValueError("episodes and steps must be positive; disturbance must be in (0, 0.2] mrad")

    machine = ServedRing(deployment, args.ca_port, args.pva_port, args.tick_s)
    try:
        preflight = machine.preflight()
        output.mkdir(parents=True)
        splits = split_map(args.episodes, args.seed)
        attempts_path = output / "attempts.jsonl"
        episodes_path = output / "episodes.jsonl"
        counts = {"train": 0, "validation": 0, "test": 0}
        started = utc_now()
        with attempts_path.open("w") as attempts_file, episodes_path.open("w") as episodes_file:
            for episode_index in range(args.episodes):
                episode_seed = args.seed + episode_index
                rng = np.random.default_rng(episode_seed)
                offsets = {sp: float(rng.uniform(-args.disturbance_mrad, args.disturbance_mrad)) for sp in SETPOINTS}
                baseline = machine.reset_episode(offsets)
                episode_id = f"ep-{episode_index:04d}"
                split = splits[episode_index]
                episode = {"schema": "hackring.xopt_episode/1", "episode_id": episode_id,
                           "split": split, "seed": episode_seed, "hidden_calibration_offsets_mrad": offsets,
                           "initial_pvs": baseline, "initial_metrics": objective(baseline)}
                episodes_file.write(json.dumps(episode, allow_nan=False) + "\n")
                episodes_file.flush()
                step = 0

                def evaluate(settings: dict[str, float]) -> dict[str, float]:
                    nonlocal step
                    commanded = {sp: float(settings[sp]) for sp in SETPOINTS}
                    before = machine.read()
                    after, written = machine.write(commanded, before=before)
                    before_metrics, after_metrics = objective(before), objective(after)
                    row = {"schema": SCHEMA, "episode_id": episode_id, "step": step,
                           "split": split, "read_before": before,
                           "write": {"setpoints_mrad": commanded, "pvs_written": written},
                           "read_after": after, "before_metrics": before_metrics,
                           "after_metrics": after_metrics, "status": "completed"}
                    attempts_file.write(json.dumps(row, allow_nan=False) + "\n")
                    attempts_file.flush()
                    step += 1
                    counts[split] += 1
                    return {"orbit_cost": after_metrics["orbit_cost"]}

                vocs = VOCS(variables={sp: [-LIMIT, LIMIT] for sp in SETPOINTS}, objectives={"orbit_cost": "MINIMIZE"})
                ordered = sorted(SETPOINTS)
                simplex = {name: [0.06 if vertex == index + 1 else 0.0 for vertex in range(len(SETPOINTS) + 1)]
                           for index, name in enumerate(ordered)}
                generator = get_generator("neldermead")(
                    vocs=vocs, initial_point={sp: 0.0 for sp in SETPOINTS}, initial_simplex=simplex,
                )
                optimizer = Xopt(generator=generator, evaluator=Evaluator(function=evaluate, max_workers=1))
                for _ in range(args.steps):
                    optimizer.step()
                if step != args.steps:
                    raise RuntimeError(f"Xopt produced {step} attempts; expected {args.steps}")
                print(f"{episode_id} [{split}] {step} attempts, orbit cost {objective(baseline)['orbit_cost']:.6g} -> {objective(machine.read())['orbit_cost']:.6g}", flush=True)

        manifest = {
            "schema": "hackring.dataset_manifest/1", "version": output.name,
            "created_utc": utc_now(), "started_utc": started,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "generator_sha256": sha256(Path(__file__)),
            "source_deck_sha256": sha256(deck), "staged_deck_sha256": provenance["staged_deck_sha256"],
            "temporary_bpm_class_conversion": provenance["temporary_bpm_class_conversion"],
            "test_orbit_max_abs_difference_m": provenance["test_orbit_max_abs_difference_m"],
            "osprey_source_commit_prefix": provenance["osprey_source_commit_prefix"],
            "temporary_wheel_metadata_version": provenance["temporary_wheel_metadata_version"],
            "pv_contract_sha256": sha256(ROOT / "data/pv_contract.json"),
            "backend": "Osprey served PyAT via EPICS Channel Access",
            "model_rpc_instance": preflight["instance"], "model_only_fault_names": preflight["fault_names"],
            "optimizer": "Xopt Nelder-Mead", "objective": "x_rms_mm^2 + y_rms_mm^2 + 0.01*sum(sp_mrad^2)",
            "variables": {sp: [-LIMIT, LIMIT] for sp in SETPOINTS},
            "run": {"episodes": args.episodes, "steps_per_episode": args.steps, "seed": args.seed,
                    "disturbance_mrad": args.disturbance_mrad, "tick_s": args.tick_s},
            "attempts": sum(counts.values()), "attempts_by_split": counts,
            "software": {name: importlib.metadata.version(name) for name in
                         ("xopt", "pyepics", "accelerator-toolbox", "osprey-framework")},
            "files_sha256": {name: sha256(output / name) for name in ("attempts.jsonl", "episodes.jsonl")},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {manifest['attempts']} attempts to {output}")
    finally:
        machine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, default=Path("/private/tmp/hackring-data-v001-monitor"))
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v001")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--disturbance-mrad", type=float, default=0.08)
    parser.add_argument("--ca-port", type=int, default=5064)
    parser.add_argument("--pva-port", type=int, default=5075)
    parser.add_argument("--tick-s", type=float, default=0.1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        if sys.platform == "darwin":
            # PyEPICS/libca can block indefinitely in ca_context_destroy at
            # interpreter atexit on macOS after p4p shared its EPICS libraries.
            os._exit(1)
        raise
    else:
        if sys.platform == "darwin":
            sys.stdout.flush()
            os._exit(0)
