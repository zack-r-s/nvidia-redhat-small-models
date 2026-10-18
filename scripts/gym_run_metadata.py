"""Write or backfill run.json for a hackring_orbit Gym evaluation run.

Records what was evaluated and on what software: the served model name and source, the exact
Hugging Face revision when it can be determined, the vLLM version and backend, the NeMo Gym
version, and the evaluation host. For a local endpoint the vLLM command is read from the process
listening on the port, so `--revision` and the executable (e.g. a vllm-metal install) are exact.
Without `--revision`, a Hugging Face model's revision comes from the local cache's refs/main and is
labeled as such. VLLM_MODEL_REVISION overrides detection.

  scripts/gym_run_metadata.py write <run_dir> <model> <base_url> <split> [extra args...]
  scripts/gym_run_metadata.py backfill <run_dir> [<base_url>]
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
RUN_FIELDS = {"served_model_name", "model_source", "max_model_len", "base_url", "split", "repo_commit",
              "extra_args", "started_utc"}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def server_command(base_url: str) -> list[str] | None:
    """The command line of a local vLLM server, found by its listening port."""
    parsed = urlparse(base_url)
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1") or not parsed.port:
        return None
    pids = run("lsof", "-nP", f"-iTCP:{parsed.port}", "-sTCP:LISTEN", "-t").split()
    command = run("ps", "-o", "command=", "-p", pids[0]) if pids else ""
    return shlex.split(command) if command else None


def hf_cache_revision(source: str) -> str | None:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    ref = hub / f"models--{source.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.is_file() else None


def model_revision(source: str | None, command: list[str] | None) -> tuple[str | None, str]:
    if os.environ.get("VLLM_MODEL_REVISION"):
        return os.environ["VLLM_MODEL_REVISION"], "VLLM_MODEL_REVISION"
    if command and "--revision" in command:
        return command[command.index("--revision") + 1], "vllm --revision"
    if source and Path(source).expanduser().exists():
        return None, "local checkpoint path (no revision)"
    if source and (revision := hf_cache_revision(source)):
        return revision, "hf cache refs/main (vllm started without --revision)"
    return None, "unknown"


def vllm_backend(command: list[str] | None) -> str | None:
    """vllm-metal when the interpreter, the vllm launcher, or the launcher's shebang points into it."""
    if not command:
        return None
    clues = []
    for token in command[:2]:
        path = Path(os.path.realpath(token))
        clues.append(str(path))
        if path.is_file():
            with path.open("rb") as handle:
                clues.append(handle.readline(512).decode(errors="replace"))
    return "vllm-metal" if any("vllm-metal" in clue for clue in clues) else "vllm"


def nemo_gym_version() -> str | None:
    line = run("gym", "--version").splitlines()
    return line[0].removeprefix("NeMo Gym v") if line else None


def host() -> dict:
    cpu = run("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin" else platform.processor()
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu or None}


def software(model: str, base_url: str) -> dict:
    served = next(m for m in get_json(f"{base_url}/models")["data"] if m.get("id") == model)
    command = server_command(base_url)
    revision, revision_source = model_revision(served.get("root"), command)
    return {
        "served_model_name": model,
        "model_source": served.get("root"),
        "model_revision": revision,
        "model_revision_source": revision_source,
        "max_model_len": served.get("max_model_len"),
        "base_url": base_url,
        "vllm_version": get_json(f"{base_url.removesuffix('/v1')}/version").get("version"),
        "vllm_backend": vllm_backend(command),
        "vllm_command": shlex.join(command) if command else None,
        "nemo_gym_version": nemo_gym_version(),
        "eval_host": host(),
    }


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("write", "backfill"):
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[2])
    path = run_dir / "run.json"
    if sys.argv[1] == "write":
        model, base_url, split, *extra = sys.argv[3:]
        record = software(model, base_url) | {
            "split": split,
            "repo_commit": run("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"),
            "extra_args": extra,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
    else:
        record = json.loads(path.read_text())
        base_url = sys.argv[3] if len(sys.argv) > 3 else record["base_url"]
        found = software(record["served_model_name"], base_url)
        if found["model_source"] != record.get("model_source"):
            raise SystemExit(f"{base_url} serves {found['model_source']}, not {record.get('model_source')}")
        # Fields the run recorded itself stay; detected software fields are added or refreshed.
        record |= {key: value for key, value in found.items() if key not in RUN_FIELDS}
        record["metadata_backfilled_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("served_model_name", "model_revision", "vllm_version",
                                                  "vllm_backend", "nemo_gym_version")}))


if __name__ == "__main__":
    main()
