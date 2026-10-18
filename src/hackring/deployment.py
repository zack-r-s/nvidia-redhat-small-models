"""Resolve the deployment identity the running accelerator actually uses."""

from pathlib import Path

import yaml


def identity(repo: Path) -> tuple[str, int, int]:
    rendered = repo / "build" / "config.yml"
    if rendered.is_file():
        config = yaml.safe_load(rendered.read_text())
        va = (config.get("services") or {}).get("virtual_accelerator")
        if va:
            return config["project_name"], int(va["port"]), int(va["pva_port"])

    profile = yaml.safe_load((repo / "profile.yml").read_text())
    va = profile["virtual_accelerator"]
    return profile["project_name"], int(va["port"]), int(va["pva_port"])
