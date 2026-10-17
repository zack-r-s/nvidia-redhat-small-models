"""Stage an isolated, served Osprey deployment without editing the ring repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOYMENT = Path("/private/tmp/hackring-data-v001-monitor")
BPM_NAMES = {f"BPM{number:02d}" for number in range(1, 5)}
PINNED_OSPREY_COMMIT = "d639869"
TEMP_WHEEL_VERSION = "2026.6.2+gd639869"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_monitor_elements(deck: Path) -> list[str]:
    """Change only the four zero-length BPM class tags in the temporary deck."""
    document = json.loads(deck.read_text())
    changed: list[str] = []
    for element in document["elements"]:
        if element.get("FamName") not in BPM_NAMES:
            continue
        if element.get("Class") != "Marker" or element.get("Length") != 0.0 or element.get("PassMethod") != "IdentityPass":
            raise RuntimeError(f"BPM is not a zero-length identity marker: {element.get('FamName')}")
        element["Class"] = "Monitor"
        changed.append(element["FamName"])
    if set(changed) != BPM_NAMES or len(changed) != 4:
        raise RuntimeError(f"expected exactly four BPM markers, found {changed}")
    deck.write_text(json.dumps(document, indent=2) + "\n")
    return changed


def prepare(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"deployment already exists: {destination}; choose a new path")
    destination.mkdir(parents=True)
    shutil.copytree(ROOT / "data", destination / "data")
    source_deck = ROOT / "data/facility/decks/RING.json"
    copy_deck = destination / "data/facility/decks/RING.json"
    changed = adapt_monitor_elements(copy_deck)
    shutil.copytree(ROOT / "rules", destination / "rules")
    shutil.copy2(ROOT / "providers.yml", destination / "providers.yml")

    profile = yaml.safe_load((ROOT / "profile.yml").read_text())
    # The small preset omits the control-assistant database consumers; the
    # checked-in facility data still supplies the exact same PyAT model.
    profile["extends"] = "hello-world"
    profile["project_name"] = "hackring-data-v001-monitor"
    profile["config"]["control_system.connector.virtual_accelerator.serving"] = "served"
    profile["config"]["archiver.type"] = "mongodb_archiver"
    profile["config"]["simulation.tick_s"] = 0.1
    profile["config"]["claude_code.servers.ariel.enabled"] = False
    profile["virtual_accelerator"] = {"port": 5064, "pva_port": 5075}
    profile["va_archiver"] = {}
    (destination / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False))

    env = (ROOT / ".env.example").read_text().rstrip() + "\n"
    env += f"VA_MODEL_WRITE_TOKEN={secrets.token_urlsafe(32)}\n"
    env_file = destination / ".env"
    env_file.write_text(env)
    env_file.chmod(0o600)

    import at
    import numpy as np

    source_ring = at.load_lattice(str(source_deck))
    staged_ring = at.load_lattice(str(copy_deck))
    if len(source_ring) != len(staged_ring):
        raise RuntimeError("staged lattice changed element count")
    for source, staged in zip(source_ring, staged_ring):
        if (source.FamName, source.Length, source.PassMethod) != (staged.FamName, staged.Length, staged.PassMethod):
            raise RuntimeError(f"staged lattice changed a physical element: {source.FamName}")
    for ring in (source_ring, staged_ring):
        next(element for element in ring if element.FamName == "H01").KickAngle[0] = 0.05e-3
    indices = [next(i for i, element in enumerate(source_ring) if element.FamName == name) for name in sorted(BPM_NAMES)]
    _, source_orbit = at.find_orbit4(source_ring, refpts=indices)
    _, staged_orbit = at.find_orbit4(staged_ring, refpts=indices)
    if not np.allclose(source_orbit, staged_orbit, rtol=0, atol=1e-12):
        raise RuntimeError("staged BPM conversion changed the test orbit")
    provenance = {
        "source_deck_sha256": digest(source_deck),
        "staged_deck_sha256": digest(copy_deck),
        "temporary_bpm_class_conversion": changed,
        "test_orbit_max_abs_difference_m": float(np.max(np.abs(source_orbit - staged_orbit))),
        "source_pv_contract_sha256": digest(ROOT / "data/pv_contract.json"),
        "staged_pv_contract_sha256": digest(destination / "data/pv_contract.json"),
        "osprey_source_commit_prefix": PINNED_OSPREY_COMMIT,
        "temporary_wheel_metadata_version": TEMP_WHEEL_VERSION,
    }
    (destination / "dataset_deployment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {destination}; four temporary BPM class conversions; test orbit difference {provenance['test_orbit_max_abs_difference_m']:.3g} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--build", action="store_true", help="run osprey build in the temporary deployment")
    args = parser.parse_args()
    prepare(args.deployment.resolve())
    if args.build:
        import osprey

        source_root = Path(osprey.__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        if not commit.startswith(PINNED_OSPREY_COMMIT):
            raise RuntimeError(f"Osprey source is {commit}, expected {PINNED_OSPREY_COMMIT}*")
        osprey = ROOT / ".venv/bin/osprey"
        env = os.environ.copy()
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = TEMP_WHEEL_VERSION
        subprocess.run(
            [str(osprey), "build", "--repo", str(args.deployment.resolve()), "--skip-deps", "--skip-lifecycle", "--dev"],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
