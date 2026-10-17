"""Fast physical and contract checks for the shared ring."""

from __future__ import annotations

import at
import numpy as np

from hackring.lattice import FACILITY
from hackring.machine import BPMS, LocalMachine, SETPOINTS


def main() -> None:
    machine = LocalMachine()
    lattice = machine.lattice
    bend_sum = sum(element.BendingAngle for element in lattice if hasattr(element, "BendingAngle"))
    assert np.isclose(bend_sum, 2 * np.pi)
    tune = at.get_optics(lattice)[1].tune
    assert np.all(np.isfinite(tune))
    assert len(BPMS) == 8 and len(SETPOINTS) == 8
    baseline = machine.reset(seed=7)
    response = machine.write({SETPOINTS[0]: 0.05})["state"]
    delta = np.array([response["pvs"][name] - baseline["pvs"][name] for name in BPMS])
    assert np.max(np.abs(delta)) > 1e-4, "a corrector write must move a BPM"
    assert (FACILITY / "decks" / "RING.json").is_file()
    print(f"ring: {len(lattice)} elements, tune={tune}, 24 PVs, BPM response={np.max(np.abs(delta)):.4f} mm")


if __name__ == "__main__":
    main()
