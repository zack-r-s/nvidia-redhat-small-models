"""Deterministic Channel Access fixture: a caproto soft IOC over the local PyAT plant.

Serves the 24 contract PVs so PyEPICS (HACKRING_BACKEND=epics) and Xopt can be
exercised without a served Osprey virtual accelerator. Test tooling only; it
does not modify the lattice or the PV contract.
"""

from __future__ import annotations

import argparse

from caproto import ChannelType
from caproto.server import PVGroup, pvproperty, run

from hackring.machine import BPMS, SETPOINTS, LocalMachine


def build_ioc(seed: int, disturbance_mrad: float, prefix: str = ""):
    machine = LocalMachine()
    machine.reset(seed, disturbance_mrad)

    initial = machine.read()["pvs"]
    attrs: dict = {}

    class Ring(PVGroup):
        async def push(self, state: dict) -> None:
            for name, value in state["pvs"].items():
                if not name.endswith(":SP"):  # SP holds the operator's put; the rest are readbacks
                    await self.pvdb[name].write(value)

    def make_setpoint(name: str):
        async def put(self, instance, value):
            try:
                result = machine.write({name: float(value)})
            except ValueError:
                return instance.value  # reject out-of-limit puts; keep prior value
            await self.push(result["state"])
            return float(value)

        return pvproperty(name=name, value=initial[name], dtype=ChannelType.DOUBLE, put=put, precision=6)

    for i, name in enumerate(BPMS):
        attrs[f"bpm{i}"] = pvproperty(name=name, value=initial[name], dtype=ChannelType.DOUBLE, read_only=True, precision=6)
    for i, name in enumerate(SETPOINTS):
        attrs[f"sp{i}"] = make_setpoint(name)
        attrs[f"rb{i}"] = pvproperty(name=name.replace(":SP", ":RB"), value=initial[name.replace(":SP", ":RB")], dtype=ChannelType.DOUBLE, read_only=True, precision=6)
    group_cls = type("RingIOC", (Ring,), attrs)
    ioc = group_cls(prefix=prefix)
    return ioc, machine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disturbance-mrad", type=float, default=0.08)
    args = parser.parse_args()
    ioc, machine = build_ioc(args.seed, args.disturbance_mrad)
    run(ioc.pvdb, interfaces=["127.0.0.1"], module_name="caproto.asyncio.server", log_pv_names=False)


if __name__ == "__main__":
    main()
