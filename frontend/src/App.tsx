import { Card, H2, HTMLSelect, Switch, Tag } from "@blueprintjs/core";
import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { Correctors, Readings, StatusBar, Unavailable } from "./Panels";
import RingScene, { WINDOWS_MM, magnificationFor } from "./RingScene";
import type { Device } from "./types";
import { bpmPv, correctorPv } from "./types";
import { useFeed } from "./useFeed";
import { useHistory } from "./useHistory";
import BeamView from "./BeamView";

// Dev-only write harness; the dynamic import is dropped from production builds.
const driverEnabled = import.meta.env.DEV && new URLSearchParams(window.location.search).get("driver") === "1";
const TestDriver = driverEnabled ? lazy(() => import("./TestDriver")) : null;

export default function App() {
  const feed = useFeed();
  const [expert, setExpert] = useState(false);
  const [selected, setSelected] = useState<Device | null>(null);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const t = window.setInterval(() => setNow(Date.now()), 250); return () => window.clearInterval(t); }, []);
  const { state } = feed;
  const history = useHistory(state);
  const [windowMm, setWindowMm] = useState(WINDOWS_MM[2]);
  const [clearedAt, setClearedAt] = useState(0);
  const shown = useMemo(() => history.filter((h) => h.t > clearedAt), [history, clearedAt]);
  const dim = feed.status === "stale" || feed.status === "disconnected";

  const inspect = (d: Device) => {
    if (!state) return null;
    const rows: [string, number | undefined, string][] =
      d.kind === "bpm" ? [["X", state.pvs[bpmPv(d.name, "X")], "mm"], ["Y", state.pvs[bpmPv(d.name, "Y")], "mm"]]
      : d.kind === "corrector" ? [["Setpoint", state.pvs[correctorPv(d.name, "SP")], "mrad"], ["Readback", state.pvs[correctorPv(d.name, "RB")], "mrad"]] : [];
    return (
      <div data-testid="inspector">
        <b>{d.name}</b> <Tag minimal>{d.kind}</Tag> <span className="muted">s = {d.s_m.toFixed(2)} m</span>
        {rows.map(([k, v, u]) => <div key={k}>{k}: {v === undefined ? "—" : v.toFixed(4)} {u}</div>)}
        {rows.length === 0 && <div className="muted">No public PVs for this element.</div>}
      </div>
    );
  };

  return (
    <div className={`app-shell bp6-dark ${dim ? "is-dim" : ""}`}>
      <header className="topbar">
        <div><div className="eyebrow">OSPREY / PYAT · READ-ONLY VIEWER</div><H2>FODO ring monitor</H2></div>
        <Switch checked={expert} onChange={(e) => setExpert(e.currentTarget.checked)} label={expert ? "Expert (PV names)" : "Simple"} />
      </header>
      <StatusBar feed={feed} now={now} />
      {TestDriver && <Suspense fallback={null}><TestDriver /></Suspense>}
      <div className="grid">
        <Card className="panel scene-card">
          <h4>Ring schematic <span className="muted">(not the physical floor plan)</span></h4>
          <div className="driver-row"><span className="muted">Tube display window</span><HTMLSelect value={windowMm} onChange={(e) => setWindowMm(Number(e.currentTarget.value))} options={WINDOWS_MM.map((w) => ({ value: w, label: `± ${w} mm` }))} data-testid="window-select" /></div>
          <div className="scene">
            {state ? <RingScene state={state} selected={selected?.name ?? null} onSelect={setSelected} dim={dim} windowMm={windowMm} /> : <div className="placeholder">{feed.status === "disconnected" ? "No machine data — viewer is disconnected." : "Waiting for machine state…"}</div>}
          </div>
          <div className="legend muted">Grey ring = reference path · translucent tube = vacuum chamber look; its cross-section is the display window (± {windowMm} mm ≈ ×{state ? Math.round(magnificationFor(windowMm, state.lattice.circumference_m || 1)) : "—"} magnification), not the physical pipe · red dots = measured BPM position, yellow = beyond the window (shown at the wall) · tables show true values · click a device to inspect · drag to orbit</div>
          <div className="inspect">{selected ? inspect(selected) : <span className="muted">No device selected.</span>}</div>
        </Card>
        <div className="col">
          {state ? <BeamView history={shown} onClear={() => setClearedAt(Date.now())} /> : null}
          {state ? <><Readings state={state} expert={expert} /><Correctors state={state} expert={expert} /></> : <Card className="panel"><span className="muted">No readings available.</span></Card>}
        </div>
        <div className="status-row">
        <Unavailable title="Agent intent" need="No agent-event contract from the backend yet." />
        <Unavailable title="Osprey action status" need="No action/approval event stream is provided yet." />
        <Unavailable title="Target & goal" need="No target or threshold contract exists, so success is not computed." />
        </div>
      </div>
    </div>
  );
}
