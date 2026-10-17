import { Button, Card, HTMLSelect, Popover, Switch, Tab, Tabs, Tag } from "@blueprintjs/core";
import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { BeamSpots, RmsHistory } from "./BeamView";
import Chat from "./Chat";
import { Correctors, Readings, StatusBar, Unavailable } from "./Panels";
import RingScene, { WINDOWS_MM, magnificationFor } from "./RingScene";
import type { Device } from "./types";
import { bpmPv, correctorPv } from "./types";
import { useFeed } from "./useFeed";
import { useHistory } from "./useHistory";

// Dev-only write harness; the dynamic import is dropped from production builds.
const driverEnabled = import.meta.env.DEV && new URLSearchParams(window.location.search).get("driver") === "1";
const TestDriver = driverEnabled ? lazy(() => import("./TestDriver")) : null;

type SideTab = "beam" | "readings" | "correctors" | "agent";

export default function App() {
  const feed = useFeed();
  const [expert, setExpert] = useState(false);
  const [selected, setSelected] = useState<Device | null>(null);
  const [tab, setTab] = useState<SideTab>("beam");
  const [driverOpen, setDriverOpen] = useState(false);
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

  const empty = <span className="muted">No readings available.</span>;
  const pane = (id: SideTab, content: React.ReactNode) => <div className="pane" hidden={tab !== id}>{content}</div>;

  return (
    <div className={`app-shell bp6-dark ${dim ? "is-dim" : ""}`}>
      <header className="topbar">
        <div className="brand"><span className="eyebrow">OSPREY / PYAT</span><strong>FODO ring monitor</strong></div>
        <StatusBar feed={feed} now={now} />
        <div className="topbar-actions">
          {TestDriver && (
            <Popover isOpen={driverOpen} onInteraction={setDriverOpen} placement="bottom-end" content={<div className="driver-pop"><Suspense fallback={null}><TestDriver /></Suspense></div>}>
              <Button small icon="wrench" intent="warning" minimal data-testid="driver-toggle">Test driver</Button>
            </Popover>
          )}
          <Switch checked={expert} onChange={(e) => setExpert(e.currentTarget.checked)} label={expert ? "Expert" : "Simple"} />
        </div>
      </header>
      <main className="workspace">
        <Card className="panel scene-card" data-testid="stage">
          <div className="panel-head">
            <h4>Ring <span className="muted">(schematic)</span></h4>
            <span className="panel-tools">
              <span className="muted">Tube window</span>
              <HTMLSelect minimal value={windowMm} onChange={(e) => setWindowMm(Number(e.currentTarget.value))} options={WINDOWS_MM.map((w) => ({ value: w, label: `± ${w} mm` }))} data-testid="window-select" />
              <Popover placement="bottom-end" content={<div className="legend-pop muted">Grey ring = reference path · tube = vacuum-chamber look; its cross-section is the display window (± {windowMm} mm ≈ ×{state ? Math.round(magnificationFor(windowMm, state.lattice.circumference_m || 1)) : "—"}), not the physical pipe · red dot = measured BPM, yellow = beyond the window (drawn at the wall) · tables show true values · click a device to inspect · drag to orbit</div>}>
                <Button minimal small icon="info-sign" aria-label="Legend" />
              </Popover>
            </span>
          </div>
          <div className="scene">
            {state ? <RingScene state={state} selected={selected?.name ?? null} onSelect={setSelected} dim={dim} windowMm={windowMm} /> : <div className="placeholder">{feed.status === "disconnected" ? "No machine data — viewer is disconnected." : "Waiting for machine state…"}</div>}
            <div className="inspect">{selected ? inspect(selected) : <span className="muted">Click a device to inspect.</span>}</div>
          </div>
        </Card>
        <aside className="right">
          <Card className="panel tabs-panel" data-testid="insight">
            <Tabs id="insight-tabs" selectedTabId={tab} onChange={(id) => setTab(id as SideTab)} animate={false}>
              <Tab id="beam" title="Beam" />
              <Tab id="readings" title="Readings" />
              <Tab id="correctors" title="Correctors" />
              <Tab id="agent" title="Agent & goal" />
            </Tabs>
            <div className="panes">
              {pane("beam", state ? <div className="beam-stack"><BeamSpots history={shown} onClear={() => setClearedAt(Date.now())} /><RmsHistory history={shown} /></div> : empty)}
              {pane("readings", state ? <Readings state={state} expert={expert} /> : empty)}
              {pane("correctors", state ? <Correctors state={state} expert={expert} /> : empty)}
              {pane("agent", <div className="stack">
                <Unavailable title="Agent intent" need="No agent-event contract yet." />
                <Unavailable title="Osprey action status" need="No action/approval events yet." />
                <Unavailable title="Target & goal" need="No target contract, so success is not computed." />
              </div>)}
            </div>
          </Card>
          <Card className="panel chat-panel" data-testid="chat-panel"><Chat /></Card>
        </aside>
      </main>
    </div>
  );
}
