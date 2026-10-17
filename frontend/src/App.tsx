import { Card, H2, Tag } from "@blueprintjs/core";
import { useEffect, useState } from "react";
import RingScene from "./RingScene";
import type { MachineState } from "./types";

const API = import.meta.env.VITE_MACHINE_URL || "http://127.0.0.1:8001";

export default function App() {
  const [state, setState] = useState<MachineState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const response = await fetch(`${API}/state`);
        if (!response.ok) throw new Error(`Machine API returned ${response.status}`);
        const next = (await response.json()) as MachineState;
        if (next.schema !== "hackring.state/1") throw new Error("Unknown machine state schema");
        if (active) { setState(next); setError(null); }
      } catch (cause) {
        if (active) setError(cause instanceof Error ? cause.message : "Machine unavailable");
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 500);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  return (
    <div className="app-shell bp6-dark">
      <header className="topbar">
        <div><div className="eyebrow">OSPREY / PYAT</div><H2>FODO ring monitor</H2></div>
        <Tag intent={error ? "danger" : "success"} minimal large>{error ? "Disconnected" : "Simulator"}</Tag>
      </header>
      <Card className="starter-card">
        <div className="meta">
          <span>{error || "Live state scaffold"}</span>
          {state && <span>X RMS {state.summary.x_rms_mm.toFixed(3)} mm · Y RMS {state.summary.y_rms_mm.toFixed(3)} mm</span>}
        </div>
        <div className="scene">{state ? <RingScene state={state} /> : <div className="placeholder">Waiting for machine state…</div>}</div>
      </Card>
    </div>
  );
}
