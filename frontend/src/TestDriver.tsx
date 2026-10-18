/**
 * DEV-ONLY manual test driver. This is deliberately NOT part of the read-only viewer:
 * it is loaded only by `vite dev` with `?driver=1`, and it is the only module that may
 * call POST /scenario or POST /write, against the local PyAT fixture.
 * It is a test harness, not the model or Osprey control path.
 */
import { Button, Callout, Card, HTMLSelect } from "@blueprintjs/core";
import { useState } from "react";
import { KICK_LIMIT_MRAD, correctorPv } from "./types";

const API = import.meta.env.VITE_MACHINE_URL || "http://127.0.0.1:8001";
const NAMES = ["H01", "V01", "H02", "V02", "H03", "V03", "H04", "V04"];
const zeroKicks = Object.fromEntries(NAMES.map((n) => [correctorPv(n, "SP"), 0]));
const kicks = (over: Record<string, number>) => ({ ...zeroKicks, ...Object.fromEntries(Object.entries(over).map(([n, v]) => [correctorPv(n, "SP"), v])) });

const post = (path: string, body: unknown) => fetch(`${API}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** Sequential steps so a scenario reset lands before any kick write. */
type Step = [string, unknown];
const PRESETS: { id: string; label: string; note: string; steps: Step[]; expectReject?: boolean }[] = [
  { id: "seed7", label: "Disturbed ring — seed 7 (0.08 mrad)", note: "Demo scenario; correctors zeroed.", steps: [["/scenario", { seed: 7, disturbance_mrad: 0.08 }]] },
  { id: "seed1", label: "Disturbed ring — seed 1 (0.08 mrad)", note: "Alternate seeded disturbance.", steps: [["/scenario", { seed: 1, disturbance_mrad: 0.08 }]] },
  { id: "seed3", label: "Disturbed ring — seed 3 (0.08 mrad)", note: "Alternate seeded disturbance.", steps: [["/scenario", { seed: 3, disturbance_mrad: 0.08 }]] },
  { id: "large", label: "Large disturbance — seed 7 (0.20 mrad, max)", note: "Strongest allowed error kicks.", steps: [["/scenario", { seed: 7, disturbance_mrad: 0.2 }]] },
  { id: "quiet", label: "No disturbance (0 mrad)", note: "Ideal ring; BPMs should read ~0.", steps: [["/scenario", { seed: 7, disturbance_mrad: 0 }]] },
  { id: "zero", label: "Zero all correctors", note: "Keeps current disturbance.", steps: [["/write", { settings: zeroKicks }]] },
  { id: "h01", label: "Single kick: H01 = +0.20 mrad", note: "Watch BPM response to one corrector.", steps: [["/write", { settings: kicks({ H01: 0.2 }) }]] },
  { id: "v03", label: "Single kick: V03 = −0.20 mrad", note: "Vertical response.", steps: [["/write", { settings: kicks({ V03: -0.2 }) }]] },
  { id: "limit", label: `Corrector at limit: H02 = +${KICK_LIMIT_MRAD} mrad`, note: "Maximum allowed kick.", steps: [["/write", { settings: kicks({ H02: KICK_LIMIT_MRAD }) }]] },
  { id: "reject", label: "Out-of-limit write: H01 = 0.50 mrad (expect rejection)", note: "Machine must reject and stay unchanged.", expectReject: true, steps: [["/write", { settings: kicks({ H01: 0.5 }) }]] },
];

export default function TestDriver({ onMachineChange }: { onMachineChange?: () => void }) {
  const [id, setId] = useState(PRESETS[0].id);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ intent: "success" | "danger" | "warning"; text: string } | null>(null);
  const preset = PRESETS.find((p) => p.id === id)!;

  const apply = async (target = preset) => {
    setBusy(true);
    try {
      for (const [path, body] of target.steps) {
        const response = await post(path, body);
        if (!response.ok) {
          const detail = await response.text();
          const expected = target.expectReject && response.status === 422;
          setResult({ intent: expected ? "success" : "danger", text: `${expected ? "Rejected as expected" : "Failed"}: ${path} → ${response.status} ${detail.slice(0, 160)}` });
          return;
        }
      }
      setResult({ intent: "success", text: `Applied: ${target.label}` });
      onMachineChange?.();
    } catch (cause) {
      setResult({ intent: "danger", text: cause instanceof Error ? cause.message : "Request failed" });
    } finally { setBusy(false); }
  };

  return (
    <Card className="panel driver" data-testid="test-driver">
      <h4>Manual test driver <span className="muted">(dev only — writes to the local fixture, not the model/Osprey path)</span></h4>
      <div className="driver-row">
        <HTMLSelect value={id} onChange={(e) => setId(e.currentTarget.value)} options={PRESETS.map((p) => ({ value: p.id, label: p.label }))} data-testid="driver-select" fill />
        <Button intent="primary" icon="play" loading={busy} onClick={() => void apply()} data-testid="driver-apply">Apply state</Button>
      </div>
      <div className="driver-row">
        <Button icon="reset" disabled={busy} onClick={() => void apply(PRESETS.find((p) => p.id === "seed7")!)} data-testid="driver-reset">Reset to seed 7</Button>
        <Button icon="clean" disabled={busy} onClick={() => void apply(PRESETS.find((p) => p.id === "zero")!)} data-testid="driver-zero">Zero correctors</Button>
        <span className="muted">{preset.note}</span>
      </div>
      {result && <Callout intent={result.intent} compact data-testid="driver-result">{result.text}</Callout>}
    </Card>
  );
}
