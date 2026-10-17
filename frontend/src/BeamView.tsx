import { Button } from "@blueprintjs/core";
import type { Sample } from "./useHistory";
import { bpmPv } from "./types";

const BPMS = ["BPM01", "BPM02", "BPM03", "BPM04"];
/** Earlier samples kept as faded dots (not joined by lines). Set to 0 for latest-only. */
const SPOT_DOTS = 8;
const COLORS = { x: "#38bdf8", y: "#f59e0b" };

/** Transverse beam spot at each BPM. Every plotted point is a real measured sample. */
export function BeamSpots({ history, onClear }: { history: Sample[]; onClear: () => void }) {
  const xy = (s: Sample, b: string) => [s.state.pvs[bpmPv(b, "X")] ?? 0, s.state.pvs[bpmPv(b, "Y")] ?? 0] as const;
  const peak = Math.max(0.1, ...history.flatMap((s) => BPMS.flatMap((b) => xy(s, b).map(Math.abs))));
  const scale = Math.ceil(peak * 1.15 * 20) / 20; // shared ± range in mm
  const px = (v: number) => 50 + (v / scale) * 45;
  const py = (v: number) => 50 - (v / scale) * 45;
  const last = history[history.length - 1];

  return (
    <div data-testid="beam-view">
      <div className="pane-head">
        <span><b>Beam spot</b> <span className="muted">at each BPM · ± {scale.toFixed(2)} mm, same scale</span></span>
        <Button minimal small icon="eraser" onClick={onClear}>Clear</Button>
      </div>
      <div className="spots">
        {BPMS.map((b) => {
          const pts = history.slice(-SPOT_DOTS - 1, -1).map((s) => xy(s, b));
          const cur = last ? xy(last, b) : null;
          return (
            <svg key={b} viewBox="0 0 100 100" className="spot" role="img" aria-label={`${b} beam position`}>
              <rect x="0" y="0" width="100" height="100" fill="#0f172a" />
              <line x1="50" y1="3" x2="50" y2="97" stroke="#475569" strokeWidth=".5" />
              <line x1="3" y1="50" x2="97" y2="50" stroke="#475569" strokeWidth=".5" />
              <circle cx="50" cy="50" r="2.5" fill="none" stroke="#94a3b8" strokeWidth=".6" />
              {pts.map(([x, y], i) => <circle key={i} cx={px(x)} cy={py(y)} r="1" fill="#fb7185" opacity={0.12 + 0.6 * (i / pts.length)} />)}
              {cur && <circle cx={px(cur[0])} cy={py(cur[1])} r="3" fill="#f43f5e" stroke="#fff" strokeWidth=".6" />}
              <text x="4" y="9" fontSize="7" fill="#e2e8f0">{b}</text>
              <text x="96" y="47" fontSize="5" fill="#94a3b8" textAnchor="end">+X</text>
              <text x="53" y="8" fontSize="5" fill="#94a3b8">+Y</text>
            </svg>
          );
        })}
      </div>
      <div className="muted">Crosshair = reference · bright dot = latest · faint = earlier samples, not joined</div>
    </div>
  );
}

/** X and Y orbit RMS over the recent measured samples. */
export function RmsHistory({ history }: { history: Sample[] }) {
  const rmsMax = Math.max(0.05, ...history.flatMap((s) => [s.state.summary.x_rms_mm, s.state.summary.y_rms_mm]));
  const line = (pick: (s: Sample) => number) => history.map((s, i) => `${history.length > 1 ? (i / (history.length - 1)) * 200 : 200},${58 - (pick(s) / rmsMax) * 54}`).join(" ");
  return (
    <div data-testid="rms-view">
      <div className="pane-head"><span><b>Orbit RMS</b> <span className="muted">last {history.length} samples · 0–{rmsMax.toFixed(2)} mm</span></span></div>
      <svg viewBox="0 0 200 60" className="rmsplot" role="img" aria-label="RMS history">
        <rect width="200" height="60" fill="#0f172a" />
        {history.length > 1 && <><polyline points={line((s) => s.state.summary.x_rms_mm)} fill="none" stroke={COLORS.x} strokeWidth="1" /><polyline points={line((s) => s.state.summary.y_rms_mm)} fill="none" stroke={COLORS.y} strokeWidth="1" /></>}
      </svg>
      <div className="muted"><span style={{ color: COLORS.x }}>■</span> X · <span style={{ color: COLORS.y }}>■</span> Y · lower = closer to reference</div>
    </div>
  );
}
