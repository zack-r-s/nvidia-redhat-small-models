import { Callout, Card, HTMLTable, Tag, Button, ProgressBar } from "@blueprintjs/core";
import type { Feed, MachineState, TurnEvidence } from "./types";
import { KICK_LIMIT_MRAD, bpmPv, correctorPv } from "./types";

const fmt = (v: number | undefined, d = 4) => (v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));

export function StatusBar({ feed, now }: { feed: Feed; now: number }) {
  const intent = { live: "success", stale: "warning", disconnected: "danger", replay: "primary" } as const;
  const liveLabel = feed.state?.backend === "osprey-epics" ? "Live (Osprey)" : feed.state?.backend === "epics" ? "Live (EPICS)" : "Live (fixture)";
  const label = { live: liveLabel, stale: "Stale", disconnected: "Disconnected", replay: "Replay" }[feed.status];
  const age = feed.receivedAt && feed.status !== "replay" ? Math.max(0, (now - feed.receivedAt) / 1000).toFixed(1) + " s ago" : null;
  return (
    <div className="statusbar" data-testid="status">
      <Tag intent={intent[feed.status]} large data-testid="status-tag">{label}</Tag>
      <span className="muted">{feed.source}</span>
      {age && <span className="muted">last good sample {age}</span>}
      {feed.error && <span className="err">{feed.error}</span>}
      {feed.replay && (
        <span className="muted">
          {feed.replay.runId && `run ${feed.replay.runId} · `}{feed.replay.seed !== undefined && `seed ${feed.replay.seed} · `}
          frame {feed.replay.frame + 1}/{feed.replay.frames} <Button minimal small icon={feed.replay.playing ? "pause" : "play"} onClick={feed.replay.toggle} />
        </span>
      )}
    </div>
  );
}

export function Readings({ state, expert }: { state: MachineState; expert: boolean }) {
  const bpms = state.lattice.devices.filter((d) => d.kind === "bpm");
  const maxAbs = Math.max(0.05, ...bpms.flatMap((b) => [state.pvs[bpmPv(b.name, "X")], state.pvs[bpmPv(b.name, "Y")]].map((v) => Math.abs(v ?? 0))));
  return (
    <Card className="panel">
      <h4>Beam position <span className="muted">(measured at 4 BPMs only)</span></h4>
      <HTMLTable compact className="tbl" data-testid="bpm-table">
        <thead><tr><th>BPM</th><th>s (m)</th><th>X (mm)</th><th>Y (mm)</th></tr></thead>
        <tbody>
          {bpms.map((b) => (
            <tr key={b.name}>
              <td>{b.name}{expert && <div className="pv">{bpmPv(b.name, "X")}</div>}</td>
              <td>{b.s_m.toFixed(1)}</td>
              <td data-testid={`pv-${bpmPv(b.name, "X")}`}>{fmt(state.pvs[bpmPv(b.name, "X")])}</td>
              <td data-testid={`pv-${bpmPv(b.name, "Y")}`}>{fmt(state.pvs[bpmPv(b.name, "Y")])}</td>
            </tr>
          ))}
        </tbody>
      </HTMLTable>
      <div className="bars" aria-label="Measured BPM deviations">
        {(["X", "Y"] as const).map((axis) => (
          <div key={axis}>
            <div className="muted">{axis} deviation, bar scale ±{maxAbs.toFixed(2)} mm</div>
            {bpms.map((b) => {
              const v = state.pvs[bpmPv(b.name, axis)] ?? 0;
              return (
                <div className="barrow" key={b.name}>
                  <span>{b.name}</span>
                  <div className="track"><div className="mid" /><div className="fill" style={{ left: v < 0 ? `${50 + (v / maxAbs) * 50}%` : "50%", width: `${(Math.abs(v) / maxAbs) * 50}%` }} /></div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <p className="rms" data-testid="rms">RMS over BPMs: X <b>{fmt(state.summary.x_rms_mm)}</b> mm · Y <b>{fmt(state.summary.y_rms_mm)}</b> mm</p>
    </Card>
  );
}

export function Correctors({ state, expert }: { state: MachineState; expert: boolean }) {
  const names = state.lattice.devices.filter((d) => d.kind === "corrector");
  return (
    <Card className="panel">
      <h4>Steering correctors <span className="muted">(limit ±{KICK_LIMIT_MRAD} mrad)</span></h4>
      <HTMLTable compact className="tbl" data-testid="corr-table">
        <thead><tr><th>Corrector</th><th>Setpoint (mrad)</th><th>Readback (mrad)</th><th>Use</th></tr></thead>
        <tbody>
          {names.map((c) => {
            const sp = state.pvs[correctorPv(c.name, "SP")];
            return (
              <tr key={c.name}>
                <td>{c.name} <span className="muted">{c.name.startsWith("H") ? "horiz." : "vert."}</span>{expert && <div className="pv">{correctorPv(c.name, "SP")}</div>}</td>
                <td data-testid={`pv-${correctorPv(c.name, "SP")}`}>{fmt(sp)}</td>
                <td data-testid={`pv-${correctorPv(c.name, "RB")}`}>{fmt(state.pvs[correctorPv(c.name, "RB")])}</td>
                <td style={{ width: 70 }}><ProgressBar value={Math.min(1, Math.abs(sp ?? 0) / KICK_LIMIT_MRAD)} stripes={false} animate={false} intent={Math.abs(sp ?? 0) > 0.9 * KICK_LIMIT_MRAD ? "warning" : "none"} /></td>
              </tr>
            );
          })}
        </tbody>
      </HTMLTable>
    </Card>
  );
}

/** Each of these needs a backend contract that does not exist yet; nothing is invented here. */
export function Unavailable({ title, need }: { title: string; need: string }) {
  return (
    <Card className="panel unavailable" data-testid={`unavailable-${title}`}>
      <h4>{title}</h4>
      <Callout icon="disable" compact>Unavailable. {need}</Callout>
    </Card>
  );
}

export function AgentGoal({ turn }: { turn: TurnEvidence | null }) {
  if (!turn) return <Unavailable title="Agent & goal" need="Send a chat request to see its measured outcome." />;
  const goal = turn.intent === "orbit_correction"
    ? `Reduce ${turn.plane === "both" ? "X and Y" : turn.plane?.toUpperCase()} BPM RMS`
    : turn.requested_write
      ? `Set ${turn.requested_write.channel} to ${turn.requested_write.value_mrad} mrad`
      : "Read or discuss the machine";
  return (
    <Card className="panel" data-testid="agent-goal">
      <h4>Agent & goal</h4>
      <p>Request: <b>{goal}</b></p>
      <p>Write status: <Tag intent={turn.action_status === "confirmed" ? "success" : turn.action_status === "rejected" ? "danger" : "none"}>{turn.action_status}</Tag></p>
      {turn.goal_met !== null && <p>Measured goal: <b>{turn.goal_met ? "met" : "not met"}</b></p>}
      <p>X RMS: {fmt(turn.before.x_rms_mm)} → {fmt(turn.after.x_rms_mm)} mm</p>
      <p>Y RMS: {fmt(turn.before.y_rms_mm)} → {fmt(turn.after.y_rms_mm)} mm</p>
      {turn.writes.map((write, index) => <div key={index} className="muted">{write.channel}: {write.outcome}{write.observed_value !== undefined ? ` · readback ${fmt(write.observed_value)} mrad` : ""}</div>)}
    </Card>
  );
}
