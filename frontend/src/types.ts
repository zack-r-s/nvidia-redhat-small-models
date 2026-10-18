export type Device = { name: string; kind: "bpm" | "corrector" | "quadrupole" | "dipole"; s_m: number };

export type MachineState = {
  schema: "hackring.state/1";
  backend?: "local-pyat" | "osprey-epics" | "caproto-epics" | "epics";
  pvs: Record<string, number>;
  summary: { x_rms_mm: number; y_rms_mm: number };
  lattice: { circumference_m: number | null; devices: Device[] };
};

export type TurnEvidence = {
  intent: "orbit_correction" | "exact_write" | "read_or_other";
  plane: "x" | "y" | "both" | null;
  requested_write: { channel: string; value_mrad: number } | null;
  action_status: "none" | "confirmed" | "rejected" | "partial";
  writes: { channel: string; outcome: string; observed_value?: number; refusal_reason?: string }[];
  before: { x_rms_mm: number; y_rms_mm: number };
  after: { x_rms_mm: number; y_rms_mm: number };
  goal_met: boolean | null;
};

export type FeedMode = "live" | "replay";
/** live: fresh /state; stale: last good snapshot is old; disconnected: no snapshot at all or fetch failing. */
export type FeedStatus = "live" | "stale" | "disconnected" | "replay";

export type Feed = {
  status: FeedStatus;
  state: MachineState | null;
  /** Wall-clock ms of the last good snapshot (live) or the replay frame time. */
  receivedAt: number | null;
  error: string | null;
  source: string;
  replay?: { frame: number; frames: number; runId?: string; seed?: number; playing: boolean; toggle: () => void };
};

export type ReplayFile = {
  run_id?: string;
  seed?: number;
  source?: string;
  frames: { t_s: number; state: MachineState }[];
};

/** Fixed by data/pv_contract.json; the viewer displays but never changes it. */
export const KICK_LIMIT_MRAD = 0.35;
export const bpmPv = (name: string, axis: "X" | "Y") => `RING:DIAG:${name}:POSITION:${axis}`;
export const correctorPv = (name: string, field: "SP" | "RB") => `RING:MAG:${name}:KICK:${field}`;
