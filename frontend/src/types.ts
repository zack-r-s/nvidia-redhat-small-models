export type Device = { name: string; kind: "bpm" | "corrector" | "quadrupole" | "dipole"; s_m: number };

export type MachineState = {
  schema: "hackring.state/1";
  pvs: Record<string, number>;
  summary: { x_rms_mm: number; y_rms_mm: number };
  lattice: { circumference_m: number | null; devices: Device[] };
};
