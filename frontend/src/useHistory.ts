import { useEffect, useState } from "react";
import type { MachineState } from "./types";

export type Sample = { t: number; state: MachineState };
export const HISTORY_LEN = 80;

/** Rolling buffer of the real snapshots the feed has delivered (no synthesis, no interpolation). */
export function useHistory(state: MachineState | null): Sample[] {
  const [history, setHistory] = useState<Sample[]>([]);
  useEffect(() => {
    if (!state) return;
    setHistory((h) => (h.length && h[h.length - 1].state === state ? h : [...h, { t: Date.now(), state }].slice(-HISTORY_LEN)));
  }, [state]);
  return history;
}
