import { useEffect, useRef, useState } from "react";
import type { Feed, MachineState, ReplayFile } from "./types";

const API = import.meta.env.VITE_MACHINE_URL || "http://127.0.0.1:8001";
const POLL_MS = 500;
const STALE_MS = 2000;
const TIMEOUT_MS = 2000;
const SCHEMA = "hackring.state/1";

function useNow(period: number) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const t = window.setInterval(() => setNow(Date.now()), period); return () => window.clearInterval(t); }, [period]);
  return now;
}

/** Read-only: only ever issues GET /state. */
function useLiveFeed(enabled: boolean): Feed {
  const [state, setState] = useState<MachineState | null>(null);
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const now = useNow(250);

  useEffect(() => {
    if (!enabled) return;
    let active = true;
    let timer: number | undefined;
    let issued = 0;
    let applied = 0;
    let inFlight: AbortController | null = null;
    // One request in flight at a time: the next poll is scheduled only after this one settles,
    // so a slow response can never be overtaken by a later one. The sequence guard is a backstop.
    const poll = async () => {
      const seq = ++issued;
      const started = Date.now();
      const controller = new AbortController();
      inFlight = controller;
      const timeout = window.setTimeout(() => controller.abort(), TIMEOUT_MS);
      try {
        const response = await fetch(`${API}/state`, { signal: controller.signal });
        if (!response.ok) throw new Error(`Machine API returned ${response.status}`);
        const next = (await response.json()) as MachineState;
        if (next.schema !== SCHEMA) throw new Error("Unknown machine state schema");
        if (active && seq > applied) { applied = seq; setState(next); setReceivedAt(Date.now()); setError(null); }
      } catch (cause) {
        if (active && seq > applied) setError(cause instanceof Error ? (cause.name === "AbortError" ? "Machine API timed out" : cause.message) : "Machine unavailable");
      } finally {
        window.clearTimeout(timeout);
        if (active) timer = window.setTimeout(() => void poll(), Math.max(0, POLL_MS - (Date.now() - started)));
      }
    };
    void poll();
    return () => { active = false; window.clearTimeout(timer); inFlight?.abort(); };
  }, [enabled]);

  const age = receivedAt === null ? Infinity : now - receivedAt;
  const status = !state || (error && age > STALE_MS * 4) ? "disconnected" : age > STALE_MS || error ? "stale" : "live";
  return { status, state, receivedAt, error, source: `${API} (local PyAT fixture, not live Osprey)` };
}

/** Steps through recorded immutable snapshots; no interpolation between frames. */
function useReplayFeed(url: string | null): Feed {
  const [file, setFile] = useState<ReplayFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const started = useRef(Date.now());

  useEffect(() => {
    if (!url) return;
    fetch(url).then((r) => { if (!r.ok) throw new Error(`Replay file returned ${r.status}`); return r.json(); })
      .then((data: ReplayFile) => {
        if (!data.frames?.length || data.frames.some((f) => f.state.schema !== SCHEMA)) throw new Error("Invalid replay file");
        setFile(data); setFrame(0);
      })
      .catch((cause) => setError(cause instanceof Error ? cause.message : "Replay failed"));
  }, [url]);

  useEffect(() => {
    if (!file || !playing) return;
    const timer = window.setInterval(() => setFrame((f) => Math.min(f + 1, file.frames.length - 1)), 500);
    return () => window.clearInterval(timer);
  }, [file, playing]);

  const current = file?.frames[frame];
  return {
    status: "replay", state: current?.state ?? null, receivedAt: current ? started.current + current.t_s * 1000 : null, error,
    source: `Recorded: ${file?.source ?? url ?? "none"}`,
    replay: file ? { frame, frames: file.frames.length, runId: file.run_id, seed: file.seed, playing, toggle: () => { if (frame >= file.frames.length - 1) setFrame(0); setPlaying((p) => !p || frame >= file.frames.length - 1); } } : undefined,
  };
}

export function useFeed(): Feed {
  const replayUrl = new URLSearchParams(window.location.search).get("replay");
  const live = useLiveFeed(!replayUrl);
  const replay = useReplayFeed(replayUrl);
  return replayUrl ? replay : live;
}
