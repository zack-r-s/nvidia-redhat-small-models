import { Button, Callout, Tag, TextArea } from "@blueprintjs/core";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TurnEvidence } from "./types";

const CHAT_URL = import.meta.env.VITE_CHAT_URL || "http://127.0.0.1:8002";

type ToolEvent = { kind: "tool" | "note"; tool?: string; phase?: string | null; rejected?: boolean; detail?: string; text?: string };
type Message = { id: number; role: "user" | "agent"; text: string; events?: ToolEvent[]; mode?: string; elapsed_s?: number; error?: boolean };
type Health = { mode: "live" | "scripted"; backend: string; fixture: { ok: boolean }; model: { ok: boolean } };

/**
 * Chat panel for the dev chat bridge. The bridge is a dev fixture, not Osprey: every reply is labelled with its
 * mode, and tool traces are shown apart from the reply text so intent, tool activity and answers stay distinct.
 * The viewer never writes to the machine; any machine change here happens inside the bridge's agent turn.
 */
export default function Chat({ onTurn, onReset, disabled = false }: { onTurn?: (turn: TurnEvidence) => void; onReset?: () => void; disabled?: boolean }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const idRef = useRef(0);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`${CHAT_URL}/chat/health`, { signal: AbortSignal.timeout(3000) });
        if (!response.ok) throw new Error(String(response.status));
        const next = (await response.json()) as Health;
        if (active) { setHealth(next); setOffline(false); }
      } catch { if (active) setOffline(true); }
    };
    void check();
    const timer = window.setInterval(() => void check(), 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [messages, busy]);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || busy || disabled) return;
    setDraft("");
    setBusy(true);
    setMessages((m) => [...m, { id: ++idRef.current, role: "user", text }]);
    try {
      const response = await fetch(`${CHAT_URL}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }) });
      if (!response.ok) throw new Error(`Chat bridge returned ${response.status}`);
      const data = await response.json();
      setMessages((m) => [...m, { id: ++idRef.current, role: "agent", text: data.reply, events: data.events, mode: data.mode, elapsed_s: data.elapsed_s, error: Boolean(data.error) }]);
      if (data.turn) onTurn?.(data.turn as TurnEvidence);
    } catch (cause) {
      setMessages((m) => [...m, { id: ++idRef.current, role: "agent", error: true, text: cause instanceof Error ? cause.message : "Chat bridge unavailable" }]);
    } finally { setBusy(false); }
  }, [draft, busy, disabled, onTurn]);

  const clear = async () => {
    setMessages([]);
    onReset?.();
    try { await fetch(`${CHAT_URL}/chat/clear`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); } catch { /* bridge offline: local clear only */ }
  };

  const reset = async () => {
    setBusy(true);
    try {
      const response = await fetch(`${CHAT_URL}/chat/reset`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ seed: 7, disturbance_mrad: 0.08 }) });
      if (!response.ok) throw new Error(`Fixture reset failed: ${response.status}`);
      setMessages([]);
      onReset?.();
    } catch (cause) {
      setMessages((m) => [...m, { id: ++idRef.current, role: "agent", error: true, text: cause instanceof Error ? cause.message : "Fixture reset failed" }]);
    } finally { setBusy(false); }
  };

  return (
    <div className="chat" data-testid="chat">
      <div className="chat-head">
        {offline || !health
          ? <Tag intent="danger" minimal data-testid="chat-status">Chat bridge offline</Tag>
          : <Tag intent={health.mode === "live" ? "success" : "warning"} minimal data-testid="chat-status">{health.mode === "live" ? "Live model" : "Scripted (no model)"} · dev fixture</Tag>}
        <Button minimal small icon="eraser" onClick={() => void clear()} disabled={messages.length === 0 || disabled}>Clear</Button>
        <Button minimal small icon="reset" onClick={() => void reset()} disabled={busy || offline || disabled} data-testid="fixture-reset">Reset beam</Button>
      </div>
      {offline && <Callout intent="danger" compact icon="disable">Start the bridge with <code>scripts/dev_up.sh</code>. Nothing here is sent anywhere else.</Callout>}
      {disabled && <Callout intent="primary" compact icon="pause">Replay is showing recorded frames. Return to the live fixture to chat.</Callout>}
      <div className="chat-log" data-testid="chat-log">
        {messages.length === 0 && <div className="muted chat-empty">Ask about the beam, e.g. “report the beam position” or “set RING:MAG:H01:KICK:SP to 0.1”. Replies come from the local dev fixture, not Osprey.</div>}
        {messages.map((m) => (
          <div key={m.id} className={`msg msg-${m.role}${m.error ? " msg-error" : ""}`} data-testid={`msg-${m.role}`}>
            {m.events && m.events.length > 0 && (
              <details className="trace" open>
                <summary>Tool activity ({m.events.length}) <span className="muted">dev fixture</span></summary>
                {m.events.map((e, i) => (
                  <div key={i} className={`trace-line${e.rejected ? " rejected" : ""}`}>
                    {e.kind === "tool" ? <><b>{e.tool}</b>{e.phase ? ` · ${e.phase}` : ""}{e.rejected ? " · rejected" : ""} <span className="muted trace-detail" title={e.detail}>{e.detail}</span></> : e.text}
                  </div>
                ))}
              </details>
            )}
            <div className="msg-text">{m.text}</div>
            {m.role === "agent" && m.mode && <div className="muted msg-meta">{m.mode}{m.elapsed_s !== undefined ? ` · ${m.elapsed_s}s` : ""}</div>}
          </div>
        ))}
        {busy && <div className="muted chat-busy">Working…</div>}
        <div ref={endRef} />
      </div>
      <div className="chat-input">
        <TextArea fill rows={2} value={draft} disabled={disabled} placeholder="Message the agent (Enter to send, Shift+Enter for a new line)" data-testid="chat-input"
          onChange={(e) => setDraft(e.currentTarget.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }} />
        <Button intent="primary" icon="send-message" loading={busy} disabled={!draft.trim() || offline || disabled} onClick={() => void send()} data-testid="chat-send">Send</Button>
      </div>
    </div>
  );
}
