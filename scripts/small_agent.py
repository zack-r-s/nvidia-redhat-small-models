"""Small-model operator agent: the Osprey `controls` MCP server driven by a local vLLM model.

Runs as an interactive REPL, so it can replace `claude` in the Osprey Web Terminal
(`web_terminal.shell`). Harness-side guards make ~1B models usable: a tool call is
forced on the first step of a turn that names a channel or asks to read/set one, and
tool arguments are coerced to the types the schema expects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory

from hackring.deployment import identity
from hackring.control import SKILL_SYSTEM, SKILL_TOOL, requested_effort, solve

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env", override=False)
BUILD = REPO / "build"
LOGS = REPO / "var" / "logs"
CONTRACT = json.loads((REPO / "data" / "pv_contract.json").read_text())
PROVIDER = yaml.safe_load((REPO / "providers.yml").read_text())["providers"]["vllm"]
PROJECT, CA_PORT, PVA_PORT = identity(REPO)
VA_CONTAINER = f"{PROJECT}-virtual-accelerator"
CHANNELS = CONTRACT["bpm"] + CONTRACT["corrector_setpoints"]
READBACKS = {name: name.removesuffix(":SP") + ":RB" for name in CONTRACT["corrector_setpoints"]}
KNOWN_READS = set(CHANNELS) | set(READBACKS.values())
# Target switching and archiver history stay out of reach; a small model misfires them.
TOOLS = {"channel_read", "channel_write", "channel_limits"}
ACTION = re.compile(r"\b(read|set|write|kick|get|measure|report|show)\b", re.IGNORECASE)
BEAM_POSITION = re.compile(r"\b(beam position|orbit|bpm)\b", re.IGNORECASE)
MOVE_BEAM = re.compile(r"\b(move|shift|steer|disturb|perturb)\b.*\b(beam|orbit)\b|\b(beam|orbit)\b.*\b(move|shift|steer|disturb|perturb)\b", re.IGNORECASE)
GOAL_VERB = re.compile(r"\b(correct|center(?:ing|ed)?|centre(?:ing|ed)?|reduce|minimi[sz]e|flatten|restore|bring)\b", re.IGNORECASE)
NEGATION = re.compile(r"\b(do not|don't|never|without)\b", re.IGNORECASE)
KICK_LIMIT = float(CONTRACT["kick_limit_mrad"])
COMMANDS = {"/clear": "forget the conversation so far", "/help": "show commands", "/exit": "quit"}

SYSTEM = (
    "You operate a simulated FODO storage ring through tools; this is not a physical machine. "
    "Read a channel before writing it. Only use channel names returned by tools. "
    "You may adjust a corrector setpoint when the user specifies the channel and kick, "
    "or choose bounded correctors when explicitly asked to correct the orbit. "
    "For a vague request to move the beam, ask for an exact corrector setpoint and kick. "
    "Answer briefly."
    f"\nBPM channels ({CONTRACT['units']['orbit']}): " + ", ".join(CONTRACT["bpm"])
    + f"\nCorrector setpoints ({CONTRACT['units']['kick']}, |kick| <= {CONTRACT['kick_limit_mrad']}): "
    + ", ".join(CONTRACT["corrector_setpoints"])
)
GOAL_SYSTEM = (
    SYSTEM + " Choose at most one batch of absolute corrector setpoints "
    "for the requested plane; values are mrad and must be within the listed limits. "
    "Use only the measured BPMs and setpoints below. Do not claim a result before the write is measured."
)


def server_params() -> StdioServerParameters:
    config = str(BUILD / "config.yml")
    return StdioServerParameters(
        command=str(REPO / ".venv" / "bin" / "python"),
        args=["-m", "osprey.mcp_server.control_system"],
        env={
            **os.environ,
            "CONFIG_FILE": config,
            "OSPREY_CONFIG": config,
            "OSPREY_MCP_TOOL_PREFIX": "controls",
            "EPICS_CA_NAME_SERVERS": f"127.0.0.1:{CA_PORT}",
            "EPICS_CA_AUTO_ADDR_LIST": "NO",
            "EPICS_CA_ADDR_LIST": "",
            "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{PVA_PORT}",
            "EPICS_PVA_AUTO_ADDR_LIST": "NO",
            "EPICS_PVA_ADDR_LIST": "",
        },
        cwd=str(BUILD),
    )


def wants_tool(prompt: str) -> bool:
    """Force a tool call only for prompts that name a channel or ask for a read/write."""
    return any(name in prompt for name in KNOWN_READS) or bool(ACTION.search(prompt) or BEAM_POSITION.search(prompt))


def goal_request(prompt: str) -> str | None:
    """Recognize an explicit orbit-correction goal, preserving exact PV commands."""
    steer_toward_center = bool(re.search(r"\bsteer\b", prompt, re.I) and
                               re.search(r"\b(toward|nearer)\b.*\b(axis|center|centre|zero)\b", prompt, re.I))
    if requested_write(prompt) or NEGATION.search(prompt) or not (GOAL_VERB.search(prompt) or steer_toward_center):
        return None
    if not re.search(r"\b(beam|orbit|bpm|transverse|correctors?)\b", prompt, re.IGNORECASE):
        return None
    horizontal = bool(re.search(r"\b(horizontal|x-plane|x orbit)\b", prompt, re.IGNORECASE))
    vertical = bool(re.search(r"\b(vertical|y-plane|y orbit)\b", prompt, re.IGNORECASE))
    return "both" if horizontal == vertical else "x" if horizontal else "y"


def move_clarification(prompt: str) -> str | None:
    """Do not let a small model invent a kick for an underspecified beam move."""
    if (any(channel in prompt for channel in CONTRACT["corrector_setpoints"])
            and re.search(r"\b(do not|don't|never|without)\b", prompt, re.IGNORECASE)):
        return "No write was made."
    if (any(channel in prompt for channel in CONTRACT["corrector_setpoints"])
            and re.search(r"\b(set|write|change|adjust)\b", prompt, re.IGNORECASE)
            and requested_write(prompt) is None):
        return (
            "No write was made. Specify an absolute numeric kick for the corrector "
            "(the setpoint unit is mrad), for example: Set RING:MAG:H01:KICK:SP to 0.2."
        )
    if MOVE_BEAM.search(prompt) and requested_write(prompt) is None and goal_request(prompt) is None:
        return (
            "I can move the simulated beam with a corrector. Please specify the corrector setpoint "
            "and kick in mrad. For a small test, say: Set RING:MAG:H01:KICK:SP to 0.05 mrad."
        )
    return None


def requested_write(prompt: str) -> tuple[str, float] | None:
    """Only explicit setpoint-and-kick requests authorize a model-issued write."""
    if not re.search(r"\b(set|write|change|adjust)\b", prompt, re.IGNORECASE):
        return None
    if re.search(r"\b(do not|don't|never|without)\b", prompt, re.IGNORECASE):
        return None
    for channel in CONTRACT["corrector_setpoints"]:
        match = re.search(
            re.escape(channel)
            + r"[^\n]*?\bto\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
            + r"(?:\s*mrad)?(?![\d.])(?=\s*(?:[,.;]|$|\bthen\b|\band\b))",
            prompt,
            re.IGNORECASE,
        )
        if match:
            return channel, float(match.group(1))
    return None


def coerce(args: dict, schema: dict | None) -> dict:
    """Wrap bare values where the schema wants an array; send numeric write values as numbers."""
    props = (schema or {}).get("properties", {})
    args = {k: [v] if props.get(k, {}).get("type") == "array" and not isinstance(v, list) else v for k, v in args.items()}
    # channel_write leaves `value` untyped, so small models send "0.1".
    for op in args.get("operations") or []:
        if isinstance(op, dict) and isinstance(op.get("value"), str):
            try:
                op["value"] = float(op["value"])
            except ValueError:
                pass
    return args


def dim(text: str) -> None:
    sys.stdout.flush()  # keep order with plain print() output
    print_formatted_text(FormattedText([("ansibrightblack", text)]))


def prompt_reader():
    """Line editor for a terminal (arrows, history, channel completion); plain input() for pipes."""
    if not sys.stdin.isatty():
        return lambda: asyncio.to_thread(input, "\n❯ ")
    session = PromptSession(
        history=FileHistory(str(LOGS / "small_agent_history")),
        auto_suggest=AutoSuggestFromHistory(),
        # WORD=True keeps colon-separated channel names whole while completing.
        completer=WordCompleter([*CHANNELS, *COMMANDS], WORD=True),
        complete_while_typing=False,
    )
    return lambda: session.prompt_async("\n❯ ")


def to_openai(tool) -> dict:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema}}


def readings_from(text: str) -> dict:
    """Extract only observed PV values from a successful Osprey channel_read result."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if data.get("status") != "success":
        return {}
    return data.get("summary", {}).get("readings", {})


def beam_position_report(text: str) -> str | None:
    readings = readings_from(text)
    if not all(pv in readings for pv in CONTRACT["bpm"]):
        return None
    lines = ["Measured beam position (mm):"]
    for index in range(0, len(CONTRACT["bpm"]), 2):
        x, y = CONTRACT["bpm"][index : index + 2]
        label = x.split(":")[2]
        lines.append(f"  {label}: X={readings[x]['value']:+.4f}, Y={readings[y]['value']:+.4f}")
    return "\n".join(lines)


def beam_plane_report(text: str) -> str | None:
    readings = readings_from(text)
    if not all(pv in readings for pv in CONTRACT["bpm"]):
        return None
    x = rms(readings, [pv for pv in CONTRACT["bpm"] if pv.endswith(":X")])
    y = rms(readings, [pv for pv in CONTRACT["bpm"] if pv.endswith(":Y")])
    if x is None or y is None:
        return None
    larger = "X" if x > y else "Y" if y > x else "neither"
    return (f"Measured RMS: X={x:.4f} mm, Y={y:.4f} mm. "
            + (f"The {larger} plane is larger." if larger != "neither" else "The planes are equal."))


def verified_report(prompt: str, readings: dict, write_request: tuple[str, float] | None,
                    write_result: dict | None, write_attempted: bool = False) -> str:
    """Render only observations and write outcomes returned by Osprey."""
    lines = []
    if write_request:
        channel, requested_value = write_request
        if write_result is None:
            lines.append((f"Write outcome unknown for {channel}; check its current setpoint."
                          if write_attempted else f"No write was made to {channel}."))
        else:
            outcome = write_result.get("outcome", "unknown")
            observed = write_result.get("observed_value")
            lines.append(f"{channel}: Osprey reported {outcome} for {requested_value:g} mrad"
                         + (f"; observed setpoint {observed} mrad." if observed is not None else "."))
    named = [pv for pv in [*CONTRACT["bpm"], *CONTRACT["corrector_setpoints"], *READBACKS.values()]
             if pv in prompt]
    for pv in named:
        if write_request and pv == write_request[0]:
            continue  # a pre-write setpoint read is not the final setpoint
        if pv in readings:
            unit = "mm" if pv in CONTRACT["bpm"] else "mrad"
            lines.append(f"{pv}: {readings[pv]['value']} {unit} (read from Osprey).")
        elif not (write_request and pv == write_request[0]):
            lines.append(f"{pv}: no verified reading returned.")
    if write_request and READBACKS[write_request[0]] in readings:
        pv = READBACKS[write_request[0]]
        lines.append(f"{pv}: {readings[pv]['value']} mrad (post-write readback).")
    return "\n".join(lines) if lines else "No verified channel result was returned."


def goal_channels(plane: str) -> tuple[list[str], list[str]]:
    bpm = [pv for pv in CONTRACT["bpm"] if plane == "both" or pv.endswith(":X" if plane == "x" else ":Y")]
    correctors = [pv for pv in CONTRACT["corrector_setpoints"]
                  if plane == "both" or (":MAG:H" in pv if plane == "x" else ":MAG:V" in pv)]
    return bpm, correctors


def rms(readings: dict, channels: list[str]) -> float | None:
    try:
        values = [float(readings[pv]["value"]) for pv in channels]
        return math.sqrt(sum(value * value for value in values) / len(values)) if all(map(math.isfinite, values)) else None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def compact_read(text: str, channels: list[str]) -> str:
    readings = readings_from(text)
    return json.dumps({"status": "success" if all(pv in readings for pv in channels) else "error",
                       "summary": {"readings": {pv: {"value": readings[pv]["value"]}
                                                for pv in channels if pv in readings}}})


def valid_goal_operations(operations: object, allowed: list[str]) -> list[dict]:
    if not isinstance(operations, list) or not 1 <= len(operations) <= len(allowed):
        raise ValueError("expected one bounded write batch for the selected plane")
    normalized = []
    seen = set()
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("every write operation must be an object")
        channel = op.get("channel", op.get("name"))
        value = op.get("value")
        if channel not in allowed or channel in seen:
            raise ValueError(f"duplicate or disallowed corrector: {channel}")
        if isinstance(value, bool):
            raise ValueError("kick must be a finite number")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("kick must be a finite number") from None
        if not math.isfinite(value) or abs(value) > KICK_LIMIT:
            raise ValueError(f"kick for {channel} exceeds ±{KICK_LIMIT:g} mrad")
        seen.add(channel)
        normalized.append({"channel": channel, "value": value})
    return normalized


async def goal_turn(prompt, plane, history, mcp, http, tools, schemas, args) -> None:
    """Let the model choose one bounded correction, then report only measured results."""
    bpm, correctors = goal_channels(plane)
    needed = bpm + correctors
    try:
        before_raw = await mcp.call_tool("channel_read", {"channels": needed})
        before_text = "\n".join(getattr(part, "text", "") for part in before_raw.content)
        before = readings_from(before_text)
        if not all(pv in before for pv in needed):
            raise ValueError("pre-write BPM or setpoint read was incomplete")
    except Exception as exc:
        print(f"No write was made: {exc}.")
        return
    dim(f"  [channel_read · pre-write] {len(needed)} measured PVs")
    write_tools = [tool for tool in tools if tool["function"]["name"] == "channel_write"]
    tool_id = "prewrite-read"
    messages = [
        {"role": "system", "content": GOAL_SYSTEM},
        *history, {"role": "user", "content": prompt},
        {"role": "assistant", "tool_calls": [{"id": tool_id, "type": "function",
          "function": {"name": "channel_read", "arguments": json.dumps({"channels": needed})}}]},
        {"role": "tool", "tool_call_id": tool_id, "content": compact_read(before_text, needed)},
    ]
    response = await http.post("/chat/completions", json={
        "model": args.model, "messages": messages, "tools": write_tools,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "tool_choice": "required",
    })
    if response.is_error:
        raise httpx.HTTPStatusError(f"{response.status_code} from vLLM: {response.text}",
                                    request=response.request, response=response)
    calls = response.json()["choices"][0]["message"].get("tool_calls") or []
    try:
        if len(calls) != 1 or calls[0]["function"]["name"] != "channel_write":
            raise ValueError("model did not propose one channel_write call")
        arguments = coerce(json.loads(calls[0]["function"]["arguments"] or "{}"),
                           schemas.get("channel_write"))
        operations = valid_goal_operations(arguments.get("operations"), correctors)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"No write was made: {exc}.")
        return
    call_args = {"operations": operations, "confirm": True}
    dim(f"  [channel_write] {json.dumps(call_args)}")
    write_error = None
    results = {}
    try:
        result = await mcp.call_tool("channel_write", call_args)
        payload = json.loads("\n".join(getattr(part, "text", "") for part in result.content))
        if payload.get("status") == "success":
            results = {entry.get("channel"): entry for entry in payload.get("summary", {}).get("results", [])}
        else:
            write_error = payload.get("message", "Osprey did not confirm the write")
    except Exception as exc:
        write_error = str(exc)
    follow_up = bpm + [READBACKS[op["channel"]] for op in operations]
    after = {}
    try:
        observed = await mcp.call_tool("channel_read", {"channels": follow_up})
        after = readings_from("\n".join(getattr(part, "text", "") for part in observed.content))
    except Exception as exc:
        write_error = f"{write_error}; post-write read failed: {exc}" if write_error else f"post-write read failed: {exc}"
    dim(f"  [channel_read · post-write] {len(after)}/{len(follow_up)} measured PVs")
    lines = []
    for op in operations:
        channel = op["channel"]
        rb = READBACKS[channel]
        outcome = results.get(channel, {}).get("outcome", "unknown")
        measured = after.get(rb, {}).get("value")
        lines.append(f"{channel}: requested {op['value']:g} mrad; Osprey {outcome}; "
                     + (f"readback {measured} mrad." if measured is not None else "no verified readback."))
    if write_error:
        lines.append(f"Write or follow-up error: {write_error}.")
    for label, channels in (("X", [pv for pv in bpm if pv.endswith(":X")]),
                            ("Y", [pv for pv in bpm if pv.endswith(":Y")])):
        if not channels:
            continue
        start, end = rms(before, channels), rms(after, channels)
        lines.append(f"{label} RMS: {start:.4f} → {end:.4f} mm (measured)." if start is not None and end is not None
                     else f"{label} RMS: no complete post-write measurement.")
    answer = "\n".join(lines)
    print(answer)
    history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    del history[: max(0, len(history) - max(1, args.history // 2) * 2)]


async def skill_goal_turn(prompt, expected_plane, history, mcp, http, args) -> None:
    """Let LFM route a goal; use measured Osprey PVs and a bounded deterministic solve."""
    response = await http.post("/chat/completions", json={
        "model": args.model,
        "messages": [{"role": "system", "content": SKILL_SYSTEM}, *history,
                     {"role": "user", "content": prompt}],
        "tools": [SKILL_TOOL], "tool_choice": "required",
        "temperature": args.temperature, "max_tokens": args.max_tokens,
    })
    if response.is_error:
        raise httpx.HTTPStatusError(f"{response.status_code} from vLLM: {response.text}",
                                    request=response.request, response=response)
    calls = response.json()["choices"][0]["message"].get("tool_calls") or []
    try:
        if len(calls) != 1 or calls[0]["function"]["name"] != "orbit_correct":
            raise ValueError("model did not select one orbit_correct skill")
        choice = json.loads(calls[0]["function"]["arguments"] or "{}")
        plane, effort = choice.get("plane"), choice.get("effort")
        if plane != expected_plane:
            raise ValueError(f"model selected {plane!r}, but the request names {expected_plane!r}")
        if effort not in ("gentle", "balanced", "aggressive"):
            raise ValueError("model selected an unknown effort level")
        intended_effort = requested_effort(prompt)
        if effort != intended_effort and intended_effort != "balanced":
            raise ValueError(f"model selected {effort!r} effort, but the request calls for {intended_effort!r}")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"No write was made: {exc}.")
        return
    if effort != intended_effort:
        dim(f"  [orbit_correct · effort normalized] model selected {effort}; using balanced for this request")
        effort = intended_effort
    dim(f"  [orbit_correct] {json.dumps({'plane': plane, 'effort': effort})}")
    bpm, correctors = goal_channels(plane)
    needed = bpm + correctors
    try:
        pre = await mcp.call_tool("channel_read", {"channels": needed})
        before = readings_from("\n".join(getattr(part, "text", "") for part in pre.content))
        if not all(pv in before for pv in needed):
            raise ValueError("pre-write BPM or corrector read incomplete")
        measured = {pv: float(before[pv]["value"]) for pv in needed}
        settings = solve(measured, plane, effort)
        operations = valid_goal_operations(
            [{"channel": pv, "value": value} for pv, value in settings.items()], correctors)
    except Exception as exc:
        print(f"No write was made: {exc}.")
        return
    dim(f"  [channel_read · pre-write] {len(needed)} measured PVs")
    dim(f"  [channel_write] {json.dumps({'operations': operations, 'confirm': True})}")
    write_error = None
    confirmed = {}
    try:
        written = await mcp.call_tool("channel_write", {"operations": operations, "confirm": True})
        payload = json.loads("\n".join(getattr(part, "text", "") for part in written.content))
        if payload.get("status") == "success":
            confirmed = {entry.get("channel"): entry for entry in payload.get("summary", {}).get("results", [])}
        else:
            write_error = payload.get("message", "Osprey did not confirm the write")
    except Exception as exc:
        write_error = str(exc)
    follow_up = bpm + [READBACKS[pv] for pv in settings]
    after = {}
    try:
        post = await mcp.call_tool("channel_read", {"channels": follow_up})
        after = readings_from("\n".join(getattr(part, "text", "") for part in post.content))
    except Exception as exc:
        write_error = f"{write_error}; read failed: {exc}" if write_error else f"read failed: {exc}"
    dim(f"  [channel_read · post-write] {len(after)}/{len(follow_up)} measured PVs")
    lines = []
    for pv, value in settings.items():
        observed = after.get(READBACKS[pv], {}).get("value")
        outcome = confirmed.get(pv, {}).get("outcome", "unknown")
        lines.append(f"{pv}: requested {value:.6f} mrad; Osprey {outcome}; "
                     + (f"readback {observed} mrad." if observed is not None else "no verified readback."))
    if write_error:
        lines.append(f"Write or follow-up error: {write_error}.")
    for label, channels in (("X", [pv for pv in bpm if pv.endswith(":X")]),
                            ("Y", [pv for pv in bpm if pv.endswith(":Y")])):
        if channels:
            start, end = rms(before, channels), rms(after, channels)
            lines.append(f"{label} RMS: {start:.4f} → {end:.4f} mm (measured)." if start is not None and end is not None
                         else f"{label} RMS: no complete post-write measurement.")
    answer = "\n".join(lines)
    print(answer)
    history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    del history[: max(0, len(history) - max(1, args.history // 2) * 2)]


async def turn(prompt, history, mcp, http, tools, schemas, args) -> None:
    plane = goal_request(prompt)
    if plane:
        await skill_goal_turn(prompt, plane, history, mcp, http, args)
        return
    messages = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": prompt}]
    force = wants_tool(prompt)
    write_request = requested_write(prompt)
    available_tools = tools if write_request else [tool for tool in tools if tool["function"]["name"] != "channel_write"]
    general_beam_request = bool(BEAM_POSITION.search(prompt)) and not any(pv in prompt for pv in KNOWN_READS)
    channel_request = bool(write_request or BEAM_POSITION.search(prompt) or any(pv in prompt for pv in KNOWN_READS))
    observed_readings: dict = {}
    post_write_observed: set[str] = set()
    write_result: dict | None = None
    write_attempted = False
    answer = ""
    for step in range(args.max_steps):
        resp = await http.post(
            "/chat/completions",
            json={
                "model": args.model, "messages": messages, "tools": available_tools,
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "tool_choice": "required" if force and step == 0 else "auto",
            },
        )
        if resp.is_error:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} from vLLM: {resp.text}", request=resp.request, response=resp
            )
        msg = resp.json()["choices"][0]["message"]
        messages.append({k: v for k, v in msg.items() if v is not None})
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (verified_report(prompt, observed_readings, write_request, write_result, write_attempted)
                      if channel_request else (msg.get("content") or ""))
            break
        measured_beam = None
        for call in calls:
            name = call["function"]["name"]
            readback_log = None
            pre_read_log = None
            rejected = None
            try:
                if name not in schemas:
                    raise ValueError(f"unknown tool {name!r}; available: {', '.join(schemas)}")
                if name == "channel_write" and write_request is None:
                    raise ValueError("no explicit corrector setpoint and kick in the operator request")
                call_args = coerce(json.loads(call["function"]["arguments"] or "{}"), schemas.get(name))
                if name == "channel_read":
                    if general_beam_request:
                        call_args["channels"] = CONTRACT["bpm"]
                    unknown = set(call_args.get("channels") or []) - KNOWN_READS
                    if unknown:
                        raise ValueError(f"unknown PV(s): {sorted(unknown)}; use the listed ring PVs")
                elif name == "channel_write":
                    operations = call_args.get("operations") or []
                    if len(operations) != 1 or not isinstance(operations[0], dict):
                        raise ValueError("expected exactly one corrector write operation")
                    requested_channel, requested_value = write_request
                    if "channel" not in operations[0] and operations[0].get("name") == requested_channel:
                        operations[0]["channel"] = operations[0].pop("name")
                    if operations[0].get("channel") != requested_channel or operations[0].get("value") != requested_value:
                        raise ValueError("model write arguments differ from the operator's requested channel or kick")
                    if write_attempted:
                        raise ValueError("already attempted the requested write this turn")
                    unknown = {
                        op.get("channel") for op in operations
                        if isinstance(op, dict) and op.get("channel") not in READBACKS
                    }
                    if unknown:
                        raise ValueError(f"not a writable corrector setpoint: {sorted(unknown)}")
                    if requested_channel not in observed_readings:
                        before = await mcp.call_tool("channel_read", {"channels": [requested_channel]})
                        before_text = "\n".join(getattr(part, "text", "") for part in before.content)
                        prior = readings_from(before_text)
                        if requested_channel not in prior:
                            raise ValueError("pre-write setpoint read failed; no write attempted")
                        observed_readings.update(prior)
                        pre_read_log = f"  [channel_read · pre-write] {requested_channel}={prior[requested_channel]['value']} mrad"
                if name == "channel_write":
                    write_attempted = True
                result = await mcp.call_tool(name, call_args)
                text = "\n".join(getattr(part, "text", "") for part in result.content)
                if name == "channel_read":
                    values = readings_from(text)
                    observed_readings.update(values)
                    if write_attempted:
                        post_write_observed.update(values)
                if name == "channel_read" and general_beam_request:
                    comparison = bool(re.search(r"\b(which|larger|compare|comparison)\b.*\b(plane|rms)\b|\b(plane|rms)\b.*\b(larger|compare)\b", prompt, re.I))
                    measured_beam = beam_plane_report(text) if comparison else beam_position_report(text)
                if name == "channel_write":
                    try:
                        envelope = json.loads(text)
                        if envelope.get("status") == "success":
                            results = envelope.get("summary", {}).get("results", [])
                            write_result = next((entry for entry in results if entry.get("channel") == requested_channel), None)
                    except (ValueError, TypeError):
                        pass
                if name == "channel_write" and write_result is not None:
                    readbacks = [
                        READBACKS[op["channel"]]
                        for op in call_args.get("operations", [])
                        if isinstance(op, dict) and op.get("channel") in READBACKS
                    ]
                    if readbacks:
                        observed = await mcp.call_tool("channel_read", {"channels": readbacks})
                        observed_text = "\n".join(getattr(part, "text", "") for part in observed.content)
                        text += "\nPost-write readback:\n" + observed_text
                        values = readings_from(observed_text)
                        observed_readings.update(values)
                        post_write_observed.update(values)
                        shown = ", ".join(f"{pv}={values[pv]['value']}" for pv in readbacks if pv in values)
                        readback_log = f"  [channel_read · post-write] {shown or json.dumps({'channels': readbacks})}"
            except Exception as exc:  # report tool errors to the model instead of crashing the REPL
                rejected, text = str(exc), f"error: {exc}"
            if pre_read_log:
                dim(pre_read_log)
            if rejected:
                dim(f"  [{name} rejected] {rejected}")
            else:
                dim(f"  [{name}] {json.dumps(call_args)}")
            if readback_log:
                dim(readback_log)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        if measured_beam:
            answer = measured_beam
            break
        if write_attempted:
            # A write is complete from the operator's perspective only after
            # every explicitly requested follow-up PV has been read again.
            requested = [pv for pv in [*CONTRACT["bpm"], *CONTRACT["corrector_setpoints"],
                                       *READBACKS.values()] if pv in prompt and pv != write_request[0]]
            missing = [pv for pv in requested if pv not in post_write_observed]
            if missing:
                for pv in missing:
                    observed_readings.pop(pv, None)  # a pre-write reading is not post-write evidence
                try:
                    follow_up = await mcp.call_tool("channel_read", {"channels": missing})
                    follow_up_text = "\n".join(getattr(part, "text", "") for part in follow_up.content)
                    values = readings_from(follow_up_text)
                    observed_readings.update(values)
                    shown = ", ".join(f"{pv}={values[pv]['value']}" for pv in missing if pv in values)
                    dim(f"  [channel_read · requested] {shown or json.dumps({'channels': missing})}")
                except Exception as exc:
                    dim(f"  [channel_read · requested failed] {exc}")
            answer = verified_report(prompt, observed_readings, write_request, write_result, write_attempted)
            break
    else:
        answer = (verified_report(prompt, observed_readings, write_request, write_result, write_attempted)
                  if channel_request else f"(stopped after {args.max_steps} steps)")
    print(answer)
    # Keep concise, verified turns. Raw MCP envelopes can be thousands of
    # tokens and exhaust the 230M model's 4096-token context in a few turns.
    history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    del history[: max(0, len(history) - max(1, args.history // 2) * 2)]


async def main(args) -> None:
    health = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Health.Status}}", VA_CONTAINER],
        capture_output=True, text=True, check=False,
    )
    if health.returncode != 0 or health.stdout.strip() != "healthy":
        raise RuntimeError(f"{VA_CONTAINER} is not healthy; start this worktree's Osprey deployment first")
    LOGS.mkdir(parents=True, exist_ok=True)
    server_log = LOGS / "small_agent_mcp.log"
    # The controls server's FastMCP logging goes to a file, not the operator's terminal.
    with server_log.open("a") as errlog:
        async with stdio_client(server_params(), errlog=errlog) as (read, write), ClientSession(read, write) as mcp:
            await mcp.initialize()
            listed = [t for t in (await mcp.list_tools()).tools if t.name in TOOLS]
            tools, schemas = [to_openai(t) for t in listed], {t.name: t.inputSchema for t in listed}
            print(f"small-agent · {args.model} @ {args.base_url} · {len(listed)} tools")
            dim(f"Tab completes channels · ↑/↓ history · /help · Ctrl-D exits · server log: {server_log}")
            read_line = prompt_reader()
            history: list[dict] = []
            async with httpx.AsyncClient(base_url=args.base_url, timeout=300) as http:
                while True:
                    try:
                        prompt = (await read_line()).strip()
                    except KeyboardInterrupt:
                        continue  # Ctrl-C clears the line, like a shell
                    except EOFError:
                        print()
                        return
                    if prompt in ("/exit", "/quit"):
                        return
                    if prompt == "/clear":
                        history.clear()
                        dim("conversation cleared")
                    elif prompt == "/help":
                        for command, text in COMMANDS.items():
                            dim(f"  {command:<7} {text}")
                    elif prompt:
                        clarification = move_clarification(prompt)
                        if clarification:
                            print(clarification)
                            continue
                        try:
                            await turn(prompt, history, mcp, http, tools, schemas, args)
                        except (httpx.HTTPError, KeyError) as exc:
                            print(f"model request failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL_NAME") or PROVIDER["default_model"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=5, help="Model calls per turn")
    parser.add_argument("--history", type=int, default=8, help="Messages kept across turns")
    # The Osprey Web Terminal appends Claude Code's conversation flags to its shell command.
    parser.add_argument("--session-id", help=argparse.SUPPRESS)
    parser.add_argument("--resume", help=argparse.SUPPRESS)
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        pass
