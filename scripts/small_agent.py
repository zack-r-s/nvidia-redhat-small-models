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
COMMANDS = {"/clear": "forget the conversation so far", "/help": "show commands", "/exit": "quit"}

SYSTEM = (
    "You operate a simulated FODO storage ring through tools; this is not a physical machine. "
    "Read a channel before writing it. Only use channel names returned by tools. "
    "You may adjust a corrector setpoint when the user specifies the channel and kick. "
    "For a vague request to move the beam, ask for an exact corrector setpoint and kick. "
    "Answer briefly."
    f"\nBPM channels ({CONTRACT['units']['orbit']}): " + ", ".join(CONTRACT["bpm"])
    + f"\nCorrector setpoints ({CONTRACT['units']['kick']}, |kick| <= {CONTRACT['kick_limit_mrad']}): "
    + ", ".join(CONTRACT["corrector_setpoints"])
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
    if MOVE_BEAM.search(prompt) and requested_write(prompt) is None:
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


async def turn(prompt, history, mcp, http, tools, schemas, args) -> None:
    messages = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": prompt}]
    start = len(messages) - 1
    force = wants_tool(prompt)
    write_request = requested_write(prompt)
    available_tools = tools if write_request else [tool for tool in tools if tool["function"]["name"] != "channel_write"]
    general_beam_request = bool(BEAM_POSITION.search(prompt)) and not any(pv in prompt for pv in KNOWN_READS)
    channel_request = bool(write_request or BEAM_POSITION.search(prompt) or any(pv in prompt for pv in KNOWN_READS))
    observed_readings: dict = {}
    write_result: dict | None = None
    write_attempted = False
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
            print(verified_report(prompt, observed_readings, write_request, write_result, write_attempted)
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
                    observed_readings.update(readings_from(text))
                if name == "channel_read" and general_beam_request:
                    measured_beam = beam_position_report(text)
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
            print(measured_beam)
            break
    else:
        print(verified_report(prompt, observed_readings, write_request, write_result, write_attempted)
              if channel_request else f"(stopped after {args.max_steps} steps)")
    history.extend(messages[start:])
    # Small models degrade with context; keep only the last few exchanges.
    # Trim at a user message so no tool result is left without its call.
    del history[: max(0, len(history) - args.history)]
    while history and history[0]["role"] != "user":
        history.pop(0)


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
    parser.add_argument("--model", default=PROVIDER["default_model"])
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
