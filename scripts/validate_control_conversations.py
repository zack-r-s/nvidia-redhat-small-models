"""Check exported LFM tool conversations and their 230M context length.

Run with a Python environment containing transformers and the cached LFM2.5-230M tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(folder: Path, model: str, max_context: int) -> dict:
    from transformers import AutoTokenizer

    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "hackring.lfm_conversations/1":
        raise ValueError("wrong conversation manifest")
    for name, expected in manifest["files_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"conversation checksum mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    counts = Counter()
    largest = 0
    for split in ("train", "validation", "test"):
        for line in (folder / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if set(row) != {"messages"}:
                raise ValueError("SFT row must contain only messages")
            messages = row["messages"]
            if messages[0]["role"] != "system" or messages[1]["role"] != "user" or messages[-1]["role"] != "assistant":
                raise ValueError("invalid role sequence")
            for message in messages:
                if set(message) != {"role", "content"} or not isinstance(message["content"], str):
                    raise ValueError("invalid message")
                if message["role"] == "assistant" and "<|tool_call_start|>" in message["content"]:
                    if message["content"].count("<|tool_call_start|>") != 1 or message["content"].count("<|tool_call_end|>") != 1:
                        raise ValueError("malformed LFM tool call")
                if message["role"] == "tool":
                    if "<|tool_response_start|>" in message["content"]:
                        raise ValueError("tool response markers must be inserted by template")
                    json.loads(message["content"])
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            input_tokens = len(tokens["input_ids"])
            largest = max(largest, input_tokens)
            if input_tokens + 512 > max_context:
                raise ValueError(f"conversation needs {input_tokens} input + 512 output tokens")
            counts[split] += 1
    if dict(counts) != manifest["counts"]:
        raise ValueError("conversation counts differ from manifest")
    return {"conversations": sum(counts.values()), "by_split": dict(counts),
            "largest_input_tokens": largest, "model": model}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=ROOT / "datasets/v002/sft")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(validate(args.folder.resolve(), args.model, args.max_context), indent=2))


if __name__ == "__main__":
    main()
