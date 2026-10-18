"""Record a lexical overlap screen for Mark IV against every SFT training prompt."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from mark_iv_common import CASES, ROOT, cases, sha

TRAIN = [ROOT / name for name in (
    "datasets/v002/sft/train.jsonl", "datasets/v003/train.jsonl", "datasets/v004/train.jsonl")]
OUT = ROOT / "evals/mark-iv/novelty-audit.json"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> None:
    training: list[tuple[str, str]] = []
    for path in TRAIN:
        for row in path.read_text().splitlines():
            for message in json.loads(row)["messages"]:
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    training.append((str(path.relative_to(ROOT)), normalize(message["content"])))
    results = []
    for case in cases():
        prompt = normalize(case["prompt"])
        ratios = [(difflib.SequenceMatcher(None, prompt, candidate).ratio(), source)
                  for source, candidate in training]
        highest, source = max(ratios)
        results.append({"id": case["id"], "exact_training_match": any(prompt == item for _, item in training),
                        "max_normalized_sequence_ratio": round(highest, 4), "closest_training_source": source})
    result = {"schema": "hackring.mark_iv_novelty_audit/1", "cases_sha256": sha(CASES),
              "method": "lowercase alphanumeric normalization; Python SequenceMatcher ratio against every training user turn; semantic overlap still requires review",
              "training_files": {str(path.relative_to(ROOT)): sha(path) for path in TRAIN},
              "training_turns": len(training), "exact_matches": sum(row["exact_training_match"] for row in results),
              "highest_ratio": max(row["max_normalized_sequence_ratio"] for row in results),
              "cases": results}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Screened {len(results)} prompts against {len(training)} training turns; "
          f"exact matches: {result['exact_matches']}; highest ratio: {result['highest_ratio']}")


if __name__ == "__main__":
    main()
