"""Offline checks for episode-scoped annotations and grounded text validation."""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import annotate_dataset as annotator


class FakeResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        field = kwargs["text"]["format"]["name"]
        context = json.loads(kwargs["input"])
        if field == "operator_prompt":
            if context["focus_kind"] == "largest_bpm":
                name = context["focus_pv"].split(":")[2]
                plane = context["focus_pv"].split(":")[-1]
                value = f"Bring {name} {plane} closer to center, then check the orbit after each correction."
            else:
                plane = context["focus_plane"]
                value = f"Bring the {plane} plane closer to center, then check the orbit after each correction."
        else:
            trend = context["orbit_cost_trend"]
            value = f"The measured orbit cost {trend} after the corrector write and follow-up read."
        return types.SimpleNamespace(status="completed", output_text=json.dumps({field: value}))


class FakeOpenAI:
    responses = FakeResponses()

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass


class AnnotatorTests(unittest.TestCase):
    def test_bad_model_observation_uses_labeled_grounded_fallback(self):
        row = next(annotator.records(annotator.DATASET / "attempts.jsonl"))[0]

        class BadResponses:
            calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                return types.SimpleNamespace(
                    status="completed",
                    output_text=json.dumps({"observation": "The measured orbit cost improved after the write."}),
                )

        responses = BadResponses()
        result, source = asyncio.run(annotator.generate(
            types.SimpleNamespace(responses=responses), row, annotator.DEFAULT_MODEL, "observation"))
        self.assertEqual(source, "deterministic_fallback")
        self.assertIn("orbit cost was unchanged", result)
        self.assertEqual(responses.calls, 3)

    def test_rejects_invented_number_and_wrong_trend(self):
        row = next(annotator.records(annotator.DATASET / "attempts.jsonl"))[0]
        with self.assertRaisesRegex(ValueError, "numeric claim"):
            annotator.validate_text("Bring BPM02 X to 9 mm and check the orbit again.", "operator_prompt", row)
        with self.assertRaisesRegex(ValueError, "focused BPM plane"):
            annotator.validate_text("Bring BPM02 Y closer to center, then check the orbit again.",
                                    "operator_prompt", row)
        with self.assertRaisesRegex(ValueError, "measured trend"):
            annotator.validate_text("The measured orbit cost improved after the corrector write.",
                                    "observation", row)
        with self.assertRaisesRegex(ValueError, "schema field name"):
            annotator.validate_text("The orbit_cost_trend was unchanged after the corrector write.",
                                    "observation", row)
        with self.assertRaisesRegex(ValueError, "moving between correctors"):
            annotator.validate_text("The write moved H02 kick setpoint to H03, and orbit cost was unchanged.",
                                    "observation", row)

    def test_one_prompt_per_episode_and_separate_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotations"
            args = argparse.Namespace(dataset=annotator.DATASET, output=output,
                                      model=annotator.DEFAULT_MODEL, limit=None,
                                      sample=8, workers=2, resume_after_edit=True)
            fake_module = types.SimpleNamespace(AsyncOpenAI=FakeOpenAI)
            FakeOpenAI.responses = FakeResponses()
            source_before = annotator.digest(annotator.DATASET / "attempts.jsonl")
            with patch.dict(sys.modules, {"openai": fake_module}), patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                annotator.run(args)
                first_calls = len(FakeOpenAI.responses.calls)
                (output / "annotated_attempts.jsonl").unlink()
                (output / "annotation_manifest.json").unlink()
                run = json.loads((output / "run.json").read_text())
                run["annotator_sha256"] = "previous-script-hash"
                annotator.write_json(output / "run.json", run)
                annotator.run(args)
            self.assertEqual(len(FakeOpenAI.responses.calls), first_calls)
            prompts = [row for row, _ in annotator.records(output / "episode_prompts.jsonl")]
            rows = [row for row, _ in annotator.records(output / "annotated_attempts.jsonl")]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(prompts), len({row["episode_id"] for row in rows}))
            prompt_by_episode = {row["custom_id"]: row["operator_prompt"] for row in prompts}
            for row in rows:
                self.assertEqual(row["annotation"]["operator_prompt"], prompt_by_episode[row["episode_id"]])
                self.assertEqual(row["annotation"]["observation_source"], "model")
            self.assertEqual(json.loads((output / "run.json").read_text())["resume_annotator_sha256"],
                             annotator.digest(Path(annotator.__file__)))
            self.assertEqual(annotator.digest(annotator.DATASET / "attempts.jsonl"), source_before)


if __name__ == "__main__":
    unittest.main()
