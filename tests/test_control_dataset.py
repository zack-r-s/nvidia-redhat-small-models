"""The response teacher and SFT examples must preserve measured PV semantics."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_control_conversations import canonical_prompt, conversation
from fit_response_matrix import fit
from generate_control_dataset import metrics, target
from generate_dataset import BPMS, SETPOINTS
import paraphrase_control_prompts as paraphrases
from paraphrase_control_prompts import request, style_for, validate_prompt


def test_v001_train_only_response_fit_predicts_heldout_changes():
    result = fit(ROOT / "datasets/v001")
    assert result["fit_split"] == "train"
    assert result["rank"] == 8
    assert result["heldout_metrics"]["test"]["rows"] == 448
    assert result["heldout_metrics"]["test"]["max_abs_error_mm"] < 1e-8


def test_bounded_teacher_improves_disturbed_orbit_predictions():
    teacher = json.loads((ROOT / "datasets/v002/response_teacher.json").read_text())
    matrix = np.asarray(teacher["matrix_mm_per_mrad"])
    episodes = [json.loads(line) for line in (ROOT / "datasets/v001/episodes.jsonl").read_text().splitlines()[:4]]
    for episode in episodes:
        before = episode["initial_pvs"]
        for goal in ("x", "y", "both"):
            settings = target(goal, before, matrix, np.random.default_rng(episode["seed"]), episode["seed"])
            delta = np.array([settings[sp] - before[sp] for sp in SETPOINTS])
            after = {**before, **settings}
            after.update({pv: before[pv] + float(matrix[i] @ delta) for i, pv in enumerate(BPMS)})
            assert all(abs(value) <= 0.35 for value in settings.values())
            assert metrics(goal, after)["goal_cost"] < metrics(goal, before)["goal_cost"]


def test_export_contains_only_public_measured_state_and_native_tool_calls():
    row = json.loads((ROOT / "datasets/v001/attempts.jsonl").read_text().splitlines()[0])
    row.update(goal="both", seed=20262000, task_id="sample")
    messages = conversation(row, canonical_prompt(row))["messages"]
    assert [message["role"] for message in messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "<|tool_call_start|>[channel_write(" in messages[4]["content"]
    assert "hidden_calibration_offsets" not in json.dumps(messages)
    assert "read_after" not in messages[1]["content"]


def test_paraphrase_rejects_changed_plane_or_invented_values():
    row = json.loads((ROOT / "datasets/v001/attempts.jsonl").read_text().splitlines()[0])
    row.update(goal="x", seed=20262000, task_id="sample")
    validate_prompt("Please center the horizontal orbit and show the measured result.", row)
    with pytest.raises(ValueError):
        validate_prompt("Please center the vertical orbit and show the result.", row)
    with pytest.raises(ValueError):
        validate_prompt("Correct the horizontal orbit with a 0.1 mrad kick.", row)


def test_paraphrase_requests_vary_style_without_sending_future_values():
    rows = [json.loads(line) for line in (ROOT / "datasets/v002/raw/actions.jsonl").read_text().splitlines()[:8]]
    styles = {style_for(row["task_id"]) for row in rows}
    assert len(styles) > 1
    for row in rows:
        payload = json.loads(request(row, "gpt-6-luna")["input"])
        assert payload == {"task": row["goal"], "canonical_operator_request": canonical_prompt(row),
                           "style_cue": style_for(row["task_id"])}


def test_paraphrase_run_resumes_completed_output_without_api(tmp_path, monkeypatch):
    async def fake_collect(rows, completed, sources, progress, model, workers):
        for row in rows:
            prompt = "Please " + canonical_prompt(row)
            validate_prompt(prompt, row)
            with progress.open("a") as stream:
                stream.write(json.dumps({"task_id": row["task_id"], "operator_prompt": prompt,
                                         "source": "gpt"}) + "\n")
            completed[row["task_id"]] = prompt
            sources[row["task_id"]] = "gpt"

    monkeypatch.setattr(paraphrases, "collect", fake_collect)
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    args = Namespace(raw=ROOT / "datasets/v002/raw", output=tmp_path / "notes", model="gpt-6-luna",
                     workers=2, limit=4, resume_after_edit=False)
    paraphrases.run(args)
    first = (args.output / "paraphrases.jsonl").read_bytes()
    monkeypatch.delenv("OPENAI_API_KEY")
    paraphrases.run(args)
    assert (args.output / "paraphrases.jsonl").read_bytes() == first


def test_exact_write_fallback_preserves_action_and_measurement():
    rows = [json.loads(line) for line in (ROOT / "datasets/v002/raw/actions.jsonl").read_text().splitlines()]
    exact = next(row for row in rows if row["goal"] == "exact_write")
    prompt = paraphrases.exact_write_fallback(exact)
    assert prompt != canonical_prompt(exact)
    validate_prompt(prompt, exact)
