"""Smoke test for the production OpenAI harness path.

Contract:
- Run 1 must fully succeed and classify every input row.
- Run 2 must perform zero incremental work because dedup should match
  on dataset + resolved_model_id + prompt_hash.

Note:
- This test intentionally uses live provider calls, so rare transient HTTP
  failures are possible. The LLM retry policy should absorb most flakiness.
  A one-off rerun is acceptable; repeated failures indicate a systematic issue.

To run:
```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.test_llm_openai
```
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from lib.constants import REPO_ROOT
from lib.timestamp_utils import get_current_timestamp

INPUT_CSV = Path(__file__).with_name("testing_data.csv")
EXPECTED_ALIAS = "openai"
EXPECTED_RESOLVED_MODEL_ID = "gpt-5-nano"


def _count_csv_data_rows(path: Path) -> int:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _latest_run_dir(output_root: Path) -> Path:
    run_dirs = [p for p in output_root.iterdir() if p.is_dir()]
    assert run_dirs, f"No timestamped run directories found under {output_root}"
    return sorted(run_dirs)[-1]


def _run_eval_once(output_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    cmd = [
        sys.executable,
        "-m",
        "evaluation.examples_test",
        "--input-path",
        str(INPUT_CSV),
        "--output-root",
        str(output_root),
        "--models",
        EXPECTED_ALIAS,
        "--batch-size",
        "10",
    ]

    # check=False so we can print useful stdout/stderr in assertion messages
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_run_succeeded(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, (
        "Evaluation harness subprocess failed.\n"
        f"STDOUT:\n{proc.stdout}\n\n"
        f"STDERR:\n{proc.stderr}"
    )


def _assert_common_artifacts_exist(run_dir: Path) -> None:
    assert (run_dir / "output.csv").exists(), f"Missing output.csv in {run_dir}"
    assert (run_dir / "metrics.json").exists(), f"Missing metrics.json in {run_dir}"
    assert (run_dir / "metadata.json").exists(), f"Missing metadata.json in {run_dir}"


def _assert_metadata_shape(metadata_path: Path) -> None:
    metadata = _read_json(metadata_path)

    assert "cli_args" in metadata
    assert metadata["cli_args"]["input_path"] == str(INPUT_CSV)
    assert metadata["cli_args"]["models"] == [EXPECTED_ALIAS]

    models = metadata.get("models")
    assert isinstance(models, list), "metadata.models must be a list"
    assert len(models) == 1, (
        f"Expected exactly one model metadata entry, got {len(models)}"
    )

    model_meta = models[0]
    assert model_meta["llm_provider_name"] == EXPECTED_ALIAS
    assert model_meta["resolved_model_id"] == EXPECTED_RESOLVED_MODEL_ID
    assert isinstance(model_meta["prompt_hash"], str) and model_meta["prompt_hash"]
    assert (
        isinstance(model_meta["prompt_template"], str) and model_meta["prompt_template"]
    )


def _assert_first_run_complete(run_dir: Path) -> None:
    output_csv = run_dir / "output.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"

    _assert_common_artifacts_exist(run_dir)
    _assert_metadata_shape(metadata_json)

    expected_rows = _count_csv_data_rows(INPUT_CSV)
    actual_rows = _read_csv_rows(output_csv)

    assert len(actual_rows) == expected_rows, (
        f"Expected first run to classify all rows. "
        f"Expected {expected_rows}, got {len(actual_rows)}"
    )

    # Verify every row is attributed to the resolved model id.
    model_values = {row["model"] for row in actual_rows}
    assert model_values == {EXPECTED_RESOLVED_MODEL_ID}

    # Verify no prediction rows are blank.
    blank_pred_rows = [row for row in actual_rows if row["pred_label"] in ("", None)]
    assert not blank_pred_rows, (
        f"Found rows with blank pred_label: {blank_pred_rows[:3]}"
    )

    metrics = _read_json(metrics_json)
    assert EXPECTED_RESOLVED_MODEL_ID in metrics, (
        f"Expected metrics entry for {EXPECTED_RESOLVED_MODEL_ID}, got keys: {list(metrics.keys())}"
    )
    assert metrics[EXPECTED_RESOLVED_MODEL_ID]["total_samples"] == expected_rows


def _assert_second_run_deduped(run_dir: Path) -> None:
    output_csv = run_dir / "output.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"

    _assert_common_artifacts_exist(run_dir)
    _assert_metadata_shape(metadata_json)

    rows = _read_csv_rows(output_csv)
    assert rows == [], (
        "Expected second run to do zero incremental work, "
        f"but found {len(rows)} output rows"
    )

    metrics = _read_json(metrics_json)
    assert metrics == {}, (
        f"Expected empty metrics on dedup-only second run, got: {metrics}"
    )


def test_openai_smoke_two_runs(tmp_path: Path) -> None:
    assert INPUT_CSV.exists(), f"Missing smoke-test dataset: {INPUT_CSV}"

    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Running first eval once...")
    first = _run_eval_once(output_root)
    _assert_run_succeeded(first)
    first_run_dir = _latest_run_dir(output_root)
    _assert_first_run_complete(first_run_dir)

    print(f"Running second eval once...")
    second = _run_eval_once(output_root)
    _assert_run_succeeded(second)
    second_run_dir = _latest_run_dir(output_root)

    # Make sure we are inspecting the new timestamped directory, not the first one again.
    assert second_run_dir != first_run_dir, (
        "Expected a distinct timestamped run directory on second run"
    )

    _assert_second_run_deduped(second_run_dir)


if __name__ == "__main__":
    timestamp = get_current_timestamp()
    test_openai_smoke_two_runs(Path(f"tmp/{timestamp}/openai_smoke_two_runs"))
