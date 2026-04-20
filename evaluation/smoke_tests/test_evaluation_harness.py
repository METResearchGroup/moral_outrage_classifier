"""Shared smoke test for the production evaluation harness (any model alias).

Contract:
- Run 1 must fully succeed and classify every input row.
- Run 2 must perform zero incremental work because dedup should match
  on dataset + resolved_model_id + prompt_hash (LLM) or dataset + model (Perspective).

Note:
- This intentionally uses live provider calls, so rare transient HTTP failures are possible.
  The LLM retry policy should absorb most flakiness. A one-off rerun is acceptable;
  repeated failures indicate a systematic issue.

Expected model identity (resolved IDs, prompt hash) is inferred from
``evaluation.model_registry.MODEL_REGISTRY`` and LLM facades, not hardcoded here.

To run a specific model, use the entrypoints under ``evaluation/smoke_tests/model_specific/``,
for example:

```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.openai
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.qwen
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.anthropic
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.minimax
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.perspective_api
```
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation.metadata import get_model_id_value
from evaluation.model_registry import MODEL_REGISTRY
from lib.constants import REPO_ROOT
from lib.timestamp_utils import get_current_timestamp
from models.llm.models import BaseHarnessLLMModel

INPUT_CSV_DEFAULT = Path(__file__).with_name("testing_data.csv")


def _resolve_model_identity(model_alias: str) -> dict[str, Any]:
    """Alias policy and resolved IDs come from the registry and facade classes."""
    model_cls = MODEL_REGISTRY[model_alias]
    if issubclass(model_cls, BaseHarnessLLMModel):
        return {
            "llm_provider_name": model_cls.get_requested_alias(),
            "resolved_model_id": model_cls.get_resolved_model_id(),
            "prompt_hash": model_cls.get_prompt_hash(),
            "is_llm": True,
        }
    return {
        "llm_provider_name": model_alias,
        "resolved_model_id": get_model_id_value(model_alias),
        "prompt_hash": None,
        "is_llm": False,
    }


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


def _run_eval_once(
    *,
    output_root: Path,
    input_csv: Path,
    model_alias: str,
    batch_size: int = 10,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    cmd = [
        sys.executable,
        "-m",
        "evaluation.examples_test",
        "--input-path",
        str(input_csv),
        "--output-root",
        str(output_root),
        "--models",
        model_alias,
        "--batch-size",
        str(batch_size),
    ]

    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        check=False,
    )


def _assert_run_succeeded(proc: subprocess.CompletedProcess[str]) -> None:
    assert proc.returncode == 0, "Evaluation harness subprocess failed."


def _assert_common_artifacts_exist(run_dir: Path) -> None:
    assert (run_dir / "output.csv").exists(), f"Missing output.csv in {run_dir}"
    assert (run_dir / "metrics.json").exists(), f"Missing metrics.json in {run_dir}"
    assert (run_dir / "metadata.json").exists(), f"Missing metadata.json in {run_dir}"


def _assert_metadata_shape(
    metadata_path: Path,
    *,
    input_csv: Path,
    model_alias: str,
    identity: dict[str, Any],
) -> None:
    metadata = _read_json(metadata_path)

    assert "cli_args" in metadata
    assert metadata["cli_args"]["input_path"] == str(input_csv)
    assert metadata["cli_args"]["models"] == [model_alias]

    models = metadata.get("models")
    assert isinstance(models, list), "metadata.models must be a list"
    assert len(models) == 1, (
        f"Expected exactly one model metadata entry, got {len(models)}"
    )

    model_meta = models[0]
    assert model_meta["llm_provider_name"] == identity["llm_provider_name"]
    assert model_meta["resolved_model_id"] == identity["resolved_model_id"]
    if identity["is_llm"]:
        assert isinstance(model_meta["prompt_hash"], str) and model_meta["prompt_hash"]
        assert (
            isinstance(model_meta["prompt_template"], str)
            and model_meta["prompt_template"]
        )
    else:
        assert model_meta["prompt_hash"] is None
        assert model_meta["prompt_template"] is None


def _assert_first_run_complete(
    run_dir: Path,
    *,
    input_csv: Path,
    model_alias: str,
    identity: dict[str, Any],
) -> None:
    output_csv = run_dir / "output.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"

    resolved = identity["resolved_model_id"]

    _assert_common_artifacts_exist(run_dir)
    _assert_metadata_shape(
        metadata_json,
        input_csv=input_csv,
        model_alias=model_alias,
        identity=identity,
    )

    expected_rows = _count_csv_data_rows(input_csv)
    actual_rows = _read_csv_rows(output_csv)

    assert len(actual_rows) == expected_rows, (
        f"Expected first run to classify all rows. "
        f"Expected {expected_rows}, got {len(actual_rows)}"
    )

    model_values = {row["model"] for row in actual_rows}
    assert model_values == {resolved}

    blank_pred_rows = [row for row in actual_rows if row["pred_label"] in ("", None)]
    assert not blank_pred_rows, (
        f"Found rows with blank pred_label: {blank_pred_rows[:3]}"
    )

    metrics = _read_json(metrics_json)
    assert resolved in metrics, (
        f"Expected metrics entry for {resolved}, got keys: {list(metrics.keys())}"
    )
    assert metrics[resolved]["total_samples"] == expected_rows


def _assert_second_run_deduped(
    run_dir: Path,
    *,
    input_csv: Path,
    identity: dict[str, Any],
    model_alias: str,
) -> None:
    output_csv = run_dir / "output.csv"
    metrics_json = run_dir / "metrics.json"
    metadata_json = run_dir / "metadata.json"

    _assert_common_artifacts_exist(run_dir)
    _assert_metadata_shape(
        metadata_json,
        input_csv=input_csv,
        model_alias=model_alias,
        identity=identity,
    )

    rows = _read_csv_rows(output_csv)
    assert rows == [], (
        "Expected second run to do zero incremental work, "
        f"but found {len(rows)} output rows"
    )

    metrics = _read_json(metrics_json)
    assert metrics == {}, (
        f"Expected empty metrics on dedup-only second run, got: {metrics}"
    )


def run_harness_smoke_test(
    model_alias: str,
    *,
    input_csv: Path | None = None,
    output_root: Path | None = None,
    cleanup: bool | None = None,
    batch_size: int = 10,
) -> None:
    """Run the two-phase harness smoke test for a single model alias.

    If ``output_root`` is omitted, uses
    ``evaluation/smoke_tests/outputs/<alias>/<timestamp>/`` and removes it after a
    successful run when cleanup is True (default for that case).

    ``batch_size`` is passed to ``evaluation.examples_test`` (default 10). Lower
    values reduce parallel LLM completions per batch (e.g. for rate limits).

    For pytest, pass ``output_root=tmp_path / "outputs"`` and ``cleanup=False``.
    """
    if model_alias not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model alias {model_alias!r}. Valid: {list(MODEL_REGISTRY.keys())}"
        )

    input_path = input_csv or INPUT_CSV_DEFAULT
    assert input_path.exists(), f"Missing smoke-test dataset: {input_path}"

    identity = _resolve_model_identity(model_alias)
    created_default_root = output_root is None
    if output_root is None:
        output_root = (
            REPO_ROOT
            / "evaluation"
            / "smoke_tests"
            / "outputs"
            / model_alias
            / get_current_timestamp()
        )
    if cleanup is None:
        cleanup = created_default_root

    print(f"Smoke test model alias: {model_alias}")
    print(f"Resolved model id (CSV/metrics key): {identity['resolved_model_id']}")
    print(f"Smoke test output directory (--output-root): {output_root}")

    try:
        output_root.mkdir(parents=True, exist_ok=True)

        print("Running first eval...")
        first = _run_eval_once(
            output_root=output_root,
            input_csv=input_path,
            model_alias=model_alias,
            batch_size=batch_size,
        )
        _assert_run_succeeded(first)
        first_run_dir = _latest_run_dir(output_root)
        _assert_first_run_complete(
            first_run_dir,
            input_csv=input_path,
            model_alias=model_alias,
            identity=identity,
        )
        print(
            "First run success: all input rows were classified and "
            "artifacts/metrics were validated."
        )

        print("Running second eval (dedup check)...")
        second = _run_eval_once(
            output_root=output_root,
            input_csv=input_path,
            model_alias=model_alias,
            batch_size=batch_size,
        )
        _assert_run_succeeded(second)
        second_run_dir = _latest_run_dir(output_root)

        assert second_run_dir != first_run_dir, (
            "Expected a distinct timestamped run directory on second run"
        )

        _assert_second_run_deduped(
            second_run_dir,
            input_csv=input_path,
            identity=identity,
            model_alias=model_alias,
        )
        print(
            "Second run success: deduplication worked and no additional labeling "
            "was performed."
        )
        print(f"Smoke test completed successfully for {model_alias!r}.")
    finally:
        if cleanup:
            shutil.rmtree(output_root, ignore_errors=True)
