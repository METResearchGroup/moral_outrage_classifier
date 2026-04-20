import json
import subprocess
import time
from pathlib import Path

import typer

from evaluation.run_evaluation_harness import (
    EvaluationHarness,
    build_models_metadata_entries,
)
from lib.timestamp_utils import get_current_timestamp


def get_git_hash():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def get_run_metadata(
    input_path: str,
    output_root: str,
    models: list[str],
    max_rows: int | float,
    batch_size: int,
    runtime_seconds: float,
    timestamp: str,
    models_metadata: list[dict],
) -> dict:
    git_hash = get_git_hash()
    return {
        "git_commit_hash": git_hash,
        "timestamp": timestamp,
        "cli_args": {
            "input_path": input_path,
            "output_root": output_root,
            "models": models,
            "max_rows": max_rows if max_rows != float("inf") else None,
            "batch_size": batch_size,
        },
        "models": models_metadata,
        "runtime_seconds": round(runtime_seconds, 4),
    }


def write_metadata_file(
    input_path: str,
    output_root: str,
    models: list[str],
    max_rows: int | float,
    batch_size: int,
    elapsed: float,
    timestamp: str,
) -> None:
    models_metadata = build_models_metadata_entries(models)
    metadata = get_run_metadata(
        input_path,
        output_root,
        models,
        max_rows,
        batch_size,
        elapsed,
        timestamp,
        models_metadata,
    )
    metadata_dir = Path(output_root) / timestamp
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main(
    input_path: str = typer.Option(..., help="Path to the input CSV file"),
    output_root: str = typer.Option(
        ...,
        "--output-root",
        "--output-path",
        help="Directory under which timestamped run folders (each with output.csv) are created",
    ),
    models: list[str] = typer.Option(
        ...,
        help="Model aliases to evaluate (e.g. perspective_api, openai, qwen, anthropic, minimax)",
    ),
    max_rows: int = typer.Option(-1, help="Maximum number of rows to process"),
    batch_size: int = typer.Option(10, help="Size of each batch for processing"),
):
    if max_rows == -1:
        max_rows = float("inf")

    EvaluationHarness.validate_models(models)

    timestamp = get_current_timestamp()
    eh = EvaluationHarness(
        input_path=input_path,
        output_path=output_root,
        batch_size=batch_size,
        models=models,
        timestamp=timestamp,
        max_rows=max_rows,
    )

    print("LOADING DATA")
    eh.load_data()

    print("DONE LOADING DATA, NOW RUNNING EVALUATION")
    start = time.perf_counter()
    eh.run_evaluation()
    elapsed = time.perf_counter() - start

    print("DONE RUNNING EVALUATION, NOW DISPLAYING RESULTS")
    eh.display_results()

    write_metadata_file(input_path, output_root, models, max_rows, batch_size, elapsed, timestamp)

if __name__ == "__main__":
    typer.run(main)
