from evaluation.run_evaluation_harness import EvaluationHarness
from lib.timestamp_utils import get_current_timestamp
from pathlib import Path

import json
import subprocess
import typer
import time

def get_git_hash():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()

def get_run_metadata(input_path, output_path, models, max_rows, batch_size, runtime_seconds, timestamp):
    git_hash = get_git_hash()
    return {
        "git_commit_hash": git_hash,
        "timestamp": timestamp,
        "cli_args": {
            "input_path": input_path,
            "output_path": output_path,
            "models": models,
            "max_rows": max_rows if max_rows != float('inf') else None,
            "batch_size": batch_size,
        },
        "runtime_seconds": round(runtime_seconds, 4),
    }

def write_metadata_dir(input_path, output_path, models, max_rows, batch_size, elapsed):
    timestamp = get_current_timestamp()
    metadata = get_run_metadata(input_path, output_path, models, max_rows, batch_size, elapsed, timestamp)

    metadata_dir = Path(output_path).parent / timestamp

    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main(
    input_path: str = typer.Option(..., help="Path to the input CSV file"),
    output_path: str = typer.Option(..., help="Path to the output CSV file"),
    models: list[str] = typer.Option(..., help="List of models to evaluate"),
    max_rows: int = typer.Option(-1, help="Maximum number of rows to process"),
    batch_size: int = typer.Option(10, help="Size of each batch for processing")
):
    
    if max_rows == -1: max_rows = float('inf')

    EvaluationHarness.validate_models(models)
    eh = EvaluationHarness(
        input_path=input_path,
        output_path=output_path,
        batch_size=batch_size,
        models=models,
        max_rows=max_rows
    )

    print("LOADING DATA")
    eh.load_data()

    print("DONE LOADING DATA, NOW RUNNING EVALUATION")
    start = time.perf_counter()
    eh.run_evaluation()
    elapsed = time.perf_counter() - start

    print("DONE RUNNING EVALUATION, NOW DISPLAYING RESULTS")
    eh.display_results()

    write_metadata_dir(input_path, output_path, models, max_rows, batch_size, elapsed)


if __name__ == "__main__":
    typer.run(main)
