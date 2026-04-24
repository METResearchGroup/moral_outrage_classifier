---
description: How to run the evaluation harness when testing and comparing the performance of moral outrage classifier models
tags: [evals, dataloaders, development, python]
---


# Local Development Runbook

This runbook covers setup and running the evaluation harness locally.

## Setup

- Python ≥3.12. Install dependencies with the command:

  ```bash
  uv sync --extra examples
  ```

## Running the Eval Script

From the repository root (with `uv sync` already run):

```bash
PYTHONPATH=. uv run python -m evaluation.examples_test \
  --input-path evaluation/sample_data/csv_input_name.csv \
  --output-root evaluation/outputs \
  --models perspective_api
```

You can pass multiple family aliases, for example `--models openai --models perspective_api`. Replace the input CSV path with your file; `--output-root` is the directory where timestamped run folders are created (each run contains `output.csv`, `metrics.json`, and `metadata.json`).

## Where to put data

1. Make sure your csv file has columns that adhere to the `column_name_conversion` variable defined in `evaluation/dataloader.py`
2. Put the csv file under `evaluation/sample_data/` (or pass any path). You do not need a pre-existing run directory; `--output-root` must point to a parent directory where the script will create a timestamped folder for each run.

## Models currently supported

Aliases are defined in `MODEL_REGISTRY` in `evaluation/model_registry.py`: `perspective_api`, `openai`, `qwen`, `anthropic`, and `minimax`. The merged `output.csv` records the **resolved** model id for LLM runs (for example `gpt-5.4` for the `openai` alias), not the CLI alias. Run `metadata.json` lists both `llm_provider_name` and `resolved_model_id` per model, plus prompt fields for LLM facades.

## Pull Request Link

[Here is the original link](https://github.com/METResearchGroup/moral_outrage_classifier/pull/6) to the PR that implemented the evaluation harness.
