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
PYTHONPATH=. uv run evaluation/examples_test.py \       
--input-path evaluation/sample_data/csv_input_name.csv \
--output-path evaluation/sample_data/csv_output_name.csv \
--models perspective_api  
```

You will need to replace csv_input_name.csv and csv_output_name with the actual file names. 

## Where to put data

1. Make sure your csv file has columns that adhere to the `column_name_conversion` variable defined in `evaluation/dataloader.py`
2. Put the csv file in `evaluation/sample_data/csv_input_name.csv`. An output file is not required before calling the script, only when calling the script, the CLI args require an output file path.

## Models currently supported 

This is defined in the variable `MODEL_REGISTRY` in `evaluation/run_evaluation_harness.py`

## Pull Request Link
https://github.com/METResearchGroup/moral_outrage_classifier/pull/6
