"""Smoke entrypoint: evaluation harness with MiniMax alias (OpenRouter).

To run:
```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.minimax
```
"""

from evaluation.smoke_tests.test_evaluation_harness import run_harness_smoke_test

if __name__ == "__main__":
    run_harness_smoke_test("minimax")
