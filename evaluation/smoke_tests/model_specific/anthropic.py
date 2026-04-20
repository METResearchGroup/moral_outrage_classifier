"""Smoke entrypoint: evaluation harness with Anthropic alias (OpenRouter).

To run:
```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.anthropic
```
"""

from evaluation.smoke_tests.test_evaluation_harness import run_harness_smoke_test

if __name__ == "__main__":
    run_harness_smoke_test("anthropic")
