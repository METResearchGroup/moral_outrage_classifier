"""Smoke entrypoint: evaluation harness with Anthropic alias (direct Anthropic API).

To run:
```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.anthropic
```
"""

from evaluation.smoke_tests.test_evaluation_harness import run_harness_smoke_test

if __name__ == "__main__":
    # Default harness batch is 10; 2 reduces parallel Anthropic TPM/RPM load.
    run_harness_smoke_test("anthropic", batch_size=2)
