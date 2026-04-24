"""Smoke entrypoint: evaluation harness with Perspective API.

To run:
```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.perspective_api
```
"""

from evaluation.smoke_tests.test_evaluation_harness import run_harness_smoke_test

if __name__ == "__main__":
    run_harness_smoke_test("perspective_api")
