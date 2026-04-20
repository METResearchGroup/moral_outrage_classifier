"""Smoke entrypoint: evaluation harness with OpenAI alias (resolved: gpt-5.4).

To run:
```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.openai
```
"""

from evaluation.smoke_tests.test_evaluation_harness import run_harness_smoke_test

if __name__ == "__main__":
    run_harness_smoke_test("openai")
