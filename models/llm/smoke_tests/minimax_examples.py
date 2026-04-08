"""OpenRouter (MiniMax) smoke-test examples built on shared helpers.

To run:

PYTHONPATH=. uv run python -m models.llm.smoke_tests.minimax_examples
"""

import logging

from models.llm.smoke_tests.base_examples import run_batch_example_query

EXAMPLE_TEXTS = [
    "How dare they let children go hungry while executives collect bonuses.",
    "I'm so glad I'm not a parent. It's so hard to raise kids these days.",
    "Why is it always the poor who have to pay for the mistakes of the rich?",
    "It's not fair that the rich get richer and the poor get poorer.",
    "The rich are getting richer and the poor are getting poorer.",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print("Batch example:")
    batch_results = run_batch_example_query(EXAMPLE_TEXTS, model="minimax/minimax-m2.5")
    for text, result in zip(EXAMPLE_TEXTS, batch_results, strict=True):
        print(f"Text: {text}\tLabel: {result.label}")
        print("-" * 100)


if __name__ == "__main__":
    main()
