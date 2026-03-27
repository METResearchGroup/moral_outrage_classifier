# Building an evaluation harness

To make testing easier (and to be able to do it repeatedly), we'll create what's called an [evaluation harness](https://www.eleuther.ai/projects/large-language-model-evaluation)

Ideally this would be called from the command line and save the results to a .csv file:

The basic shape would look something like:

```markdown
for each row:
  for each predictor (your model, GPT prompt, Perspective):
    call predictor(text) -> pred_label or pred_score
    store row in results
```

The CLI flags would be something like:

```bash
--data data/reddit_labeled.csv
--model your_api,gpt,perspective (or default all)
--output results/reddit_run_20260327.csv
--max-rows 500 # this is optional, though recommended (any idea why this would help?)
```

The models that we'd want to test against are:

- Perspective API
- LLMs:
  - Closed-source models: GPT, Claude
  - Open-sourced models: Qwen, DeepSeek

In the saved .csv file, we'd want something like:

```plaintext
- id: some identifier for the text
- dataset: the path to the .csv file
- gold_label: 0/1
- model_name: model name
- pred_label: predicted label
- pred_score: predicted score (float, between 0 and 1). Only applicable for Perspective API, else will be None for any LLMs (those will give binary scores, not probability floats).
- is_correct: 0/1 (binary)
```

The [structured output](https://nanonets.com/cookbooks/structured-llm-outputs) contract that we want is something like:

```python
class MoralOutrage:
    label: int # 0/1
```

We want to make sure that this is enforced by all LLM callers.

Some things to consider:

- Should these all be run one-at-a-time? Or in parallel?
- What happens if one model call fails? Should that screw up everything else?
- What are the [idempotency guarantees](https://medium.com/cache-me-out/understanding-idempotency-68a50a837fc1)? For example, what happens if we label a post twice? We ideally don't want to label something multiple times (that just wastes money, for example). How do we protect against this? (hint: explore different caching principles as well as checking if a label exists before you label it). We really want to make sure we don't double-classify posts unnecessarily.

Suggested implementation:

- PR 1: Get the Perspective API to work on the provided datasets (Twitter/Reddit training sets).
- PR 2: Build a V1 evaluation harness, and run it on *just* the Perspective API.
- PR 3: Set up [OpenRouter](https://openrouter.ai/)
- PR 4: Expand the harness to work on the various ML models.

## PR 1: Get the Perspective API to work on the provided datasets

Goal: Get the Perspective API to work on the provided datasets.

Two datasets:

- Twitter: the original training set for Dr. Brady's moral outrage classifier
- Reddit: a supplemental dataset.

Suggested implementation:

- `models/perspective_api/model.py`: exposes a `class PerspectiveApiModel` with a method `def batch_classify(texts: list[str])`
- `models/perspective_api/examples_test.py`: example cases
- `evaluate.py`: the CLI application that will run all the testing.
- `data/twitter.csv` and `data/reddit.csv`: the input files.
- `results/`: where to put the experimental results.

End result: something like this:

```bash
python evaluate.py --model perspective_api --data data/twitter.csv  --output results/
```

I will provide the API key needed to access the Perspective API.
