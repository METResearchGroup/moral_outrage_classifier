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
--text-col text --label-col label --id-col id
--predictors your_api,gpt,perspective (or default all)
--out results/reddit_run_20260327.csv
--max-rows 500
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

Suggested implementation:

- PR 1: Get the Perspective API to work.
- PR 2: Build the V1 evaluation harness, and run it on *just* the Perspective API.
- PR 3: Set up [OpenRouter](https://openrouter.ai/)