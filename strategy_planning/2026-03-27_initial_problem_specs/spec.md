# Moral outrage classifier development

## Problem statement

(TODO: Mark)

## Project components (high-level)

We plan on this project to have the following high-level phases

1. Do some basic exploration of the dataset and clean up anything as needed.
2. Train/fine-tune a few model variations.
3. Validate the model (test it against other similar APIs like the Perspective API as well as [Dr. Brady's original paper](https://osf.io/preprints/psyarxiv/gf7t5_v1) and [follow-up paper](https://osf.io/preprints/osf/k5dzr_v1)).
4. Deploy it as a REST API.

Steps 1-3 are enough for completion of the research component of the project. Step 4 is necessary for shipping a completed product like an ML engineer would.

This 4-step plan is basically what shipping a full end-to-end ML project in industry looks like. You can add layers of complexity here and there, but being an ML engineer, in large part, consists of work related to some combination of these steps.

### Part 1: Basic data exploration

A key component of any ML project is initial exploration. Some questions that you'll want to explore include:

1. Is the data all English?
2. How old is the data? Do we think that a model trained on the dataset would be representative for the task? Why or why not?
3. Do we have an even split of class labels? We ideally want something where close to 50% of the training data has moral outrage, and the rest do not. Think about what would happen if this weren't true, and what we could do to resolve it if it weren't true.

...

(TODO: Mark)

### Part 2: Train/fine-tune a few model variations

#### 2.1. Do initial experiments

Dr. Brady's original paper tried Naive Bayes and RNNs and LSTMs, if I recall correctly, and I think it would be good to test against those same baselines and then add a BERT model and then some open-source LLM (e.g., Qwen, Llama).

##### Phase 1: Non-transformer models

(TODO: Mark)

##### Phase 2: BERT model

(TODO: Mark)

##### Phase 3: Open-sourced LLM

(TODO: Mark)

#### 2.2. Train models like a proper ML engineer

Once you've figured out a basic family of models that work, let's train it systematically to perfect the results.

Tools to learn:

- Weights and Biases: for viewing model training curves
- Optuna: for hyperparameter tuning
- (Optional) MLFlow: for saving model training parameters + artifacts (may be overkill, we will revisit).

Here, we'll use hyperparameter tuning and review the training curves across model runs. Our goal is to maximize the model performance as much as possible.

### Part 3: Validate the model

Once we have a working model, we'll want to validate the results against other similar ML models. We'll want to compare it against the following baselines:

1. A basic LLM prompt (e.g., asking ChatGPT "does this post have outrage?")
2. The Perspective API (Google's in-house API endpoint for this task).

We'll also want to test against a few other datasets (which will be provided).

1. Reddit text.
2. Twitter text.

#### Part 3.1: Build a simple evaluation harness to make this efficient

To make testing easier (and to be able to do it repeatedly), we'll create what's called an [evaluation harness](https://www.eleuther.ai/projects/large-language-model-evaluation)

See [Building an evaluation harness](./eval_harness_spec.md) for more.

### Part 4: Deploy as a REST API

Deploy model in [Modal](https://modal.com/). Host model weights in [HuggingFace](https://huggingface.co/).

See [Deploying the REST API](./rest_api_spec.md) for more.

## Implementation order

We actually want to start with implementing "Part 3.1: Build a simple evaluation harness to make this efficient" first, as it'll give us a baseline for what already exists. Before we train any models, let's see how existing models already do.

Once we've done that, we can implement each step in order.
