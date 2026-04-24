# Planning a generic evaluation harness

The current harness is a problem-specific moral-outrage labeling and evaluation flow.

- V1 should expose that existing harness through a simple Streamlit UI. Let's get this out and see if people have a use for it.
- V2 can generalize the system into a prompt-driven batch text-classification tool with user-defined structured outputs.

## Happy Flow

1. User uploads a CSV file through the Streamlit app. Required columns are `id` and `text`; `gold_label` is optional.
2. The app persists the uploaded dataset so runs can be associated with a specific uploaded file.
3. The user selects one previously uploaded dataset from a dropdown.
4. The user selects one model to run. In V1, users can only choose a single model at a time.
5. The app starts a synchronous run in Streamlit by calling the existing moral-outrage evaluation harness.
6. The user sees run progress in the UI, including a progress bar that mirrors the current `tqdm` progress as closely as possible.
7. When the run finishes, the user sees a preview of the labeled output, the number of failed/deadlettered rows, and evaluation metrics if `gold_label` was present in the uploaded CSV.
8. The user exports the resulting CSV. If some rows failed, the user manually reruns labeling as a new session.

## V1 Product Requirements

### Scope

- V1 is a Streamlit frontend over the existing moral-outrage evaluator.
- V1 is synchronous inside the Streamlit app.
- V1 keeps the current harness behavior.
- V1 keeps the existing prompt, structured output, and output schema used by the current harness.
- V1 treats refreshes or browser closes naively. If someone closes the app, the session refreshes and someone would have to restart from scratch.

### Dataset contract

- Required input columns: `id`, `text`
- Optional input column: `gold_label`
- If `gold_label` is present, the app should show evaluation metrics from the current harness.
- If `gold_label` is absent, the app should still support labeling and CSV export.

### Runtime and scale assumptions

- We should optimize V1 around runs of fewer than 500 samples (some of the OpenRouter endpoints currently run slowly, which we may have to address at some point).
- Let's have a typical V1 run complete in under 5 minutes. We'll revisit this once we see how long the OpenRouter models run on larger scales and we can figure out a few options for it.
- We should not design V1 around long-running background jobs, resumability, etc.

### Failure handling

- Partial failure is acceptable in V1.
- If some rows fail and land in deadletter output, that is acceptable V1 behavior.
- The UI should clearly report how many rows failed labeling.
- Users will manually rerun the dataset themselves if they want to recover failed rows, and we will treat that as a new session.

### User-facing run states

Here are some of the statuses that we'll want to show in the Streamlit UI.

- Ready
- Uploading dataset
- Running labeling
- Completed
- Completed with failed rows
- Failed before completion

## V2 Direction

V2 is the actual generic batch text-classification tool. That version can allow users to supply:

- their own prompt
- their own structured output schema
- potentially multiple models per run

That generalization should happen only after the V1 Streamlit wrapper proves useful. Let's get a V1 out and shipped and put it in front of the lab to see if they're interested in something more generic.

## Architecture Direction

### Frontend

Use a simple single-page Streamlit app first. This keeps the first iteration Python-only and reduces the amount of product infrastructure we need before validating demand.

### Backend

Do not require a separate backend for the first cut of V1. The first implementation can call the existing harness directly from Streamlit.

After that, we can introduce a FastAPI backend so the UI and execution layer are cleanly separated.

### Deployment

Deployment will look something like this:

- First, get the Streamlit-only version working locally.
- Then, introduce a FastAPI backend.
- Then, get a local Streamlit + FastAPI version working end to end.
- After local integration is working, deploy:
  - FastAPI backend via Docker on Railway
  - Streamlit frontend via Streamlit Community Cloud

## Alternative Approaches

- We could try to make the system generic immediately, but that would force us to redesign the prompt contract, output schema handling, run metadata, and evaluation semantics all at once.
- We could also start with FastAPI first, but that adds infrastructure before we have validated whether the Streamlit wrapper is sufficient.
- Let's ship the narrowest usable wrapper first, then generalize only after we learn from real usage.

## Manual Verification

- Run the Streamlit app locally and confirm a user can upload a CSV with `id` and `text`.
- Confirm a CSV with `id`, `text`, and `gold_label` produces both labeled output and evaluation metrics.
- Confirm a CSV with only `id` and `text` still produces labeled output without metrics.
- Confirm the app only allows one model selection in V1.
- Confirm the UI exposes progress during execution.
- Confirm the final screen shows:
  - output preview
  - export affordance
  - failed-row count
  - evaluation metrics when `gold_label` exists
- Confirm a refresh or browser close resets the session and does not attempt recovery.

## Scoping

### V1

V1 is a synchronous Streamlit labeling app over the existing moral-outrage evaluation harness. It supports one model per run, optional evaluation when `gold_label` is present, progress visibility during execution, CSV export, and basic failed-row reporting.

### V2

V2 is a generic batch text-classification tool where users can supply their own prompt and structured schema.

## Suggested PR sequence

### PR 1: Add Streamlit wrapper for existing harness

- Deliverables:
  - basic Streamlit app for dataset upload, dataset selection, single-model selection, run trigger, results preview, and CSV export
  - direct integration with the existing moral-outrage harness
  - progress UI and failed-row count in the app
- Success looks like:
  - a user can run the current harness from a local Streamlit UI without using the CLI
  - runs work for datasets under 500 rows
  - labeling works with optional evaluation when `gold_label` is present

### PR 2: Add FastAPI backend around harness execution

- Deliverables:
  - FastAPI endpoints for dataset selection, run execution, and results retrieval
  - backend-side wrapping of current harness behavior without introducing V2 generalization
  - initial contract between UI and backend for single-model runs
- Success looks like:
  - the harness can be invoked through HTTP locally.
  - the backend preserves the same V1 semantics as the Streamlit-only version.

### PR 3: Get local Streamlit + FastAPI working end to end

- Deliverables:
  - Streamlit app updated to call the FastAPI backend rather than invoking the harness directly
  - local developer workflow for running both services together
  - successful end-to-end local run for labeling and optional evaluation
- Success looks like:
  - a full local flow works from upload through export using Streamlit + FastAPI
  - the UI still shows status/progress and failed-row counts clearly

### PR 4: Deploy FastAPI backend with Docker on Railway

- Deliverables:
  - Dockerized FastAPI service
  - Railway deployment configuration
  - deployed backend environment suitable for the Streamlit app
- Success looks like:
  - the FastAPI backend is deployed and reachable in Railway
  - the deployed service matches local behavior for the V1 flow

### PR 5: Deploy Streamlit app to Streamlit Community Cloud

- Deliverables:
  - Streamlit Community Cloud deployment
  - configuration pointing the app at the deployed Railway backend
  - smoke-tested hosted user flow
- Success looks like:
  - an end user can complete the V1 flow in the hosted app
  - the hosted UI works against the deployed FastAPI backend for small runs
