# Moral outrage classifier development

## Problem statement

## Project components (high-level)

We plan on this project to have the following high-level phases

1. Do some basic exploration of the dataset and clean up anything as needed.
2. Train/fine-tune a few model variations.
3. Validate the model (test it against other similar APIs like the Perspective API as well as [Dr. Brady's original paper](https://osf.io/preprints/psyarxiv/gf7t5_v1) and [follow-up paper](https://osf.io/preprints/osf/k5dzr_v1)).
4. Deploy it as a REST API on AWS.

### Part 1: Basic data exploration

A key component of any ML project is initial exploration. Some questions that you'll want to explore include:

1. Is the data all English?
2. How old is the data? Do we think that a model trained on the dataset would be representative for the task? Why or why not?
3. ...

### Part 2: Train/fine-tune a few model variations

#### 2.1. Do initial experiments

#### 2.2. Train models like a proper ML engineer

Tools to learn:

- Weights and Biases: for viewing model training curves
- Optuna: for hyperparameter tuning
- (Optional) MLFlow: for saving model training parameters + artifacts (may be overkill, we will revisit).

### Part 3: Validate the model

### Part 4: Deploy as a REST API

Deploy model in [Modal](https://modal.com/). Host model weights in [HuggingFace](https://huggingface.co/).

The following sketch assumes that we'll end up with a fine-tuned 7B model (which is what I'm expecting). Conceptually, the code would look something like:

```python
import os
import modal

APP_NAME = "moral-outrage-llm"
MODEL_ID = "your-org/your-7b-finetune"  # or any HF hub id

# CUDA stack + libs — trim/extend for your actual runtime (vLLM vs transformers, etc.)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn",
        "huggingface_hub",
        "transformers",
        "torch",
        "accelerate",
        "safetensors",
    )
    # If you need a specific CUDA wheel line, follow Modal’s GPU image examples instead of debian_slim.
)

app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="A10G",  # pick something that fits your 7B + dtype + context
    secrets=[modal.Secret.from_name("moral-outrage-llm-secrets")],
    timeout=60 * 10,
    scaledown_window=60 * 5,  # idle behavior — tune for cost vs cold start
)
class Server:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # HF_TOKEN env is injected if you put hf_token=... in Modal secrets
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field

        web = FastAPI()

        class InferRequest(BaseModel):
            prompt: str
            max_new_tokens: int = Field(default=128, le=512)

        def _auth(authorization: str | None) -> None:
            # Secret should set API_BEARER_TOKEN in the Modal dashboard / CLI
            expected = os.environ.get("API_BEARER_TOKEN")
            if not expected:
                raise HTTPException(500, "Server misconfigured")
            if authorization is None or not authorization.startswith("Bearer "):
                raise HTTPException(401, "Missing bearer token")
            token = authorization.removeprefix("Bearer ").strip()
            if token != expected:
                raise HTTPException(403, "Invalid token")

        @web.get("/health")
        def health():
            return {"ok": True, "model": MODEL_ID}

        @web.post("/v1/complete")
        def complete(
            body: InferRequest,
            authorization: str | None = Header(default=None),
        ):
            _auth(authorization)

            import torch

            inputs = self.tokenizer(body.prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=body.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)

            return {"text": text}

        return web
```

cURL request shape would look something like:

```bash
curl -sS -X POST 'https://YOUR_MODAL_URL/v1/complete' \
  -H 'Authorization: Bearer YOUR_LONG_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Classify the moral outrage in: ...","max_new_tokens":128}'
```

We choose to do it this way as this is the lightweight, easy-to-ship, way to deploy an ML application as a REST API.
