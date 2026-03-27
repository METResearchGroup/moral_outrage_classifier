# Deploying the REST API

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

## Why Modal over AWS?

AWS is great, and my initial plan had considered using an AWS stack (specifically Lambda + SageMaker). Upon deeper planning, I noticed that it's not best for what we're building here. We want to use the best tools for the current project. There's a lot of tradeoffs made when choosing tools, and we want to choose a stack that'll help us best accomplish our stated goal (creating an API to serve an ML model) without losing lots of time to other problems (e.g., VPC plumbing on AWS).

AWS is the right default for some orgs, hence why it's often a common thing in job descriptions. For this project the bottleneck is ML validation and API design, not random things like multi-account VPC design. Modal so will let us hit real deploy/auth/GPU/cost tradeoffs in the time we have. AWS-first for this project will mean teaching CloudFormation-by-accident instead of working on ML engineering.
