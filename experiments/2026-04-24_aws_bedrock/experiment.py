"""Smoke test for invoking Qwen3 32B through Amazon Bedrock.

Run from the repository root:

PYTHONPATH=. uv run python experiments/2026-04-24_aws_bedrock/experiment.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_MODEL_ID = "qwen.qwen3-32b-v1:0"
DEFAULT_REGION = "us-east-2"
DEFAULT_PROMPT = "Write a concise summary of Amazon Bedrock."


def extract_text(payload: dict[str, Any]) -> str:
    """Best-effort text extraction while still printing the raw response below."""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first_choice.get("text"), str):
                return first_choice["text"]

    for key in ("text", "output_text", "generated_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value

    output = payload.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        message = output.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                text_blocks = [
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                ]
                if text_blocks:
                    return "\n".join(text_blocks)

    return ""


def invoke_qwen(
    *,
    prompt: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    profile: str | None,
) -> dict[str, Any]:
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("bedrock-runtime", region_name=DEFAULT_REGION)

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ),
    )

    return json.loads(response["body"].read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-id", default=os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--profile",
        default=os.getenv("AWS_PROFILE"),
        help="Optional AWS profile name. Defaults to AWS_PROFILE when set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    try:
        payload = invoke_qwen(
            prompt=args.prompt,
            model_id=args.model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            profile=args.profile,
        )
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"Bedrock invocation failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    text = extract_text(payload)
    if text:
        print("\nGenerated text:")
        print(text)

    print(f"\nElapsed: {elapsed_ms:.0f} ms")
    print("\nRaw response:")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
