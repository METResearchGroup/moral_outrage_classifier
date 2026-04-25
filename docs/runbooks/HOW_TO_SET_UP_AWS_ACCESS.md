---
description: How to configure AWS CLI access for running Bedrock-backed models
tags: [aws, bedrock, iam, cli, onboarding]
---

# AWS Access Setup

This runbook is for teammates who need AWS access to run the Bedrock-backed Qwen model from this repository.

The tech lead should give you a dedicated AWS access key ID and secret access key for this project. Do not use another teammate's personal AWS credentials.

## What To Ask For

- Your AWS access key ID.
- Your AWS secret access key.
- Confirmation that the key belongs to a dedicated project IAM user with the `met-research-permissions` policy attached.

This repo currently uses:

- Runtime AWS region: `us-east-2`
- Bedrock model ID: `qwen.qwen3-32b-v1:0`
- Local AWS profile name recommended by this runbook: `met-research-bedrock`

## Install The AWS CLI

You need AWS CLI v2.

On macOS, install it with Homebrew:

```bash
brew install awscli
```

Verify the install:

```bash
aws --version
```

## Configure Your AWS CLI Profile

Run:

```bash
aws configure --profile met-research-bedrock
```

Use the credentials you received:

```text
AWS Access Key ID: <AWS access key ID>
AWS Secret Access Key: <AWS secret access key>
Default region name: us-east-2
Default output format: json
```

The profile name `met-research-bedrock` is just a local alias on your machine. It does not need to exist in AWS. The actual AWS identity comes from the access key ID and secret access key.

Confirm the profile works:

```bash
aws sts get-caller-identity --profile met-research-bedrock
```

That command should print the AWS account and IAM user identity. If it fails, the profile is not ready yet.

## Run The Bedrock Smoke Tests

From the repository root, install dependencies if you have not already:

```bash
uv sync
```

Set the profile for this shell:

```bash
export AWS_PROFILE=met-research-bedrock
export AWS_REGION=us-east-2
```

Run the direct Bedrock smoke test:

```bash
PYTHONPATH=. uv run python experiments/2026-04-24_aws_bedrock/experiment.py
```

Run the LLM service Qwen example:

```bash
PYTHONPATH=. uv run python -m models.llm.smoke_tests.qwen_examples
```

Run the evaluation smoke test for the Qwen alias:

```bash
PYTHONPATH=. uv run python -m evaluation.smoke_tests.model_specific.qwen
```

## How Access Works

The `met-research-permissions` policy should grant access to invoke the project's Bedrock model without granting broad AWS permissions.

The expected Bedrock permissions are:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeQwenOnBedrock",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:us-east-2::foundation-model/qwen.qwen3-32b-v1:0"
    }
  ]
}
```

Treat the AWS secret access key like a password:

- Do not commit it to the repository.
- Do not put it in `.env`.
- Do not send it in Slack, email, or issue comments.
- Store it in a password manager or another approved secret-sharing tool.

When IAM Identity Center is available, the team should migrate to SSO and delete the long-lived IAM access key.

## Troubleshooting

- `Unable to locate credentials`: Confirm you ran `aws configure --profile met-research-bedrock`, then retry with `--profile met-research-bedrock` or `AWS_PROFILE=met-research-bedrock`.
- `The security token included in the request is invalid`: Re-run `aws configure --profile met-research-bedrock` and confirm the access key ID and secret access key were copied correctly.
- `AccessDeniedException`: Ask tech lead to confirm that your IAM user has the correct policy and that it allows `bedrock:InvokeModel` for `qwen.qwen3-32b-v1:0` in `us-east-2`.
- `ResourceNotFoundException`: Confirm `AWS_REGION=us-east-2` and the model ID is `qwen.qwen3-32b-v1:0`.
