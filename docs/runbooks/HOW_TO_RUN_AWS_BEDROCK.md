---
description: How to provision IAM access and run the Amazon Bedrock Qwen smoke test
tags: [aws, bedrock, qwen, terraform, experiments]
---

# Amazon Bedrock Qwen Runbook

This runbook verifies that this repo can invoke Qwen3 32B through Amazon Bedrock without provisioning a model endpoint.

## What This Uses

- AWS region: `us-east-2`
- Bedrock Runtime model ID: `qwen.qwen3-32b-v1:0`
- Python entry point: `experiments/2026-04-24_aws_bedrock/experiment.py`
- Terraform entry point: `terraform/main.tf`

Use `us-east-2` for AWS.

## Prerequisites

Install local dependencies:

```bash
uv sync
```

Authenticate to AWS with a principal that can create IAM policies and, if desired, attach them:

```bash
aws sts get-caller-identity
```

If you use AWS SSO:

```bash
aws sso login --profile <your-profile>
export AWS_PROFILE=<your-profile>
export AWS_REGION=us-east-2
```

## Provision IAM Permissions

The Terraform creates a least-privilege managed IAM policy that allows invoking only the Qwen3 32B Bedrock foundation model in `us-east-2`.

From the repository root:

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

That creates the policy and prints its ARN. If you want Terraform to attach the policy to an IAM role or user during apply, pass the relevant name:

```bash
terraform apply \
  -var='attach_to_iam_role_names=["<role-name>"]'
```

or:

```bash
terraform apply \
  -var='attach_to_iam_user_names=["<user-name>"]'
```

If you are using an AWS SSO permission-set role, you may need to attach the generated policy through your organization's IAM Identity Center workflow instead of attaching it directly with this Terraform.

Return to the repository root:

```bash
cd ..
```

## Run The Smoke Test

With AWS credentials available in your shell:

```bash
PYTHONPATH=. uv run python experiments/2026-04-24_aws_bedrock/experiment.py
```

To use an explicit profile:

```bash
PYTHONPATH=. uv run python experiments/2026-04-24_aws_bedrock/experiment.py \
  --profile <your-profile>
```

To customize the prompt:

```bash
PYTHONPATH=. uv run python experiments/2026-04-24_aws_bedrock/experiment.py \
  --prompt "Classify whether this sentence expresses moral outrage: How dare they let children go hungry while executives collect bonuses."
```

Expected output includes:

- The model ID and region being invoked.
- Generated text, when it can be extracted from the response.
- Elapsed wall-clock time in milliseconds.
- The raw JSON response from Bedrock.

## Troubleshooting

- `AccessDeniedException`: Confirm the active principal has `bedrock:InvokeModel` on `arn:aws:bedrock:us-east-2::foundation-model/qwen.qwen3-32b-v1:0`.
- `ResourceNotFoundException`: Confirm the region is `us-east-2` and the model ID is `qwen.qwen3-32b-v1:0`.
- `Unable to locate credentials`: Run `aws sts get-caller-identity`; if it fails, refresh your AWS credentials or run `aws sso login`.
- `ValidationException`: Check that the request body uses `messages`, `max_tokens`, and `temperature`, not Titan-style `inputText`.
- Slow first request: Bedrock is managed/serverless, but first calls can be slower than warm calls. Run the script a few times before comparing latency with OpenRouter.
