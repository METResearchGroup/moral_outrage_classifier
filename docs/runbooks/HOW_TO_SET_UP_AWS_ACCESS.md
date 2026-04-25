---
description: How to configure AWS CLI access for project work
tags: [aws, iam, cli, onboarding]
---

# AWS Access Setup

This runbook is for teammates who need AWS CLI access for project work.

The tech lead should give you a dedicated AWS access key ID and secret access key for this project. Do not use another teammate's personal AWS credentials.

## What To Ask For

- Your AWS access key ID.
- Your AWS secret access key.
- Confirmation that the key belongs to a dedicated project IAM user with the `met-research-permissions` policy attached.
- The default AWS region to use.
- The local AWS profile name the team wants you to use. This runbook uses `met-research-aws`.

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
aws configure --profile met-research-aws
```

Use the credentials you received:

```text
AWS Access Key ID: <AWS access key ID>
AWS Secret Access Key: <AWS secret access key>
Default region name: <AWS region from the tech lead>
Default output format: json
```

The profile name `met-research-aws` is just a local alias on your machine. It does not need to exist in AWS. The actual AWS identity comes from the access key ID and secret access key.

Confirm the profile works:

```bash
aws sts get-caller-identity --profile met-research-aws
```

That command should print the AWS account and IAM user identity. If it fails, the profile is not ready yet.

## Use The Profile

For commands that support an explicit AWS profile, pass:

```bash
--profile met-research-aws
```

For code that relies on the default AWS credential lookup chain, set the profile in your shell before running project commands:

```bash
export AWS_PROFILE=met-research-aws
export AWS_REGION=<AWS region from the tech lead>
```

## How Access Works

The `met-research-permissions` policy should grant only the AWS permissions needed for project work. The exact IAM policy is managed by the tech lead and can be changed later if the project needs access to more AWS services or resources.

Treat the AWS secret access key like a password:

- Do not commit it to the repository.
- Do not put it in `.env`.
- Do not send it in Slack, email, or issue comments.
- Store it in a password manager or another approved secret-sharing tool.

When IAM Identity Center is available, the team should migrate to SSO and delete the long-lived IAM access key.

## Troubleshooting

- `Unable to locate credentials`: Confirm you ran `aws configure --profile met-research-aws`, then retry with `--profile met-research-aws` or `AWS_PROFILE=met-research-aws`.
- `The security token included in the request is invalid`: Re-run `aws configure --profile met-research-aws` and confirm the access key ID and secret access key were copied correctly.
- `AccessDeniedException`: Ask the tech lead to confirm that your IAM user has the correct project policy attached.
- Wrong account or user: Run `aws sts get-caller-identity --profile met-research-aws` and send the output to the tech lead.
