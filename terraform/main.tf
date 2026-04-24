terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  description = "AWS region for Bedrock runtime calls."
  type        = string
  default     = "us-east-2"
}

variable "bedrock_model_id" {
  description = "Bedrock foundation model ID to allow invoking."
  type        = string
  default     = "qwen.qwen3-32b-v1:0"
}

variable "attach_to_iam_user_names" {
  description = "Optional IAM user names that should receive the Bedrock invoke policy."
  type        = set(string)
  default     = []
}

variable "attach_to_iam_role_names" {
  description = "Optional IAM role names that should receive the Bedrock invoke policy."
  type        = set(string)
  default     = []
}

provider "aws" {
  region = var.aws_region
}

data "aws_partition" "current" {}

locals {
  qwen_model_arn = "arn:${data.aws_partition.current.partition}:bedrock:${var.aws_region}::foundation-model/${var.bedrock_model_id}"
}

data "aws_iam_policy_document" "bedrock_qwen_invoke" {
  statement {
    sid    = "InvokeQwenOnBedrock"
    effect = "Allow"

    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]

    resources = [local.qwen_model_arn]
  }
}

resource "aws_iam_policy" "bedrock_qwen_invoke" {
  name        = "BedrockQwenInvoke"
  description = "Allows invoking ${var.bedrock_model_id} through Amazon Bedrock Runtime."
  policy      = data.aws_iam_policy_document.bedrock_qwen_invoke.json
}

resource "aws_iam_user_policy_attachment" "bedrock_qwen_invoke" {
  for_each = var.attach_to_iam_user_names

  user       = each.value
  policy_arn = aws_iam_policy.bedrock_qwen_invoke.arn
}

resource "aws_iam_role_policy_attachment" "bedrock_qwen_invoke" {
  for_each = var.attach_to_iam_role_names

  role       = each.value
  policy_arn = aws_iam_policy.bedrock_qwen_invoke.arn
}

output "bedrock_qwen_model_arn" {
  description = "ARN of the Bedrock Qwen foundation model allowed by this policy."
  value       = local.qwen_model_arn
}

output "bedrock_qwen_policy_arn" {
  description = "Managed IAM policy ARN for invoking the Bedrock Qwen model."
  value       = aws_iam_policy.bedrock_qwen_invoke.arn
}
