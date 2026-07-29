"""Wrapper client for Amazon Bedrock Runtime."""

from __future__ import annotations

import os
from typing import Any

import boto3

DEFAULT_BEDROCK_REGION = "us-east-2"


class BedrockClient:
    """Wrapper class for Bedrock Runtime access."""

    def __init__(
        self,
        *,
        region_name: str = DEFAULT_BEDROCK_REGION,
        profile_name: str | None = None,
    ) -> None:
        self.region_name = region_name
        self.profile_name = (
            profile_name if profile_name is not None else os.getenv("AWS_PROFILE")
        )
        session = (
            boto3.Session(profile_name=self.profile_name)
            if self.profile_name
            else boto3.Session()
        )
        self.client = session.client("bedrock-runtime", region_name=self.region_name)

    def __getattr__(self, name: str) -> Any:
        """Delegate Bedrock Runtime API access to the boto3 client."""
        try:
            return getattr(self.client, name)
        except AttributeError:
            raise AttributeError(f"'BedrockClient' object has no attribute '{name}'")
