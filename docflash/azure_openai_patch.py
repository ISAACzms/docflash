"""
Azure OpenAI Language Model monkey patch for LangExtract.
This adds AzureOpenAILanguageModel to the langextract.inference module without modifying the core package.
"""

import concurrent.futures
import dataclasses
import json
from collections.abc import Iterator, Sequence
from typing import Any

import yaml
from langextract import data, schema

# Import necessary classes from langextract
from langextract.inference import BaseLanguageModel, InferenceOutputError, ScoredOutput


@dataclasses.dataclass
class AzureOpenAILanguageModel(BaseLanguageModel):
    """Language model inference using Azure OpenAI's API with structured output."""

    model_id: str = "gpt-4o-mini"
    deployment_name: str | None = None
    api_key: str | None = None
    azure_endpoint: str | None = None
    api_version: str = "2024-02-01"
    organization: str | None = None
    azure_ad_token_provider: Any | None = None
    format_type: data.FormatType = data.FormatType.JSON
    temperature: float = 0.0
    max_workers: int = 10
    _client: Any | None = dataclasses.field(default=None, repr=False, compare=False)
    _extra_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        deployment_name: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-01",
        organization: str | None = None,
        azure_ad_token_provider: Any | None = None,
        format_type: data.FormatType = data.FormatType.JSON,
        temperature: float = 0.0,
        max_workers: int = 10,
        **kwargs,
    ) -> None:
        """Initialize the Azure OpenAI language model."""
        self.model_id = model_id
        self.deployment_name = deployment_name or model_id
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.organization = organization
        self.azure_ad_token_provider = azure_ad_token_provider
        self.format_type = format_type
        self.temperature = temperature
        self.max_workers = max_workers
        self._extra_kwargs = kwargs or {}

        if not self.azure_endpoint:
            raise ValueError("Azure endpoint not provided.")

        # Initialize the Azure OpenAI client
        try:
            from openai import AzureOpenAI

            if self.azure_ad_token_provider:
                # Use Azure AD token provider
                self._client = AzureOpenAI(
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token_provider=self.azure_ad_token_provider,
                    api_version=self.api_version,
                    organization=self.organization,
                )
            elif self.api_key:
                # Use API key
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.api_version,
                    organization=self.organization,
                )
            else:
                raise ValueError(
                    "Either api_key or azure_ad_token_provider must be provided."
                )

        except ImportError as e:
            raise ImportError(
                "Azure OpenAI library not found. Install with: pip install openai"
            ) from e

        super().__init__(
            constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
        )

    def _process_single_prompt(self, prompt: str, config: dict) -> ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            # Prepare the system message for structured output
            system_message = ""
            if self.format_type == data.FormatType.JSON:
                system_message = (
                    "You are a helpful assistant that responds in JSON format."
                )
            elif self.format_type == data.FormatType.YAML:
                system_message = (
                    "You are a helpful assistant that responds in YAML format."
                )

            # Create the chat completion using Azure OpenAI
            response = self._client.chat.completions.create(
                model=self.deployment_name,  # Use deployment name for Azure
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.get("temperature", self.temperature),
                max_tokens=config.get("max_output_tokens"),
                top_p=config.get("top_p"),
                n=1,
            )

            # Extract the response text
            output_text = response.choices[0].message.content
            return ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise InferenceOutputError(f"Azure OpenAI API error: {str(e)}") from e

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Runs inference on a list of prompts via Azure OpenAI's API."""
        config = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if "max_output_tokens" in kwargs:
            config["max_output_tokens"] = kwargs["max_output_tokens"]
        if "top_p" in kwargs:
            config["top_p"] = kwargs["top_p"]

        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(
                        self._process_single_prompt, prompt, config.copy()
                    ): i
                    for i, prompt in enumerate(batch_prompts)
                }
                results: list[ScoredOutput | None] = [None] * len(batch_prompts)
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise InferenceOutputError(
                            f"Parallel inference error: {str(e)}"
                        ) from e

                for result in results:
                    if result is None:
                        raise InferenceOutputError(
                            "Failed to process one or more prompts"
                        )
                    yield [result]
        else:
            # Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt, config.copy())
                yield [result]

    def parse_output(self, output: str) -> Any:
        """Parses Azure OpenAI output as JSON or YAML."""
        try:
            if self.format_type == data.FormatType.JSON:
                return json.loads(output)
            else:
                return yaml.safe_load(output)
        except Exception as e:
            raise ValueError(
                f"Failed to parse output as {self.format_type.name}: {str(e)}"
            ) from e


def apply_azure_openai_patch():
    """Apply the AzureOpenAI monkey patch to langextract.inference module."""
    import langextract.inference

    # Add the AzureOpenAILanguageModel class to the langextract.inference module
    langextract.inference.AzureOpenAILanguageModel = AzureOpenAILanguageModel

    print("✅ Applied AzureOpenAI monkey patch to langextract.inference")