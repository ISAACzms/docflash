"""
LLM Provider Factory for Doc Flash
Supports Azure OpenAI, OpenAI, and Google Gemini
"""

import os
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Apply Azure OpenAI monkey patch before importing
from .azure_openai_patch import apply_azure_openai_patch

apply_azure_openai_patch()

from langextract.inference import (
    AzureOpenAILanguageModel,  # Our custom class from monkey patch
    BaseLanguageModel,
)
from langextract.providers.gemini import GeminiLanguageModel
from langextract.providers.openai import OpenAILanguageModel

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import google.generativeai as genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None

from .config import LLMConfig


class ModelFactory:
    """Factory class to create language models based on provider configuration"""

    @staticmethod
    def create_model(config: LLMConfig) -> Any:
        """Create a language model instance based on configuration"""

        if config.provider == "azure_openai":
            return ModelFactory._create_azure_openai_model(config)
        elif config.provider == "openai":
            return ModelFactory._create_openai_model(config)
        elif config.provider == "gemini":
            return ModelFactory._create_gemini_model(config)
        elif config.provider == "ollama":
            return ModelFactory._create_ollama_model(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    @staticmethod
    def _create_azure_openai_model(config: LLMConfig) -> AzureOpenAILanguageModel:
        """Create Azure OpenAI model instance"""

        if not config.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        # Check if API key is explicitly provided
        import os

        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if api_key:
            # Use API key authentication if explicitly provided
            return AzureOpenAILanguageModel(
                model_id=config.model_id,
                deployment_name=config.azure_deployment_name,
                api_key=api_key,
                azure_endpoint=config.azure_endpoint,
                api_version=config.azure_api_version,
                temperature=config.temperature,
            )
        else:
            # Use managed identity authentication (default)
            credential = DefaultAzureCredential(
                exclude_cli_credential=False,
                exclude_managed_identity_credential=False,
                exclude_environment_credential=True,
                exclude_powershell_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_workload_identity_credential=False,
                exclude_developer_cli_credential=True,
                exclude_visual_studio_code_credential=True,
            )

            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )

            return AzureOpenAILanguageModel(
                model_id=config.model_id,
                deployment_name=config.azure_deployment_name,
                azure_endpoint=config.azure_endpoint,
                api_version=config.azure_api_version,
                azure_ad_token_provider=token_provider,
                temperature=config.temperature,
            )

    @staticmethod
    def _create_openai_model(config: LLMConfig):
        """Create OpenAI model instance"""

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Create OpenAI client
        client_kwargs = {"api_key": config.openai_api_key}

        if config.openai_base_url:
            client_kwargs["base_url"] = config.openai_base_url

        client = OpenAI(**client_kwargs)

        # Wrap in a compatible interface
        return OpenAIWrapper(
            client=client, model_id=config.model_id, temperature=config.temperature
        )

    @staticmethod
    def _create_gemini_model(config: LLMConfig) -> GeminiLanguageModel:
        """Create Google Gemini model instance"""

        if not config.google_api_key:
            raise ValueError("Google API key is required")

        return GeminiLanguageModel(
            model_id=config.model_id,
            api_key=config.google_api_key,
            temperature=config.temperature,
        )

    @staticmethod
    def _create_ollama_model(config: LLMConfig):
        """Create Ollama model instance using LangExtract's native support"""
        
        # For Ollama, we'll create a wrapper that uses LangExtract's direct extraction
        # since LangExtract has native Ollama support
        return OllamaWrapper(
            model_id=config.model_id,
            base_url=config.ollama_base_url or "http://localhost:11434",
            temperature=config.temperature,
        )


class OpenAIWrapper:
    """Wrapper to make OpenAI compatible with LangExtract interface"""

    def __init__(self, client: OpenAI, model_id: str, temperature: float):
        self.client = client
        self.model_id = model_id
        self.temperature = temperature
        self.deployment_name = model_id  # For compatibility

    def infer(self, prompts: list[str]):
        """Generate responses for a batch of prompts"""

        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )

                # Yield in LangExtract compatible format
                class MockResponse:
                    def __init__(self, output: str):
                        self.output = output

                yield [MockResponse(response.choices[0].message.content)]

            except Exception as e:
                print(f"OpenAI API error: {e}")
                raise


class GeminiWrapper:
    """Wrapper to make Gemini compatible with LangExtract interface"""

    def __init__(self, model, model_id: str, temperature: float):
        self.model = model
        self.model_id = model_id
        self.temperature = temperature
        self.deployment_name = model_id  # For compatibility

    def infer(self, prompts: list[str]):
        """Generate responses for a batch of prompts"""

        for prompt in prompts:
            try:
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature
                )

                response = self.model.generate_content(
                    prompt, generation_config=generation_config
                )

                # Yield in LangExtract compatible format
                class MockResponse:
                    def __init__(self, output: str):
                        self.output = output

                yield [MockResponse(response.text)]

            except Exception as e:
                print(f"Gemini API error: {e}")
                raise


class OllamaWrapper:
    """Wrapper for Ollama models using requests"""

    def __init__(self, model_id: str, base_url: str, temperature: float):
        self.model_id = model_id
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.deployment_name = model_id  # For compatibility
        # Add attributes needed for LangExtract compatibility
        self.azure_endpoint = None
        self.api_version = None
        self.azure_ad_token_provider = None

    def infer(self, prompts: list[str]):
        """Generate responses for a batch of prompts using Ollama API"""
        import requests
        import json

        for prompt in prompts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                        }
                    },
                    timeout=300  # 5 minute timeout for local models
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Yield in LangExtract compatible format
                class MockResponse:
                    def __init__(self, output: str):
                        self.output = output

                yield [MockResponse(result.get("response", ""))]

            except Exception as e:
                print(f"Ollama API error: {e}")
                raise