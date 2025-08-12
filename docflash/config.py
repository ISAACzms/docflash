"""
Configuration file for Doc Flash
Supports multiple LLM providers: Azure OpenAI, OpenAI, Google Gemini
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ProviderType = Literal["azure_openai", "openai", "gemini", "vllm", "ollama"]


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: ProviderType
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 16000

    # Azure OpenAI specific
    azure_endpoint: Optional[str] = None
    azure_deployment_name: Optional[str] = None
    azure_api_version: Optional[str] = None

    # OpenAI specific
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # Google Gemini specific
    google_api_key: Optional[str] = None
    google_project_id: Optional[str] = None

    # Ollama specific
    ollama_base_url: Optional[str] = None

    # OCR specific settings
    ocr_provider: ProviderType = "azure_openai"
    ocr_model_id: str = "gpt-4.1-mini"


class Config:
    """Main configuration class"""

    def __init__(self):
        # Get provider from environment variable
        self.provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

        # Validate provider
        if self.provider not in ["azure_openai", "openai", "gemini", "vllm", "ollama"]:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # OCR provider (can be different from main LLM provider)
        self.ocr_provider = os.getenv("OCR_PROVIDER", self.provider).lower()

        # Load provider-specific configuration
        self.llm_config = self._load_llm_config()
        self.ocr_config = self._load_ocr_config()

    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration based on provider"""

        if self.provider == "azure_openai":
            return LLMConfig(
                provider="azure_openai",
                model_id=os.getenv("AZURE_OPENAI_MODEL_ID", "gpt-4.1-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16000")),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                azure_api_version=os.getenv(
                    "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
                ),
            )

        elif self.provider == "openai":
            return LLMConfig(
                provider="openai",
                model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1-mini"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16000")),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
            )

        elif self.provider == "gemini":
            return LLMConfig(
                provider="gemini",
                model_id=os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16000")),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                google_project_id=os.getenv("GOOGLE_PROJECT_ID"),
            )

        elif self.provider == "ollama":
            return LLMConfig(
                provider="ollama",
                model_id=os.getenv("OLLAMA_MODEL_ID", "gemma2:2b"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16000")),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _load_ocr_config(self) -> LLMConfig:
        """Load OCR-specific configuration"""

        if self.ocr_provider == "azure_openai":
            return LLMConfig(
                provider="azure_openai",
                model_id=os.getenv("OCR_AZURE_OPENAI_MODEL_ID", "gpt-4.1-mini"),
                temperature=float(os.getenv("OCR_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("OCR_MAX_TOKENS", "16000")),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment_name=os.getenv(
                    "OCR_AZURE_OPENAI_DEPLOYMENT_NAME",
                    os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                ),
                azure_api_version=os.getenv(
                    "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
                ),
            )

        elif self.ocr_provider == "openai":
            return LLMConfig(
                provider="openai",
                model_id=os.getenv("OCR_OPENAI_MODEL_ID", "gpt-4.1-mini"),
                temperature=float(os.getenv("OCR_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("OCR_MAX_TOKENS", "16000")),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
            )

        elif self.ocr_provider == "gemini":
            return LLMConfig(
                provider="gemini",
                model_id=os.getenv("OCR_GEMINI_MODEL_ID", "gemini-1.5-flash"),
                temperature=float(os.getenv("OCR_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("OCR_MAX_TOKENS", "16000")),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                google_project_id=os.getenv("GOOGLE_PROJECT_ID"),
            )

        elif self.ocr_provider == "vllm":
            # vLLM OCR configuration - uses external service
            return LLMConfig(
                provider="vllm",
                model_id=os.getenv("VLLM_MODEL_ID", "nanonets-ocr"),
                temperature=float(os.getenv("OCR_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("OCR_MAX_TOKENS", "16000")),
            )

        elif self.ocr_provider == "ollama":
            return LLMConfig(
                provider="ollama",
                model_id=os.getenv("OCR_OLLAMA_MODEL_ID", "gemma2:2b"),
                temperature=float(os.getenv("OCR_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("OCR_MAX_TOKENS", "16000")),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

        else:
            raise ValueError(f"Unsupported OCR provider: {self.ocr_provider}")

    def get_available_models(self) -> dict:
        """Get available models for current provider"""

        models = {
            "azure_openai": [
                "gpt-5",
                "gpt-5-mini",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-35-turbo",
            ],
            "openai": [
                "gpt-5",
                "gpt-5-mini",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ],
            "gemini": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
            "ollama": [
                "gemma2:2b",
                "gemma2:9b",
                "gemma2:27b",
                "llama3.2:1b",
                "llama3.2:3b",
                "llama3.1:8b",
                "llama3.1:70b",
                "mistral:7b",
                "mistral-nemo:12b",
                "qwen2.5:7b",
                "qwen2.5:14b",
                "deepseek-coder:6.7b",
                "codellama:7b",
                "phi3:3.8b",
            ],
        }

        return {
            "main": models.get(self.provider, []),
            "ocr": models.get(self.ocr_provider, []),
        }


# Global configuration instance
config = Config()