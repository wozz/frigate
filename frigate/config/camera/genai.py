from enum import Enum
from typing import Any, Optional

from pydantic import Field

from ..base import FrigateBaseModel
from ..env import EnvString

__all__ = ["GenAIConfig", "GenAIProviderEnum"]


class GenAIProviderEnum(str, Enum):
    openai = "openai"
    azure_openai = "azure_openai"
    gemini = "gemini"
    ollama = "ollama"


class GenAIConfig(FrigateBaseModel):
    """Primary GenAI Config to define GenAI Provider."""

    api_key: Optional[EnvString] = Field(default=None, title="Provider API key.")
    base_url: Optional[str] = Field(default=None, title="Provider base url.")
    model: str = Field(default="gpt-4o", title="GenAI model.")
    embedding_model: str = Field(
        default="text-embedding-3-small", title="GenAI embedding model."
    )
    vision_model_prompt: str = Field(
        default="A detailed description of the image for semantic search.",
        title="Prompt for the vision model to describe the image for embedding.",
    )
    provider: GenAIProviderEnum | None = Field(default=None, title="GenAI provider.")
    provider_options: dict[str, Any] = Field(
        default={}, title="GenAI Provider extra options."
    )
