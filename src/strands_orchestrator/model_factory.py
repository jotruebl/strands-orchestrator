"""Model factory — maps model alias strings to Strands model instances."""

from __future__ import annotations

import logging
from typing import Any

from strands.models.anthropic import AnthropicModel
from strands.models.openai import OpenAIModel

logger = logging.getLogger(__name__)

# Default alias → (provider, model_id) mapping
DEFAULT_ALIASES: dict[str, tuple[str, str]] = {
    # Anthropic
    "sonnet": ("anthropic", "claude-sonnet-4-20250514"),
    "opus": ("anthropic", "claude-opus-4-20250514"),
    "haiku": ("anthropic", "claude-haiku-4-5-20251001"),
    # OpenAI
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "o3-mini": ("openai", "o3-mini"),
}

PROVIDER_CLASSES: dict[str, type] = {
    "anthropic": AnthropicModel,
    "openai": OpenAIModel,
}


class ModelFactory:
    """Creates Strands model instances from alias strings.

    Supports custom alias overrides via the `custom_aliases` parameter.
    Falls back to treating the string as a literal Anthropic model ID
    if no alias matches.
    """

    def __init__(self, custom_aliases: dict[str, str] | None = None):
        self._aliases = dict(DEFAULT_ALIASES)
        if custom_aliases:
            for alias, model_id in custom_aliases.items():
                provider = self._infer_provider(model_id)
                self._aliases[alias] = (provider, model_id)

    def create(self, model_string: str, **kwargs: Any) -> Any:
        """Create a Strands model instance from an alias or model ID.

        Args:
            model_string: Model alias (e.g., "sonnet") or full model ID.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            A Strands model instance (AnthropicModel, OpenAIModel, etc.)
        """
        if model_string in self._aliases:
            provider, model_id = self._aliases[model_string]
        else:
            provider = self._infer_provider(model_string)
            model_id = model_string

        model_class = PROVIDER_CLASSES.get(provider)
        if model_class is None:
            logger.warning(
                "Unknown provider '%s' for model '%s'. Falling back to Anthropic.",
                provider,
                model_string,
            )
            model_class = AnthropicModel

        return model_class(model_id=model_id, **kwargs)

    @staticmethod
    def _infer_provider(model_id: str) -> str:
        """Infer provider from model ID string."""
        if model_id.startswith(("gpt-", "o1-", "o3-", "o4-")):
            return "openai"
        # Default to Anthropic for claude-* and unknown models
        return "anthropic"
