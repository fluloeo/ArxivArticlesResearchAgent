import logging
from typing import Any, Dict, Iterator, List

from openai import OpenAI

from .base import Conversation, LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """Опциональный облачный бэкенд поверх OpenRouter (OpenAI-совместимый API)."""

    def __init__(self, api_key: str, model_name: str = "qwen/qwen3-30b-a3b-instruct-2507"):
        if not api_key:
            raise ValueError("OpenRouterProvider требует OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_name = model_name

    def generate(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        if not conversations:
            return []
        results = []
        for conversation in conversations:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=list(conversation),
                    temperature=sampling_params.get("temperature", 0),
                    max_tokens=sampling_params.get("max_tokens", 1024),
                    # frequency_penalty раньше молча отбрасывался (передавались только
                    # temperature/max_tokens), хотя это штатный параметр OpenAI-совместимого
                    # API и он объявлен в NodeGenerationConfig для map-фазы суммаризации.
                    frequency_penalty=sampling_params.get("frequency_penalty", 0.0),
                    extra_body={"include_reasoning": False},
                )
                message = response.choices[0].message
                content = getattr(message, "content", None)
                reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
                results.append(content or reasoning or "")
            except Exception as e:
                logger.exception("OpenRouter generation failed")
                results.append(f"Error: {e}")
        return results

    def generate_stream(self, conversation: Conversation, sampling_params: Dict[str, Any]) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=list(conversation),
                temperature=sampling_params.get("temperature", 0),
                max_tokens=sampling_params.get("max_tokens", 1024),
                frequency_penalty=sampling_params.get("frequency_penalty", 0.0),
                extra_body={"include_reasoning": False},
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception:
            logger.exception("OpenRouter streaming generation failed")
