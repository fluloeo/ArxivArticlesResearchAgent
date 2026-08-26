import logging
from typing import Any, Dict, Optional

from .llm.base import Conversation

logger = logging.getLogger(__name__)


class PromptResolver:
    """Резолвит промпт для узла: LangSmith Hub (если use_hub) -> локальный fallback -> сырое значение,
    и форматирует его в List[ChatMessage].

    Раньше эта логика была продублирована почти дословно в ArxivAgent и SummarizationPipeline
    (включая json.dumps/json.loads prompt'а как строки — этого больше нет: format_chat
    сразу отдаёт структурированный Conversation, который LLMProvider понимает нативно).
    """

    def __init__(self, ls_client: Optional[Any], local_prompts: Optional[Dict[str, Any]] = None, use_hub: bool = False):
        self.ls_client = ls_client
        self.local_prompts = local_prompts or {}
        self.use_hub = use_hub

    def resolve(self, key: str, hub_ref: Any) -> Any:
        if self.use_hub and self.ls_client is not None and isinstance(hub_ref, str):
            try:
                logger.info("Pulling prompt from LangSmith: %s", hub_ref)
                return self.ls_client.pull_prompt(hub_ref)
            except Exception:
                logger.warning("Failed to pull prompt '%s' from hub, falling back to local", hub_ref, exc_info=True)

        if key in self.local_prompts:
            return self.local_prompts[key]

        return hub_ref

    def resolve_all(self, prompts: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self.resolve(key, val) for key, val in prompts.items()}

    @staticmethod
    def format_chat(prompt_data: Any, variables: Dict[str, Any], system_fallback: str = "") -> Conversation:
        if hasattr(prompt_data, "format_messages"):
            messages = prompt_data.format_messages(**variables)
            return [{"role": "system" if m.type == "system" else "user", "content": m.content} for m in messages]

        if isinstance(prompt_data, dict):
            return [
                {"role": "system", "content": str(prompt_data.get("system", system_fallback))},
                {"role": "user", "content": str(prompt_data.get("user", "")).format(**variables)},
            ]

        return [
            {"role": "system", "content": system_fallback or "You are a helpful assistant."},
            {"role": "user", "content": str(prompt_data).format(**variables)},
        ]
