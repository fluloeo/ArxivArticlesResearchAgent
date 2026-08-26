import json
import logging
import time
from typing import Any, Dict, List

from .base import Conversation, LLMProvider

logger = logging.getLogger(__name__)


class VLLMProvider(LLMProvider):
    """Опциональный GPU-бэкенд поверх vLLM. Требует уже созданный `llm_engine`
    (vllm.LLM) и `tokenizer` — сам пакет vllm здесь не импортируется, поэтому
    этот модуль безопасно импортировать без vllm в окружении.
    """

    def __init__(self, llm_engine, sampling_params_class, tokenizer, model_name: str):
        self.llm = llm_engine
        self.params_factory = sampling_params_class
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.generations_log: List[Dict[str, Any]] = []

    def _format_prompt(self, conversation: Conversation) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                list(conversation), add_generation_prompt=True, tokenize=False
            )
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in conversation) + "\n\nassistant:"

    def generate(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        if not conversations:
            return []
        prompts = [self._format_prompt(c) for c in conversations]
        vllm_params = self.params_factory(**sampling_params)
        outputs = self.llm.generate(prompts, vllm_params)
        texts = [output.outputs[0].text for output in outputs]

        for p, t in zip(prompts, texts):
            self.generations_log.append(
                {"timestamp": time.time(), "model": self.model_name, "prompt": p, "response": t, "params": sampling_params}
            )
        return texts

    def save_log_to_json(self, filename: str = "debug_generations.json") -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.generations_log, f, ensure_ascii=False, indent=2)
