from typing import Any, Dict, Tuple

from .llm.base import LLMProvider
from .prompt_resolver import PromptResolver


class SummarizationPipeline:
    """Map-Reduce суммаризация статьи по чанкам с перекрытиями (см. ArticleProcessor).

    Провайдеры LLM и логика резолва промптов больше не живут здесь — см.
    modules.llm и modules.prompt_resolver соответственно (устраняет дублирование,
    которое раньше было почти дословной копией того, что делал ArxivAgent).
    """

    def __init__(self, provider: LLMProvider, prompts: Dict[str, Any], prompt_resolver: PromptResolver):
        self.provider = provider
        self.prompt_resolver = prompt_resolver
        self.resolved_prompts = prompt_resolver.resolve_all(prompts)

    def run(
        self,
        overlap_dict: Dict[str, Dict[str, str]],
        map_params: Dict[str, Any],
        reduce_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, str]]:
        titles = list(overlap_dict.keys())

        map_conversations = [
            self.prompt_resolver.format_chat(
                self.resolved_prompts["map"], {"title": t, **p}, str(self.resolved_prompts.get("system_map", ""))
            )
            for t, p in overlap_dict.items()
        ]
        chunk_summaries = self.provider.generate(map_conversations, map_params)
        combined = "\n\n".join(f"### {t}\n{s}" for t, s in zip(titles, chunk_summaries))

        reduce_conversation = self.prompt_resolver.format_chat(
            self.resolved_prompts["reduce"], {"summaries": combined}, str(self.resolved_prompts.get("system_reduce", ""))
        )
        final_report = self.provider.generate([reduce_conversation], reduce_params)[0]

        return final_report, dict(zip(titles, chunk_summaries))
