from typing import Any, Dict, Iterator, Tuple

from .llm.base import LLMProvider
from .prompt_resolver import PromptResolver
from .streaming import ChunkDoneEvent, ProgressEvent, RateEstimator, StreamEvent, TextDeltaEvent


def _combine_for_reduce(titles, chunk_summaries: Dict[str, str]) -> str:
    # НЕ markdown-заголовки ("### {title}") — та разметка визуально подсказывала
    # reduce-модели, что от неё ждут ту же структуру на выходе, и в паре с прежней
    # инструкцией "сохраняя структуру по разделам" итоговый отчёт вырождался в
    # конкатенацию map-выжимок под теми же заголовками вместо синтеза (см. промпты ниже).
    return "\n\n".join(f"[{t}]\n{chunk_summaries[t]}" for t in titles)


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
        combined = _combine_for_reduce(titles, dict(zip(titles, chunk_summaries)))

        reduce_conversation = self.prompt_resolver.format_chat(
            self.resolved_prompts["reduce"], {"summaries": combined}, str(self.resolved_prompts.get("system_reduce", ""))
        )
        final_report = self.provider.generate([reduce_conversation], reduce_params)[0]

        return final_report, dict(zip(titles, chunk_summaries))

    def run_stream(
        self,
        overlap_dict: Dict[str, Dict[str, str]],
        map_params: Dict[str, Any],
        reduce_params: Dict[str, Any],
    ) -> Iterator[StreamEvent]:
        """Тот же Map-Reduce, что run(), но стримит прогресс по map-чанкам (ChunkDoneEvent)
        и токены reduce-стадии (TextDeltaEvent) по мере готовности — для UI, которому нужно
        показывать процесс, а не ждать возврата функции. run() не трогается: харнесс
        (evaluation/) и прочие невизуальные вызовы им и продолжают пользоваться."""
        titles = list(overlap_dict.keys())
        n = len(titles)
        rate = RateEstimator()
        chunk_summaries: Dict[str, str] = {}

        map_conversations = [
            self.prompt_resolver.format_chat(
                self.resolved_prompts["map"], {"title": t, **overlap_dict[t]},
                str(self.resolved_prompts.get("system_map", "")),
            )
            for t in titles
        ]
        # generate_as_completed, а не по одному чанку последовательным generate() — иначе
        # параллельность OpenRouterProvider (см. modules/llm/openrouter_provider.py) не
        # проявляется вовсе: батч из N однострочных вызовов не даёт провайдеру ни одного
        # шанса отправить их конкурентно. Порядок готовности не совпадает с порядком
        # чанков на параллельных бэкендах — chunk_summaries индексируется по title, а не
        # по порядку yield'а, поэтому reduce ниже собирает их в исходном порядке статьи.
        done = 0
        for index, summary in self.provider.generate_as_completed(map_conversations, map_params):
            title = titles[index]
            chunk_summaries[title] = summary
            done += 1
            yield ChunkDoneEvent(title=title, summary=summary, index=done, total=n)
            yield ProgressEvent(
                stage="map_reduce_summarize", message=f"Обработан раздел «{title}»",
                current=done, total=n, elapsed_s=rate.elapsed(), eta_s=rate.eta(done, n),
            )

        combined = _combine_for_reduce(titles, chunk_summaries)
        reduce_conversation = self.prompt_resolver.format_chat(
            self.resolved_prompts["reduce"], {"summaries": combined}, str(self.resolved_prompts.get("system_reduce", ""))
        )
        yield ProgressEvent(stage="map_reduce_summarize", message="Синтезирую итоговый отчёт...", current=n, total=n)
        for delta in self.provider.generate_stream(reduce_conversation, reduce_params):
            yield TextDeltaEvent(text=delta)
