"""Событийные типы для потокового исполнения агента (ArxivAgent.invoke_stream /
summarize_article_stream) — transport-agnostic: grpc_service/server.py транслирует их в
protobuf AskEvent, но сам модуль ничего не знает про gRPC, и modules.summarization тоже
использует их напрямую (SummarizationPipeline.run_stream).

Почему не через LangGraph.stream(): тот даёт видимость только на ГРАНИЦАХ узлов
(app.stream(stream_mode="debug"), см. evaluation/tracing/recorder.py) — этого достаточно
для наблюдения (харнесс), но не для токенного стриминга текста и прогресса по чанкам
ВНУТРИ map_reduce_summarize, который сам по себе один непрозрачный узел с циклом по
десяткам чанков. invoke_stream/summarize_article_stream поэтому напрямую вызывают методы
узлов по порядку (тот же порядок, что задают рёбра графа в _build_graph/_add_summarize_chain)
вместо компиляции отдельного LangGraph-графа под стриминг.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class ProgressEvent:
    """Прогресс долгой стадии — узла целиком (fetch/chunk) или итерации внутри узла
    (i-й чанк map-стадии, i-я итерация research_step)."""

    stage: str
    message: str
    current: int = 0
    total: int = 0
    elapsed_s: float = 0.0
    eta_s: Optional[float] = None


@dataclass
class TextDeltaEvent:
    """Добавочный фрагмент финального текста (сейчас — только reduce-стадия
    суммаризации, единственный узел с генерацией связной прозы, а не structured output)."""

    text: str


@dataclass
class ChunkDoneEvent:
    """Одна законченная map-выжимка — стримится по готовности, а не всем списком в конце."""

    title: str
    summary: str
    index: int
    total: int


@dataclass
class FinalEvent:
    """Финал потока — тот же dict, что раньше возвращал agent.invoke()/summarize_article()
    (state после слияния всех дельт узлов). Ровно одно FinalEvent на поток, последним."""

    result: Dict[str, Any] = field(default_factory=dict)


StreamEvent = Union[ProgressEvent, TextDeltaEvent, ChunkDoneEvent, FinalEvent]


class RateEstimator:
    """ETA по средней скорости уже пройденных элементов — тот же принцип, что у tqdm:
    не пытается угадывать заранее, просто линейно экстраполирует уже наблюдённый темп."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def eta(self, current: int, total: int) -> Optional[float]:
        if current <= 0 or total <= 0 or current > total:
            return None
        per_item = self.elapsed() / current
        return per_item * (total - current)
