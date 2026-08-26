"""RecordingProvider — необязательный (--record-llm-io) декоратор над LLMProvider,
даёт харнессу число LLM-вызовов, размер батча, время и (опционально) сами prompt/response
на узел, без единой правки в modules.llm.*.

Атрибуция по CURRENT_NODE безопасна здесь именно потому, что RecordingProvider.generate()
вызывается СИНХРОННО на графовом потоке (том самом, где NodeWrapper выставил ContextVar) —
он лишь оборачивает и дожидается internal_provider.generate(...), а не сам передаёт работу
в другой поток. См. подробное объяснение границы потоков в context.py.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modules.llm.base import Conversation, LLMProvider

from .context import CURRENT_NODE
from .trace import LLMCall


@dataclass
class _CallSink:
    calls: List[LLMCall] = field(default_factory=list)

    def append(self, call: LLMCall) -> None:
        self.calls.append(call)


class RecordingProvider(LLMProvider):
    """Оборачивает произвольный LLMProvider, логируя каждый generate() в sink.calls как
    LLMCall, атрибутированный текущим узлом. Прозрачно проксирует .tokenizer, если он есть
    у внутреннего провайдера (modules.bootstrap._tokenizer_for_chunking читает этот атрибут
    через getattr — прокси не должен его прятать)."""

    def __init__(self, inner: LLMProvider, sink: Optional[_CallSink] = None, capture_io: bool = False):
        self._inner = inner
        self.sink = sink if sink is not None else _CallSink()
        self.capture_io = capture_io

    def __getattr__(self, name: str) -> Any:
        # Проксируем всё, чего нет у самого RecordingProvider (в первую очередь .tokenizer
        # у MLXProvider) — не перехватываем через явные свойства, чтобы не рассинхронизироваться
        # при появлении новых атрибутов у конкретных провайдеров.
        return getattr(self._inner, name)

    def generate(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        node, occurrence = CURRENT_NODE.get()
        chars_in = sum(len(m["content"]) for conv in conversations for m in conv)

        t0 = time.perf_counter()
        responses = self._inner.generate(conversations, sampling_params)
        duration_s = time.perf_counter() - t0

        self.sink.append(
            LLMCall(
                node=node,
                occurrence=occurrence,
                n_conversations=len(conversations),
                duration_s=duration_s,
                sampling_params=dict(sampling_params),
                chars_in=chars_in,
                chars_out=sum(len(r) for r in responses),
                prompts=tuple(conversations) if self.capture_io else None,
                responses=tuple(responses) if self.capture_io else None,
            )
        )
        return responses
