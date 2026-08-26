"""Перехват существующих логов modules.* и превращение их в атрибутированные события.

Идея: агент уже логирует всё, что нужно харнессу (research_step: normalized incomplete
tool call, structured_output repair failed, fetch failed и т.д.) — просто это никогда не
попадало в состояние графа. Ничего не меняем в инференс-пути: вешаем logging.Handler на
дерево "modules" на время прогона кейса и снимаем в finally.

Ключуем по record.msg (НЕформатированный %-шаблон, стабильный идентификатор), а не по
готовому тексту сообщения — текст меняется вместе с аргументами, шаблон нет. Если кто-то
переформулирует лог в agent.py и забудет обновить _EVENT_KEYS, событие не потеряется -
оно попадёт под ключ "other" (см. test), просто не получит специализированное имя.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, List

from .context import CURRENT_NODE
from .trace import LogEvent

# record.msg (сырой %-шаблон) -> стабильное имя события для агрегации в отчётах.
_EVENT_KEYS = {
    "classifier: intent=%s query=%r": "classifier_decision",
    "resolve_target_article: explicit arXiv ID %s in query": "explicit_id_used",
    "resolve_target_article: no explicit ID, returning %d candidates for user to pick": "candidates_returned",
    "research_step: iteration=%d decision.action=%s tool=%s confidence=%s": "research_decision",
    "research_step: overriding premature final_answer — forcing a grounding tool "
    "call (min_research_iterations=%d not yet met)": "premature_final_overridden",
    "research_step: synthesizing final answer from evidence (exhausted=%s repeated_call=%s)": "forced_final",
    "research_step: called tool=%s ok=%s": "tool_called",
    "research_step: normalized incomplete tool call -> %s(%s)": "tool_call_normalized",
    "Structured output repair failed for schema=%s, falling back to default: %s": "structured_output_fallback",
    "Corrupt cache entry for %s, will refetch": "cache_corrupt",
    "Could not fetch full text for arXiv:%s (ar5iv and PDF both failed)": "fetch_failed",
    "arXiv API rejected request (HTTP %d, %s): %s": "arxiv_api_rejected",
    "arXiv API request failed after %d attempts (%s): %s": "arxiv_api_failed",
    "arXiv API request failed (attempt %d/%d), retrying: %s": "arxiv_api_retry",
    "Failed to parse arXiv Atom feed": "arxiv_feed_parse_error",
    "Tool %s failed with args=%r": "tool_exception",
}


def _jsonable(args: Any) -> Any:
    """record.args — либо tuple позиционных аргументов, либо dict (%(name)s-стиль), либо
    None. Приводим к JSON-совместимому виду, не падая на непредвиденных типах (Exception,
    произвольные объекты и т.п. — logging этого не запрещает)."""
    if args is None:
        return None
    if isinstance(args, dict):
        items = args.items()
    elif isinstance(args, (tuple, list)):
        items = enumerate(args)
    else:
        return repr(args)

    out: Any = {} if isinstance(args, dict) else [None] * len(args)
    for key, value in items:
        try:
            json.dumps(value, ensure_ascii=False)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = repr(value)
    return out


@dataclass
class _Sink:
    events: List[LogEvent]

    def append(self, event: LogEvent) -> None:
        self.events.append(event)


class NodeLogHandler(logging.Handler):
    """Собирает LogRecord'ы дерева "modules" в sink, атрибутируя их текущим узлом
    через CURRENT_NODE (см. context.py). Уровень — WARNING и выше плюс явно
    зарегистрированные в _EVENT_KEYS INFO-события; остальной INFO-шум не нужен харнессу."""

    def __init__(self, sink: _Sink):
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        event = _EVENT_KEYS.get(record.msg, "other")
        if event == "other" and record.levelno < logging.WARNING:
            return  # непомеченный INFO-шум пропускаем, WARNING/ERROR — всегда интересны

        node, occurrence = CURRENT_NODE.get()
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)

        self.sink.append(
            LogEvent(
                node=node,
                occurrence=occurrence,
                logger=record.name,
                level=record.levelname,
                event=event,
                msg_template=str(record.msg),
                args=_jsonable(record.args),
                message=message,
            )
        )


class capture_node_logs:
    """Контекстный менеджер: `with capture_node_logs() as sink:` — вешает NodeLogHandler
    на logger "modules" на время блока, снимает в любом случае. sink.events накапливает
    LogEvent по мере исполнения; GraphRecorder разносит их по NodeVisit постфактум."""

    def __init__(self) -> None:
        self.sink = _Sink(events=[])
        self._handler = NodeLogHandler(self.sink)
        self._logger = logging.getLogger("modules")

    def __enter__(self) -> _Sink:
        self._logger.addHandler(self._handler)
        return self.sink

    def __exit__(self, *exc: Any) -> None:
        self._logger.removeHandler(self._handler)
