"""Модель данных трейса: один прогон графа = GraphTrace = список NodeVisit.

Собирается GraphRecorder (recorder.py) из трёх источников: app.stream(stream_mode="debug")
для топологии/пред-состояния/дельты, NodeWrapper для устойчивого к задержкам потребителя
тайминга, NodeLogHandler для атрибутированных лог-событий. См. docstring GraphRecorder.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


@dataclass(frozen=True)
class ErrorInfo:
    type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class LogEvent:
    """Одна атрибутированная запись из логов modules.* — см. log_capture.py."""

    node: Optional[str]
    occurrence: int
    logger: str
    level: str
    event: str  # стабильный ключ из _EVENT_KEYS, либо "other"
    msg_template: str  # record.msg — неформатированный шаблон, не текст
    args: Any  # JSON-совместимое представление record.args
    message: str  # готовое сообщение, для человека


@dataclass(frozen=True)
class LLMCall:
    """Один вызов LLMProvider.generate(), перехваченный RecordingProvider."""

    node: Optional[str]
    occurrence: int
    n_conversations: int
    duration_s: float
    sampling_params: Dict[str, Any]
    chars_in: int
    chars_out: int
    # Полные prompt/response — только если capture_io=True (--record-llm-io).
    prompts: Optional[Tuple[Any, ...]] = None
    responses: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class NodeVisit:
    """Одно исполнение одного узла графа. research_step в одном кейсе даёт несколько
    NodeVisit с одинаковым node, но растущим occurrence."""

    visit_id: str  # task id из langgraph
    node: str  # modules.node_names.NodeName значение, либо "__start__"
    graph: str  # "app" | "summarize_app"
    step: int
    occurrence: int
    triggers: Tuple[str, ...]
    input_state: Dict[str, Any]  # task.payload.input — полное состояние ДО узла
    output_delta: Dict[str, Any]  # task_result.payload.result; {} если узел ничего не вернул
    duration_s: float
    timing_source: Literal["wrapper", "debug_timestamps"]
    status: Literal["ok", "error"]
    error: Optional[ErrorInfo] = None
    log_events: Tuple[LogEvent, ...] = field(default_factory=tuple)
    llm_calls: Tuple[LLMCall, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GraphTrace:
    """Полный трейс одного прогона одного кейса через один граф (app | summarize_app)."""

    case_id: str
    graph: str
    visits: Tuple[NodeVisit, ...]
    final_state: Dict[str, Any]  # свёрнуто из init_state + все output_delta по порядку
    path: Tuple[str, ...]  # последовательность имён узлов, включая повторы research_step
    terminal_error: Optional[ErrorInfo]
    total_s: float
