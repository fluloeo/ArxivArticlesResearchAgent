"""GraphRecorder — прогоняет один кейс через скомпилированный LangGraph-граф ArxivAgent
и возвращает полный GraphTrace, не трогая рантайм-путь агента.

Три источника данных, объединяемые по (node, occurrence):
  1. app.stream(state, stream_mode="debug") — топология, step, task id, пред-состояние узла
     (task.payload.input) и его дельта (task_result.payload.result).
  2. NodeWrapper — подменяет app.nodes[name].bound.func на засекающую время обёртку.
     Нужен потому, что таймстемпы stream_mode="debug" искажаются задержкой ПОТРЕБИТЕЛЯ
     потока (замерено: узел 0.30 с показывал 0.815 с при 0.5-секундном стойле консьюмера).
     .bound.func — приватный атрибут langgraph (langgraph._internal._runnable), поэтому
     обёртка снимается в finally и есть safe-фоллбек на таймстемпы debug, если структура
     когда-нибудь изменится (_supports_node_wrapping).
  3. capture_node_logs() — атрибутированные логи modules.* за время исполнения узла.

Критично: цикл потребления generator'а из stream() должен быть МИНИМАЛЬНЫМ (только
raw.append(ev)) — весь разбор, сопоставление и запись выполняются ПОСЛЕ того, как
generator исчерпан, иначе искажаются те самые таймстемпы debug, на которые опирается
фоллбек-тайминг.
"""
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .context import CURRENT_NODE
from .log_capture import capture_node_logs
from .trace import ErrorInfo, GraphTrace, LogEvent, NodeVisit

_START = "__start__"


def _supports_node_wrapping(app: Any) -> bool:
    return all(hasattr(getattr(node, "bound", None), "func") for name, node in app.nodes.items() if name != _START)


class _WrapperTimings:
    """Пер-узловые тайминги, снятые обёрткой .bound.func — по (node, occurrence)."""

    def __init__(self) -> None:
        self._durations: Dict[tuple, float] = {}
        self._counters: Dict[str, int] = {}

    def record_call(self, node: str) -> "_TimingHandle":
        occurrence = self._counters.get(node, 0) + 1
        self._counters[node] = occurrence
        return _TimingHandle(self, node, occurrence)

    def get(self, node: str, occurrence: int) -> Optional[float]:
        return self._durations.get((node, occurrence))


class _TimingHandle:
    def __init__(self, owner: _WrapperTimings, node: str, occurrence: int):
        self.owner, self.node, self.occurrence = owner, node, occurrence

    def __enter__(self) -> "_TimingHandle":
        self._t0 = time.perf_counter()
        tok_state = CURRENT_NODE.set((self.node, self.occurrence))
        self._tok = tok_state
        return self

    def __exit__(self, *exc: Any) -> None:
        self.owner._durations[(self.node, self.occurrence)] = time.perf_counter() - self._t0
        CURRENT_NODE.reset(self._tok)


@contextmanager
def _node_wrapping(app: Any, timings: _WrapperTimings) -> Iterator[bool]:
    """Подменяет .bound.func каждого узла на таймер+ContextVar шим. Возвращает через yield,
    поддерживается ли обёртка вообще — если нет, вызывающий код читает тайминг из debug-
    таймстемпов и помечает manifest.timing_source="debug_timestamps"."""
    if not _supports_node_wrapping(app):
        yield False
        return

    originals: Dict[str, Any] = {}
    for name, node in app.nodes.items():
        if name == _START:
            continue
        bound = node.bound
        original_func = bound.func
        originals[name] = original_func

        def make_wrapped(node_name: str, fn: Any) -> Any:
            def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
                with timings.record_call(node_name):
                    return fn(state, *args, **kwargs)

            return wrapped

        bound.func = make_wrapped(name, original_func)

    try:
        yield True
    finally:
        for name, node in app.nodes.items():
            if name in originals:
                node.bound.func = originals[name]


def _fold_state(init_state: Dict[str, Any], deltas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = dict(init_state)
    for delta in deltas:
        state.update(delta or {})
    return state


class GraphRecorder:
    """Прогоняет один кейс через app.stream(..., stream_mode="debug") и собирает GraphTrace.

    Не модифицирует состояние или поведение агента — только читает поток событий и (опционально)
    подменяет node-обёртки на время вызова run(), возвращая всё как было в finally. Один
    GraphRecorder можно переиспользовать для многих кейсов подряд на одном agent.app.
    """

    def __init__(self, app: Any, graph_name: str) -> None:
        self.app = app
        self.graph_name = graph_name

    def run(
        self,
        init_state: Dict[str, Any],
        *,
        config: Optional[Dict[str, Any]] = None,
        stop_after: Optional[str] = None,
        max_visits: int = 200,
        case_id: str = "",
    ) -> GraphTrace:
        timings = _WrapperTimings()
        raw: List[dict] = []
        occurrence_counters: Dict[str, int] = {}
        terminal_error: Optional[BaseException] = None
        t_start = time.perf_counter()

        with _node_wrapping(self.app, timings) as wrapping_active, capture_node_logs() as log_sink:
            try:
                stream = self.app.stream(init_state, config=config, stream_mode="debug")
                for ev in stream:
                    raw.append(ev)  # см. docstring модуля — тело цикла должно быть ровно этим
                    if len(raw) > max_visits * 2:  # task + task_result на визит
                        stream.close()
                        break
                    if stop_after is not None and ev.get("type") == "task_result":
                        if ev.get("payload", {}).get("name") == stop_after:
                            stream.close()
                            break
            except BaseException as exc:  # noqa: BLE001 — намеренно широко: любой сбой узла
                terminal_error = exc
        total_s = time.perf_counter() - t_start

        # --- разбор: task/task_result попарно по task id ---
        tasks: Dict[str, dict] = {}
        results: Dict[str, dict] = {}
        order: List[str] = []
        for ev in raw:
            payload = ev.get("payload", {})
            tid = payload.get("id")
            if ev.get("type") == "task":
                tasks[tid] = ev
                order.append(tid)
            elif ev.get("type") == "task_result":
                results[tid] = ev

        # события логов не несут task id — разносим их по (node, occurrence) вручную,
        # используя те же счётчики occurrence, что породил wrapper (либо, без wrapper'а,
        # пересчитываем occurrence по порядку появления в debug-потоке ниже).
        events_by_node: Dict[tuple, List[LogEvent]] = {}
        for log_event in log_sink.events:
            events_by_node.setdefault((log_event.node, log_event.occurrence), []).append(log_event)

        visits: List[NodeVisit] = []
        path: List[str] = []
        for tid in order:
            task_ev = tasks[tid]
            payload = task_ev.get("payload", {})
            node = payload.get("name", "")
            if node == _START:
                continue

            occurrence = occurrence_counters.get(node, 0) + 1
            occurrence_counters[node] = occurrence
            path.append(node)

            result_ev = results.get(tid)
            if result_ev is None:
                # Узел начал исполняться, но task_result не пришёл — узел упал (langgraph
                # не эмитит task_result на исключении, см. docstring модуля) либо стрим
                # был остановлен раньше (stop_after/max_visits). Различаем по terminal_error.
                status = "error" if terminal_error is not None else "ok"
                error = (
                    ErrorInfo(
                        type=type(terminal_error).__name__,
                        message=str(terminal_error),
                        traceback="".join(traceback.format_exception(terminal_error)),
                    )
                    if terminal_error is not None
                    else None
                )
                delta: Dict[str, Any] = {}
            else:
                result_payload = result_ev.get("payload", {})
                status = "error" if result_payload.get("error") else "ok"
                error = (
                    ErrorInfo(type="GraphError", message=str(result_payload.get("error")), traceback="")
                    if result_payload.get("error")
                    else None
                )
                delta = result_payload.get("result") or {}

            wrapper_duration = timings.get(node, occurrence) if wrapping_active else None
            if wrapper_duration is not None:
                duration, timing_source = wrapper_duration, "wrapper"
            else:
                duration, timing_source = self._debug_duration(task_ev, result_ev), "debug_timestamps"

            visits.append(
                NodeVisit(
                    visit_id=tid,
                    node=node,
                    graph=self.graph_name,
                    step=task_ev.get("step", -1),
                    occurrence=occurrence,
                    triggers=tuple(payload.get("triggers") or ()),
                    input_state=payload.get("input") or {},
                    output_delta=delta,
                    duration_s=duration,
                    timing_source=timing_source,  # type: ignore[arg-type]
                    status=status,  # type: ignore[arg-type]
                    error=error,
                    log_events=tuple(events_by_node.get((node, occurrence), ())),
                )
            )

        final_state = _fold_state(init_state, [v.output_delta for v in visits])
        return GraphTrace(
            case_id=case_id,
            graph=self.graph_name,
            visits=tuple(visits),
            final_state=final_state,
            path=tuple(path),
            terminal_error=(
                ErrorInfo(
                    type=type(terminal_error).__name__,
                    message=str(terminal_error),
                    traceback="".join(traceback.format_exception(terminal_error)),
                )
                if terminal_error is not None
                else None
            ),
            total_s=total_s,
        )

    @staticmethod
    def _debug_duration(task_ev: dict, result_ev: Optional[dict]) -> float:
        if result_ev is None:
            return 0.0
        try:
            from datetime import datetime

            t0 = datetime.fromisoformat(task_ev["timestamp"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(result_ev["timestamp"].replace("Z", "+00:00"))
            return max(0.0, (t1 - t0).total_seconds())
        except (KeyError, ValueError):
            return 0.0
