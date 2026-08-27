"""ExperimentRunner — связывает вместе dataset (Suite/EvalCase), tracing (GraphRecorder),
checks (детерминированные проверки), metrics (MetricsRunner, опционально) и runlog
(RunWriter) в один прогон сьюта.

Метрики (faithfulness/coverage/answer_relevancy) подключаются, только если вызывающий код
передал `metrics_runner` — по плану (шаг 3) харнесс должен работать и приносить пользу на
одних только детерминированных проверках (ноль LLM-вызовов судьи) без Judge вовсе; Judge
требует явного --judge-model (см. evaluation/metrics/judge.py — харнесс не предлагает
"same as main model" шорткот, самосудейство обесценивает сравнение моделей).
"""
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.checks import CHECKS, CheckContext, check_graph_level
from evaluation.dataset.case import EvalCase, Suite
from evaluation.metrics_runner import MetricsRunner
from evaluation.runlog.run_writer import RunWriter
from evaluation.tracing.provider_wrapper import RecordingProvider
from evaluation.tracing.recorder import GraphRecorder
from evaluation.tracing.trace import GraphTrace
from modules.agent import ArxivAgent

logger = logging.getLogger(__name__)


@dataclass
class CaseOutcome:
    case_id: str
    trace: GraphTrace
    n_checks_passed: int
    n_checks_failed: int
    n_checks_warned: int


def _run_checks_for_trace(trace: GraphTrace, case: EvalCase, context: CheckContext) -> List[tuple]:
    """Возвращает [(node, occurrence, CheckResult), ...] — по визитам плюс graph-level."""
    out = []
    for visit in trace.visits:
        for fn in CHECKS.get(visit.node, []):
            for result in fn(visit, case, context):
                out.append((visit.node, visit.occurrence, result))
    for result in check_graph_level(trace, case, context):
        out.append(("__graph__", 0, result))
    return out


class ExperimentRunner:
    def __init__(
        self,
        agent: ArxivAgent,
        context: CheckContext,
        writer: RunWriter,
        recording_provider: Optional[RecordingProvider] = None,
        metrics_runner: Optional[MetricsRunner] = None,
        max_visits: int = 200,
    ):
        self.agent = agent
        self.context = context
        self.writer = writer
        self.recording_provider = recording_provider
        self.metrics_runner = metrics_runner
        self.max_visits = max_visits
        self._recorders = {
            "app": GraphRecorder(agent.app, "app"),
            "summarize_app": GraphRecorder(agent.summarize_app, "summarize_app"),
        }

    def run_case(self, case: EvalCase, suite: Suite) -> CaseOutcome:
        recorder = self._recorders[case.entry]
        if self.recording_provider is not None:
            self.recording_provider.sink.calls.clear()

        trace = recorder.run(
            case.initial_state(),
            stop_after=case.stop_after,
            max_visits=self.max_visits,
            case_id=case.case_id,
        )

        for visit in trace.visits:
            self.writer.write_node_visit(case.case_id, visit)
            self.writer.write_events(case.case_id, visit)

        check_results = _run_checks_for_trace(trace, case, self.context)
        for node, occurrence, result in check_results:
            self.writer.write_check(case.case_id, node, occurrence, result)

        passed = sum(1 for *_x, r in check_results if r.passed)
        failed = sum(1 for *_x, r in check_results if not r.passed and r.severity == "error")
        warned = sum(1 for *_x, r in check_results if not r.passed and r.severity == "warning")

        if self.metrics_runner is not None and suite.metrics and not trace.terminal_error:
            try:
                self.metrics_runner.run_case(trace, case, suite, self.writer)
            except Exception:
                logger.exception("Кейс %s: сбой при подсчёте метрик (checks уже записаны)", case.case_id)

        self.writer.write_case(case.case_id, trace, [r for *_x, r in check_results], scores={})

        if trace.terminal_error:
            logger.warning("Кейс %s завершился ошибкой: %s", case.case_id, trace.terminal_error.message)

        return CaseOutcome(
            case_id=case.case_id, trace=trace,
            n_checks_passed=passed, n_checks_failed=failed, n_checks_warned=warned,
        )

    def run_suite(self, suite: Suite, limit: Optional[int] = None) -> List[CaseOutcome]:
        cases = suite.cases[:limit] if limit else suite.cases
        self.writer.set_suite_info(path=None, n_cases=len(cases), case_ids=[c.case_id for c in cases])
        outcomes = []
        for i, case in enumerate(cases, 1):
            logger.info("[%d/%d] %s: %s", i, len(cases), case.case_id, case.query[:60])
            try:
                outcomes.append(self.run_case(case, suite))
            except Exception:
                logger.exception("Кейс %s упал вне графа (bug в самом харнессе)", case.case_id)
        return outcomes


def run_suite_concurrent(
    runners: List["ExperimentRunner"], suite: Suite, limit: Optional[int] = None
) -> List[CaseOutcome]:
    """Параллельная версия run_suite: несколько статей (кейсов) генерируются и судятся
    ОДНОВРЕМЕННО, а не одна за другой — раньше вся суть параллелизма ограничивалась
    чанками ВНУТРИ одной статьи (MetricsRunner._run_per_chunk), а сами статьи всё равно
    шли строго по очереди, отсюда счёт на десятки минут для сколь-нибудь заметной выборки.

    Требует ОТДЕЛЬНОГО ArxivAgent (и значит, отдельного скомпилированного LangGraph-графа)
    на каждый параллельный воркер — GraphRecorder на время .run() монки-патчит
    app.nodes[name].bound.func НА САМОМ graph-объекте (см. tracing/recorder.py), и это не
    потокобезопасно на ОДНОМ разделяемом agent: два конкурентных прогона патчили бы/
    восстанавливали одни и те же атрибуты друг у друга под ногами. Отсюда пул из N
    независимых runners (каждый со своим agent), а не N потоков поверх одного runner.
    Judge/LLMProvider безопасно шарить между воркерами — каждый generate()-вызов создаёт
    свой event loop и свой AsyncOpenAI-клиент (см. modules/llm/openrouter_provider.py).
    """
    cases = suite.cases[:limit] if limit else suite.cases
    writer = runners[0].writer
    writer.set_suite_info(path=None, n_cases=len(cases), case_ids=[c.case_id for c in cases])

    pool: "queue.Queue[ExperimentRunner]" = queue.Queue()
    for r in runners:
        pool.put(r)

    outcomes: List[CaseOutcome] = []
    outcomes_lock = threading.Lock()
    progress = {"done": 0}

    def _work(case: EvalCase) -> None:
        runner = pool.get()
        try:
            outcome = runner.run_case(case, suite)
        except Exception:
            logger.exception("Кейс %s упал вне графа (bug в самом харнессе)", case.case_id)
            outcome = None
        finally:
            pool.put(runner)
        if outcome is not None:
            with outcomes_lock:
                outcomes.append(outcome)
                progress["done"] += 1
                logger.info("[%d/%d] завершён %s", progress["done"], len(cases), case.case_id)

    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        list(executor.map(_work, cases))
    return outcomes


def summarize_outcomes(outcomes: List[CaseOutcome]) -> Dict[str, Any]:
    n = len(outcomes)
    n_errors = sum(1 for o in outcomes if o.trace.terminal_error is not None)
    return {
        "n_cases": n,
        "n_errored": n_errors,
        "n_ok": n - n_errors,
        "total_checks_passed": sum(o.n_checks_passed for o in outcomes),
        "total_checks_failed": sum(o.n_checks_failed for o in outcomes),
        "total_checks_warned": sum(o.n_checks_warned for o in outcomes),
        "total_duration_s": sum(o.trace.total_s for o in outcomes),
    }
