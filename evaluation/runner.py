"""ExperimentRunner — связывает вместе dataset (Suite/EvalCase), tracing (GraphRecorder),
checks (детерминированные проверки) и runlog (RunWriter) в один прогон сьюта.

Метрики (faithfulness/coverage/answer_relevancy, evaluation/metrics/) в этой версии ЕЩЁ
НЕ подключены — по плану (шаг 3) харнесс сначала должен работать и приносить пользу на
одних только детерминированных проверках, которые стоят ноль LLM-вызовов судьи; метрики
добавляются отдельным шагом (evaluation/metrics/), не блокируя эту часть.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evaluation.checks import CHECKS, CheckContext, check_graph_level
from evaluation.dataset.case import EvalCase, Suite
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
        max_visits: int = 200,
    ):
        self.agent = agent
        self.context = context
        self.writer = writer
        self.recording_provider = recording_provider
        self.max_visits = max_visits
        self._recorders = {
            "app": GraphRecorder(agent.app, "app"),
            "summarize_app": GraphRecorder(agent.summarize_app, "summarize_app"),
        }

    def run_case(self, case: EvalCase) -> CaseOutcome:
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
                outcomes.append(self.run_case(case))
            except Exception:
                logger.exception("Кейс %s упал вне графа (bug в самом харнессе)", case.case_id)
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
