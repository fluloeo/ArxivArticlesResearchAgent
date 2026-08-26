from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit

from .base import CheckContext, CheckResult

_ALLOWED_TOOLS = {"search_arxiv", "get_fulltext"}


def check_research_step(visit: NodeVisit, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    """Один визит = одна итерация research_step (см. modules/agent.py). Часть проверок
    читает атрибутированные лог-события — те самые research_step: normalized/overriding/
    synthesizing строки, которые раньше были видны только в консоли сервера
    (evaluation/tracing/log_capture.py)."""
    results: List[CheckResult] = []
    delta = visit.output_delta
    is_final = "final_answer" in delta

    if not is_final:
        tool = (delta.get("evidence") or [{}])[-1].get("tool") if delta.get("evidence") else None
        results.append(
            CheckResult(
                check="tool_in_allowlist",
                passed=tool in _ALLOWED_TOOLS,
                severity="error",
                observed={"tool": tool},
                expected={"one_of": sorted(_ALLOWED_TOOLS)},
            )
        )

    evidence = visit.input_state.get("evidence") or []
    new_evidence = delta.get("evidence")
    if new_evidence is not None:
        results.append(
            CheckResult(
                check="evidence_grows_monotonically",
                passed=len(new_evidence) == len(evidence) + 1,
                severity="error",
                observed={"before": len(evidence), "after": len(new_evidence)},
            )
        )

    if case.max_iterations is not None:
        results.append(
            CheckResult(
                check="respects_max_iterations",
                passed=visit.occurrence <= case.max_iterations + 1,  # +1: финальный синтез-ответ после лимита
                severity="warning",
                observed={"occurrence": visit.occurrence},
                expected={"max_iterations": case.max_iterations},
            )
        )

    normalized = [e for e in visit.log_events if e.event == "tool_call_normalized"]
    results.append(
        CheckResult(
            check="tool_call_was_well_formed",
            passed=not normalized,
            severity="warning",
            observed={"normalized": bool(normalized)},
            message="модель прислала неполный/пустой tool_call — потребовалась нормализация" if normalized else "",
        )
    )

    if is_final:
        results.append(
            CheckResult(
                check="grounded_final_answer",
                passed=bool(evidence) or context.app_config.min_research_iterations == 0,
                severity="error",
                observed={"n_evidence_entries": len(evidence)},
                message="финальный ответ без единого вызова инструмента при min_research_iterations>0",
            )
        )

        if case.expected_sources:
            sources = set(delta.get("sources") or [])
            found_ids = {s.split(":", 1)[0].strip() for s in sources}
            hit = any(gold in found_ids for gold in case.expected_sources)
            results.append(
                CheckResult(
                    check="gold_source_retrieved",
                    passed=hit,
                    severity="warning",
                    observed={"sources": sorted(sources)},
                    expected={"one_of": case.expected_sources},
                )
            )

    return results
