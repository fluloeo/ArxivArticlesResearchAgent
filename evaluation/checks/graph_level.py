from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import GraphTrace

from .base import CheckContext, CheckResult


def check_graph_level(trace: GraphTrace, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    results: List[CheckResult] = []

    if case.expected_path is not None:
        # Совпадение по неповторяющейся форме пути — зацикленные узлы (research_step)
        # схлопываем в один элемент, иначе ожидание пришлось бы писать под конкретное
        # число итераций, которое зависит от решений модели, а не от топологии.
        collapsed = []
        for node in trace.path:
            if not collapsed or collapsed[-1] != node:
                collapsed.append(node)
        results.append(
            CheckResult(
                check="path_match",
                passed=collapsed == list(case.expected_path),
                severity="error",
                observed={"path": list(trace.path), "collapsed": collapsed},
                expected={"path": case.expected_path},
            )
        )

    final_answer = trace.final_state.get("final_answer")
    candidates = trace.final_state.get("candidates")
    terminal_valid = bool(final_answer) != bool(candidates)  # ровно один из двух, не оба и не ни одного
    results.append(
        CheckResult(
            check="terminal_state_valid",
            passed=terminal_valid or trace.terminal_error is not None,
            severity="error",
            observed={"has_final_answer": bool(final_answer), "has_candidates": bool(candidates)},
        )
    )

    results.append(
        CheckResult(
            check="no_node_errors",
            passed=trace.terminal_error is None and all(v.status == "ok" for v in trace.visits),
            severity="error",
            observed={
                "terminal_error": trace.terminal_error.type if trace.terminal_error else None,
                "failed_visits": [v.node for v in trace.visits if v.status == "error"],
            },
        )
    )

    results.append(
        CheckResult(
            check="no_recursion_limit_hit",
            passed=trace.terminal_error is None or trace.terminal_error.type != "GraphRecursionError",
            severity="error",
        )
    )

    return results
