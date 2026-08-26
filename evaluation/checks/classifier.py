from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit

from .base import CheckContext, CheckResult


def check_classifier(visit: NodeVisit, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    results = []
    intent = visit.output_delta.get("intent")

    results.append(
        CheckResult(
            check="intent_in_enum",
            passed=intent in ("summarize", "research", "other"),
            severity="error",
            observed={"intent": intent},
            expected={"one_of": ["summarize", "research", "other"]},
        )
    )

    fallback_events = [e for e in visit.log_events if e.event == "structured_output_fallback"]
    results.append(
        CheckResult(
            check="no_structured_fallback",
            passed=not fallback_events,
            severity="warning",
            observed={"fallback_count": len(fallback_events)},
            message="classifier скатился в safe-default (обычно intent='other') из-за невалидного JSON"
            if fallback_events
            else "",
        )
    )

    if case.expected_intent is not None:
        results.append(
            CheckResult(
                check="intent_matches_expected",
                passed=intent == case.expected_intent,
                severity="error",
                observed={"intent": intent},
                expected={"intent": case.expected_intent},
            )
        )

    return results
