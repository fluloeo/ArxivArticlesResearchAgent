import re
from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit

from .base import CheckContext, CheckResult

_PDF_URL_RE = re.compile(r"^https?://.*arxiv\.org/pdf/")


def check_fetch_fulltext(visit: NodeVisit, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    results: List[CheckResult] = []
    delta = visit.output_delta

    fetch_failed_events = [e for e in visit.log_events if e.event == "fetch_failed"]
    succeeded = bool(delta.get("raw_sections"))

    results.append(
        CheckResult(
            check="fetch_succeeded",
            passed=succeeded,
            severity="error",
            observed={"has_sections": succeeded, "fetch_failed_logged": bool(fetch_failed_events)},
        )
    )

    if not succeeded:
        return results  # остальные проверки бессмысленны без текста

    sections = delta["raw_sections"]
    results.append(
        CheckResult(
            check="no_empty_sections",
            passed=all(bool(text.strip()) for text in sections.values()),
            severity="warning",
            observed={"n_sections": len(sections), "n_empty": sum(1 for t in sections.values() if not t.strip())},
        )
    )
    results.append(
        CheckResult(
            check="title_non_empty",
            passed=bool(delta.get("article_title", "").strip()),
            severity="warning",
        )
    )
    pdf_url = delta.get("article_pdf_url", "")
    results.append(
        CheckResult(
            check="pdf_url_wellformed",
            passed=bool(_PDF_URL_RE.match(pdf_url)),
            severity="warning",
            observed={"pdf_url": pdf_url},
        )
    )

    return results
