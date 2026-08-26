from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit

from .base import CheckContext, CheckResult


def check_resolve_target_article(visit: NodeVisit, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    """resolve_target_article_node (modules/agent.py) возвращает РОВНО один из трёх
    исходов: target_article_id (явный ID в запросе), candidates (поиск), final_answer
    (ничего не найдено). Смешение или отсутствие всех трёх — баг в узле."""
    results: List[CheckResult] = []
    delta = visit.output_delta

    has_id = "target_article_id" in delta
    has_candidates = "candidates" in delta
    has_final = "final_answer" in delta
    outcomes = sum([has_id, has_candidates, has_final])

    results.append(
        CheckResult(
            check="exactly_one_outcome",
            passed=outcomes == 1,
            severity="error",
            observed={"target_article_id": has_id, "candidates": has_candidates, "final_answer": has_final},
        )
    )

    if has_candidates:
        candidates = delta["candidates"]
        valid_schema = all(
            isinstance(c, dict) and c.get("arxiv_id") and c.get("title") is not None and c.get("abstract") is not None
            for c in candidates
        )
        results.append(
            CheckResult(
                check="candidate_schema_valid",
                passed=valid_schema,
                severity="error",
                observed={"n_candidates": len(candidates)},
            )
        )
        ids = [c.get("arxiv_id") for c in candidates]
        results.append(
            CheckResult(
                check="no_duplicate_candidate_ids",
                passed=len(ids) == len(set(ids)),
                severity="warning",
                observed={"ids": ids},
            )
        )

    if case.expects_explicit_id is not None:
        results.append(
            CheckResult(
                check="explicit_id_extracted_when_expected",
                passed=has_id == case.expects_explicit_id,
                severity="error",
                observed={"explicit_id_found": has_id},
                expected={"explicit_id_found": case.expects_explicit_id},
            )
        )

    if case.expected_article_id is not None:
        if has_id:
            passed = delta["target_article_id"] == case.expected_article_id
            observed = {"target_article_id": delta["target_article_id"]}
        elif has_candidates:
            found_ids = [c.get("arxiv_id") for c in delta["candidates"]]
            passed = case.expected_article_id in found_ids
            # rank — 1-based позиция для MRR (evaluation/reporting агрегирует 1/rank);
            # None, если не найдена вовсе (MRR-вклад 0, а не деление на None).
            rank = found_ids.index(case.expected_article_id) + 1 if passed else None
            observed = {"candidate_ids": found_ids, "rank": rank}
        else:
            passed, observed = False, {"outcome": "final_answer (nothing found)"}
        results.append(
            CheckResult(
                check="gold_id_reachable",
                passed=passed,
                severity="error",
                observed=observed,
                expected={"article_id": case.expected_article_id},
            )
        )

    return results
