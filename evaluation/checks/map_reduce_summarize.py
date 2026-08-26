from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit

from .base import CheckContext, CheckResult

_MIN_FINAL_ANSWER_CHARS = 200


def check_map_reduce_summarize(visit: NodeVisit, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    """Только детерминированные проверки — faithfulness/coverage считает
    evaluation/metrics/ отдельно поверх этих же visit.input_state/output_delta
    (см. runner.py, разрезы map_stage/reduce_stage/end_to_end/vs_reference)."""
    results: List[CheckResult] = []
    delta = visit.output_delta
    chunks = visit.input_state.get("article_chunks") or {}
    debug_data = delta.get("debug_data") or {}
    final_answer = delta.get("final_answer", "")

    results.append(
        CheckResult(
            check="no_chunk_dropped",
            passed=len(debug_data) == len(chunks),
            severity="error",
            observed={"n_chunks": len(chunks), "n_map_summaries": len(debug_data)},
        )
    )

    empty_summaries = [title for title, summary in debug_data.items() if not summary.strip()]
    results.append(
        CheckResult(
            check="no_empty_map_summary",
            passed=not empty_summaries,
            severity="error",
            observed={"empty_chunk_titles": empty_summaries[:5], "n_empty": len(empty_summaries)},
        )
    )

    # modules/agent.py::map_reduce_summarize_node — header это "# {title}\n🔗 [PDF](...)\n\n"
    results.append(
        CheckResult(
            check="header_present",
            passed=final_answer.startswith("# ") and "🔗 [PDF](" in final_answer[:200],
            severity="warning",
            observed={"final_answer_head": final_answer[:80]},
        )
    )

    results.append(
        CheckResult(
            check="final_answer_min_length",
            passed=len(final_answer) >= _MIN_FINAL_ANSWER_CHARS,
            severity="error",
            observed={"length": len(final_answer)},
            expected={"min_length": _MIN_FINAL_ANSWER_CHARS},
        )
    )

    if chunks and debug_data:
        ratios = []
        for title, chunk in chunks.items():
            summary = debug_data.get(title, "")
            main_len = len(chunk.get("main_text", ""))
            if main_len:
                ratios.append(len(summary) / main_len)
        n_longer = sum(1 for r in ratios if r >= 1.0)
        results.append(
            CheckResult(
                check="map_summary_shorter_than_chunk",
                passed=n_longer == 0,
                severity="warning",
                observed={"n_summaries_not_shorter": n_longer, "n_total": len(ratios)},
            )
        )

    # n_llm_calls == n_chunks + 1 (map на каждый чанк + один reduce) — если llm_calls не
    # писался (RecordingProvider не подключён, --record-llm-io выключен), проверку пропускаем.
    if visit.llm_calls:
        expected_calls = sum(c.n_conversations for c in visit.llm_calls)
        results.append(
            CheckResult(
                check="llm_call_count_matches_chunks",
                passed=expected_calls == len(chunks) + 1,
                severity="warning",
                observed={"n_llm_conversations": expected_calls},
                expected={"n_chunks_plus_reduce": len(chunks) + 1},
            )
        )

    return results
