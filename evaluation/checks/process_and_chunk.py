from typing import List

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit

from .base import CheckContext, CheckResult


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def check_process_and_chunk(visit: NodeVisit, case: EvalCase, context: CheckContext) -> List[CheckResult]:
    """modules/agent.py::process_and_chunk_node -> modules.processing.ArticleProcessor.
    article_chunks: {title: {past_overlap, main_text, future_overlap}}."""
    results: List[CheckResult] = []
    chunks = visit.output_delta.get("article_chunks")

    results.append(CheckResult(check="chunks_non_empty", passed=bool(chunks), severity="error"))
    if not chunks:
        return results

    overlap_len = context.app_config.chunk_overlap_chars
    min_tokens = context.app_config.min_chunk_tokens
    max_tokens = context.app_config.max_chunk_tokens
    items = list(chunks.items())

    token_counts = [len(context.tokenizer.encode(v["main_text"])) for _, v in items]
    n_below_min = sum(1 for t in token_counts if t < min_tokens)
    n_above_max = sum(1 for t in token_counts if t > max_tokens)
    # Единственная секция и последний осколок после разбиения — легитимные исключения из
    # min_tokens (см. ArticleProcessor._merge_small_chunks: `len(processed_chunks) == 1`
    # выходит из слияния сразу); не считаем это нарушением, если чанков всего 1.
    results.append(
        CheckResult(
            check="tokens_within_bounds",
            passed=n_above_max == 0 and (n_below_min == 0 or len(items) == 1),
            severity="error" if n_above_max else "warning",
            observed={
                "n_chunks": len(items),
                "n_below_min": n_below_min,
                "n_above_max": n_above_max,
                "p50": _percentile(token_counts, 0.5),
                "p90": _percentile(token_counts, 0.9),
                "p99": _percentile(token_counts, 0.99),
                "min": min(token_counts),
                "max": max(token_counts),
            },
            expected={"min_tokens": min_tokens, "max_tokens": max_tokens},
        )
    )

    # overlap_correct: past_overlap чанка i должен быть ТОЧНО хвостом main_text чанка i-1
    # длины overlap_len (см. ArticleProcessor.create_overlap_dict) — не приблизительно.
    overlap_mismatches = []
    for i in range(1, len(items)):
        prev_main = items[i - 1][1]["main_text"]
        expected_past = prev_main[-overlap_len:] if prev_main else ""
        actual_past = items[i][1]["past_overlap"]
        if actual_past != expected_past:
            overlap_mismatches.append(items[i][0])
    for i in range(len(items) - 1):
        next_main = items[i + 1][1]["main_text"]
        expected_future = next_main[:overlap_len] if next_main else ""
        actual_future = items[i][1]["future_overlap"]
        if actual_future != expected_future:
            overlap_mismatches.append(items[i][0])
    results.append(
        CheckResult(
            check="overlap_correct",
            passed=not overlap_mismatches,
            severity="error",
            observed={"mismatched_chunks": overlap_mismatches[:5], "n_mismatched": len(overlap_mismatches)},
        )
    )

    first_past = items[0][1]["past_overlap"]
    last_future = items[-1][1]["future_overlap"]
    results.append(
        CheckResult(
            check="boundary_overlaps_empty",
            passed=first_past == "" and last_future == "",
            severity="warning",
            observed={"first_past_overlap": first_past[:30], "last_future_overlap": last_future[:30]},
        )
    )

    raw_sections = visit.input_state.get("raw_sections", {})
    total_raw = sum(len(t) for t in raw_sections.values())
    total_main = sum(len(v["main_text"]) for _, v in items)
    retention = (total_main / total_raw) if total_raw else 1.0
    results.append(
        CheckResult(
            check="text_retention_ratio",
            passed=0.9 <= retention <= 1.15,  # некоторый рост допустим из-за нормализации пробелов в сплиттере
            severity="warning",
            observed={"retention_ratio": round(retention, 4), "total_raw_chars": total_raw, "total_main_chars": total_main},
        )
    )

    return results
