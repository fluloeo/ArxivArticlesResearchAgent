"""Связывает evaluation/metrics/ (faithfulness/coverage/answer_relevancy поверх Judge) с
остальным харнессом: по объявлению `Suite.metrics` (node -> scope -> [metric names],
evaluation/suites/*.yaml) считает метрики по GraphTrace и пишет их в metrics.jsonl через
RunWriter.write_metric.

Отдельный модуль, а не часть runner.py — runner.py собирается на шаге 3 плана ДО судьи
(детерминированные checks стоят ноль LLM-вызовов и должны работать без Judge вовсе);
MetricsRunner — опциональный слой поверх него, включаемый только если вызывающий код
(evaluation/cli.py) передал Judge (т.е. пользователь указал --judge-model).

Источник контекста/кандидата — `trace.final_state` (свёрнутое состояние), а не отдельные
NodeVisit: article_chunks/debug_data/final_answer к моменту завершения графа уже слиты
воедино, и это ровно то, что видит конечный пользователь.
"""
import logging
from typing import Any, Dict, List, Optional

from evaluation.dataset.case import EvalCase, Suite
from evaluation.metrics.answer_relevancy import compute_answer_relevancy
from evaluation.metrics.applicability import filter_requested, is_applicable
from evaluation.metrics.base import MetricResult
from evaluation.metrics.cache import JudgeCache
from evaluation.metrics.coverage import compute_coverage, compute_factual_f1
from evaluation.metrics.faithfulness import compute_faithfulness
from evaluation.metrics.judge import Judge
from evaluation.runlog.run_writer import RunWriter
from evaluation.tracing.trace import GraphTrace

logger = logging.getLogger(__name__)


def _article_full_text(chunks: Dict[str, Any]) -> str:
    return "\n\n".join(c["main_text"] for c in chunks.values())


def _evidence_text(evidence: List[Dict[str, Any]]) -> str:
    return "\n\n".join(e.get("content", "") for e in evidence)


def _result_row(node: str, scope: str, result: MetricResult, chunk: Optional[str] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "node": node,
        "scope": scope,
        "metric": result.metric,
        "status": result.status,
        "score": result.score,
        "na_reason": result.na_reason,
        "error": result.error,
        "detail": result.detail,
    }
    if chunk is not None:
        row["chunk"] = chunk
    return row


class MetricsRunner:
    def __init__(self, judge: Judge, cache: JudgeCache, judge_model_name: str, embed_model: Any = None):
        self.judge = judge
        self.cache = cache
        self.judge_model_name = judge_model_name
        self.embed_model = embed_model

    def run_case(self, trace: GraphTrace, case: EvalCase, suite: Suite, writer: RunWriter) -> None:
        for node, scopes in (suite.metrics or {}).items():
            for scope, requested in scopes.items():
                allowed, rejected = filter_requested(node, scope, requested)
                for metric in rejected:
                    writer.write_metric(
                        case.case_id,
                        _result_row(node, scope, MetricResult.na(metric, "not_applicable_for_node")),
                    )
                if node == "map_reduce_summarize":
                    self._run_summarize_scope(trace, case, scope, allowed, writer)
                elif node == "research_step":
                    self._run_research_scope(trace, case, scope, allowed, writer)
                else:
                    logger.warning("MetricsRunner: неизвестный узел %s в Suite.metrics — пропущен", node)

        # coverage терминального research-ответа условна на наличии эталона (см. docstring
        # плана §3: "coverage здесь только если у кейса есть reference_answer — решает
        # runner") — считаем её, даже если сьют явно не запросил её в YAML, при условии что
        # applicability вообще её разрешает для этого узла/среза.
        research_declared = (suite.metrics or {}).get("research_step", {})
        if case.reference_answer and "coverage" not in research_declared.get("terminal", []):
            if is_applicable("research_step", "terminal", "coverage"):
                self._run_research_scope(trace, case, "terminal", ["coverage"], writer)

    # ------------------------------------------------------------------ summarization

    def _run_summarize_scope(
        self, trace: GraphTrace, case: EvalCase, scope: str, metrics: List[str], writer: RunWriter
    ) -> None:
        if not metrics:
            return
        state = trace.final_state
        chunks: Dict[str, Any] = state.get("article_chunks") or {}
        map_summaries: Dict[str, str] = state.get("debug_data") or {}
        final_answer: str = state.get("final_answer") or ""
        if not chunks or not final_answer:
            for metric in metrics:
                writer.write_metric(case.case_id, _result_row("map_reduce_summarize", scope, MetricResult.na(metric, "no_summarize_output")))
            return

        if scope == "map_stage":
            self._run_per_chunk(chunks, map_summaries, metrics, case, writer)
        elif scope == "reduce_stage":
            context = "\n\n".join(map_summaries.values())
            self._score_pair("map_reduce_summarize", scope, metrics, case, writer, context=context, candidate=final_answer)
        elif scope == "end_to_end":
            context = _article_full_text(chunks)
            self._score_pair("map_reduce_summarize", scope, metrics, case, writer, context=context, candidate=final_answer)
        elif scope == "vs_reference":
            if not case.reference_summary:
                for metric in metrics:
                    writer.write_metric(case.case_id, _result_row("map_reduce_summarize", scope, MetricResult.na(metric, "no_reference_summary")))
                return
            self._score_pair(
                "map_reduce_summarize", scope, metrics, case, writer,
                context=case.reference_summary, candidate=final_answer,
            )
        else:
            logger.warning("MetricsRunner: неизвестный scope=%s для map_reduce_summarize", scope)

    def _run_per_chunk(
        self, chunks: Dict[str, Any], map_summaries: Dict[str, str], metrics: List[str], case: EvalCase, writer: RunWriter
    ) -> None:
        per_metric_scores: Dict[str, List[float]] = {m: [] for m in metrics}
        for title, chunk in chunks.items():
            candidate = map_summaries.get(title, "")
            context = chunk.get("main_text", "")
            results = self._compute_pair(metrics, context=context, candidate=candidate)
            for result in results:
                writer.write_metric(case.case_id, _result_row("map_reduce_summarize", "map_stage", result, chunk=title))
                if result.status == "ok":
                    per_metric_scores[result.metric].append(result.score)
        for metric, scores in per_metric_scores.items():
            if scores:
                writer.write_metric(
                    case.case_id,
                    _result_row(
                        "map_reduce_summarize", "map_stage",
                        MetricResult.ok(metric, sum(scores) / len(scores), n_chunks=len(scores), aggregate=True),
                    ),
                )

    def _score_pair(
        self, node: str, scope: str, metrics: List[str], case: EvalCase, writer: RunWriter, context: str, candidate: str
    ) -> None:
        results = {r.metric: r for r in self._compute_pair(metrics, context=context, candidate=candidate)}
        for result in results.values():
            writer.write_metric(case.case_id, _result_row(node, scope, result))
        if results.get("faithfulness") and results.get("coverage"):
            f, c = results["faithfulness"], results["coverage"]
            if f.status == "ok" and c.status == "ok":
                f1 = compute_factual_f1(f.score, c.score)
                writer.write_metric(case.case_id, _result_row(node, scope, MetricResult.ok("factual_f1", f1)))

    def _compute_pair(self, metrics: List[str], context: str, candidate: str) -> List[MetricResult]:
        out = []
        if "faithfulness" in metrics:
            out.append(compute_faithfulness(self.judge, candidate, context))
        if "coverage" in metrics:
            out.append(compute_coverage(self.judge, self.cache, self.judge_model_name, context, candidate))
        return out

    # ------------------------------------------------------------------ research

    def _run_research_scope(
        self, trace: GraphTrace, case: EvalCase, scope: str, metrics: List[str], writer: RunWriter
    ) -> None:
        if not metrics or scope != "terminal":
            return
        state = trace.final_state
        final_answer: str = state.get("final_answer") or ""
        evidence = state.get("evidence") or []
        if not final_answer:
            for metric in metrics:
                writer.write_metric(case.case_id, _result_row("research_step", scope, MetricResult.na(metric, "no_final_answer")))
            return

        context = _evidence_text(evidence)
        if "faithfulness" in metrics:
            result = compute_faithfulness(self.judge, final_answer, context)
            writer.write_metric(case.case_id, _result_row("research_step", scope, result))
        if "answer_relevancy" in metrics:
            if self.embed_model is None:
                writer.write_metric(case.case_id, _result_row("research_step", scope, MetricResult.na("answer_relevancy", "no_embed_model")))
            else:
                result = compute_answer_relevancy(self.judge, self.embed_model, case.query, final_answer)
                writer.write_metric(case.case_id, _result_row("research_step", scope, result))
        if "coverage" in metrics:
            if case.reference_answer:
                result = compute_coverage(self.judge, self.cache, self.judge_model_name, case.reference_answer, final_answer)
            else:
                result = MetricResult.na("coverage", "no_reference_answer")
            writer.write_metric(case.case_id, _result_row("research_step", scope, result))
