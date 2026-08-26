"""Общий протокол для LLM-judge метрик (faithfulness/coverage/answer_relevancy) —
в отличие от evaluation/checks/ (детерминированные, бесплатные), эти стоят реальных
LLM-вызовов и оцениваются числом, а не pass/fail.

status различает ТРИ разных случая, которые в прежней реализации (modules/ragas_eval.py)
были неразличимы — оба возвращали None:
  - "ok"    — посчитано, score осмыслен.
  - "na"    — метрика неприменима или не смогла посчитаться по содержательной причине
              (нет контекста, LLM не извлекла ни одного утверждения/тезиса, ответ пуст) —
              НИКОГДА не приводится к 0.0, чтобы не выглядеть как «плохой результат».
  - "error" — судья упал технически (сеть, невалидный JSON даже после repair).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

Status = Literal["ok", "na", "error"]


@dataclass(frozen=True)
class MetricResult:
    metric: str
    status: Status
    score: Optional[float] = None
    na_reason: Optional[str] = None
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)  # numerator/denominator и т.п. — для отчёта

    @staticmethod
    def ok(metric: str, score: float, **detail: Any) -> "MetricResult":
        return MetricResult(metric=metric, status="ok", score=score, detail=detail)

    @staticmethod
    def na(metric: str, reason: str, **detail: Any) -> "MetricResult":
        return MetricResult(metric=metric, status="na", na_reason=reason, detail=detail)

    @staticmethod
    def error_result(metric: str, message: str) -> "MetricResult":
        return MetricResult(metric=metric, status="error", error=message)
