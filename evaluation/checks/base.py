"""Общий протокол детерминированных проверок узла — дешёвых (без LLM), pass/fail,
запускаемых на КАЖДОМ визите узла. Метрики (evaluation/metrics/) — отдельная, платная
категория поверх judge LLM; проверки здесь её не используют и не заменяют.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from evaluation.dataset.case import EvalCase
from evaluation.tracing.trace import NodeVisit
from modules.config import AppConfig, NodeGenerationConfig

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class CheckResult:
    check: str
    passed: bool
    severity: Severity
    observed: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class CheckContext:
    """То, что проверкам нужно знать о прогоне, но чего нет в самом NodeVisit — в первую
    очередь ТОТ ЖЕ tokenizer, что использовал ArticleProcessor при прогоне (иначе
    tokens_within_bounds считал бы по другой токенизации, чем сам агент — числа не совпали
    бы с реальным поведением) и границы min/max_chunk_tokens из AppConfig."""

    tokenizer: Any  # объект с .encode(text) -> Sized, как modules.processing.ArticleProcessor ждёт
    app_config: AppConfig
    node_gen: NodeGenerationConfig


# check(visit, case, context) -> список результатов (обычно один, но, например,
# tokens_within_bounds логически одна проверка с богатым observed — оставляем List на будущее).
NodeCheckFn = Callable[[NodeVisit, EvalCase, CheckContext], List[CheckResult]]
