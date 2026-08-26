"""Structured-output схемы для LLM-судьи. ClaimExtraction/ClaimVerdict/GeneratedQuestions
перенесены из modules/schemas.py verbatim (это была рабочая, проверенная живьём реализация
faithfulness/answer_relevancy до выноса метрик из инференс-пути — поведение не меняется).
KeyPoint/KeyPoints/PointVerdict — новые, для coverage (см. coverage.py).
"""
from typing import List, Literal

from pydantic import BaseModel, Field


class ClaimExtraction(BaseModel):
    """Faithfulness, шаг 1: разложение ответа на атомарные проверяемые утверждения."""

    claims: List[str] = Field(default_factory=list, description="Список атомарных фактических утверждений")


class ClaimVerdict(BaseModel):
    """Faithfulness, шаг 2: подтверждается ли конкретное утверждение контекстом.

    Одно поле намеренно: раньше здесь было ещё обязательное `claim`, и модель должна была
    дословно повторить проверяемое утверждение в JSON — на ограниченном токен-бюджете
    длинное утверждение обрезалось на полуслове, JSON становился невалиден, и после repair
    срабатывал safe default `supported=False`, то есть faithfulness систематически
    занижался. Соответствие вердикта утверждению задаётся порядком: verdicts[i] относится
    к claims[i] — это гарантирует sequential вызов generate_structured() (тот же порядок
    conversations -> тот же порядок результатов).
    """

    supported: bool = Field(description="True, если утверждение можно вывести из контекста")


class GeneratedQuestions(BaseModel):
    """Answer Relevancy: гипотетические вопросы, на которые ответ был бы хорошим ответом."""

    questions: List[str] = Field(default_factory=list)
    noncommittal: bool = Field(default=False, description="True, если ответ уклончивый/неполный/не по существу")


class KeyPoint(BaseModel):
    """Coverage, шаг 1: один ключевой тезис исходного текста."""

    text: str
    importance: Literal["core", "supporting"] = Field(
        description="'core' — центральный результат/вывод статьи; 'supporting' — вспомогательная деталь"
    )


class KeyPoints(BaseModel):
    points: List[KeyPoint] = Field(default_factory=list)


class PointVerdict(BaseModel):
    """Coverage, шаг 2: отражён ли конкретный тезис источника в кандидате (обзоре).
    Одно поле — по той же причине, что и ClaimVerdict."""

    covered: bool = Field(description="True, если тезис присутствует (пусть и перефразированный) в кандидате")
