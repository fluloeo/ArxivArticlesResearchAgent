from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ClassifierResult(BaseModel):
    """Результат классификации намерения пользователя."""

    intent: Literal["summarize", "research", "other"] = Field(
        description=(
            "'summarize' — просят обзор/суммаризацию конкретной статьи или темы; "
            "'research' — вопрос, требующий поиска фактов/статей; "
            "'other' — не относится к научной тематике"
        )
    )


class ResearchDecision(BaseModel):
    """Решение агента на одном шаге research-цикла (прошитый function calling)."""

    action: Literal["final_answer", "call_tool"] = Field(
        description="'final_answer' — дать ответ сейчас; 'call_tool' — сначала вызвать инструмент поиска по arXiv"
    )
    tool: Optional[Literal["search_arxiv", "get_fulltext"]] = Field(
        default=None, description="Какой инструмент вызвать, если action='call_tool'"
    )
    tool_args: Optional[dict] = Field(
        default=None,
        description="Аргументы инструмента: {'query': str} для search_arxiv, {'article_id': str} для get_fulltext",
    )
    answer: Optional[str] = Field(default=None, description="Итоговый ответ пользователю, если action='final_answer'")
    confidence: Optional[Literal["low", "medium", "high"]] = Field(
        default=None, description="Насколько уверен агент в ответе"
    )


class ClaimExtraction(BaseModel):
    """RAGAS Faithfulness, шаг 1: разложение ответа на атомарные проверяемые утверждения."""

    claims: List[str] = Field(default_factory=list, description="Список атомарных фактических утверждений")


class ClaimVerdict(BaseModel):
    """RAGAS Faithfulness, шаг 2: подтверждается ли конкретное утверждение контекстом.

    Схема намеренно состоит из одного поля: раньше здесь было ещё обязательное `claim`,
    и модель должна была дословно повторить проверяемое утверждение в JSON. На бюджете
    ragas_verdict (200 токенов) длинное утверждение обрезалось на полуслове -> JSON
    невалиден -> repair -> safe default `supported=False`, то есть faithfulness
    систематически занижался. Соответствие вердикта утверждению и так задаётся порядком:
    verdicts[i] относится к claims[i].
    """

    supported: bool = Field(description="True, если утверждение можно вывести из контекста")


class GeneratedQuestions(BaseModel):
    """RAGAS Answer Relevancy: гипотетические вопросы, на которые ответ был бы хорошим ответом."""

    questions: List[str] = Field(default_factory=list)
    noncommittal: bool = Field(
        default=False, description="True, если ответ уклончивый/неполный/не по существу"
    )
