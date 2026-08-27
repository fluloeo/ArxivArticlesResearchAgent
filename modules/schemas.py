from typing import Literal, Optional

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
