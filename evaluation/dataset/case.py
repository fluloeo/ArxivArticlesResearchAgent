"""EvalCase — один прогоняемый кейс харнесса. Общая форма для всех сьютов
(classifier_intents, search_recall, summarization, research_qa) — разные сьюты просто
заполняют разные необязательные поля; runner.py решает, что именно проверять, по тому,
какие поля непусты (см. evaluation/checks/registry.py).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str

    # ожидания — заполняются в зависимости от сьюта, каждое опционально
    expected_intent: Optional[str] = None
    expected_article_id: Optional[str] = None
    expects_explicit_id: Optional[bool] = None
    expected_path: Optional[List[str]] = None
    expected_sources: Optional[List[str]] = None
    reference_summary: Optional[str] = None  # для vs_reference-разреза суммаризации
    reference_answer: Optional[str] = None  # для coverage терминального research-ответа

    # управление прогоном конкретного кейса
    entry: str = "app"  # "app" | "summarize_app"
    stop_after: Optional[str] = None
    target_article_id: Optional[str] = None  # прямой вход для summarize_app
    max_iterations: Optional[int] = None

    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def initial_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"query": self.query}
        if self.target_article_id is not None:
            state["target_article_id"] = self.target_article_id
        return state


@dataclass(frozen=True)
class Suite:
    name: str
    entry: str
    metrics: Dict[str, Dict[str, List[str]]]  # node -> scope -> [metric names]
    cases: List[EvalCase]
    stop_after: Optional[str] = None
    description: str = ""
