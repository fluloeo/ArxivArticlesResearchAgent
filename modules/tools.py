import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .article_store import ArticleStore
from .arxiv_source.search import ArxivSearchClient

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    content: str
    ok: bool = True
    sources: List[str] = field(default_factory=list)
    """Список "{arxiv_id}: {title}" статей, которые этот вызов затронул — используется
    research_step_node для формирования итогового списка источников ответа."""


class ArxivToolkit:
    """Инструменты, которые research-агент вызывает через structured function calling:
    `search_arxiv` — найти статьи по теме, `get_fulltext` — получить (усечённый) текст статьи.

    Переиспользуется и в research-цикле, и при подборе статьи для суммаризации, так что
    сама реализация поиска/чтения статьи существует в одном месте.
    """

    def __init__(
        self,
        search_client: ArxivSearchClient,
        article_store: ArticleStore,
        excerpt_chars: int = 4000,
        max_candidates: int = 5,
    ):
        self.search_client = search_client
        self.article_store = article_store
        self.excerpt_chars = excerpt_chars
        self.max_candidates = max_candidates

    def search_arxiv(self, query: str) -> ToolResult:
        candidates = self.search_client.search(query, max_results=self.max_candidates)
        if not candidates:
            return ToolResult(name="search_arxiv", args={"query": query}, content="Ничего не найдено.", ok=False)

        lines = [f"- {c.arxiv_id}: {c.title}\n  {c.abstract[:300]}" for c in candidates]
        sources = [f"{c.arxiv_id}: {c.title}" for c in candidates]
        return ToolResult(name="search_arxiv", args={"query": query}, content="\n".join(lines), sources=sources)

    def get_fulltext(self, article_id: str) -> ToolResult:
        record = self.article_store.get(article_id)
        if record is None:
            return ToolResult(
                name="get_fulltext",
                args={"article_id": article_id},
                content=f"Не удалось получить текст статьи {article_id}.",
                ok=False,
            )

        body = "\n\n".join(f"## {title}\n{text}" for title, text in record.sections.items())
        content = f"# {record.title} ({record.article_id})\n\n{body[: self.excerpt_chars]}"
        return ToolResult(
            name="get_fulltext",
            args={"article_id": article_id},
            content=content,
            sources=[f"{record.article_id}: {record.title}"],
        )

    def dispatch(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "search_arxiv":
                return self.search_arxiv(str(args.get("query", "")))
            if name == "get_fulltext":
                return self.get_fulltext(str(args.get("article_id", "")))
        except Exception:
            logger.exception("Tool %s failed with args=%r", name, args)
            return ToolResult(name=name, args=args, content=f"Инструмент {name} завершился с ошибкой.", ok=False)

        return ToolResult(name=name, args=args, content=f"Неизвестный инструмент: {name}", ok=False)
