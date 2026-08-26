import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .article_store import ArticleStore
from .arxiv_source.identifiers import extract_arxiv_id
from .arxiv_source.search import ArxivPaperMeta, ArxivSearchClient
from .query_rewriter import QueryRewriter, build_arxiv_query_ladder

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
        rewriter: Optional[QueryRewriter] = None,
    ):
        self.search_client = search_client
        self.article_store = article_store
        self.excerpt_chars = excerpt_chars
        self.max_candidates = max_candidates
        self.rewriter = rewriter

    def find_candidates(self, query: str) -> List[ArxivPaperMeta]:
        """Единая точка поиска статей — используется и `search_arxiv` (research-инструмент),
        и `resolve_target_article_node` (summarize-ветка), так что "вставили id/ссылку —
        не искать, взять статью напрямую" гарантируется в одном месте для обоих путей,
        а не дублируется/расходится по коду.

        Если query сам оказывается arXiv ID или ссылкой на конкретную статью — НИКАКОГО
        поиска (и тем более вызова rewriter'а с LLM) не происходит: сразу `get_by_id`.
        Иначе — если есть rewriter, лестница запросов возрастающей широты
        (modules.query_rewriter.build_arxiv_query_ladder); без rewriter'а — прежняя
        эвристика `ArxivSearchClient.search()` (ti:/all:).
        """
        explicit_id = extract_arxiv_id(query)
        if explicit_id:
            logger.info("find_candidates: query is itself an arXiv id/url (%s) — skipping search", explicit_id)
            meta = self.search_client.get_by_id(explicit_id)
            return [meta] if meta else []

        if self.rewriter is None:
            return self.search_client.search(query, max_results=self.max_candidates)

        plan = self.rewriter.rewrite(query)
        for level, search_query in build_arxiv_query_ladder(plan, fallback_query=query):
            results = self.search_client.run_raw_query(search_query, max_results=self.max_candidates)
            if results:
                logger.info("find_candidates: level=%s query=%r -> %d results", level, search_query, len(results))
                return results
        return []

    def search_arxiv(self, query: str) -> ToolResult:
        candidates = self.find_candidates(query)
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
