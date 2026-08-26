"""Офлайн-заменители сетевых коллабораторов ArxivAgent — для сьютов, где корпус уже
известен заранее (в первую очередь summarization по article_sample.json) и поход в сеть
не только не нужен, но и вреден: 429 от arXiv, 3-секундный троттлинг между запросами и
недетерминированность сетевого ответа между прогонами делают сравнение моделей нечестным.

InMemoryArticleStore реализует тот же ABC (modules.article_store.ArticleStore), что и
боевой SqliteArxivArticleStore — ArxivAgent/ArxivToolkit зависят только от интерфейса
`.get(article_id) -> Optional[ArticleRecord]`, подмена безопасна и не требует правок
в modules/*.

FrozenSearchClient — тот же duck-type, что modules.arxiv_source.search.ArxivSearchClient
(.search(query, max_results), .get_by_id(arxiv_id)), backed либо записанными фикстурами
(evaluation/fixtures/arxiv/*.json), либо, для summarization-сьюта, напросто метаданными
из уже загруженного article_sample.json — там реальный поиск не нужен вовсе, так как
кейсы обращаются к статьям по явному id.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from modules.arxiv_source.search import ArxivPaperMeta
from modules.article_store import ArticleRecord, ArticleStore

logger = logging.getLogger(__name__)


class InMemoryArticleStore(ArticleStore):
    """Отдаёт статьи из уже загруженного в память словаря {id: ArticleRecord}
    (обычно — результат evaluation.dataset.assets.load_article_sample). Не создаёт
    файлов, не делает сетевых запросов, полностью детерминирован между прогонами."""

    def __init__(self, records: Dict[str, ArticleRecord]):
        self._records = records

    def get(self, article_id: str) -> Optional[ArticleRecord]:
        return self._records.get(article_id)


class FrozenSearchClient:
    """duck-type ArxivSearchClient, отдающий заранее записанные результаты вместо
    сетевых запросов. Два независимых источника:

    - `fixtures_dir` — записанные JSON-фикстуры вида {query: [ArxivPaperMeta...]},
      для search_recall-сьюта, где важно именно поведение поиска;
    - `article_index` — метаданные напрямую из article_sample.json, для
      summarization-сьюта, где search() не должен вызываться вовсе (кейсы обращаются
      по явному id), а get_by_id() нужен только для заголовка/pdf_url.
    """

    def __init__(
        self,
        fixtures_dir: Optional[Path] = None,
        article_index: Optional[Dict[str, ArticleRecord]] = None,
    ):
        self._fixtures_dir = fixtures_dir
        self._article_index = article_index or {}
        self._fixture_cache: Dict[str, List[ArxivPaperMeta]] = {}

    def search(self, query: str, max_results: int = 5) -> List[ArxivPaperMeta]:
        if self._fixtures_dir is None:
            logger.warning("FrozenSearchClient.search(%r) без fixtures_dir — возвращаю пусто", query)
            return []
        if query not in self._fixture_cache:
            self._fixture_cache[query] = self._load_fixture(query)
        return self._fixture_cache[query][:max_results]

    def run_raw_query(self, search_query: str, max_results: int = 5) -> List[ArxivPaperMeta]:
        """duck-type ArxivSearchClient.run_raw_query — нужен, чтобы ArxivToolkit.rewriter
        (лестница field-scoped запросов, modules.query_rewriter) мог работать в офлайн-
        сьютах без AttributeError, даже если под конкретный search_query фикстур нет
        (тогда честно вернёт пусто, как и search())."""
        return self.search(search_query, max_results=max_results)

    def get_by_id(self, arxiv_id: str) -> Optional[ArxivPaperMeta]:
        record = self._article_index.get(arxiv_id)
        if record is None:
            return None
        return ArxivPaperMeta(arxiv_id=record.article_id, title=record.title, abstract="", pdf_url=record.pdf_url)

    def _load_fixture(self, query: str) -> List[ArxivPaperMeta]:
        # Имя файла — не сырой query (может содержать что угодно, включая '/'), а его
        # позиция в предзаписанном индексе фикстур query -> filename.
        index_path = self._fixtures_dir / "index.json"
        if not index_path.exists():
            logger.warning("Нет индекса фикстур %s — search(%r) вернёт пусто", index_path, query)
            return []
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        filename = index.get(query)
        if filename is None:
            logger.warning("Запрос %r отсутствует в фикстурах %s — вернёт пусто", query, index_path)
            return []
        with open(self._fixtures_dir / filename, "r", encoding="utf-8") as f:
            rows = json.load(f)
        return [ArxivPaperMeta(**row) for row in rows]
