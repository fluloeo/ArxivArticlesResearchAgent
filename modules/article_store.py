import contextlib
import datetime
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from .arxiv_source.fulltext import fetch_sections
from .arxiv_source.rate_limit import RateLimiter
from .arxiv_source.search import ArxivSearchClient

logger = logging.getLogger(__name__)


@dataclass
class ArticleRecord:
    article_id: str
    title: str
    sections: Dict[str, str]
    pdf_url: str


class ArticleStore(ABC):
    """Зеркалит уже существовавший `BaseRetriever` — источник полного текста статьи
    спрятан за интерфейсом так же, как раньше был спрятан только векторный поиск."""

    @abstractmethod
    def get(self, article_id: str) -> Optional[ArticleRecord]:
        ...


class SqliteArxivArticleStore(ArticleStore):
    """Заменяет прямой psycopg2-доступ к утраченной Postgres-таблице `articles`.

    На промахе кэша тянет статью с живого arXiv (modules.arxiv_source) и кладёт
    результат в локальный SQLite, чтобы повторные запросы по той же статье не
    требовали повторного сетевого похода/парсинга PDF.
    """

    def __init__(
        self,
        db_path: str,
        search_client: Optional[ArxivSearchClient] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._search_client = search_client or ArxivSearchClient()
        # Тот же общий лимитер, что у search_client — get_by_id() (поиск метаданных) и
        # fetch_sections() (ar5iv/PDF) ниже вызываются подряд на промахе кэша; без общего
        # лимитера это несогласованный всплеск запросов к arXiv (см.
        # modules/arxiv_source/rate_limit.py).
        self._rate_limiter = rate_limiter
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """`with sqlite3.connect(...) as conn` фиксирует транзакцию, но НЕ закрывает
        соединение — на каждый get() оставался висеть открытый дескриптор до сборки мусора.
        Соединение открывается на вызов (а не на объект) намеренно: store дёргается из
        разных потоков (gRPC-воркеры), а sqlite3-соединения не потокобезопасны."""
        conn = sqlite3.connect(self._db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    pdf_url TEXT,
                    sections_json TEXT,
                    source TEXT,
                    fetched_at TEXT
                )
                """
            )

    def get(self, article_id: str) -> Optional[ArticleRecord]:
        cached = self._read_cache(article_id)
        if cached is not None:
            return cached
        return self._fetch_and_cache(article_id)

    def _read_cache(self, article_id: str) -> Optional[ArticleRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title, pdf_url, sections_json FROM articles WHERE id = ?", (article_id,)
            ).fetchone()
        if not row:
            return None

        title, pdf_url, sections_json = row
        try:
            sections = json.loads(sections_json)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Corrupt cache entry for %s, will refetch", article_id)
            return None
        return ArticleRecord(article_id=article_id, title=title, sections=sections, pdf_url=pdf_url)

    def _fetch_and_cache(self, article_id: str) -> Optional[ArticleRecord]:
        title, pdf_url = self._resolve_metadata(article_id)
        sections, source = fetch_sections(article_id, pdf_url=pdf_url, rate_limiter=self._rate_limiter)
        if not sections:
            logger.error("Could not fetch full text for arXiv:%s (ar5iv and PDF both failed)", article_id)
            return None

        record = ArticleRecord(article_id=article_id, title=title, sections=sections, pdf_url=pdf_url)
        self._write_cache(record, source)
        return record

    def _resolve_metadata(self, article_id: str) -> Tuple[str, str]:
        meta = self._search_client.get_by_id(article_id)
        if meta:
            return meta.title, meta.pdf_url
        return article_id, f"https://arxiv.org/pdf/{article_id}"

    def _write_cache(self, record: ArticleRecord, source: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO articles (id, title, pdf_url, sections_json, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.article_id,
                    record.title,
                    record.pdf_url,
                    json.dumps(record.sections, ensure_ascii=False),
                    source,
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                ),
            )
