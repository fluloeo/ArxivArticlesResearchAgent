import hashlib
import json
import logging
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests

from .identifiers import extract_arxiv_id, extract_id_from_atom_url  # noqa: F401 — extract_arxiv_id
# сюда больше не вызывается напрямую (см. _parse_feed ниже), но реэкспортируется для
# обратной совместимости: modules/agent.py исторически импортировал его отсюда, теперь
# переключён на modules.arxiv_source.identifiers напрямую, но другой код мог остаться
# на старом пути импорта.
from .rate_limit import RateLimiter

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_USER_AGENT = "ArxivArticlesResearchAgent/1.0 (mailto:research-agent@example.com)"

# 2 попытки с экспоненциальным backoff — короткий троттлинг-эпизод переживается за
# несколько секунд без превращения в многоминутное ожидание. Раньше было 4 попытки на
# КАЖДЫЙ из до 6 уровней лестницы query_rewriter — при устойчивом (не кратковременном)
# отказе export.arxiv.org (наблюдалось живьём: сам API-хост не отвечает даже на голый
# curl извне приложения, пока обычный arxiv.org отвечает нормально — похоже на IP-level
# пенальти, а не на обычный 429-всплеск) это давало ~70с на уровень и 5-7 минут итого,
# выглядящих как зависший UI. Лестница сама по себе даёт избыточность через РАЗНЫЕ
# запросы — незачем ещё и повторять один и тот же запрос 4 раза. При 429 с заголовком
# Retry-After он в приоритете над расчётным backoff.
_RETRIES = 2
_BACKOFF_BASE_SEC = 2.0
_BACKOFF_MAX_SEC = 10.0
_BACKOFF_JITTER = 0.25

# Кэш результатов поиска — лестница query_rewriter (modules.query_rewriter) на один
# user-facing поиск может сделать до 6 последовательных запросов (title_phrase ->
# and_filtered -> and_unfiltered -> or_broadened -> all_terms -> raw_fallback), и без
# кэша каждый повтор того же search_query (тот же research-цикл, повторный запрос
# пользователя, прогон харнесса) платит за них снова. TTL умеренный — arXiv индексирует
# новые статьи не мгновенно, потеря нескольких часов свежести не критична здесь.
_SEARCH_CACHE_TTL_SEC = 6 * 3600

_MAX_TITLE_WORDS = 12
_QUESTION_WORDS = {
    "what", "how", "why", "which", "who", "when", "where", "is", "are", "does", "do", "can", "explain",
    "что", "как", "почему", "зачем", "какой", "какая", "какие", "чем", "где", "когда", "кто", "объясни",
    "расскажи", "сравни",
}


def _looks_like_title(query: str) -> bool:
    """Похож ли запрос на название статьи (а не на вопрос).

    Фразовый поиск `ti:"..."` осмыслен только для названия. Для естественного вопроса он
    (а) заведомо ничего не находит и (б) на длинной фразе оказывается дорогим для arXiv:
    живьём такие запросы стабильно ловили 429/таймаут, из-за чего следом отваливался и
    полезный fallback-запрос. Поэтому для вопросов сразу идём в `all:`.
    """
    words = query.split()
    if not words or len(words) > _MAX_TITLE_WORDS:
        return False
    if query.rstrip().endswith("?"):
        return False
    return words[0].strip('«"\'').lower() not in _QUESTION_WORDS


def _retry_after_seconds(exc: requests.RequestException) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class ArxivPaperMeta:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str


class ArxivSearchClient:
    """Поиск кандидатов по arXiv Atom API — замена лексической частью того,
    что раньше давал retrieval по LanceDB. Без стороннего пакета `arxiv`,
    только `requests` + stdlib XML."""

    def __init__(
        self,
        timeout: float = 10.0,
        rate_limiter: Optional[RateLimiter] = None,
        cache_path: Optional[str] = None,
        cache_ttl_sec: float = _SEARCH_CACHE_TTL_SEC,
    ):
        self.timeout = timeout
        # Session, а не голый requests.get: search() делает до двух запросов подряд
        # (ti:"..." и fallback all:...), и без пула соединений каждый из них платил
        # за отдельный TCP+TLS-хендшейк к export.arxiv.org.
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        # Свой RateLimiter, если не передан общий, — сохраняет прежнее поведение при
        # прямом создании клиента (тесты, ноутбуки); в проде modules.bootstrap передаёт
        # ОДИН общий лимитер и сюда, и в fetch_sections (см. modules/arxiv_source/rate_limit.py).
        self._rate_limiter = rate_limiter or RateLimiter()
        self._cache_path = cache_path
        self._cache_ttl_sec = cache_ttl_sec
        if self._cache_path:
            self._init_cache()

    def _init_cache(self) -> None:
        Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS search_cache "
                "(query_hash TEXT PRIMARY KEY, search_query TEXT, results_json TEXT, fetched_at REAL)"
            )

    @staticmethod
    def _cache_key(search_query: str, max_results: int) -> str:
        return hashlib.sha256(f"{search_query}\x00{max_results}".encode("utf-8")).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[List[ArxivPaperMeta]]:
        if not self._cache_path:
            return None
        with sqlite3.connect(self._cache_path) as conn:
            row = conn.execute(
                "SELECT results_json, fetched_at FROM search_cache WHERE query_hash = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        results_json, fetched_at = row
        if time.time() - fetched_at > self._cache_ttl_sec:
            return None
        try:
            return [ArxivPaperMeta(**d) for d in json.loads(results_json)]
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _cache_put(self, cache_key: str, search_query: str, results: List[ArxivPaperMeta]) -> None:
        if not self._cache_path:
            return
        payload = json.dumps([asdict(r) for r in results], ensure_ascii=False)
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO search_cache (query_hash, search_query, results_json, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                (cache_key, search_query, payload, time.time()),
            )
            conn.commit()

    def _get(self, params: Dict[str, Any]) -> Optional[str]:
        """GET к arXiv API с общим троттлингом и ретраями. Публичный API arXiv отдаёт
        429/таймауты под нагрузкой; без ретрая единичный сбой выглядел для агента как
        «ничего не найдено», и он честно отвечал пользователю, что статей по теме нет."""
        for attempt in range(1, _RETRIES + 1):
            self._rate_limiter.wait()
            try:
                response = self._session.get(_ARXIV_API_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                # 4xx кроме 429 — это наша собственная ошибка в запросе (например пустой
                # id_list), повтор её не исправит и только тратит время и лимит запросов.
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    logger.error("arXiv API rejected request (HTTP %d, %s): %s", status, params, e)
                    return None
                if attempt == _RETRIES:
                    logger.error("arXiv API request failed after %d attempts (%s): %s", attempt, params, e)
                    return None
                retry_after = _retry_after_seconds(e)
                if retry_after is not None:
                    backoff = retry_after
                else:
                    backoff = min(_BACKOFF_BASE_SEC * (2 ** (attempt - 1)), _BACKOFF_MAX_SEC)
                    backoff += random.uniform(0, backoff * _BACKOFF_JITTER)
                logger.warning(
                    "arXiv API request failed (attempt %d/%d, status=%s), retrying in %.1fs: %s",
                    attempt, _RETRIES, status, backoff, e,
                )
                time.sleep(backoff)
        return None

    def search(self, query: str, max_results: int = 5) -> List[ArxivPaperMeta]:
        """Если запрос похож на НАЗВАНИЕ статьи («attention is all you need») — сначала
        пробуем фразовый поиск по заголовку `ti:"..."`, который находит именно её; обычный
        `all:query` (OR по словам, без учёта позиции) на таком запросе выдаёт шум из статей,
        где просто встречаются те же слова. Для вопросов (и когда по заголовку пусто) —
        обычный keyword-поиск по всем полям.
        """
        if _looks_like_title(query):
            title_results = self._run_query(f'ti:"{query}"', max_results)
            if title_results:
                return title_results
        return self._run_query(f"all:{query}", max_results)

    def run_raw_query(self, search_query: str, max_results: int = 5) -> List[ArxivPaperMeta]:
        """Публичный вход для уже готовой `search_query`-строки (field-scoped булев запрос:
        `ti:`/`abs:`/`all:`/`au:`/`cat:`/`submittedDate:`) — используется
        modules.query_rewriter/ArxivToolkit.find_candidates для лестницы запросов
        возрастающей широты, минуя эвристику `ti:`/`all:` из search()."""
        return self._run_query(search_query, max_results)

    def _run_query(self, search_query: str, max_results: int) -> List[ArxivPaperMeta]:
        cache_key = self._cache_key(search_query, max_results)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("arXiv search cache hit: %r", search_query)
            return cached

        xml_text = self._get(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        results = self._parse_feed(xml_text) if xml_text else []
        # Кэшируем и пустой результат: та же пустая ступень лестницы запросов иначе бьёт
        # по сети заново на каждый повтор похожего запроса в рамках TTL.
        self._cache_put(cache_key, search_query, results)
        return results

    def get_by_id(self, arxiv_id: str) -> Optional[ArxivPaperMeta]:
        """Точечный lookup метаданных по arXiv ID (для полнотекстового кэша),
        через `id_list` — правильный способ адресной выборки в Atom API arXiv
        (`search_query=id:...` не является валидным полем)."""
        if not arxiv_id.strip():
            return None

        xml_text = self._get({"id_list": arxiv_id, "max_results": 1})
        if not xml_text:
            return None

        results = self._parse_feed(xml_text)
        return results[0] if results else None

    def _parse_feed(self, xml_text: str) -> List[ArxivPaperMeta]:
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            logger.exception("Failed to parse arXiv Atom feed")
            return []

        results: List[ArxivPaperMeta] = []
        for entry in root.findall(f"{_ATOM_NS}entry"):
            raw_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
            # <id> Atom-фида ВСЕГДА "http://arxiv.org/abs/{id}v{n}" — extract_id_from_atom_url
            # знает эту форму точно (см. modules/arxiv_source/identifiers.py). Раньше здесь
            # был `extract_arxiv_id(raw_id) or raw_id.rsplit("/", 1)[-1]`: для старого формата
            # ID (например math/0702019) старая _ARXIV_ID_RE не матчилась вовсе, и срабатывал
            # rsplit-фоллбек — "http://arxiv.org/abs/math/0702019v1".rsplit("/", 1)[-1] даёт
            # "0702019v1": префикс архива теряется, версия остаётся. Испорченный id уходил
            # в ключ SQLite-кэша и в URL PDF. Баг был не гипотетическим — такая статья есть
            # в article_sample.json (см. evaluation/suites/summarization.yaml, sum-math-0702019).
            arxiv_id = extract_id_from_atom_url(raw_id)
            if not arxiv_id:
                continue

            title = " ".join((entry.findtext(f"{_ATOM_NS}title") or "").split())
            abstract = " ".join((entry.findtext(f"{_ATOM_NS}summary") or "").split())

            pdf_url = ""
            for link in entry.findall(f"{_ATOM_NS}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            results.append(ArxivPaperMeta(arxiv_id=arxiv_id, title=title, abstract=abstract, pdf_url=pdf_url))
        return results
