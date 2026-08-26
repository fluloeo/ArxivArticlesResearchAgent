import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_USER_AGENT = "ArxivArticlesResearchAgent/1.0 (mailto:research-agent@example.com)"
_RETRIES = 2
_RETRY_BACKOFF_SEC = 5.0
# arXiv в условиях использования API просит не чаще одного запроса в 3 секунды.
# search() делает ДВА запроса подряд (ti:"..." + fallback all:...), и без паузы между
# ними API отвечал 429 Too Many Requests, а следом переставал отвечать вовсе — снаружи
# это выглядело как «ReadTimeout» и приводило к ответу «статей по теме не найдено».
_MIN_REQUEST_INTERVAL_SEC = 3.0

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


@dataclass
class ArxivPaperMeta:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str


def extract_arxiv_id(text: str) -> Optional[str]:
    """Достаёт arXiv ID вида 2301.12345 из произвольного текста/URL пользователя."""
    match = _ARXIV_ID_RE.search(text)
    return match.group(1) if match else None


class ArxivSearchClient:
    """Поиск кандидатов по arXiv Atom API — замена лексической частью того,
    что раньше давал retrieval по LanceDB. Без стороннего пакета `arxiv`,
    только `requests` + stdlib XML."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        # Session, а не голый requests.get: search() делает до двух запросов подряд
        # (ti:"..." и fallback all:...), и без пула соединений каждый из них платил
        # за отдельный TCP+TLS-хендшейк к export.arxiv.org.
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._rate_limit_lock = threading.Lock()
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        """Выдерживает минимальный интервал между обращениями к arXiv API.
        Лок держится и на время сна намеренно: это и есть сериализация запросов,
        нужная чтобы не превышать лимит при параллельных gRPC-вызовах."""
        with self._rate_limit_lock:
            wait = _MIN_REQUEST_INTERVAL_SEC - (time.monotonic() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()

    def _get(self, params: Dict[str, Any]) -> Optional[str]:
        """GET к arXiv API с троттлингом и ретраями. Публичный API arXiv отдаёт 429/таймауты
        под нагрузкой; без ретрая единичный сбой выглядел для агента как «ничего не найдено»,
        и он честно отвечал пользователю, что статей по теме нет."""
        for attempt in range(1, _RETRIES + 1):
            self._throttle()
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
                logger.warning("arXiv API request failed (attempt %d/%d), retrying: %s", attempt, _RETRIES, e)
                time.sleep(_RETRY_BACKOFF_SEC)
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

    def _run_query(self, search_query: str, max_results: int) -> List[ArxivPaperMeta]:
        xml_text = self._get(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        return self._parse_feed(xml_text) if xml_text else []

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
            arxiv_id = extract_arxiv_id(raw_id) or raw_id.rsplit("/", 1)[-1]
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
