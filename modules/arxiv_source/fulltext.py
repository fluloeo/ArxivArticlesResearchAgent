import logging
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz
import requests
from bs4 import BeautifulSoup

from .rate_limit import RateLimiter

logger = logging.getLogger(__name__)

_AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_USER_AGENT = "ArxivArticlesResearchAgent/1.0 (mailto:research-agent@example.com)"
_MIN_AR5IV_CHARS = 500

# Та же схема ретраев, что modules.arxiv_source.search — раньше здесь не было ни одной
# попытки повтора: единичный сбой ar5iv (тот же хост, что и Atom API, — под общим
# троттлингом/капризами arXiv) сразу проваливался в PDF-фоллбек, а сбой PDF-запроса —
# сразу в пустой результат.
_RETRIES = 3
_BACKOFF_BASE_SEC = 2.0
_BACKOFF_MAX_SEC = 20.0
_BACKOFF_JITTER = 0.25

_HEADING_NUMBER_RE = re.compile(r"^\d+(\.\d+)*\.?\s+")
_PDF_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+([A-Z][A-Za-z0-9 ,\-:]{2,60})|([A-Z][A-Z ]{3,40}))$",
    re.MULTILINE,
)


def _clean_title(raw: str) -> str:
    title = " ".join(raw.split())
    return _HEADING_NUMBER_RE.sub("", title).strip() or title


def _parse_ar5iv_html(html: str) -> Dict[str, str]:
    """Разбирает LaTeXML-рендер статьи (ar5iv) на {заголовок_секции: текст}.

    Секция ограничивается диапазоном между текущим h2/h3 и следующим — это
    воспроизводит форму исходного `section_text_new` из утраченной Postgres-таблицы.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup

    sections: Dict[str, List[str]] = {}

    abstract_tag = soup.find(class_="ltx_abstract")
    if abstract_tag:
        text = abstract_tag.get_text(" ", strip=True)
        if text:
            sections["Abstract"] = [text]

    for heading in article.find_all(["h2", "h3"]):
        title = _clean_title(heading.get_text(" ", strip=True))
        if not title:
            continue

        parts: List[str] = []
        for node in heading.find_all_next():
            if node.name in ("h2", "h3"):
                break
            if node.name == "p" or (node.name == "div" and "ltx_para" in (node.get("class") or [])):
                text = node.get_text(" ", strip=True)
                if text:
                    parts.append(text)

        if parts:
            sections.setdefault(title, [])
            sections[title].extend(parts)

    return {title: "\n\n".join(parts) for title, parts in sections.items() if parts}


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


def _get_with_retry(
    url: str, timeout: float, rate_limiter: Optional[RateLimiter], accept_404: bool = False
) -> Optional[requests.Response]:
    """GET с общим rate limiter'ом (тем же, что ArxivSearchClient — см.
    modules/arxiv_source/rate_limit.py) и ретраями на 429/5xx/таймауты. 404 (статья
    существует, но, например, ar5iv её ещё не отрендерил) не ретраится — не наша ошибка,
    но и не временный сбой, следующая попытка даст тот же результат."""
    for attempt in range(1, _RETRIES + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
            if accept_404 and response.status_code == 404:
                return response
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                return None
            if attempt == _RETRIES:
                logger.warning("Request to %s failed after %d attempts: %s", url, attempt, e)
                return None
            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                backoff = retry_after
            else:
                backoff = min(_BACKOFF_BASE_SEC * (2 ** (attempt - 1)), _BACKOFF_MAX_SEC)
                backoff += random.uniform(0, backoff * _BACKOFF_JITTER)
            logger.warning(
                "Request to %s failed (attempt %d/%d, status=%s), retrying in %.1fs", url, attempt, _RETRIES, status, backoff
            )
            time.sleep(backoff)
    return None


def _fetch_pdf_text(pdf_url: str, timeout: float, rate_limiter: Optional[RateLimiter]) -> str:
    response = _get_with_retry(pdf_url, timeout, rate_limiter)
    if response is None:
        return ""
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _split_pdf_sections(text: str) -> Dict[str, str]:
    """Эвристическое деление сырого текста PDF на секции по заголовкам
    ("1 Introduction", "INTRODUCTION" и т.п.). Fallback, если ar5iv недоступен."""
    matches = list(_PDF_HEADING_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return {"Main": stripped} if stripped else {}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        title = (m.group(1) or m.group(2) or "").strip()
        if not title:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        sections[title] = f"{sections[title]}\n\n{body}" if title in sections else body

    stripped = text.strip()
    return sections or ({"Main": stripped} if stripped else {})


def fetch_sections(
    arxiv_id: str,
    pdf_url: Optional[str] = None,
    timeout: float = 20.0,
    rate_limiter: Optional[RateLimiter] = None,
) -> Tuple[Dict[str, str], str]:
    """Достаёт полный текст статьи по arXiv ID и делит его на секции.

    Порядок попыток: ar5iv (LaTeXML HTML, точная структура секций) -> PDF-текст +
    regex-эвристика заголовков (грубее, но работает почти всегда). Возвращает
    (sections, source), где source в {"ar5iv", "pdf", "none"} — используется для логирования.

    `rate_limiter` — тот же общий лимитер, что у ArxivSearchClient (передаётся из
    modules.bootstrap): ar5iv и arxiv.org/pdf — та же инфраструктура arXiv, что и Atom API
    поиска, и без единого троттлинга на все три эти запросы дают несогласованный всплеск,
    который arXiv в какой-то момент начинает откровенно резать (см. docstring rate_limit.py).
    """
    response = _get_with_retry(_AR5IV_URL.format(arxiv_id=arxiv_id), timeout, rate_limiter, accept_404=True)
    if response is not None and response.status_code == 200:
        sections = _parse_ar5iv_html(response.text)
        if sections and sum(len(v) for v in sections.values()) >= _MIN_AR5IV_CHARS:
            return sections, "ar5iv"

    try:
        text = _fetch_pdf_text(pdf_url or f"https://arxiv.org/pdf/{arxiv_id}", timeout, rate_limiter)
        sections = _split_pdf_sections(text) if text else {}
        if sections:
            return sections, "pdf"
    except Exception:
        logger.warning("PDF fallback fetch failed for %s", arxiv_id, exc_info=True)

    return {}, "none"
