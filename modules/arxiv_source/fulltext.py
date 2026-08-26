import logging
import re
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_USER_AGENT = "ArxivArticlesResearchAgent/1.0 (mailto:research-agent@example.com)"
_MIN_AR5IV_CHARS = 500

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


def _fetch_pdf_text(pdf_url: str, timeout: float) -> str:
    response = requests.get(pdf_url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
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


def fetch_sections(arxiv_id: str, pdf_url: Optional[str] = None, timeout: float = 20.0) -> Tuple[Dict[str, str], str]:
    """Достаёт полный текст статьи по arXiv ID и делит его на секции.

    Порядок попыток: ar5iv (LaTeXML HTML, точная структура секций) -> PDF-текст +
    regex-эвристика заголовков (грубее, но работает почти всегда). Возвращает
    (sections, source), где source в {"ar5iv", "pdf", "none"} — используется для логирования.
    """
    try:
        response = requests.get(
            _AR5IV_URL.format(arxiv_id=arxiv_id), timeout=timeout, headers={"User-Agent": _USER_AGENT}
        )
        if response.status_code == 200:
            sections = _parse_ar5iv_html(response.text)
            if sections and sum(len(v) for v in sections.values()) >= _MIN_AR5IV_CHARS:
                return sections, "ar5iv"
    except requests.RequestException:
        logger.warning("ar5iv fetch failed for %s", arxiv_id, exc_info=True)

    try:
        text = _fetch_pdf_text(pdf_url or f"https://arxiv.org/pdf/{arxiv_id}", timeout)
        sections = _split_pdf_sections(text)
        if sections:
            return sections, "pdf"
    except Exception:
        logger.warning("PDF fallback fetch failed for %s", arxiv_id, exc_info=True)

    return {}, "none"
