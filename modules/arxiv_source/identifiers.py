"""Единая точка разбора arXiv-идентификаторов и ссылок — заменяет прежний
`extract_arxiv_id` (modules/arxiv_source/search.py), который ловил только новый формат
(`\\d{4}\\.\\d{4,5}`) и был единственным местом распознавания ID в проекте.

Найденный вживую баг, который этот модуль чинит: старый формат ID (до апреля 2007 —
"архив/номер", например `math/0702019`, `hep-th/9901001`, `cs/0112017`) не только не
распознавался в `resolve_target_article_node` (запрос со старым ID уходил в поиск вместо
прямого fetch), но и ПОРТИЛСЯ при разборе Atom-фида: `_parse_feed` брал
`raw_id.rsplit("/", 1)[-1]` от `http://arxiv.org/abs/math/0702019v1`, получая `"0702019v1"`
— префикс архива терялся, версия оставалась. Испорченный id уходил в ключ SQLite-кэша и
в URL PDF. При этом сам arXiv API старый формат принимает нормально (`id_list=math/0702019`
даёт HTTP 200) — статья в собственном корпусе проекта (article_sample.json) есть,
и баг был не гипотетическим.

Проверено живьём (export.arxiv.org): у самых старых статей предметный класс НЕ входит в id
(`math/9201301`, не `math.GT/9201301`) — суффикс класса поэтому опционален в регулярке.
"""
import re
from typing import Optional

# Архивы старой (до апреля 2007) таксономии arXiv — https://arxiv.org/archive/ ,
# whitelist вместо общего `[a-z-]+`, чтобы НЕ ловить произвольный "текст/1234567"
# (случайное слово + слэш + 7 цифр) как arXiv ID.
_OLD_ARCHIVES = (
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph",
    "math", "cs", "q-bio", "stat",
)
_OLD_ARCHIVES_ALT = "|".join(re.escape(a) for a in sorted(_OLD_ARCHIVES, key=len, reverse=True))

# Общий шаблон "формы ID" (новый ИЛИ старый формат), переиспользуется везде ниже. Раньше
# URL-регулярки строились на нежадном "любые word-символы и слэши" ([\w.\-/]+?) — для
# старого формата это ловушка: внутренний "/" в самом id (math/0702019) неотличим от "/"
# как разделителя URL-пути для нежадного квантификатора, он останавливался на первом же
# "/" и обрезал id до "math" (проверено, было реальным провалом теста). Явная альтернация
# двух точных форм устраняет неоднозначность в принципе, а не сужением жадности.
_ID_FORM = rf"(?:\d{{4}}\.\d{{4,5}}|(?:{_OLD_ARCHIVES_ALT})(?:\.[A-Z]{{2,3}})?/\d{{7}})"

_NEW_STYLE_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_OLD_STYLE_RE = re.compile(
    rf"\b((?:{_OLD_ARCHIVES_ALT})(?:\.[A-Z]{{2,3}})?/\d{{7}})(?:v\d+)?\b"
)
# DOI-форма: 10.48550/arXiv.<id>, id может быть в любом из двух форматов выше.
_DOI_RE = re.compile(rf"10\.48550/arXiv\.({_ID_FORM})", re.IGNORECASE)

_ABS_URL_RE = re.compile(rf"arxiv\.org/(?:abs|pdf)/({_ID_FORM})(?:v\d+)?(?:\.pdf)?")
_AR5IV_URL_RE = re.compile(rf"ar5iv\.[\w.]*arxiv\.org/html/({_ID_FORM})(?:v\d+)?")


def normalize(raw_id: str) -> str:
    """Снимает версионный суффикс (`v3`), пробелы по краям. Не трогает "/" — он часть
    самого id у старого формата, а не разделитель, который нужно чистить."""
    raw_id = raw_id.strip()
    for pattern in (_NEW_STYLE_RE, _OLD_STYLE_RE):
        m = pattern.fullmatch(raw_id)
        if m:
            return m.group(1)
    return re.sub(r"v\d+$", "", raw_id)


def extract_arxiv_id(text: str) -> Optional[str]:
    """Достаёт arXiv ID из произвольного текста: голый id (новый/старый формат), URL
    (`arxiv.org/abs|pdf/...`, `ar5iv.../html/...`), DOI (`10.48550/arXiv...`). Первое
    найденное совпадение — приоритет: URL/DOI (однозначны) -> новый формат -> старый.
    """
    for url_pattern in (_ABS_URL_RE, _AR5IV_URL_RE):
        m = url_pattern.search(text)
        if m:
            return normalize(m.group(1))

    m = _DOI_RE.search(text)
    if m:
        return normalize(m.group(1))

    m = _NEW_STYLE_RE.search(text)
    if m:
        return m.group(1)

    m = _OLD_STYLE_RE.search(text)
    if m:
        return m.group(1)

    return None


def extract_id_from_atom_url(url: str) -> Optional[str]:
    """Специализированный разбор конкретно поля `<id>` Atom-фида arXiv — оно ВСЕГДА имеет
    форму `http://arxiv.org/abs/{id}v{n}` (см. modules/arxiv_source/search.py::_parse_feed).
    Раньше это место обслуживалось общим `extract_arxiv_id(raw_id) or raw_id.rsplit("/", 1)[-1]`
    — общий экстрактор для старого формата не справлялся (см. docstring модуля до фикса
    _OLD_STYLE_RE), а fallback `rsplit("/", 1)[-1]` портил id. Отдельная функция, а не общий
    _ABS_URL_RE, потому что здесь ГАРАНТИРОВАННО известна форма URL — не нужно гадать по
    произвольному тексту, что не является/является частью пути."""
    m = _ABS_URL_RE.search(url)
    return normalize(m.group(1)) if m else extract_arxiv_id(url)
