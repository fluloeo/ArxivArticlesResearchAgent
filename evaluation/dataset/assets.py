"""Загрузчики исторических офлайн-датасетов, уже лежащих в корне репозитория:
article_sample.json (50 статей из утраченной Postgres) и dataset_gemini_final.json
(эталонные обзоры gemini-2.5-pro) — используются как исходный текст и `vs_reference`-эталон
для суммаризации.

Файлы `audit_*.json` (faithfulness от прошлых прогонов на deepeval) и
`dataset_candidate_qwen{7b,30b}.json` СОЗНАТЕЛЬНО не читаются этим модулем: метрики в них
считались некорректно (подтверждено на практике) — переиспользовать их как валидатор новой
метрики или как готовые кандидатские обзоры нельзя. Новые кандидатские обзоры харнесс
генерирует сам, прогоняя agent.summarize_article() по article_sample.json.

Оба используемых файла гитигнорены (`*.json` в .gitignore) — это НАМЕРЕННО не
переиспользуемые в общем случае артефакты, специфичные для этой машины/сессии, поэтому
loader'ы падают с понятным сообщением, а не голым FileNotFoundError.

ДВЕ ПРОВЕРЕННЫЕ ЛОВУШКИ в этих файлах:
  1. `section_text_new` в article_sample.json НЕОДНОРОДНА между записями: у 40 из 50 статей
     это уже нативный JSON-объект, у оставшихся 10 — Python-repr строка (одинарные кавычки),
     которую нужно разбирать через `ast.literal_eval`. См. `_parse_sections`. (Изначально
     это было принято за единый формат по одной проверенной записи, которая случайно
     оказалась в string-меньшинстве — отсюда неверный первый вывод "везде repr-строка".)
  2. `article_id` в dataset_gemini_final.json местами сериализован как JSON-число, а не
     строка: наивная загрузка даёт пересечение id с article_sample.json 34/50 вместо 50/50.
     Лечится `json.load(f, parse_float=str)` (проверено).
"""
import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from modules.article_store import ArticleRecord

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReferenceSummary:
    article_id: str
    title: str
    summary: str
    model_used: str
    method: str


def _load_json_list(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} не найден по пути {path}. Этот файл гитигнорен (*.json в .gitignore) "
            "и не является частью репозитория — получите его отдельно (см. README, раздел "
            "«Эксперименты») либо передайте другой путь через --corpus-dir."
        )
    with open(path, "r", encoding="utf-8") as f:
        # parse_float=str — без этого числовые article_id (например 1503.02656 без кавычек
        # в исходном JSON) теряют точность/форму и пересечение с article_sample.json падает
        # с 50/50 до 34/50. См. docstring модуля.
        return json.load(f, parse_float=str)


def _parse_sections(raw: object, article_id: str) -> Optional[Dict[str, str]]:
    """`section_text_new` в article_sample.json неоднородна между записями: у 40 из 50
    статей это уже нативный JSON-объект (dict), у оставшихся 10 — Python-repr строка
    (одинарные кавычки), которую нужно разобрать через ast.literal_eval. Раньше это
    проверялось только на первой записи файла, которая случайно попала в string-меньшинство —
    отсюда ошибочный вывод, что формат единообразен. Обрабатываем оба случая."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as e:
            logger.warning("article_sample: не удалось разобрать section_text_new (str) для id=%s: %s", article_id, e)
            return None
        if not isinstance(parsed, dict):
            logger.warning("article_sample: section_text_new для id=%s распарсился не в словарь", article_id)
            return None
        return parsed
    logger.warning("article_sample: section_text_new для id=%s неожиданного типа %s", article_id, type(raw).__name__)
    return None


def load_article_sample(path: Optional[Path] = None) -> Dict[str, ArticleRecord]:
    """article_sample.json -> {id: ArticleRecord}. См. _parse_sections про неоднородность
    формата `section_text_new` между записями."""
    path = path or (REPO_ROOT / "article_sample.json")
    records: Dict[str, ArticleRecord] = {}
    for row in _load_json_list(path):
        sections = _parse_sections(row.get("section_text_new"), row.get("id", "?"))
        if sections is None:
            continue
        records[row["id"]] = ArticleRecord(
            article_id=row["id"],
            title=row.get("title", ""),
            sections=sections,
            pdf_url=f"https://arxiv.org/pdf/{row['id']}",
        )
    return records


def load_reference_summaries(path: Path) -> Dict[str, ReferenceSummary]:
    """dataset_gemini_final.json / dataset_candidate_*.json -> {article_id: ReferenceSummary}."""
    out: Dict[str, ReferenceSummary] = {}
    for row in _load_json_list(path):
        out[row["article_id"]] = ReferenceSummary(
            article_id=row["article_id"],
            title=row.get("title", ""),
            summary=row.get("generated_summary", ""),
            model_used=row.get("model_used", ""),
            method=row.get("method", ""),
        )
    return out
