"""Загрузка YAML-сьютов (evaluation/suites/*.yaml) в Suite/EvalCase.

YAML, а не JSON — `.gitignore` глушит `*.json` в корне репозитория, а суйты должны быть
частью репозитория (в отличие от корпусных article_sample.json/dataset_gemini_final.json,
которые намеренно вне git, см. assets.py). Два способа задать кейсы, можно комбинировать:

  cases:       — явный список кейсов, один YAML-объект на EvalCase-поле.
  cases_from:  — генератор из уже загруженного корпуса (article_sample.json), чтобы не
                 переписывать вручную 50 почти одинаковых кейсов суммаризации. explicit
                 `cases` того же id перекрывает сгенерированный (полезно для регрессионных
                 кейсов вроде math/0702019 — см. suites/summarization.yaml).
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .assets import load_article_sample, load_reference_summaries
from .case import EvalCase, Suite

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"


def _slug(article_id: str) -> str:
    """case_id используется как компонент файлового пути (evaluation/runs/.../artifacts/
    <case_id>/...) — старые arXiv id вида "math/0702019" содержат "/", которое там
    интерпретировалось бы как разделитель директорий. Не только math/0702019 сегодня:
    любой старый-формата id в будущей выборке попал бы в ту же ловушку."""
    return article_id.replace("/", "-")


def _case_from_dict(raw: Dict[str, Any]) -> EvalCase:
    raw = dict(raw)
    if "id" in raw and "case_id" not in raw:  # YAML пишет "id:", поле дата-класса — case_id
        raw["case_id"] = raw.pop("id")
    known = {f for f in EvalCase.__dataclass_fields__}
    kwargs = {k: v for k, v in raw.items() if k in known and k != "extra"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return EvalCase(**kwargs, extra=extra)  # type: ignore[arg-type]


def _generate_summarization_cases(spec: Dict[str, Any]) -> List[EvalCase]:
    """`cases_from.asset: article_sample.json` — один кейс на статью, entry=summarize_app,
    target_article_id=id; если задан `reference`, подтягивает эталонный обзор gemini
    для vs_reference-разреза."""
    sample = load_article_sample()
    ids = list(sample.keys())
    if "limit" in spec:
        ids = ids[: spec["limit"]]

    references: Dict[str, str] = {}
    ref_spec = spec.get("reference")
    if ref_spec:
        ref_path = REPO_ROOT / ref_spec["asset"]
        loaded = load_reference_summaries(ref_path)
        references = {aid: r.summary for aid, r in loaded.items()}

    prefix = spec.get("case_id_prefix", "gen")
    expected_path = spec.get("expected_path")
    cases = []
    for article_id in ids:
        cases.append(
            EvalCase(
                case_id=f"{prefix}-{_slug(article_id)}",
                query=f"Summarize arXiv:{article_id}",
                entry="summarize_app",
                target_article_id=article_id,
                expected_path=list(expected_path) if expected_path else None,
                reference_summary=references.get(article_id),
            )
        )
    return cases


def _generate_search_cases(spec: Dict[str, Any]) -> List[EvalCase]:
    """`cases_from.asset` + `query_from: title` — запрос = реальное название статьи,
    ожидаем, что тот же id вернётся в топ-k кандидатов."""
    sample = load_article_sample()
    ids = list(sample.keys())
    if "limit" in spec:
        ids = ids[: spec["limit"]]
    prefix = spec.get("case_id_prefix", "gen")
    query_from = spec.get("query_from", "title")

    cases = []
    for article_id in ids:
        record = sample[article_id]
        query = record.title if query_from == "title" else article_id
        if not query:
            continue
        cases.append(
            EvalCase(case_id=f"{prefix}-{_slug(article_id)}", query=query, expected_article_id=article_id)
        )
    return cases


_GENERATORS = {
    "summarization": _generate_summarization_cases,
    "search_recall": _generate_search_cases,
}


def _generate_cases(suite_name: str, spec: Dict[str, Any]) -> List[EvalCase]:
    generator = _GENERATORS.get(suite_name)
    if generator is None:
        raise ValueError(f"cases_from не поддерживается для сьюта {suite_name!r} — нет генератора в loader.py")
    return generator(spec)


def load_suite(path: Path) -> Suite:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    name = doc["suite"]
    cases_by_id: Dict[str, EvalCase] = {}

    cases_from = doc.get("cases_from")
    if cases_from:
        for case in _generate_cases(name, cases_from):
            cases_by_id[case.case_id] = case

    for raw in doc.get("cases") or []:
        case = _case_from_dict(raw)
        cases_by_id[case.case_id] = case  # явные кейсы перекрывают сгенерированные с тем же id

    if not cases_by_id:
        logger.warning("Сьют %s (%s) не содержит ни одного кейса", name, path)

    return Suite(
        name=name,
        entry=doc.get("entry", "app"),
        metrics=doc.get("metrics") or {},
        cases=list(cases_by_id.values()),
        stop_after=doc.get("stop_after"),
        description=doc.get("description", ""),
    )


def load_suite_by_name(name: str, suites_dir: Optional[Path] = None) -> Suite:
    directory = suites_dir or SUITES_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in directory.glob("*.yaml")) if directory.exists() else []
        raise FileNotFoundError(f"Сьют {name!r} не найден ({path}). Доступные сьюты: {available}")
    return load_suite(path)
