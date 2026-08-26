"""Полноценный rewriter пользовательского запроса для гибридного (field-scoped булев,
без векторного индекса) поиска по arXiv. Раньше единственной адаптацией запроса была
эвристика `_looks_like_title` (ti:"..." vs all:...) — она не переводит русский запрос на
английский и не использует структуру запроса (авторы, категории, годы), поэтому русские
вопросы находили на arXiv ноль результатов (arXiv лексический и англоязычный).

`QueryRewriter.rewrite()` — тот же паттерн structured-output "function calling", что и
classifier/research_step (modules.structured_output.generate_structured): LLM извлекает
SearchPlan, код детерминированно строит из него лестницу запросов возрастающей широты.
Планы кэшируются (QueryPlanCache) — повторный запрос и прогоны экспериментов не платят за
LLM дважды.
"""
import hashlib
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import GenerationParams
from .llm.base import LLMProvider
from .prompt_resolver import PromptResolver
from .structured_output import generate_structured

logger = logging.getLogger(__name__)


class SearchPlan(BaseModel):
    """Структурированный план поиска, извлечённый LLM из запроса пользователя (в т.ч. с
    переводом терминов на английский — arXiv лексический и практически англоязычный)."""

    title_phrase: Optional[str] = Field(
        default=None, description="Если запрос похож на название статьи — само название на английском"
    )
    key_terms: List[str] = Field(default_factory=list, description="Ключевые английские технические термины")
    phrases: List[str] = Field(default_factory=list, description="Многословные устойчивые фразы на английском")
    authors: List[str] = Field(default_factory=list, description="Фамилии авторов, если упомянуты")
    categories: List[str] = Field(default_factory=list, description="Коды категорий arXiv, например cs.CL, cs.LG")
    year_from: Optional[int] = None
    year_to: Optional[int] = None


def _quote(s: str) -> str:
    return '"' + s.replace('"', "").strip() + '"'


def build_arxiv_query_ladder(plan: SearchPlan, fallback_query: str) -> List[Tuple[str, str]]:
    """[(имя_уровня, search_query), ...] по убыванию специфичности — вызывающий код
    останавливается на первом уровне с непустым результатом, а фактический сработавший
    уровень пишется в лог (метрика recall@5 "до/после rewriter'а" по уровням).

    Синтаксис проверен живьём против export.arxiv.org (см. план, §6.2):
      - `abs:"фраза" OR abs:"фраза2"` — работает.
      - `(abs:"фраза" AND abs:"фраза2") AND cat:cs.CL` — работает, точное попадание в тему.
      - `abs:"фраза" AND cat:cs.CL AND submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` — работает.
    """
    ladder: List[Tuple[str, str]] = []

    if plan.title_phrase:
        ladder.append(("title_phrase", f"ti:{_quote(plan.title_phrase)}"))

    filters = []
    if plan.categories:
        filters.append("(" + " OR ".join(f"cat:{c}" for c in plan.categories) + ")")
    if plan.year_from or plan.year_to:
        y_from = f"{plan.year_from or 1990}01010000"
        y_to = f"{plan.year_to or 2100}12312359"
        filters.append(f"submittedDate:[{y_from} TO {y_to}]")

    and_terms = [f"abs:{_quote(p)}" for p in plan.phrases] + [f"all:{t}" for t in plan.key_terms]
    if plan.authors:
        and_terms += [f"au:{_quote(a)}" for a in plan.authors]

    if and_terms:
        core = "(" + " AND ".join(and_terms) + ")"
        if filters:
            ladder.append(("and_filtered", core + " AND " + " AND ".join(filters)))
        ladder.append(("and_unfiltered", core))

    or_terms = [f"abs:{_quote(p)}" for p in plan.phrases] + [f"abs:{_quote(t)}" for t in plan.key_terms]
    if len(or_terms) >= 2:
        ladder.append(("or_broadened", " OR ".join(or_terms)))

    if plan.key_terms or plan.phrases:
        ladder.append(("all_terms", "all:" + " ".join(plan.key_terms + plan.phrases)))

    # Безусловный последний рубеж: сырой исходный запрос — если извлечение плана не дало
    # ничего пригодного (пустой SearchPlan), поиск не должен просто не произойти вовсе.
    ladder.append(("raw_fallback", f"all:{fallback_query}"))
    return ladder


_DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "query_plan_cache.sqlite"


class QueryPlanCache:
    """SQLite-кэш SearchPlan по тексту запроса — повторный запрос того же текста
    (частое явление: пользователь перезапускает после сетевого сбоя, эксперимент
    прогоняется несколько раз) не платит за LLM-вызов снова."""

    def __init__(self, db_path: Path = _DEFAULT_CACHE_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS query_plan_cache (query_hash TEXT PRIMARY KEY, plan_json TEXT)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[SearchPlan]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT plan_json FROM query_plan_cache WHERE query_hash = ?", (self._hash(query),)
            ).fetchone()
        if row is None:
            return None
        try:
            return SearchPlan.model_validate_json(row[0])
        except Exception:
            return None

    def put(self, query: str, plan: SearchPlan) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_plan_cache (query_hash, plan_json) VALUES (?, ?)",
                (self._hash(query), plan.model_dump_json()),
            )
            conn.commit()


class QueryRewriter:
    def __init__(
        self,
        llm: LLMProvider,
        prompt_resolver: PromptResolver,
        prompts: Dict[str, Any],
        params: GenerationParams,
        cache: Optional[QueryPlanCache] = None,
    ):
        self.llm = llm
        self.resolved_prompt = prompt_resolver.resolve("query_rewrite", prompts.get("query_rewrite"))
        self.prompt_resolver = prompt_resolver
        self.params = asdict(params)
        self.cache = cache or QueryPlanCache()

    def rewrite(self, query: str) -> SearchPlan:
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        conv = self.prompt_resolver.format_chat(self.resolved_prompt, {"query": query})
        # Дефолт при провале — не пустой план (тогда rewrite() выродился бы в "искать
        # нечего"), а эвристический: первые слова запроса как key_terms, работает не хуже
        # прежнего all:query поведения.
        default = SearchPlan(key_terms=query.split()[:8])
        plan = generate_structured(self.llm, [conv], SearchPlan, self.params, [default])[0]
        self.cache.put(query, plan)
        return plan
