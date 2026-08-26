"""Judge — LLM-as-a-judge для всех трёх метрик (faithfulness/coverage/answer_relevancy).

В отличие от инференс-пути (где judge мог "same" — переиспользовать основную модель),
харнесс НЕ предлагает такого шорткота: сравнение 4B vs 30B, где 4B ещё и судит сама себя,
не значит ничего (см. план, риск «самосудейство» и требование явного --judge-model в CLI).
Judge всегда строится явно из своего backend/model — вызывающий код (evaluation/cli.py)
обязан передать их, а не унаследовать от AppConfig основной модели.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from modules.config import GenerationParams
from modules.llm.base import LLMProvider
from modules.local_prompts import load_local_prompts
from modules.prompt_resolver import PromptResolver

_PROMPTS_PATH_KEY = "judge"  # local_prompts["judge"] — см. evaluation/prompts/judge_prompts.yaml


@dataclass(frozen=True)
class JudgeGenParams:
    claims: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=1200))
    verdict: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=200))
    questions: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=300))
    coverage_points: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=1500))
    coverage_verdict: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=200))


@dataclass(frozen=True)
class JudgeConfig:
    backend: str  # "mlx" | "openrouter"
    model: str
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    params: JudgeGenParams = field(default_factory=JudgeGenParams)


class Judge:
    """Тонкая обвязка: LLMProvider + PromptResolver(judge_prompts.yaml) + generate_structured
    per prompt kind. Метрики (faithfulness.py/coverage.py/answer_relevancy.py) вызывают
    judge.ask(kind, variables, schema, defaults) — сами не знают про промпты/параметры."""

    def __init__(self, llm: LLMProvider, prompt_resolver: PromptResolver, params: JudgeGenParams):
        self.llm = llm
        self.prompt_resolver = prompt_resolver
        prompt_keys = ["ragas_claims", "ragas_verdict", "ragas_questions", "coverage_points", "coverage_verdict"]
        self.resolved_prompts = prompt_resolver.resolve_all({k: k for k in prompt_keys})
        self.params: Dict[str, Dict[str, Any]] = {
            "ragas_claims": asdict(params.claims),
            "ragas_verdict": asdict(params.verdict),
            "ragas_questions": asdict(params.questions),
            "coverage_points": asdict(params.coverage_points),
            "coverage_verdict": asdict(params.coverage_verdict),
        }

    def format(self, kind: str, variables: Dict[str, Any]):
        return self.prompt_resolver.format_chat(self.resolved_prompts[kind], variables)

    def prompt_hash(self, kind: str) -> str:
        """Стабильный хэш содержимого промпта — используется как часть ключа кэша
        (coverage.py: смена формулировки промпта должна давать промах кэша, а не тихо
        подсунуть тезисы, извлечённые по старой инструкции) и для assert_comparable в
        отчётности (сравнение прогонов с разными промптами судьи бессмысленно)."""
        import hashlib
        import json

        return hashlib.sha256(json.dumps(self.resolved_prompts[kind], sort_keys=True, default=str).encode()).hexdigest()


def build_judge(config: JudgeConfig) -> Judge:
    from evaluation.agent_factory import build_llm_provider  # локальный импорт: избегаем цикла metrics<->agent_factory
    from modules.config import AppConfig

    # Судья строится через тот же _build_provider/build_llm_provider, что и основная
    # модель, — тем же путём (без RecordingProvider, судье не нужна трассировка узлов
    # графа, у него нет узла).
    app_config = AppConfig(llm_backend=config.backend, mlx_model=config.model, openrouter_model=config.model)
    llm = build_llm_provider(app_config, record_llm_io=False)

    local_prompts = load_local_prompts_with_judge()
    resolver = PromptResolver(None, local_prompts.get(_PROMPTS_PATH_KEY, {}), use_hub=False)
    return Judge(llm=llm, prompt_resolver=resolver, params=config.params)


def load_local_prompts_with_judge() -> Dict[str, Any]:
    """local_prompts.yaml (modules/prompts_local.yaml) больше не содержит judge-промптов —
    они в evaluation/prompts/judge_prompts.yaml (харнесс не часть инференс-пути). Объединяем
    в тот же словарь, чтобы PromptResolver не знал о существовании двух файлов."""
    from pathlib import Path

    import yaml

    prompts = dict(load_local_prompts())
    judge_path = Path(__file__).resolve().parents[1] / "prompts" / "judge_prompts.yaml"
    with open(judge_path, "r", encoding="utf-8") as f:
        judge_doc = yaml.safe_load(f) or {}
    prompts[_PROMPTS_PATH_KEY] = judge_doc.get(_PROMPTS_PATH_KEY, judge_doc)
    return prompts
