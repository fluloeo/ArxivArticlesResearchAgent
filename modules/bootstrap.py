import logging
from typing import Any, Dict, Optional

from .agent import ArxivAgent
from .article_store import SqliteArxivArticleStore
from .arxiv_source.search import ArxivSearchClient
from .config import AppConfig
from .llm.base import LLMProvider
from .local_prompts import load_local_prompts
from .processing import ArticleProcessor
from .prompt_resolver import PromptResolver
from .ragas_eval import RagasEvaluator
from .summarization import SummarizationPipeline
from .tools import ArxivToolkit

logger = logging.getLogger(__name__)


class _ApproxTokenizer:
    """Грубая оценка длины в токенах (по словам) для провайдеров без своего токенизатора
    (например OpenRouter) — используется только ArticleProcessor'ом для нарезки чанков."""

    def encode(self, text: str):
        return text.split()


def _build_provider(backend: str, model_name: str, config: AppConfig) -> LLMProvider:
    if backend == "mlx":
        from .llm.mlx_provider import MLXProvider

        return MLXProvider(model_name)

    if backend == "openrouter":
        from .llm.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(api_key=config.openrouter_api_key or "", model_name=model_name)

    raise ValueError(
        f"backend={backend!r}: vLLM требует вручную созданный движок (GPU) — "
        "соберите VLLMProvider самостоятельно и передайте в build_agent_with_provider()."
    )


def _build_llm_provider(config: AppConfig) -> LLMProvider:
    return _build_provider(config.llm_backend, config.mlx_model if config.llm_backend == "mlx" else config.openrouter_model, config)


def _build_judge_provider(config: AppConfig, main_llm: LLMProvider) -> LLMProvider:
    """LLM-as-a-judge для RAGAS. По умолчанию — тот же LLM, что отвечает пользователю;
    можно указать отдельную модель (в т.ч. через API — OpenRouter), чтобы оценка не зависела
    от той же модели, которую оценивает."""
    if config.ragas_judge_backend == "same":
        return main_llm

    model_name = config.ragas_judge_model or (
        config.mlx_model if config.ragas_judge_backend == "mlx" else config.openrouter_model
    )
    logger.info("Building separate RAGAS judge provider: backend=%s model=%s", config.ragas_judge_backend, model_name)
    return _build_provider(config.ragas_judge_backend, model_name, config)


def _tokenizer_for_chunking(provider: LLMProvider):
    return getattr(provider, "tokenizer", None) or _ApproxTokenizer()


def _hub_ref_map(keys, prefix: str = "fluloeo/arxiv-") -> Dict[str, str]:
    return {key: f"{prefix}{key.replace('_', '-')}" for key in keys}


def build_agent_with_provider(config: AppConfig, llm: LLMProvider) -> ArxivAgent:
    """Composition root: собирает все зависимости ArxivAgent из конфига и готового LLM-провайдера.

    Единая точка сборки — используется и grpc_service/server.py, и (опционально) ноутбуками,
    вместо того чтобы каждый раз вручную собирать зависимости в ячейках Kaggle-ноутбука.
    """
    from langsmith import Client as LangSmithClient

    ls_client: Optional[Any] = LangSmithClient() if config.use_hub else None
    local_prompts = load_local_prompts()

    agent_prompt_resolver = PromptResolver(ls_client, local_prompts.get("agent", {}), use_hub=config.use_hub)
    summarization_prompt_resolver = PromptResolver(
        ls_client, local_prompts.get("summarization", {}), use_hub=config.use_hub
    )
    ragas_prompt_resolver = PromptResolver(ls_client, local_prompts.get("ragas", {}), use_hub=config.use_hub)

    search_client = ArxivSearchClient()
    article_store = SqliteArxivArticleStore(config.cache_db_path, search_client=search_client)
    toolkit = ArxivToolkit(
        search_client=search_client,
        article_store=article_store,
        excerpt_chars=config.fulltext_excerpt_chars,
        max_candidates=config.arxiv_search_max_candidates,
    )

    processor = ArticleProcessor(
        tokenizer=_tokenizer_for_chunking(llm),
        min_tokens=config.min_chunk_tokens,
        max_tokens=config.max_chunk_tokens,
        overlap_len=config.chunk_overlap_chars,
    )

    sum_pipeline = SummarizationPipeline(
        provider=llm,
        prompts=_hub_ref_map(["system_map", "map", "system_reduce", "reduce"]),
        prompt_resolver=summarization_prompt_resolver,
    )

    ragas_evaluator = None
    if config.use_ragas:
        from sentence_transformers import SentenceTransformer

        judge_llm = _build_judge_provider(config, llm)
        embed_model = SentenceTransformer(config.embed_model_name)
        ragas_evaluator = RagasEvaluator(
            llm=judge_llm,
            prompts=_hub_ref_map(["ragas_claims", "ragas_verdict", "ragas_questions"]),
            prompt_resolver=ragas_prompt_resolver,
            embed_model=embed_model,
            claims_params=config.node_gen.ragas_claims,
            verdict_params=config.node_gen.ragas_verdict,
            questions_params=config.node_gen.ragas_questions,
        )

    agent = ArxivAgent(
        llm=llm,
        toolkit=toolkit,
        article_store=article_store,
        processor=processor,
        sum_pipeline=sum_pipeline,
        prompt_resolver=agent_prompt_resolver,
        prompts=_hub_ref_map(["classifier", "research_step"]),
        node_gen=config.node_gen,
        ragas_evaluator=ragas_evaluator,
        use_ragas=config.use_ragas,
        debug_mode=config.debug_mode,
        max_research_iterations=config.max_research_iterations,
        min_research_iterations=config.min_research_iterations,
    )
    logger.info(
        "ArxivAgent built: backend=%s use_ragas=%s use_hub=%s", config.llm_backend, config.use_ragas, config.use_hub
    )
    return agent


def build_agent(config: AppConfig) -> ArxivAgent:
    return build_agent_with_provider(config, _build_llm_provider(config))
