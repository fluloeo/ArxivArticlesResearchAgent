"""Composition root харнесса — зеркалит modules.bootstrap.build_agent_with_provider, но
с двумя отличиями:
  1. коллабораторы (article_store, search_client) подменяемы — offline=True подставляет
     InMemoryArticleStore/FrozenSearchClient (evaluation.dataset.offline_store), офлайн-
     сьюты (summarization) не бьют по сети и не троттлятся;
  2. НЕ собирает RagasEvaluator/SentenceTransformer — измерение качества (faithfulness/
     coverage/answer_relevancy) в харнессе отдельный слой (evaluation/metrics/), поверх
     GraphTrace, а не узел графа. `use_ragas=False` уже сегодня убирает ragas_eval из
     обоих скомпилированных графов агента (config-level no-op) — сборка здесь просто не
     тратит время и память на sentence-transformers/torch вовсе.

modules.bootstrap не переиспользуется напрямую (build_agent_with_provider жёстко
собирает SqliteArxivArticleStore/ArxivSearchClient) — здесь минимальное преднамеренное
дублирование его тела ради инжектируемых коллабораторов, а не полный форк логики.
"""
import logging
from typing import Any, Dict, Optional

from modules.agent import ArxivAgent
from modules.article_store import ArticleStore
from modules.arxiv_source.search import ArxivSearchClient
from modules.bootstrap import _build_llm_provider, _tokenizer_for_chunking
from modules.config import AppConfig
from modules.llm.base import LLMProvider
from modules.local_prompts import load_local_prompts
from modules.processing import ArticleProcessor
from modules.prompt_resolver import PromptResolver
from modules.query_rewriter import QueryRewriter
from modules.summarization import SummarizationPipeline
from modules.tools import ArxivToolkit

from .dataset.assets import load_article_sample
from .dataset.offline_store import FrozenSearchClient, InMemoryArticleStore
from .tracing.provider_wrapper import RecordingProvider

logger = logging.getLogger(__name__)


def build_llm_provider(config: AppConfig, record_llm_io: bool = False):
    provider: LLMProvider = _build_llm_provider(config)
    if record_llm_io:
        provider = RecordingProvider(provider, capture_io=True)
    return provider


def _build_collaborators(config: AppConfig, llm: LLMProvider, offline: bool, use_rewriter: bool = True):
    if offline:
        sample = load_article_sample()
        article_store: ArticleStore = InMemoryArticleStore(sample)
        search_client: Any = FrozenSearchClient(article_index=sample)
    else:
        from modules.article_store import SqliteArxivArticleStore

        search_client = ArxivSearchClient()
        article_store = SqliteArxivArticleStore(config.cache_db_path, search_client=search_client)

    rewriter = None
    if use_rewriter:
        local_prompts = load_local_prompts()
        agent_resolver = PromptResolver(None, local_prompts.get("agent", {}), use_hub=False)
        rewriter = QueryRewriter(
            llm=llm,
            prompt_resolver=agent_resolver,
            prompts={"query_rewrite": "query_rewrite"},
            params=config.node_gen.query_rewrite,
        )

    toolkit = ArxivToolkit(
        search_client=search_client,
        article_store=article_store,
        excerpt_chars=config.fulltext_excerpt_chars,
        max_candidates=config.arxiv_search_max_candidates,
        rewriter=rewriter,
    )
    processor = ArticleProcessor(
        tokenizer=_tokenizer_for_chunking(llm),
        min_tokens=config.min_chunk_tokens,
        max_tokens=config.max_chunk_tokens,
        overlap_len=config.chunk_overlap_chars,
    )
    return article_store, toolkit, processor


def build_agent_for_eval(
    config: AppConfig,
    llm: LLMProvider,
    offline: bool = False,
    use_rewriter: bool = True,
    prompt_overrides: Optional[Dict[str, Any]] = None,
) -> ArxivAgent:
    local_prompts = load_local_prompts()
    agent_resolver = PromptResolver(None, local_prompts.get("agent", {}), use_hub=False)
    summarization_resolver = PromptResolver(None, local_prompts.get("summarization", {}), use_hub=False)

    article_store, toolkit, processor = _build_collaborators(config, llm, offline, use_rewriter=use_rewriter)

    sum_pipeline = SummarizationPipeline(
        provider=llm,
        prompts={"map": "map", "reduce": "reduce", "system_map": "system_map", "system_reduce": "system_reduce"},
        prompt_resolver=summarization_resolver,
    )

    agent = ArxivAgent(
        llm=llm,
        toolkit=toolkit,
        article_store=article_store,
        processor=processor,
        sum_pipeline=sum_pipeline,
        prompt_resolver=agent_resolver,
        prompts={"classifier": "classifier", "research_step": "research_step"},
        node_gen=config.node_gen,
        ragas_evaluator=None,
        use_ragas=False,
        debug_mode=config.debug_mode,
        max_research_iterations=config.max_research_iterations,
        min_research_iterations=config.min_research_iterations,
    )
    logger.info("Eval-агент собран: backend=%s offline=%s", config.llm_backend, offline)
    return agent
