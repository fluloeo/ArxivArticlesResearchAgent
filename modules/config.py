import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.0
    max_tokens: int = 1024
    frequency_penalty: float = 0.0


@dataclass(frozen=True)
class NodeGenerationConfig:
    """Sampling-параметры по каждому узлу графа — раньше были литералами внутри agent.py."""

    classifier: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=40))
    research_step: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=800))
    # SearchPlan (modules.query_rewriter) — до 5 списков полей + 2 опциональных года,
    # с запасом на репертуар терминов длинного запроса.
    query_rewrite: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=400))
    summarization_map: GenerationParams = field(
        default_factory=lambda: GenerationParams(temperature=0.15, max_tokens=2048, frequency_penalty=1.2)
    )
    summarization_reduce: GenerationParams = field(default_factory=lambda: GenerationParams(max_tokens=4096))


@dataclass(frozen=True)
class AppConfig:
    llm_backend: Literal["mlx", "openrouter", "vllm"] = "mlx"

    # На практике Qwen3-30B-A3B-4bit (~15-17 ГБ) на 24 ГБ unified memory M4 приводил к своп-
    # трэшингу под обычной десктопной нагрузкой (VSCode, браузер и т.п.: `vm.swapusage`
    # показывал единицы ГБ в свопе) — суммаризация одной статьи занимала по 5-7+ минут
    # вместо секунд. Qwen3-4B — современная, компактная (~2-2.5 ГБ) модель, без риска свопа.
    mlx_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "qwen/qwen3-30b-a3b-instruct-2507"

    use_hub: bool = False
    debug_mode: bool = False

    cache_db_path: str = "data/arxiv_cache.sqlite"
    arxiv_search_max_candidates: int = 5
    max_research_iterations: int = 3
    min_research_iterations: int = 1
    fulltext_excerpt_chars: int = 4000

    min_chunk_tokens: int = 700
    max_chunk_tokens: int = 2000
    chunk_overlap_chars: int = 250

    grpc_port: int = 50051
    grpc_host: str = "localhost"

    # Файловый лог каждой суммаризации (id, чанки, map-выжимки, финальный отчёт, тайминги) —
    # тот же формат, что evaluation/ кладёт в artifacts/, один парсер на оба случая.
    # Пустая строка отключает запись.
    summarization_log_dir: str = "logs/summarizations"

    node_gen: NodeGenerationConfig = field(default_factory=NodeGenerationConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm_backend=os.environ.get("APP_LLM_BACKEND", "mlx"),  # type: ignore[arg-type]
            mlx_model=os.environ.get("APP_MLX_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("APP_OPENROUTER_MODEL", "qwen/qwen3-30b-a3b-instruct-2507"),
            use_hub=os.environ.get("APP_USE_HUB", "false").lower() == "true",
            debug_mode=os.environ.get("APP_DEBUG_MODE", "false").lower() == "true",
            cache_db_path=os.environ.get("APP_CACHE_DB_PATH", "data/arxiv_cache.sqlite"),
            arxiv_search_max_candidates=int(os.environ.get("APP_ARXIV_MAX_CANDIDATES", "5")),
            max_research_iterations=int(os.environ.get("APP_MAX_RESEARCH_ITERATIONS", "3")),
            min_research_iterations=int(os.environ.get("APP_MIN_RESEARCH_ITERATIONS", "1")),
            fulltext_excerpt_chars=int(os.environ.get("APP_FULLTEXT_EXCERPT_CHARS", "4000")),
            grpc_port=int(os.environ.get("APP_GRPC_PORT", "50051")),
            grpc_host=os.environ.get("APP_GRPC_HOST", "localhost"),
            summarization_log_dir=os.environ.get("APP_SUMMARIZATION_LOG_DIR", "logs/summarizations"),
        )
