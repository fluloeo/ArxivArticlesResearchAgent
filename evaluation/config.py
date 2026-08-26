"""Сборка AppConfig для прогона харнесса — тонкая обвязка над AppConfig.from_env(), плюс
CLI-переопределения (--llm-backend/--model), которые нужны только здесь, не в проде."""
import dataclasses
from typing import Optional

from modules.config import AppConfig


def build_app_config(
    llm_backend: Optional[str] = None,
    model: Optional[str] = None,
    debug_mode: bool = False,
) -> AppConfig:
    config = AppConfig.from_env()
    overrides = {}
    if llm_backend:
        overrides["llm_backend"] = llm_backend
    if model:
        overrides["mlx_model" if (llm_backend or config.llm_backend) == "mlx" else "openrouter_model"] = model
    if debug_mode:
        overrides["debug_mode"] = True
    return dataclasses.replace(config, **overrides) if overrides else config
