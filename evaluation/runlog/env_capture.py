"""Снимок окружения прогона для manifest.json: git-состояние, версии пакетов, конфиг
(с редакцией секретов). Редакция ОБЯЗАТЕЛЬНА — AppConfig.openrouter_api_key читается из
переменной окружения и без неё уехал бы в манифест открытым текстом при каждом прогоне
с OpenRouter-бэкендом."""
import hashlib
import logging
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from importlib import metadata
from typing import Any, Dict

logger = logging.getLogger(__name__)

_SECRET_FIELDS = {"openrouter_api_key", "api_key"}
_TRACKED_PACKAGES = (
    "langgraph", "langchain-core", "pydantic", "sentence-transformers", "grpcio", "protobuf",
)


def _redact(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _redact(asdict(obj))
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k in _SECRET_FIELDS and v else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def _run_git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=None, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("git %s failed: %s", " ".join(args), e)
        return ""


def capture_git_state() -> Dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--porcelain")
    dirty = bool(status.strip())
    diff = _run_git("diff", "HEAD") if dirty else ""
    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty": dirty,
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


def capture_package_versions() -> Dict[str, str]:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def capture_runtime() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": capture_package_versions(),
    }


def redact_config(config: Any) -> Dict[str, Any]:
    """config — любой dataclass (AppConfig, NodeGenerationConfig, будущий JudgeConfig)."""
    return _redact(config)
