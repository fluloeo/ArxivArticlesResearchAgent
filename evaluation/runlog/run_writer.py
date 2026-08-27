"""RunWriter — потоковая (с flush() на каждую запись) запись одного прогона харнесса в
evaluation/runs/<UTC>__<suite>__<label>__<git-sha>/. Поток записи, а не накопление в памяти
и сброс в конце — прогон, убитый на середине (Ctrl+C, OOM, упавший MLX), остаётся частично
анализируемым: nodes.jsonl/checks.jsonl содержат всё, что успело пройти, manifest.json
пишется ПЕРВЫМ с status="running", чтобы упавший прогон был опознаваем, а не тихо потерян.

nodes.jsonl хранит ДАЙДЖЕСТЫ состояния (длины, sha256, счётчики), а не сами тексты статей —
иначе 50 статей x полный текст x пред-состояние на узел дают сотни мегабайт (см. план,
раздел про nodes.jsonl). Полные payload'ы — только в artifacts/, только если
save_artifacts=True.
"""
import hashlib
import json
import logging
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.checks.base import CheckResult
from evaluation.tracing.trace import GraphTrace, NodeVisit

from .env_capture import capture_git_state, capture_runtime, redact_config
from .schema import SCHEMA_VERSION

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _count_chars(value: Any, _depth: int = 0) -> int:
    """Суммарная длина всех строковых листьев на любой глубине — вложенные структуры
    состояния графа не всегда плоские: article_chunks, например, это
    {title: {past_overlap, main_text, future_overlap}}, а не {title: str}. Без рекурсии
    total_chars для таких полей молча считался бы нулём (реальный баг, найденный на
    живом прогоне: 'article_chunks.total_chars': 0 при реально непустых чанках).
    _depth ограничивает рекурсию на случай нестандартных данных — состояние графа
    в этом проекте не глубже двух уровней."""
    if _depth > 6:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_count_chars(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_chars(v, _depth + 1) for v in value)
    return 0


def _digest_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        out: Dict[str, Any] = {"len": len(value)}
        if len(value) > 200:
            out["sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return out
    if isinstance(value, dict):
        return {"n": len(value), "total_chars": _count_chars(value)}
    if isinstance(value, (list, tuple)):
        return {"n": len(value), "total_chars": _count_chars(value)}
    if isinstance(value, (int, float, bool)) or value is None:
        return {"value": value}
    return {"repr_len": len(repr(value))}


def digest_state(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state.items():
        for subkey, subvalue in _digest_value(value).items():
            out[f"{key}.{subkey}"] = subvalue
    return out


class RunWriter:
    def __init__(
        self,
        suite: str,
        label: str,
        app_config: Any,
        node_gen: Any,
        extra_manifest: Optional[Dict[str, Any]] = None,
        save_artifacts: bool = False,
        runs_dir: Optional[Path] = None,
        cli_argv: Optional[List[str]] = None,
    ):
        git_state = capture_git_state()
        sha = (git_state.get("commit") or "nogit")[:8]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{ts}__{suite}__{label}__{sha}"

        base = runs_dir or RUNS_DIR
        self.run_dir = base / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if save_artifacts:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.save_artifacts = save_artifacts

        self._files = {
            name: open(self.run_dir / f"{name}.jsonl", "a", encoding="utf-8")
            for name in ("nodes", "checks", "metrics", "events", "cases")
        }
        # Пул кейсов при --case-workers>1 (evaluation/runner.py::run_suite_concurrent)
        # пишет в один и тот же writer из нескольких потоков — без лока f.write() из
        # разных потоков может интерливиться и дать битую строку JSONL.
        self._lock = threading.Lock()

        self.manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "label": label,
            "started_at": _now_iso(),
            "finished_at": None,
            "status": "running",
            "suite": {"name": suite},
            "git": git_state,
            "app_config": redact_config(app_config),
            "node_gen": redact_config(node_gen),
            "runtime": capture_runtime(),
            "cli_argv": cli_argv or [],
            **(extra_manifest or {}),
        }
        self._write_manifest()

    # ------------------------------------------------------------------ writers

    def _write_jsonl(self, name: str, row: Dict[str, Any]) -> None:
        f = self._files[name]
        line = json.dumps({"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **row}, ensure_ascii=False) + "\n"
        with self._lock:
            f.write(line)
            f.flush()

    def write_node_visit(self, case_id: str, visit: NodeVisit) -> None:
        self._write_jsonl(
            "nodes",
            {
                "case_id": case_id,
                "ts": _now_iso(),
                "visit_id": visit.visit_id,
                "graph": visit.graph,
                "node": visit.node,
                "step": visit.step,
                "occurrence": visit.occurrence,
                "triggers": list(visit.triggers),
                "duration_s": visit.duration_s,
                "timing_source": visit.timing_source,
                "status": visit.status,
                "error": asdict(visit.error) if visit.error else None,
                "input_digest": digest_state(visit.input_state),
                "output_digest": digest_state(visit.output_delta),
                "llm": {
                    "calls": len(visit.llm_calls),
                    "conversations": sum(c.n_conversations for c in visit.llm_calls),
                    "seconds": sum(c.duration_s for c in visit.llm_calls),
                }
                if visit.llm_calls
                else None,
                "n_log_events": len(visit.log_events),
            },
        )
        if self.save_artifacts and (visit.llm_calls or visit.output_delta):
            self._save_artifact(case_id, f"{visit.node}_{visit.occurrence}.json", {
                "output_delta": visit.output_delta,
                "llm_calls": [asdict(c) for c in visit.llm_calls],
            })

    def write_check(self, case_id: str, node: str, occurrence: int, result: CheckResult) -> None:
        self._write_jsonl(
            "checks",
            {
                "case_id": case_id,
                "node": node,
                "occurrence": occurrence,
                "check": result.check,
                "severity": result.severity,
                "passed": result.passed,
                "observed": result.observed,
                "expected": result.expected,
                "message": result.message,
            },
        )

    def write_metric(self, case_id: str, row: Dict[str, Any]) -> None:
        self._write_jsonl("metrics", {"case_id": case_id, "ts": _now_iso(), **row})

    def write_events(self, case_id: str, visit: NodeVisit) -> None:
        for event in visit.log_events:
            self._write_jsonl(
                "events",
                {
                    "case_id": case_id,
                    "node": event.node,
                    "occurrence": event.occurrence,
                    "logger": event.logger,
                    "level": event.level,
                    "event": event.event,
                    "msg_template": event.msg_template,
                    "args": event.args,
                    "message": event.message,
                },
            )

    def write_case(self, case_id: str, trace: GraphTrace, check_results: List[CheckResult], scores: Dict[str, Any]) -> None:
        passed = sum(1 for r in check_results if r.passed)
        failed = sum(1 for r in check_results if not r.passed and r.severity == "error")
        warned = sum(1 for r in check_results if not r.passed and r.severity == "warning")
        self._write_jsonl(
            "cases",
            {
                "case_id": case_id,
                "suite": self.manifest["suite"]["name"],
                "status": "error" if trace.terminal_error else "ok",
                "path": list(trace.path),
                "duration_s": trace.total_s,
                "n_visits": len(trace.visits),
                "scores": scores,
                "checks": {"passed": passed, "failed": failed, "warned": warned},
                "error": asdict(trace.terminal_error) if trace.terminal_error else None,
            },
        )

    def _save_artifact(self, case_id: str, filename: str, content: Any) -> None:
        case_dir = self.artifacts_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, open(case_dir / filename, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------ lifecycle

    def _write_manifest(self) -> None:
        with open(self.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def set_suite_info(self, **kwargs: Any) -> None:
        self.manifest["suite"].update(kwargs)
        self._write_manifest()

    def finalize(self, status: str = "completed", summary: Optional[Dict[str, Any]] = None) -> None:
        for f in self._files.values():
            f.close()
        self.manifest["status"] = status
        self.manifest["finished_at"] = _now_iso()
        self._write_manifest()
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary or {}, f, ensure_ascii=False, indent=2)
        logger.info("Run %s завершён: status=%s dir=%s", self.run_id, status, self.run_dir)
