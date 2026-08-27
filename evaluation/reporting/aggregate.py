"""Загрузка одного прогона (evaluation/runs/<run_id>/) в pandas DataFrame — только чтение
уже записанных JSONL/manifest.json, никакой мутации. compare.py/render.py строятся поверх
этого, а не парсят файлы прогона сами.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from evaluation.runlog.run_writer import RUNS_DIR


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class RunData:
    run_id: str
    run_dir: Path
    manifest: Dict[str, Any]
    cases: pd.DataFrame = field(repr=False)
    metrics: pd.DataFrame = field(repr=False)
    checks: pd.DataFrame = field(repr=False)

    @property
    def judge_model(self) -> Optional[str]:
        return self.manifest.get("suite", {}).get("judge_model")

    @property
    def suite_name(self) -> Optional[str]:
        return self.manifest.get("suite", {}).get("name")

    @property
    def case_ids(self) -> Set[str]:
        if self.cases.empty or "case_id" not in self.cases.columns:
            return set()
        return set(self.cases["case_id"])


def load_run(run_id_or_path: str) -> RunData:
    """Принимает либо имя директории прогона (evaluation/runs/<это>), либо полный путь."""
    run_dir = Path(run_id_or_path)
    if not run_dir.exists():
        run_dir = RUNS_DIR / run_id_or_path
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Прогон не найден: {run_id_or_path!r} (искали {run_dir}). "
            f"См. evaluation/runs/ или `run_eval report --list`."
        )
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{run_dir} не похож на директорию прогона — нет manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return RunData(
        run_id=run_dir.name,
        run_dir=run_dir,
        manifest=manifest,
        cases=pd.DataFrame(_read_jsonl(run_dir / "cases.jsonl")),
        metrics=pd.DataFrame(_read_jsonl(run_dir / "metrics.jsonl")),
        checks=pd.DataFrame(_read_jsonl(run_dir / "checks.jsonl")),
    )


def list_runs(suite: Optional[str] = None) -> List[str]:
    if not RUNS_DIR.exists():
        return []
    names = sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    if suite:
        names = [n for n in names if f"__{suite}__" in n]
    return names
