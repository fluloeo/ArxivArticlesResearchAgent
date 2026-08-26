"""SQLite-кэш для дорогих, но детерминированных (temperature=0 у судьи) шагов метрик —
в первую очередь извлечение ключевых тезисов в coverage.py: один и тот же исходный текст
извлекается ОДИН РАЗ и затем сверяется с разными кандидатскими обзорами (qwen7b, qwen30b,
gemini, ...). Без кэша сравнение N моделей стоило бы N одинаковых дорогих LLM-проходов по
исходнику вместо одного — и делало бы сравнение менее честным (разные модели получали бы
чуть разный рубрикатор тезисов, если бы судья не был строго temperature=0).

Ключ обязан включать judge-модель и хэш промпта — смена любого из них должна дать промах
кэша, а не тихо вернуть устаревший результат, посчитанный другим судьёй/промптом.
"""
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "judge_cache.sqlite"


def cache_key(kind: str, text: str, judge_model: str, prompt: str) -> str:
    h = hashlib.sha256()
    for part in (kind, text, judge_model, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class JudgeCache:
    def __init__(self, db_path: Path = DEFAULT_CACHE_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS judge_cache (key TEXT PRIMARY KEY, value_json TEXT, created_at TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        # См. modules/article_store.py: sqlite3-соединения не потокобезопасны, открываем
        # на вызов, а не храним одно на объект — та же причина, тот же паттерн.
        return sqlite3.connect(self._db_path)

    def get(self, key: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM judge_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: Any) -> None:
        import datetime

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO judge_cache (key, value_json, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            conn.commit()
