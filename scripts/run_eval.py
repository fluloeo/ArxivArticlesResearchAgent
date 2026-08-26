#!/usr/bin/env python3
"""Тонкий шим: `python scripts/run_eval.py ...` == `python -m evaluation.cli ...`.
sys.path insert — тот же паттерн, что ui/streamlit_app.py использует для запуска не из
корня репозитория."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
