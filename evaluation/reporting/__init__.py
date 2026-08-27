"""Сравнение прогонов харнесса (evaluation/runs/<...>/) — последний слой плана
(§(f)/§5 порядка работ): читает то, что MetricsRunner/RunWriter уже записали, сам не
запускает ни агента, ни судью.
"""
from .aggregate import RunData, list_runs, load_run
from .compare import ComparisonRow, NotComparableError, assert_comparable, compare_runs
from .render import render_markdown

__all__ = [
    "RunData",
    "list_runs",
    "load_run",
    "ComparisonRow",
    "NotComparableError",
    "assert_comparable",
    "compare_runs",
    "render_markdown",
]
