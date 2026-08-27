"""Рендер сравнения прогонов в markdown-таблицу — единственный формат, план не требовал
CSV/HTML. Колонка n(ok/na) обязательна: `34/16` на строке vs_reference сразу говорит, что
строка стоит на МЕНЬШЕЙ выборке, чем кажется по одному только среднему — не наблюдение из
контекста, а неотъемлемая часть числа (см. план, раздел «План экспериментов»).
"""
from typing import List, Optional

from .aggregate import RunData
from .compare import ComparisonRow


def _fmt(x: Optional[float], digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _fmt_ok_na(n_ok: int, n_total: int) -> str:
    return f"{n_ok}/{max(n_total - n_ok, 0)}"


def render_markdown(rows: List[ComparisonRow], label_a: str, label_b: str) -> str:
    lines = [
        f"| node | scope | metric | {label_a} mean | n(ok/na) | {label_b} mean | n(ok/na) | Δ | 95% CI | n paired |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = "—" if r.ci_low is None else f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        delta = "—" if r.delta is None else f"{r.delta:+.3f}"
        lines.append(
            f"| {r.node} | {r.scope} | {r.metric} | {_fmt(r.mean_a)} | {_fmt_ok_na(r.n_ok_a, r.n_a)} "
            f"| {_fmt(r.mean_b)} | {_fmt_ok_na(r.n_ok_b, r.n_b)} | {delta} | {ci} | {r.n_paired} |"
        )
    return "\n".join(lines)


def render_run_summary(run: RunData) -> str:
    """Один прогон без сравнения — среднее по (node, scope, metric), без парного бутстрапа
    (нет второй стороны для сравнения). Используется, когда `report` вызван с одним прогоном."""
    df = run.metrics
    header = f"# {run.run_id}\n\njudge_model={run.judge_model or '—'}  suite={run.suite_name or '—'}\n"
    if df.empty:
        return header + "\n(метрики не считались — прогон без --judge-model, либо сьют без Suite.metrics)"

    lines = [header, "| node | scope | metric | mean | n(ok/na) |", "|---|---|---|---|---|"]
    subset = df
    if "chunk" in subset.columns:
        subset = subset[subset["chunk"].isna()]
    for (node, scope, metric), group in subset.groupby(["node", "scope", "metric"]):
        ok = group[group["status"] == "ok"]
        mean = ok["score"].mean() if not ok.empty else None
        lines.append(f"| {node} | {scope} | {metric} | {_fmt(mean)} | {_fmt_ok_na(len(ok), len(group))} |")
    return "\n".join(lines)
