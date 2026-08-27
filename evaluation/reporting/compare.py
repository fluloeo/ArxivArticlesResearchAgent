"""Сравнение прогонов. Методологический гейт из плана (раздел «План экспериментов»):
сравнение 4B vs 30B с разными судьями, разными промптами судьи или разным набором
case_id не значит ничего — assert_comparable отказывается строить таблицу в этих
случаях, а не молча считает дельту по несопоставимым числам.

Сравнение — парный бутстрап по ПЕРЕСЕЧЕНИЮ case_id (не по объединению): n обычно мало
(~50 кейсов на сьют), статьи в выборке резко разной сложности, и неспаренное сравнение
(mean(A) - mean(B) без учёта того, что это одни и те же статьи) шумнее, чем нужно.
"""
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .aggregate import RunData


class NotComparableError(Exception):
    pass


def assert_comparable(runs: List[RunData]) -> None:
    if len(runs) < 2:
        return
    judge_models = {r.judge_model for r in runs}
    if None in judge_models:
        missing = [r.run_id for r in runs if r.judge_model is None]
        raise NotComparableError(
            f"Прогоны без judge_model (метрики не считались, --judge-model не был передан): {missing}"
        )
    if len(judge_models) > 1:
        raise NotComparableError(f"Разные модели судьи в сравниваемых прогонах: {judge_models}")

    suites = {r.suite_name for r in runs}
    if len(suites) > 1:
        raise NotComparableError(f"Разные сьюты: {suites} — сравнение между сьютами не имеет смысла")

    common = common_case_ids(runs)
    if not common:
        raise NotComparableError("Нет пересекающихся case_id между прогонами")


def common_case_ids(runs: List[RunData]) -> List[str]:
    case_sets = [r.case_ids for r in runs]
    if not case_sets:
        return []
    return sorted(set.intersection(*case_sets))


def _non_chunk_rows(run: RunData, node: str, scope: str, metric: str):
    df = run.metrics
    if df.empty or "node" not in df.columns:
        return df
    subset = df[(df["node"] == node) & (df["scope"] == scope) & (df["metric"] == metric)]
    if "chunk" in subset.columns:
        # map_stage пишет и по-чанковые строки, и одну агрегатную (среднюю) на кейс —
        # для сравнения между прогонами нужна ровно одна цифра на кейс, всегда агрегатная.
        subset = subset[subset["chunk"].isna()]
    return subset


def _case_scores(run: RunData, node: str, scope: str, metric: str) -> Dict[str, float]:
    subset = _non_chunk_rows(run, node, scope, metric)
    if subset.empty:
        return {}
    ok = subset[subset["status"] == "ok"]
    return {row["case_id"]: row["score"] for _, row in ok.iterrows()}


def paired_bootstrap(a: List[float], b: List[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95%-й доверительный интервал для mean(b) - mean(a), бутстрап по индексам ПАР
    (a[i], b[i] — один и тот же кейс в обоих прогонах)."""
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        boot_means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot_means.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = min(int(0.975 * n_boot), n_boot - 1)
    return (boot_means[lo_idx], boot_means[hi_idx])


@dataclass
class ComparisonRow:
    node: str
    scope: str
    metric: str
    n_a: int  # всего строк (ok+na+error) в baseline для этой (node,scope,metric)
    n_ok_a: int
    n_b: int
    n_ok_b: int
    mean_a: Optional[float]
    mean_b: Optional[float]
    n_paired: int  # пересечение case_id с обеими сторонами status=ok
    delta: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]


def _combos(run: RunData) -> List[Tuple[str, str, str]]:
    df = run.metrics
    if df.empty or not {"node", "scope", "metric"}.issubset(df.columns):
        return []
    cols = df[["node", "scope", "metric"]].drop_duplicates()
    return [tuple(r) for r in cols.itertuples(index=False, name=None)]


def compare_runs(baseline: RunData, candidate: RunData) -> List[ComparisonRow]:
    assert_comparable([baseline, candidate])
    combos = sorted(set(_combos(baseline)) | set(_combos(candidate)))

    rows: List[ComparisonRow] = []
    for node, scope, metric in combos:
        n_a = len(_non_chunk_rows(baseline, node, scope, metric))
        n_b = len(_non_chunk_rows(candidate, node, scope, metric))
        scores_a = _case_scores(baseline, node, scope, metric)
        scores_b = _case_scores(candidate, node, scope, metric)

        paired_ids = sorted(set(scores_a) & set(scores_b))
        a_vals = [scores_a[c] for c in paired_ids]
        b_vals = [scores_b[c] for c in paired_ids]

        mean_a = sum(scores_a.values()) / len(scores_a) if scores_a else None
        mean_b = sum(scores_b.values()) / len(scores_b) if scores_b else None

        delta = ci_low = ci_high = None
        if paired_ids:
            delta = sum(b_vals) / len(b_vals) - sum(a_vals) / len(a_vals)
            if len(paired_ids) >= 2:
                # Бутстрап с n=1 вырожден (CI = точка) — с одним общим кейсом дельта есть,
                # доверительный интервал по ней не значит ничего и не показывается.
                ci_low, ci_high = paired_bootstrap(a_vals, b_vals)

        rows.append(
            ComparisonRow(
                node=node, scope=scope, metric=metric,
                n_a=n_a, n_ok_a=len(scores_a), n_b=n_b, n_ok_b=len(scores_b),
                mean_a=mean_a, mean_b=mean_b, n_paired=len(paired_ids),
                delta=delta, ci_low=ci_low, ci_high=ci_high,
            )
        )
    return rows
