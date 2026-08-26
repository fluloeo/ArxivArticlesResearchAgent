"""Coverage — новая метрика, зеркальная faithfulness: faithfulness проверяет «не выдумал
ли кандидат лишнего» (precision против источника), coverage — «не упустил ли кандидат
важного» (recall против источника). Обе вместе дают содержательную картину — faithfulness
в одиночку награждает пустой ответ (нечего опровергнуть), coverage в одиночку награждает
дословную копию источника; см. compute_factual_f1 ниже.

Асимметрия контекста в отличие от faithfulness ЗДЕСЬ ОБРАТНАЯ, и это определяет реализацию:
у faithfulness контекст длинный (вся статья) и утверждение короткое — отсюда
_select_context (context_select.py). У coverage длинный — источник ключевых тезисов
(извлекается ОДИН РАЗ и КЭШИРУЕТСЯ, см. cache.py), а candidate (обзор), который сверяется
с каждым тезисом, обычно короткий — весь целиком помещается в один вызов проверки, поэтому
_select_context здесь не нужен в типичном случае; страхуемся усечением только если сам
кандидат неожиданно огромен.
"""
from typing import List

from modules.structured_output import generate_structured

from .base import MetricResult
from .cache import JudgeCache, cache_key
from .context_select import MAX_CONTEXT_CHARS
from .judge import Judge
from .schemas import KeyPoint, KeyPoints, PointVerdict

_WEIGHT = {"core": 2, "supporting": 1}


def _extract_key_points(judge: Judge, cache: JudgeCache, judge_model_name: str, source: str) -> List[KeyPoint]:
    key = cache_key("coverage_points", source, judge_model_name, judge.prompt_hash("coverage_points"))
    cached = cache.get(key)
    if cached is not None:
        return [KeyPoint(**p) for p in cached]

    conv = judge.format("coverage_points", {"source": source})
    result = generate_structured(judge.llm, [conv], KeyPoints, judge.params["coverage_points"], [KeyPoints(points=[])])[0]
    cache.put(key, [p.model_dump() for p in result.points])
    return result.points


def compute_coverage(
    judge: Judge,
    cache: JudgeCache,
    judge_model_name: str,
    source: str,
    candidate: str,
) -> MetricResult:
    if not source.strip():
        return MetricResult.na("coverage", "no_source")
    if not candidate.strip():
        return MetricResult.na("coverage", "empty_candidate")

    points = _extract_key_points(judge, cache, judge_model_name, source)
    if not points:
        return MetricResult.na("coverage", "no_key_points", source_len=len(source))

    # candidate — контекст для КАЖДОЙ проверки, а не наоборот (см. docstring модуля);
    # усечение — только страховка на нетипично длинный кандидат.
    candidate_context = candidate if len(candidate) <= MAX_CONTEXT_CHARS else candidate[:MAX_CONTEXT_CHARS]

    verdict_conversations = [
        judge.format("coverage_verdict", {"candidate": candidate_context, "point": p.text}) for p in points
    ]
    defaults = [PointVerdict(covered=False) for _ in points]
    verdicts = generate_structured(
        judge.llm, verdict_conversations, PointVerdict, judge.params["coverage_verdict"], defaults
    )

    n_covered = sum(1 for v in verdicts if v.covered)
    score = n_covered / len(points)

    total_weight = sum(_WEIGHT[p.importance] for p in points)
    covered_weight = sum(_WEIGHT[p.importance] for p, v in zip(points, verdicts) if v.covered)
    weighted_score = covered_weight / total_weight if total_weight else score

    return MetricResult.ok(
        "coverage", score, n_points=len(points), n_covered=n_covered, weighted_score=weighted_score
    )


def compute_factual_f1(faithfulness: float, coverage: float) -> float:
    """Гармоническое среднее — по одиночке ни одна из метрик не показательна (см. docstring
    модуля); возвращается 0.0, если сумма нулевая, чтобы избежать деления на ноль."""
    if faithfulness + coverage == 0:
        return 0.0
    return 2 * faithfulness * coverage / (faithfulness + coverage)
