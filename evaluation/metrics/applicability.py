"""Декларативная таблица «где какая метрика вообще имеет смысл» — независимый от YAML-сьютов
код-level гард (требование 2 из плана: суммаризация -> faithfulness+coverage,
ответ на вопрос -> answer_relevancy+faithfulness, не считать метрики там, где они не имеют
смысла). Суйты (evaluation/suites/*.yaml) декларируют, ЧТО они хотят посчитать; эта таблица
решает, разрешено ли это — запрос answer_relevancy на map_reduce_summarize отфильтровывается
здесь ДО обращения к судье, а не тихо считается и даёт бессмысленное число."""
from typing import Dict, List, Set, Tuple

# node -> scope -> {applicable metric names}
_APPLICABLE: Dict[str, Dict[str, Set[str]]] = {
    "map_reduce_summarize": {
        "map_stage": {"faithfulness", "coverage"},
        "reduce_stage": {"faithfulness", "coverage"},
        "end_to_end": {"faithfulness", "coverage"},
        # faithfulness здесь — согласованность с ЭТАЛОННЫМ ОБЗОРОМ (gemini), а не с
        # первоисточником (это уже даёт end_to_end): истинное утверждение, которое gemini
        # просто не упомянул(а), формально засчитается как "неподтверждённое". Полезно как
        # проверка согласованности между двумя независимыми обзорами, но не путать с
        # "выдумал ли факты против статьи" — за это отвечает end_to_end.
        "vs_reference": {"faithfulness", "coverage"},
    },
    "research_step": {
        # coverage здесь условна (только если у кейса есть reference_answer) — это решает
        # вызывающий код (runner), applicability лишь не запрещает её как класс.
        "terminal": {"faithfulness", "answer_relevancy", "coverage"},
    },
}


def is_applicable(node: str, scope: str, metric: str) -> bool:
    return metric in _APPLICABLE.get(node, {}).get(scope, set())


def filter_requested(node: str, scope: str, requested: List[str]) -> Tuple[List[str], List[str]]:
    """(разрешённые, отклонённые) — отклонённые вызывающий код должен записать как
    MetricResult.na("not_applicable_for_node"), а не молча выбросить."""
    allowed = _APPLICABLE.get(node, {}).get(scope, set())
    applicable = [m for m in requested if m in allowed]
    rejected = [m for m in requested if m not in allowed]
    return applicable, rejected
