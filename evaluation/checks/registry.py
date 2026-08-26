"""Единая точка сопоставления имени узла (modules.node_names.NodeName) с функциями проверки.
Узлы без записи здесь (other_handler и т.д.) просто не проверяются — не ошибка."""
from typing import Dict, List

from modules.node_names import NodeName

from .base import NodeCheckFn
from .classifier import check_classifier
from .fetch_fulltext import check_fetch_fulltext
from .map_reduce_summarize import check_map_reduce_summarize
from .process_and_chunk import check_process_and_chunk
from .research_step import check_research_step
from .resolve_target_article import check_resolve_target_article

CHECKS: Dict[str, List[NodeCheckFn]] = {
    NodeName.CLASSIFIER.value: [check_classifier],
    NodeName.RESOLVE_TARGET_ARTICLE.value: [check_resolve_target_article],
    NodeName.FETCH_FULLTEXT.value: [check_fetch_fulltext],
    NodeName.PROCESS_AND_CHUNK.value: [check_process_and_chunk],
    NodeName.MAP_REDUCE_SUMMARIZE.value: [check_map_reduce_summarize],
    NodeName.RESEARCH_STEP.value: [check_research_step],
}
