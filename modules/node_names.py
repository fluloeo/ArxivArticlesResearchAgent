from enum import Enum


class NodeName(str, Enum):
    """Единый источник имён узлов графа ArxivAgent.

    Используется и при построении графа в agent.py, и в eval.py при
    сопоставлении LangSmith-ранов с узлами — раньше эти два места
    хранили независимые копии одних и тех же строк.
    """

    CLASSIFIER = "classifier"
    OTHER_HANDLER = "other_handler"
    RESOLVE_TARGET_ARTICLE = "resolve_target_article"
    FETCH_FULLTEXT = "fetch_fulltext"
    PROCESS_AND_CHUNK = "process_and_chunk"
    MAP_REDUCE_SUMMARIZE = "map_reduce_summarize"
    RESEARCH_STEP = "research_step"
