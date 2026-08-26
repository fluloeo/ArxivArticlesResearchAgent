"""Лексический отбор контекста под конкретное утверждение — перенесено из
modules/ragas_eval.py::RagasEvaluator._select_context (там было приватным методом класса,
здесь свободные функции, переиспользуемые faithfulness.py и, при необходимости, coverage.py).

Раньше на проверку каждого утверждения подавались просто первые _MAX_CONTEXT_CHARS символов
контекста. Для суммаризации статьи контекст — весь её текст (десятки тысяч символов),
поэтому любое утверждение из второй половины статьи гарантированно оказывалось
«не подтверждено контекстом», и faithfulness занижался тем сильнее, чем длиннее статья.

Отбор лексический (пересечение словоформ) — это НЕ возврат к RAG-ретриву: он живёт строго
внутри метрики и не влияет на ответ пользователю. В оригинальном RAGAS на этом месте стоят
`retrieved_contexts`, которых у нас нет, раз ретривера нет.
"""
import re
from typing import List, Set

MAX_CONTEXT_CHARS = 6000
_BLOCK_CHARS = 1500
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)


def terms(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def split_blocks(context: str) -> List[str]:
    """Режет контекст на блоки ~<=_BLOCK_CHARS, чтобы один гигантский абзац не съедал
    весь бюджет контекста при отборе под конкретное утверждение."""
    blocks: List[str] = []
    for paragraph in (p.strip() for p in context.split("\n\n")):
        if not paragraph:
            continue
        for start in range(0, len(paragraph), _BLOCK_CHARS):
            blocks.append(paragraph[start : start + _BLOCK_CHARS])
    return blocks


def select_context(claim: str, blocks: List[str], block_terms: List[Set[str]]) -> str:
    """Подбирает под КОНКРЕТНОЕ утверждение наиболее близкие ему куски контекста."""
    if sum(len(b) for b in blocks) <= MAX_CONTEXT_CHARS:
        return "\n\n".join(blocks)

    claim_terms = terms(claim)
    ranked = sorted(range(len(blocks)), key=lambda i: (-len(claim_terms & block_terms[i]), i))

    chosen: List[int] = []
    used = 0
    for i in ranked:
        if used + len(blocks[i]) > MAX_CONTEXT_CHARS:
            continue
        chosen.append(i)
        used += len(blocks[i])
    # Возвращаем в исходном порядке — так контекст читается связно, а не вперемешку.
    return "\n\n".join(blocks[i] for i in sorted(chosen))
