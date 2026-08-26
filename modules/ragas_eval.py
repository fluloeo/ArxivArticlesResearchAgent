import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .config import GenerationParams
from .llm.base import LLMProvider
from .prompt_resolver import PromptResolver
from .schemas import ClaimExtraction, ClaimVerdict, GeneratedQuestions
from .structured_output import generate_structured

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 6000
_BLOCK_CHARS = 1500
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)


def _terms(text: str) -> Set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _split_blocks(context: str) -> List[str]:
    """Режет контекст на блоки ~<=_BLOCK_CHARS, чтобы один гигантский абзац не съедал
    весь бюджет контекста при отборе под конкретное утверждение."""
    blocks: List[str] = []
    for paragraph in (p.strip() for p in context.split("\n\n")):
        if not paragraph:
            continue
        for start in range(0, len(paragraph), _BLOCK_CHARS):
            blocks.append(paragraph[start : start + _BLOCK_CHARS])
    return blocks


@dataclass
class RagasScores:
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]


class RagasEvaluator:
    """Faithfulness и Answer Relevancy по методике RAGAS — заменяет прежний узел критика.

    В отличие от критика, ничего не переписывает и не корректирует отчёт — только измеряет
    и возвращает метрики. Единственное место, где после отказа от RAG снова понадобились
    эмбеддинги (для answer relevancy) — не для retrieval, а только для этой метрики.
    """

    def __init__(
        self,
        llm: LLMProvider,
        prompts: Dict[str, Any],
        prompt_resolver: PromptResolver,
        embed_model: Any,
        claims_params: GenerationParams,
        verdict_params: GenerationParams,
        questions_params: GenerationParams,
    ):
        self.llm = llm
        self.prompt_resolver = prompt_resolver
        self.resolved_prompts = prompt_resolver.resolve_all(prompts)
        self.embed_model = embed_model
        self.claims_params = asdict(claims_params)
        self.verdict_params = asdict(verdict_params)
        self.questions_params = asdict(questions_params)

    def evaluate(self, question: str, answer: str, context: str) -> RagasScores:
        if not answer.strip():
            return RagasScores(faithfulness=None, answer_relevancy=None)
        return RagasScores(
            faithfulness=self._faithfulness(answer, context) if context.strip() else None,
            answer_relevancy=self._answer_relevancy(question, answer),
        )

    def _faithfulness(self, answer: str, context: str) -> Optional[float]:
        claims_conv = self.prompt_resolver.format_chat(self.resolved_prompts["ragas_claims"], {"answer": answer})
        claims_result = generate_structured(
            self.llm, [claims_conv], ClaimExtraction, self.claims_params, [ClaimExtraction(claims=[])]
        )[0]
        claims = [c.strip() for c in claims_result.claims if c.strip()]
        if not claims:
            logger.info("ragas_eval: no claims extracted from answer, faithfulness=None")
            return None

        blocks = _split_blocks(context)
        block_terms = [_terms(b) for b in blocks]
        verdict_conversations = [
            self.prompt_resolver.format_chat(
                self.resolved_prompts["ragas_verdict"],
                {"context": self._select_context(claim, blocks, block_terms), "claim": claim},
            )
            for claim in claims
        ]
        defaults = [ClaimVerdict(supported=False) for _ in claims]
        verdicts = generate_structured(self.llm, verdict_conversations, ClaimVerdict, self.verdict_params, defaults)

        score = sum(1 for v in verdicts if v.supported) / len(verdicts)
        logger.info("ragas_eval: faithfulness=%.2f (%d/%d claims supported)", score, sum(1 for v in verdicts if v.supported), len(verdicts))
        return score

    @staticmethod
    def _select_context(claim: str, blocks: List[str], block_terms: List[Set[str]]) -> str:
        """Подбирает под КОНКРЕТНОЕ утверждение наиболее близкие ему куски контекста.

        Раньше на проверку каждого утверждения подавались просто первые 6000 символов
        контекста. Для суммаризации статьи контекст — это весь её текст (десятки тысяч
        символов), поэтому любое утверждение из второй половины статьи гарантированно
        оказывалось «не подтверждено контекстом», и faithtfulness занижался тем сильнее,
        чем длиннее статья.

        Отбор лексический (пересечение словоформ) — это не возврат к RAG-ретриву: он живёт
        строго внутри метрики и никак не влияет на ответ пользователю. В оригинальном RAGAS
        на этом месте стоят `retrieved_contexts`, которых у нас нет, раз ретривера нет.
        """
        if sum(len(b) for b in blocks) <= _MAX_CONTEXT_CHARS:
            return "\n\n".join(blocks)

        claim_terms = _terms(claim)
        ranked = sorted(range(len(blocks)), key=lambda i: (-len(claim_terms & block_terms[i]), i))

        chosen: List[int] = []
        used = 0
        for i in ranked:
            if used + len(blocks[i]) > _MAX_CONTEXT_CHARS:
                continue
            chosen.append(i)
            used += len(blocks[i])
        # Возвращаем в исходном порядке — так контекст читается связно, а не вперемешку.
        return "\n\n".join(blocks[i] for i in sorted(chosen))

    def _answer_relevancy(self, question: str, answer: str) -> Optional[float]:
        conv = self.prompt_resolver.format_chat(self.resolved_prompts["ragas_questions"], {"answer": answer})
        default = GeneratedQuestions(questions=[], noncommittal=True)
        result = generate_structured(self.llm, [conv], GeneratedQuestions, self.questions_params, [default])[0]

        if result.noncommittal or not result.questions:
            logger.info("ragas_eval: answer flagged noncommittal or no questions generated, answer_relevancy=0.0")
            return 0.0

        embeddings = self.embed_model.encode([question, *result.questions])
        original, generated = np.asarray(embeddings[0]), np.asarray(embeddings[1:])
        similarities = generated @ original / (np.linalg.norm(generated, axis=1) * np.linalg.norm(original) + 1e-8)
        score = float(np.mean(similarities))
        logger.info("ragas_eval: answer_relevancy=%.2f (%d generated questions)", score, len(result.questions))
        return score
