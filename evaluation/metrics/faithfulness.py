"""Faithfulness — перенесено из modules/ragas_eval.py::RagasEvaluator._faithfulness
(behavior-preserving: та же двухшаговая методика, тот же лексический отбор контекста),
поверх Judge вместо самодостаточного класса — остальные метрики используют того же judge.

Известное ограничение (не устранённое при переносе, чтобы не трогать
modules.structured_output.generate_structured): если claim extraction обрежется по
max_tokens (уже случалось на бюджете 400, отсюда params.claims.max_tokens=1200 в
JudgeGenParams), generate_structured либо провалит парсинг JSON целиком (тогда сработает
repair -> safe default -> claims=[] -> корректно даст status='na'), либо, если модель
успела закрыть JSON раньше лимита, вернёт частичный, но валидный список утверждений —
это неотличимо от «модель сама решила извлечь меньше утверждений» без доступа к сырому
тексту ответа (generate_structured не возвращает флаг truncated). Заметный разрыв между
n_claims и объёмом answer в detail — сигнал смотреть на это глазами при подозрении.
"""
from typing import List

from modules.structured_output import generate_structured

from .base import MetricResult
from .context_select import select_context, split_blocks, terms
from .judge import Judge
from .schemas import ClaimExtraction, ClaimVerdict


def compute_faithfulness(judge: Judge, answer: str, context: str) -> MetricResult:
    if not answer.strip():
        return MetricResult.na("faithfulness", "empty_answer")
    if not context.strip():
        return MetricResult.na("faithfulness", "no_context")

    claims_conv = judge.format("ragas_claims", {"answer": answer})
    claims_result = generate_structured(
        judge.llm, [claims_conv], ClaimExtraction, judge.params["ragas_claims"], [ClaimExtraction(claims=[])]
    )[0]
    claims: List[str] = [c.strip() for c in claims_result.claims if c.strip()]
    if not claims:
        return MetricResult.na("faithfulness", "no_claims_extracted", answer_len=len(answer))

    blocks = split_blocks(context)
    block_terms = [terms(b) for b in blocks]
    verdict_conversations = [
        judge.format("ragas_verdict", {"context": select_context(claim, blocks, block_terms), "claim": claim})
        for claim in claims
    ]
    defaults = [ClaimVerdict(supported=False) for _ in claims]
    verdicts = generate_structured(judge.llm, verdict_conversations, ClaimVerdict, judge.params["ragas_verdict"], defaults)

    n_supported = sum(1 for v in verdicts if v.supported)
    return MetricResult.ok("faithfulness", n_supported / len(verdicts), n_claims=len(claims), n_supported=n_supported)
