"""Answer Relevancy — перенесено из modules/ragas_eval.py::RagasEvaluator._answer_relevancy
(behavior-preserving: генерация гипотетических вопросов -> косинус с исходным вопросом).

Только для research-ветки (см. evaluation/metrics/applicability.py) — в суммаризации нет
настоящего вопроса пользователя, и прежняя попытка синтезировать псевдовопрос
(f"Summarize arXiv:{id}") сравнивала эмбеддинг с голым ID без смысловой нагрузки, что и
привело к систематически заниженным (0.1-0.2) значениям независимо от качества обзора —
эта категория ошибки устраняется не починкой формулы, а тем, что метрика здесь просто не
считается, где у неё нет входа, ради которого она была придумана.
"""
import numpy as np

from modules.structured_output import generate_structured

from .base import MetricResult
from .judge import Judge
from .schemas import GeneratedQuestions


def compute_answer_relevancy(judge: Judge, embed_model, question: str, answer: str) -> MetricResult:
    if not answer.strip():
        return MetricResult.na("answer_relevancy", "empty_answer")
    if not question.strip():
        return MetricResult.na("answer_relevancy", "no_question")

    conv = judge.format("ragas_questions", {"answer": answer})
    default = GeneratedQuestions(questions=[], noncommittal=True)
    result = generate_structured(judge.llm, [conv], GeneratedQuestions, judge.params["ragas_questions"], [default])[0]

    if result.noncommittal or not result.questions:
        return MetricResult.ok("answer_relevancy", 0.0, noncommittal=True, n_questions=len(result.questions))

    embeddings = embed_model.encode([question, *result.questions])
    original, generated = np.asarray(embeddings[0]), np.asarray(embeddings[1:])
    similarities = generated @ original / (np.linalg.norm(generated, axis=1) * np.linalg.norm(original) + 1e-8)
    return MetricResult.ok("answer_relevancy", float(np.mean(similarities)), n_questions=len(result.questions))
