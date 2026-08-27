import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .article_store import ArticleStore
from .arxiv_source.identifiers import extract_arxiv_id
from .config import NodeGenerationConfig
from .llm.base import LLMProvider
from .node_names import NodeName
from .processing import ArticleProcessor
from .prompt_resolver import PromptResolver
from .schemas import ClassifierResult, ResearchDecision
from .streaming import ChunkDoneEvent, FinalEvent, ProgressEvent, RateEstimator, StreamEvent, TextDeltaEvent
from .structured_output import generate_structured
from .summarization import SummarizationPipeline
from .tools import ArxivToolkit

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    final_answer: str

    # ветка summarize
    target_article_id: Optional[str]
    article_title: str
    article_pdf_url: str
    raw_sections: Dict[str, str]
    article_chunks: Dict[str, Any]
    debug_data: dict
    candidates: List[Dict[str, str]]

    # ветка research (function calling вместо RAG)
    evidence: List[Dict[str, Any]]
    iterations: int
    sources: List[str]


class ArxivAgent:
    """LangGraph-агент: classifier -> {summarize | research | other}.

    - summarize: по явному arXiv ID в запросе — сразу тянет статью и суммаризирует;
      без явного ID — возвращает top-N кандидатов (`candidates`) и ждёт, пока конкретную
      статью выберет пользователь (см. `summarize_article`), а не выбирает сама.
    - research: цикл со structured-output "function calling" — модель сама решает,
      достаточно ли ей знаний для ответа, либо вызывает search_arxiv/get_fulltext.
      Минимум `min_research_iterations` вызовов инструмента обязателен, прежде чем
      разрешён финальный ответ — иначе модель может ответить, ничего не проверив.
    - Качество финального ответа (faithfulness/coverage/answer_relevancy) измеряется не
      здесь, а офлайн-харнессом (evaluation/), поверх трассировки графа — см. README,
      «Эксперименты и оценка качества».
    """

    def __init__(
        self,
        llm: LLMProvider,
        toolkit: ArxivToolkit,
        article_store: ArticleStore,
        processor: ArticleProcessor,
        sum_pipeline: SummarizationPipeline,
        prompt_resolver: PromptResolver,
        prompts: Dict[str, Any],
        node_gen: NodeGenerationConfig,
        debug_mode: bool = False,
        max_research_iterations: int = 3,
        min_research_iterations: int = 1,
        summarization_log_dir: Optional[str] = None,
    ):
        self.llm = llm
        self.toolkit = toolkit
        self.article_store = article_store
        self.processor = processor
        self.sum_pipeline = sum_pipeline
        self.prompt_resolver = prompt_resolver
        self.resolved_prompts = prompt_resolver.resolve_all(prompts)
        self.node_gen = node_gen
        self.debug_mode = debug_mode
        self.max_research_iterations = max_research_iterations
        self.min_research_iterations = min_research_iterations
        self.summarization_log_dir = summarization_log_dir or None

        self.app = self._build_graph()
        self.summarize_app = self._build_summarize_subgraph()

    # ---------------------------------------------------------------- nodes

    def classifier_node(self, state: AgentState) -> Dict[str, Any]:
        conversation = self.prompt_resolver.format_chat(self.resolved_prompts["classifier"], {"query": state["query"]})
        default = ClassifierResult(intent="other")
        result = generate_structured(
            self.llm, [conversation], ClassifierResult, asdict(self.node_gen.classifier), [default]
        )[0]
        logger.info("classifier: intent=%s query=%r", result.intent, state["query"])
        return {"intent": result.intent}

    def other_node(self, state: AgentState) -> Dict[str, Any]:
        msg = (
            "Я — специализированный научный ассистент по базе arXiv. Пожалуйста, задайте "
            "вопрос, касающийся научных исследований, или укажите тему/ID статьи для суммаризации."
        )
        return {"final_answer": msg}

    def resolve_target_article_node(self, state: AgentState) -> Dict[str, Any]:
        """Явный arXiv ID в запросе — используем сразу. Иначе отдаём кандидатов пользователю
        на выбор (см. summarize_article) вместо того, чтобы угадывать LLM'ом."""
        query = state["query"]

        explicit_id = extract_arxiv_id(query)
        if explicit_id:
            logger.info("resolve_target_article: explicit arXiv ID %s in query", explicit_id)
            return {"target_article_id": explicit_id}

        candidates = self.toolkit.find_candidates(query)
        if not candidates:
            return {"final_answer": "Не удалось найти на arXiv статью, подходящую под ваш запрос."}

        logger.info("resolve_target_article: no explicit ID, returning %d candidates for user to pick", len(candidates))
        return {"candidates": [{"arxiv_id": c.arxiv_id, "title": c.title, "abstract": c.abstract} for c in candidates]}

    def fetch_fulltext_node(self, state: AgentState) -> Dict[str, Any]:
        article_id = state["target_article_id"]

        if self.debug_mode:
            return {"final_answer": f"DEBUG MODE: цель — статья {article_id}. Загрузка текста пропущена."}

        record = self.article_store.get(article_id)
        if record is None:
            return {"final_answer": f"Не удалось получить текст статьи {article_id} с arXiv (ar5iv и PDF недоступны)."}

        return {
            "article_title": record.title,
            "article_pdf_url": record.pdf_url,
            "raw_sections": record.sections,
        }

    def process_and_chunk_node(self, state: AgentState) -> Dict[str, Any]:
        clean_sections = self.processor.process(state["raw_sections"])
        overlap_data = self.processor.create_overlap_dict(clean_sections)
        return {"article_chunks": overlap_data}

    def map_reduce_summarize_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        report, chunk_summaries = self.sum_pipeline.run(
            state["article_chunks"],
            map_params=asdict(self.node_gen.summarization_map),
            reduce_params=asdict(self.node_gen.summarization_reduce),
        )
        duration_s = time.perf_counter() - t0
        header = f"# {state.get('article_title', '')}\n🔗 [PDF]({state.get('article_pdf_url', '')})\n\n"
        self._log_summarization(state, chunk_summaries, report, duration_s)
        return {"final_answer": header + report, "debug_data": chunk_summaries}

    def _log_summarization(
        self, state: AgentState, chunk_summaries: Dict[str, str], report: str, duration_s: float
    ) -> None:
        """Файловый лог каждой суммаризации — id, размеры чанков, map-выжимки, финальный
        отчёт, параметры генерации, тайминги. Тот же набор полей, что evaluation/ кладёт
        в artifacts/ (см. evaluation/runlog/run_writer.py), чтобы один парсер годился и для
        продакшена, и для экспериментов. Отключается пустым summarization_log_dir."""
        if not self.summarization_log_dir:
            return
        try:
            chunks = state.get("article_chunks") or {}
            article_id = state.get("target_article_id") or "unknown"
            safe_id = article_id.replace("/", "-")  # старый формат ID содержит "/" — не путь
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

            record = {
                "article_id": article_id,
                "article_title": state.get("article_title", ""),
                "article_pdf_url": state.get("article_pdf_url", ""),
                "n_chunks": len(chunks),
                "chunk_char_lengths": {title: len(c.get("main_text", "")) for title, c in chunks.items()},
                "map_summaries": chunk_summaries,
                "final_report": report,
                "params": {
                    "map": asdict(self.node_gen.summarization_map),
                    "reduce": asdict(self.node_gen.summarization_reduce),
                },
                "duration_s": duration_s,
                "logged_at": datetime.now(timezone.utc).isoformat(),
            }

            log_dir = Path(self.summarization_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / f"{safe_id}_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            # Логирование — вспомогательная функция; сбой записи не должен ломать сам ответ.
            logger.exception("Не удалось записать лог суммаризации для %s", state.get("target_article_id"))

    def research_step_node(self, state: AgentState) -> Dict[str, Any]:
        if self.debug_mode:
            return {"final_answer": "DEBUG MODE: research-цикл пропущен.", "evidence": [], "iterations": 0}

        evidence = state.get("evidence") or []
        iterations = state.get("iterations", 0)

        evidence_text = (
            "\n\n".join(f"[{i + 1}] {e['tool']}({e['args']}):\n{e['content']}" for i, e in enumerate(evidence))
            or "(пока ничего не найдено)"
        )
        conversation = self.prompt_resolver.format_chat(
            self.resolved_prompts["research_step"], {"query": state["query"], "evidence": evidence_text}
        )

        fallback_answer = "Не удалось найти достаточно информации на arXiv для уверенного ответа."
        default = ResearchDecision(action="final_answer", answer=fallback_answer, confidence="low")
        decision = generate_structured(
            self.llm, [conversation], ResearchDecision, asdict(self.node_gen.research_step), [default]
        )[0]
        logger.info(
            "research_step: iteration=%d decision.action=%s tool=%s confidence=%s",
            iterations, decision.action, decision.tool, decision.confidence,
        )

        if iterations < self.min_research_iterations and decision.action == "final_answer":
            logger.info(
                "research_step: overriding premature final_answer — forcing a grounding tool "
                "call (min_research_iterations=%d not yet met)", self.min_research_iterations,
            )
            decision = ResearchDecision(action="call_tool", tool="search_arxiv", tool_args={"query": state["query"]})

        if decision.action == "call_tool":
            decision = self._normalize_tool_call(decision, state["query"])

        # Лимит исчерпан — либо модель зациклилась и просит ровно тот же вызов, что уже
        # успешно отработал (наблюдалось живьём: два одинаковых search_arxiv подряд).
        # В обоих случаях новых данных не появится, и итерацию тратить не на что.
        exhausted = iterations >= self.max_research_iterations
        repeated = decision.action == "call_tool" and self._already_called(evidence, decision)
        if decision.action == "call_tool" and (exhausted or repeated):
            logger.warning(
                "research_step: synthesizing final answer from evidence (exhausted=%s repeated_call=%s)",
                exhausted, repeated,
            )
            decision = self._force_final_answer(conversation, decision, fallback_answer)

        if decision.action == "final_answer":
            sources = sorted({s for e in evidence for s in e.get("sources", [])})
            return {
                "final_answer": decision.answer or fallback_answer,
                "evidence": evidence,
                "iterations": iterations,
                "sources": sources,
            }

        tool_result = self.toolkit.dispatch(decision.tool or "", decision.tool_args or {})
        logger.info("research_step: called tool=%s ok=%s", tool_result.name, tool_result.ok)
        new_evidence = evidence + [
            {"tool": tool_result.name, "args": tool_result.args, "content": tool_result.content, "sources": tool_result.sources}
        ]
        return {"evidence": new_evidence, "iterations": iterations + 1}

    def _force_final_answer(
        self, conversation: List[Dict[str, str]], decision: ResearchDecision, fallback_answer: str
    ) -> ResearchDecision:
        """Лимит вызовов инструментов исчерпан, а модель всё ещё просит инструмент.

        Раньше здесь просто бралось `decision.answer or fallback_answer` — но при
        action="call_tool" поле `answer` практически всегда пустое, поэтому пользователь
        получал канцелярское «не удалось найти достаточно информации» ДАЖЕ ЕСЛИ агент к
        этому моменту успешно скачал полные тексты статей. Весь собранный evidence
        выбрасывался. Теперь делаем один дополнительный вызов, явно сняв инструменты со
        стола, чтобы модель сформулировала ответ по тому, что уже найдено.
        """
        forced_conversation = [
            *conversation,
            {
                "role": "user",
                "content": (
                    "Лимит вызовов инструментов исчерпан, инструменты больше недоступны. "
                    'Верни action="final_answer" и дай содержательный ответ строго по уже '
                    "собранным выше данным. Если их не хватает для полного ответа — честно "
                    "скажи об этом, но обязательно изложи то, что удалось найти."
                ),
            },
        ]
        default = ResearchDecision(
            action="final_answer", answer=decision.answer or fallback_answer, confidence="low"
        )
        forced = generate_structured(
            self.llm, [forced_conversation], ResearchDecision, asdict(self.node_gen.research_step), [default]
        )[0]
        return ResearchDecision(
            action="final_answer",
            answer=forced.answer or decision.answer or fallback_answer,
            confidence=forced.confidence or "low",
        )

    @staticmethod
    def _already_called(evidence: List[Dict[str, Any]], decision: ResearchDecision) -> bool:
        """Был ли ровно такой (успешный) вызов уже сделан. Неудачные вызовы не считаем —
        повторить упавший запрос может быть осмысленно."""
        return any(
            e["tool"] == decision.tool and e["args"] == (decision.tool_args or {}) and e.get("sources")
            for e in evidence
        )

    @staticmethod
    def _normalize_tool_call(decision: ResearchDecision, query: str) -> ResearchDecision:
        """Малые модели регулярно возвращают action="call_tool" без `tool`/без аргументов.
        Без этой нормализации dispatch("") отдавал бы «Неизвестный инструмент», а
        search_arxiv — искал по пустой строке; и то и другое просто засоряло evidence
        мусором и сжигало итерацию."""
        tool = decision.tool or "search_arxiv"
        args = dict(decision.tool_args or {})
        # get_fulltext без article_id заведомо провалится и просто сожжёт итерацию —
        # осмысленнее потратить её на поиск, который как раз и даёт ID.
        if tool == "get_fulltext" and not str(args.get("article_id", "")).strip():
            tool, args = "search_arxiv", {}
        if tool == "search_arxiv" and not str(args.get("query", "")).strip():
            args["query"] = query
        if tool != decision.tool or args != (decision.tool_args or {}):
            logger.info("research_step: normalized incomplete tool call -> %s(%s)", tool, args)
        return ResearchDecision(action="call_tool", tool=tool, tool_args=args)

    # ---------------------------------------------------------------- graphs

    def _add_summarize_chain(self, wf: StateGraph) -> None:
        """Общая часть графа summarize-пути: fetch -> process -> map-reduce. Используется
        и основным графом (после resolve_target_article), и отдельным summarize_app
        (после явного выбора статьи пользователем)."""
        wf.add_node(NodeName.FETCH_FULLTEXT.value, self.fetch_fulltext_node)
        wf.add_node(NodeName.PROCESS_AND_CHUNK.value, self.process_and_chunk_node)
        wf.add_node(NodeName.MAP_REDUCE_SUMMARIZE.value, self.map_reduce_summarize_node)

        def route_after_fetch(state: AgentState) -> str:
            return "ok" if state.get("raw_sections") else "stop"

        wf.add_conditional_edges(
            NodeName.FETCH_FULLTEXT.value,
            route_after_fetch,
            {"ok": NodeName.PROCESS_AND_CHUNK.value, "stop": END},
        )
        wf.add_edge(NodeName.PROCESS_AND_CHUNK.value, NodeName.MAP_REDUCE_SUMMARIZE.value)
        wf.add_edge(NodeName.MAP_REDUCE_SUMMARIZE.value, END)

    def _build_summarize_subgraph(self):
        wf = StateGraph(AgentState)
        self._add_summarize_chain(wf)
        wf.set_entry_point(NodeName.FETCH_FULLTEXT.value)
        return wf.compile()

    def _build_graph(self):
        wf = StateGraph(AgentState)

        wf.add_node(NodeName.CLASSIFIER.value, self.classifier_node)
        wf.add_node(NodeName.OTHER_HANDLER.value, self.other_node)
        wf.add_node(NodeName.RESOLVE_TARGET_ARTICLE.value, self.resolve_target_article_node)
        wf.add_node(NodeName.RESEARCH_STEP.value, self.research_step_node)
        self._add_summarize_chain(wf)

        wf.set_entry_point(NodeName.CLASSIFIER.value)

        def route_after_classifier(state: AgentState) -> str:
            return {
                "summarize": NodeName.RESOLVE_TARGET_ARTICLE.value,
                "research": NodeName.RESEARCH_STEP.value,
            }.get(state["intent"], NodeName.OTHER_HANDLER.value)

        wf.add_conditional_edges(
            NodeName.CLASSIFIER.value,
            route_after_classifier,
            {
                NodeName.OTHER_HANDLER.value: NodeName.OTHER_HANDLER.value,
                NodeName.RESOLVE_TARGET_ARTICLE.value: NodeName.RESOLVE_TARGET_ARTICLE.value,
                NodeName.RESEARCH_STEP.value: NodeName.RESEARCH_STEP.value,
            },
        )

        def route_after_resolve(state: AgentState) -> str:
            return "found" if state.get("target_article_id") else "stop"

        wf.add_conditional_edges(
            NodeName.RESOLVE_TARGET_ARTICLE.value,
            route_after_resolve,
            {"found": NodeName.FETCH_FULLTEXT.value, "stop": END},
        )

        def route_after_research(state: AgentState) -> str:
            return "final" if state.get("final_answer") else "continue"

        wf.add_conditional_edges(
            NodeName.RESEARCH_STEP.value,
            route_after_research,
            {"final": END, "continue": NodeName.RESEARCH_STEP.value},
        )

        wf.add_edge(NodeName.OTHER_HANDLER.value, END)

        return wf.compile()

    def invoke(self, query: str) -> Dict[str, Any]:
        # research_step — циклический узел, и каждая его итерация тратит шаг из общего
        # лимита рекурсии LangGraph (по умолчанию 25). При большом
        # max_research_iterations граф падал бы с GraphRecursionError раньше, чем
        # сработал бы собственный лимит агента.
        recursion_limit = max(25, self.max_research_iterations * 2 + 10)
        return self.app.invoke({"query": query}, config={"recursion_limit": recursion_limit})

    def summarize_article(self, article_id: str) -> Dict[str, Any]:
        """Суммаризация конкретной, уже выбранной пользователем статьи — минует
        classifier/resolve_target_article (ID уже известен, выбор уже сделан)."""
        query = f"Summarize arXiv:{article_id}"
        return self.summarize_app.invoke({"query": query, "target_article_id": article_id})

    # ---------------------------------------------------------------- streaming

    def _summarize_chain_stream(self, state: AgentState) -> Iterator[StreamEvent]:
        """fetch -> chunk -> map-reduce, стримингом (state уже содержит target_article_id).
        Вызывает те же node-методы, что и обычный граф, напрямую — см. modules/streaming.py
        про то, почему не через LangGraph.stream()."""
        t0 = time.perf_counter()
        yield ProgressEvent(
            stage="fetch_fulltext", message=f"Загружаю текст статьи {state.get('target_article_id')}…"
        )
        state.update(self.fetch_fulltext_node(state))
        if not state.get("raw_sections"):
            yield FinalEvent(result=dict(state))
            return

        yield ProgressEvent(stage="process_and_chunk", message="Разбиваю статью на фрагменты…")
        state.update(self.process_and_chunk_node(state))

        chunks = state.get("article_chunks") or {}
        yield ProgressEvent(
            stage="map_reduce_summarize", message=f"Извлекаю факты из {len(chunks)} фрагментов…", total=len(chunks)
        )

        chunk_summaries: Dict[str, str] = {}
        final_parts: List[str] = []
        for event in self.sum_pipeline.run_stream(
            chunks,
            map_params=asdict(self.node_gen.summarization_map),
            reduce_params=asdict(self.node_gen.summarization_reduce),
        ):
            if isinstance(event, ChunkDoneEvent):
                chunk_summaries[event.title] = event.summary
            elif isinstance(event, TextDeltaEvent):
                final_parts.append(event.text)
            yield event

        report = "".join(final_parts)
        header = f"# {state.get('article_title', '')}\n🔗 [PDF]({state.get('article_pdf_url', '')})\n\n"
        state["final_answer"] = header + report
        state["debug_data"] = chunk_summaries
        self._log_summarization(state, chunk_summaries, report, time.perf_counter() - t0)
        yield FinalEvent(result=dict(state))

    def _research_stream(self, state: AgentState) -> Iterator[StreamEvent]:
        rate = RateEstimator()
        hard_cap = self.max_research_iterations + 3  # тот же запас, что recursion_limit в invoke()
        for _ in range(hard_cap):
            current_iter = state.get("iterations", 0)
            yield ProgressEvent(
                stage="research_step", message=f"Ищу и анализирую источники (шаг {current_iter + 1})…",
                current=current_iter, total=self.max_research_iterations,
                elapsed_s=rate.elapsed(), eta_s=rate.eta(current_iter, self.max_research_iterations),
            )
            delta = self.research_step_node(state)
            state.update(delta)
            if delta.get("final_answer") is not None:
                break
        yield FinalEvent(result=dict(state))

    def invoke_stream(self, query: str) -> Iterator[StreamEvent]:
        """Потоковый эквивалент invoke(): progress-события по ходу узлов + токены
        reduce-стадии для summarize-ветки, финальный dict — тем же FinalEvent, что раньше
        возвращал invoke() целиком (в т.ч. candidates, если explicit ID не нашёлся)."""
        state: AgentState = {"query": query}

        yield ProgressEvent(stage="classifier", message="Определяю тип запроса…")
        state.update(self.classifier_node(state))
        intent = state.get("intent")

        if intent == "other":
            state.update(self.other_node(state))
            yield FinalEvent(result=dict(state))
            return

        if intent == "summarize":
            yield ProgressEvent(stage="resolve_target_article", message="Ищу статью на arXiv…")
            state.update(self.resolve_target_article_node(state))
            if not state.get("target_article_id"):
                yield FinalEvent(result=dict(state))
                return
            yield from self._summarize_chain_stream(state)
            return

        yield from self._research_stream(state)

    def summarize_article_stream(self, article_id: str) -> Iterator[StreamEvent]:
        """Потоковый эквивалент summarize_article()."""
        query = f"Summarize arXiv:{article_id}"
        state: AgentState = {"query": query, "target_article_id": article_id}
        yield from self._summarize_chain_stream(state)
