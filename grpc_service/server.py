import logging
import threading
from concurrent import futures
from typing import Any, Dict, Iterator, Optional

import grpc

from grpc_service.generated import arxiv_agent_pb2, arxiv_agent_pb2_grpc
from modules.bootstrap import build_agent
from modules.config import AppConfig
from modules.streaming import ChunkDoneEvent, FinalEvent, ProgressEvent, StreamEvent, TextDeltaEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_ask_response(result: Dict[str, Any], intent_override: Optional[str] = None):
    tool_calls = [f"{e['tool']}({e['args']})" for e in result.get("evidence") or []]
    candidates = [
        arxiv_agent_pb2.ArticleCandidate(arxiv_id=c["arxiv_id"], title=c["title"], abstract=c["abstract"])
        for c in result.get("candidates") or []
    ]
    map_summaries = [
        arxiv_agent_pb2.ChunkSummary(title=title, summary=summary)
        for title, summary in (result.get("debug_data") or {}).items()
    ]

    return arxiv_agent_pb2.AskResponse(
        final_answer=result.get("final_answer", ""),
        intent=intent_override or result.get("intent", ""),
        candidates=candidates,
        sources=result.get("sources") or [],
        tool_calls=tool_calls,
        map_summaries=map_summaries,
    )


def _event_to_ask_event(event: StreamEvent, intent_override: Optional[str] = None) -> "arxiv_agent_pb2.AskEvent":
    if isinstance(event, ProgressEvent):
        return arxiv_agent_pb2.AskEvent(
            progress=arxiv_agent_pb2.ProgressUpdate(
                stage=event.stage, message=event.message, current=event.current, total=event.total,
                elapsed_s=event.elapsed_s, eta_s=event.eta_s if event.eta_s is not None else -1.0,
            )
        )
    if isinstance(event, TextDeltaEvent):
        return arxiv_agent_pb2.AskEvent(delta=arxiv_agent_pb2.TextDelta(text=event.text))
    if isinstance(event, ChunkDoneEvent):
        return arxiv_agent_pb2.AskEvent(
            map_summary=arxiv_agent_pb2.ChunkSummary(title=event.title, summary=event.summary)
        )
    if isinstance(event, FinalEvent):
        return arxiv_agent_pb2.AskEvent(final=_build_ask_response(event.result, intent_override=intent_override))
    raise TypeError(f"Неизвестный тип события стрима: {event!r}")


class ArxivAgentServicer(arxiv_agent_pb2_grpc.ArxivAgentServiceServicer):
    """gRPC-обвязка вокруг ArxivAgent. Сам ArxivAgent (и LLM) собирается один раз при
    старте сервера через modules.bootstrap.build_agent — единую точку сборки зависимостей.

    Ask/SummarizeArticle — server-streaming (см. .proto): граф исполняется минутами,
    поэтому вместо одного unary-ответа отдаём поток AskEvent (прогресс, токены reduce-
    стадии, map-выжимки по готовности), последним элементом — payload=final.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.agent = build_agent(config)
        # MLX/vLLM-модель — общий изменяемый ресурс; сериализуем обращения к ней,
        # чтобы параллельные gRPC-запросы не дрались за один и тот же инференс-движок.
        # `with` вокруг генератора удерживает лок на всё время стрима — тот же эффект,
        # что раньше давала блокирующая unary-реализация.
        self._inference_lock = threading.Lock()

    def Ask(self, request, context) -> Iterator["arxiv_agent_pb2.AskEvent"]:
        query = request.query.strip()
        if not query:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query не может быть пустым")
            return

        try:
            with self._inference_lock:
                for event in self.agent.invoke_stream(query):
                    yield _event_to_ask_event(event)
        except Exception:
            logger.exception("agent.invoke_stream failed for query=%r", query)
            yield arxiv_agent_pb2.AskEvent(
                final=arxiv_agent_pb2.AskResponse(error="Внутренняя ошибка агента. Подробности — в логах сервера.")
            )

    def SummarizeArticle(self, request, context) -> Iterator["arxiv_agent_pb2.AskEvent"]:
        article_id = request.article_id.strip()
        if not article_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("article_id не может быть пустым")
            return

        try:
            with self._inference_lock:
                for event in self.agent.summarize_article_stream(article_id):
                    yield _event_to_ask_event(event, intent_override="summarize")
        except Exception:
            logger.exception("agent.summarize_article_stream failed for article_id=%r", article_id)
            yield arxiv_agent_pb2.AskEvent(
                final=arxiv_agent_pb2.AskResponse(error="Внутренняя ошибка агента. Подробности — в логах сервера.")
            )

    def HealthCheck(self, request, context):
        return arxiv_agent_pb2.HealthCheckResponse(ok=True, backend=self.config.llm_backend)


def serve() -> None:
    config = AppConfig.from_env()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    arxiv_agent_pb2_grpc.add_ArxivAgentServiceServicer_to_server(ArxivAgentServicer(config), server)

    address = f"{config.grpc_host}:{config.grpc_port}"
    server.add_insecure_port(address)
    server.start()
    logger.info("ArxivAgent gRPC server listening on %s (backend=%s)", address, config.llm_backend)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
