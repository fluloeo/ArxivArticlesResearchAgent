import logging
import threading
from concurrent import futures
from typing import Any, Dict, Optional

import grpc

from grpc_service.generated import arxiv_agent_pb2, arxiv_agent_pb2_grpc
from modules.bootstrap import build_agent
from modules.config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_ask_response(result: Dict[str, Any], intent_override: Optional[str] = None):
    tool_calls = [f"{e['tool']}({e['args']})" for e in result.get("evidence") or []]
    candidates = [
        arxiv_agent_pb2.ArticleCandidate(arxiv_id=c["arxiv_id"], title=c["title"], abstract=c["abstract"])
        for c in result.get("candidates") or []
    ]
    # debug_data (modules/agent.py::map_reduce_summarize_node) — map-выжимки по разделам,
    # раньше терялись на границе gRPC (были в state, но AskResponse их не нёс): достаются
    # только в проводе, через notebook_utils.visualize() в ноутбуке или прямой agent.invoke().
    map_summaries = [
        arxiv_agent_pb2.ChunkSummary(title=title, summary=summary)
        for title, summary in (result.get("debug_data") or {}).items()
    ]

    kwargs: Dict[str, Any] = dict(
        final_answer=result.get("final_answer", ""),
        intent=intent_override or result.get("intent", ""),
        candidates=candidates,
        sources=result.get("sources") or [],
        tool_calls=tool_calls,
        map_summaries=map_summaries,
    )
    if result.get("faithfulness") is not None:
        kwargs["faithfulness"] = result["faithfulness"]
    if result.get("answer_relevancy") is not None:
        kwargs["answer_relevancy"] = result["answer_relevancy"]

    return arxiv_agent_pb2.AskResponse(**kwargs)


class ArxivAgentServicer(arxiv_agent_pb2_grpc.ArxivAgentServiceServicer):
    """gRPC-обвязка вокруг ArxivAgent. Сам ArxivAgent (и LLM) собирается один раз при
    старте сервера через modules.bootstrap.build_agent — единую точку сборки зависимостей.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.agent = build_agent(config)
        # MLX/vLLM-модель — общий изменяемый ресурс; сериализуем обращения к ней,
        # чтобы параллельные gRPC-запросы не дрались за один и тот же инференс-движок.
        self._inference_lock = threading.Lock()

    def Ask(self, request, context):
        query = request.query.strip()
        if not query:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query не может быть пустым")
            return arxiv_agent_pb2.AskResponse()

        try:
            with self._inference_lock:
                result = self.agent.invoke(query, compute_metrics=not request.skip_metrics)
        except Exception:
            logger.exception("agent.invoke failed for query=%r", query)
            return arxiv_agent_pb2.AskResponse(error="Внутренняя ошибка агента. Подробности — в логах сервера.")

        return _build_ask_response(result)

    def SummarizeArticle(self, request, context):
        article_id = request.article_id.strip()
        if not article_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("article_id не может быть пустым")
            return arxiv_agent_pb2.AskResponse()

        try:
            with self._inference_lock:
                result = self.agent.summarize_article(article_id, compute_metrics=not request.skip_metrics)
        except Exception:
            logger.exception("agent.summarize_article failed for article_id=%r", article_id)
            return arxiv_agent_pb2.AskResponse(error="Внутренняя ошибка агента. Подробности — в логах сервера.")

        return _build_ask_response(result, intent_override="summarize")

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
