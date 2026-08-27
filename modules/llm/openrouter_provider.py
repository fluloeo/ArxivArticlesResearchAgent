import asyncio
import logging
import queue
import threading
from typing import Any, Dict, Iterator, List, Tuple

from openai import AsyncOpenAI

from .base import Conversation, LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """Опциональный облачный бэкенд поверх OpenRouter (OpenAI-совместимый API).

    В отличие от MLXProvider (один процесс держит одну локальную модель — конкурентные
    вызовы физически некому обслуживать, отсюда строго последовательный executor), здесь
    каждый элемент батча — независимый HTTP-запрос к чужому серверу, и они не делят общий
    ресурс. Реализовано через `asyncio` + `AsyncOpenAI` (не через ThreadPoolExecutor):
    для I/O-bound HTTP-запросов это тот же результат по факту, но без накладных расходов
    ОС на поток под каждый одновременный запрос и с точным контролем конкурентности через
    `asyncio.Semaphore`, а не через размер пула потоков.

    Публичный интерфейс (`generate`, `generate_as_completed`) остаётся синхронным — вся
    остальная кодовая база (agent.py, SummarizationPipeline) синхронная, и городить async
    через неё только ради одного провайдера не стоит. Каждый синхронный вызов открывает
    свой event loop через `asyncio.run(...)` и закрывает его по завершении — вызовы
    `generate`/`generate_as_completed` не переиспользуют loop между собой, но это и не
    нужно: они не вызываются рекурсивно друг из друга.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen/qwen3-30b-a3b-instruct-2507",
        max_concurrency: int = 8,
    ):
        if not api_key:
            raise ValueError("OpenRouterProvider требует OPENROUTER_API_KEY")
        self._api_key = api_key
        self._base_url = "https://openrouter.ai/api/v1"
        self.model_name = model_name
        self.max_concurrency = max_concurrency

    def _new_client(self) -> AsyncOpenAI:
        """Новый клиент на каждый independent event loop (см. docstring класса — generate/
        generate_as_completed/generate_stream каждый крутят свой asyncio.run()). Один общий
        AsyncOpenAI, переживший несколько таких прогонов, держит httpx-пул соединений,
        привязанный к ПЕРВОМУ из них: когда тот loop закрывается, а следующий вызов (другой
        поток/loop) пытается переиспользовать/закрыть тот же пул, падает
        `RuntimeError: Event loop is closed` — например map-стадия суммаризации отрабатывала,
        а reduce (в отдельном потоке/loop сразу следом) молча падал с этой ошибкой внутри
        try/except и не выдавал ни одного TextDeltaEvent. AsyncOpenAI дёшев в создании (без
        сетевого I/O в конструкторе), так что новый клиент на каждый loop ничего не стоит."""
        return AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)

    async def _acreate_one(
        self, client: AsyncOpenAI, conversation: Conversation, sampling_params: Dict[str, Any]
    ) -> str:
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=list(conversation),
                temperature=sampling_params.get("temperature", 0),
                max_tokens=sampling_params.get("max_tokens", 1024),
                # frequency_penalty раньше молча отбрасывался (передавались только
                # temperature/max_tokens), хотя это штатный параметр OpenAI-совместимого
                # API и он объявлен в NodeGenerationConfig для map-фазы суммаризации.
                frequency_penalty=sampling_params.get("frequency_penalty", 0.0),
                extra_body={"include_reasoning": False},
            )
            message = response.choices[0].message
            content = getattr(message, "content", None)
            reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
            return content or reasoning or ""
        except Exception as e:
            logger.exception("OpenRouter generation failed")
            return f"Error: {e}"

    async def _agather_ordered(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        client = self._new_client()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _bounded(conv: Conversation) -> str:
            async with semaphore:
                return await self._acreate_one(client, conv, sampling_params)

        return await asyncio.gather(*(_bounded(c) for c in conversations))

    async def _acreate_single(self, conversation: Conversation, sampling_params: Dict[str, Any]) -> str:
        client = self._new_client()
        return await self._acreate_one(client, conversation, sampling_params)

    def generate(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        if not conversations:
            return []
        if len(conversations) == 1:
            return [asyncio.run(self._acreate_single(conversations[0], sampling_params))]
        return asyncio.run(self._agather_ordered(conversations, sampling_params))

    def generate_as_completed(
        self, conversations: List[Conversation], sampling_params: Dict[str, Any]
    ) -> Iterator[Tuple[int, str]]:
        """Реальная параллельная версия: (индекс, текст) отдаются по мере готовности
        КАЖДОГО запроса, а не по завершении всего батча — на этом строится
        прогресс/ETA map-стадии суммаризации (evaluation.streaming.ChunkDoneEvent).

        asyncio.run() блокирует поток вызова до завершения event loop, а нам нужно
        отдавать результаты вызывающему коду ПО ХОДУ — поэтому event loop крутится в
        отдельном потоке, а результаты передаются наружу через thread-safe очередь (тот
        же паттерн генератор-через-очередь, что MLXProvider.generate_stream использует
        для мостика между mlx-потоком и потоком вызова)."""
        if not conversations:
            return
        q: "queue.Queue[Any]" = queue.Queue()
        _SENTINEL = object()

        async def _runner() -> None:
            client = self._new_client()
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _one(i: int, conv: Conversation) -> None:
                async with semaphore:
                    text = await self._acreate_one(client, conv, sampling_params)
                    q.put((i, text))

            await asyncio.gather(*(_one(i, c) for i, c in enumerate(conversations)))

        def _thread_main() -> None:
            try:
                asyncio.run(_runner())
            finally:
                q.put(_SENTINEL)

        threading.Thread(target=_thread_main, daemon=True).start()
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            yield item

    def generate_stream(self, conversation: Conversation, sampling_params: Dict[str, Any]) -> Iterator[str]:
        q: "queue.Queue[Any]" = queue.Queue()
        _SENTINEL = object()

        async def _runner() -> None:
            try:
                client = self._new_client()
                stream = await client.chat.completions.create(
                    model=self.model_name,
                    messages=list(conversation),
                    temperature=sampling_params.get("temperature", 0),
                    max_tokens=sampling_params.get("max_tokens", 1024),
                    frequency_penalty=sampling_params.get("frequency_penalty", 0.0),
                    extra_body={"include_reasoning": False},
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        q.put(delta)
            except Exception:
                logger.exception("OpenRouter streaming generation failed")
            finally:
                q.put(_SENTINEL)

        def _thread_main() -> None:
            asyncio.run(_runner())

        threading.Thread(target=_thread_main, daemon=True).start()
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            yield item
