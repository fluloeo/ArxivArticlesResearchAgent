"""Общий rate limiter для ВСЕХ обращений к инфраструктуре arXiv — Atom API поиска
(export.arxiv.org), ar5iv (ar5iv.labs.arxiv.org) и PDF (arxiv.org/pdf/...).

Раньше троттлинг был только у ArxivSearchClient (свой `_throttle`, приватный для этого
класса) — modules.arxiv_source.fulltext.fetch_sections вообще не троттлился. А
`SqliteArxivArticleStore._fetch_and_cache` вызывает `search_client.get_by_id()` (поиск,
троттлился) и сразу следом `fetch_sections()` (ar5iv+PDF, НЕ троттлился) — со стороны
arXiv это выглядит как всплеск нескоординированных запросов. После достаточного числа
таких всплесков IP временно попадает под более жёсткий троттлинг на стороне arXiv, и это
проявляется как «arxiv в какой-то момент перестаёт отвечать», хотя отдельные запросы вроде
бы и укладывались в лимит 1/3с — просто каждый укладывался в лимит СВОЕГО класса запросов,
а не общий.

Один `RateLimiter`, инжектируемый и в ArxivSearchClient, и в fetch_sections, устраняет
это: все запросы делят один и тот же минимальный интервал, независимо от того, какой это
хост и какой код его вызвал.
"""
import threading
import time


class RateLimiter:
    def __init__(self, min_interval_sec: float = 3.0):
        self.min_interval_sec = min_interval_sec
        self._lock = threading.Lock()
        self._last_request_ts = 0.0

    def wait(self) -> None:
        """Блокирует вызывающий поток ровно настолько, чтобы с последнего запроса (любого,
        через любой класс) прошло не меньше min_interval_sec. Лок держится и на время сна
        намеренно — это и есть сериализация конкурентных запросов (несколько потоков gRPC-
        воркеров) в один поток к arXiv."""
        with self._lock:
            wait = self.min_interval_sec - (time.monotonic() - self._last_request_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_request_ts = time.monotonic()
