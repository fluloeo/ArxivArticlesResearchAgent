"""Единственный источник "текущий узел" для атрибуции логов и LLM-вызовов.

Устанавливается NodeWrapper (recorder.py) вокруг каждого исполнения узла-функции — то есть
на том же потоке, на котором LangGraph синхронно вызывает node_fn(state). ContextVar, а не
обычная переменная — потому что research_step зациклен на себя и несколько разных "текущих
узлов" (в т.ч. вложенные вызовы) не должны путаться при параллельных прогонах кейсов.

ВАЖНО — граница потока: ContextVar НЕ пробрасывается автоматически через
`concurrent.futures.ThreadPoolExecutor.submit()` (проверено: обычный submit() в дочернем
потоке видит только default, значение из чтения на другом потоке теряется — нужен явный
`contextvars.copy_context().run(...)`, который submit() сам не делает). MLXProvider.generate()
(modules/llm/mlx_provider.py) устроен как раз так: submit() в свой единственный inference-
поток + .result() — но это неважно для RecordingProvider (provider_wrapper.py), потому что
он читает CURRENT_NODE ДО этого submit(), на графовом потоке, где и был установлен контекст;
.result() лишь синхронно блокирует тот же графовый поток. Читать CURRENT_NODE изнутри
mlx_provider._generate_on_mlx_thread напрямую — уже ловушка, и делать это не нужно.
"""
import contextvars
from typing import Optional, Tuple

# (имя_узла, порядковый_номер_визита) — occurrence нужен, потому что research_step
# зациклен на себя и одно и то же имя узла посещается несколько раз за кейс.
CURRENT_NODE: "contextvars.ContextVar[Tuple[Optional[str], int]]" = contextvars.ContextVar(
    "CURRENT_NODE", default=(None, 0)
)
