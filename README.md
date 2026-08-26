# AI-агент для поиска и суммаризации научных статей (arXiv.org)

Данный репозиторий содержит исходный код системы для анализа научной литературы на базе LLM. Проект автоматизирует цикл работы с ArXiv: от суммаризации отдельной статьи до ответа на исследовательские вопросы через агента с function calling.

## Зачем это нужно

**Кому.** Исследователю, аспиранту, ML-инженеру, который следит за потоком arXiv (~20 тыс. статей в месяц).

**Боль.** Аннотации не хватает, чтобы понять, нужна ли статья — приходится вычитывать методику и результаты целиком. Поиск arXiv лексический и практически англоязычный: он не понимает вопрос на естественном языке, тем более на русском.

**Ценность.** Путь «вопрос → обоснованный ответ со ссылками на статьи» — с часов до минут.

**Две работы, которые продукт делает (Jobs To Be Done):**
1. **«Разобраться в конкретной статье»** — структурированный обзор статьи целиком, а не пересказ аннотации. Успех = обзор ничего не выдумал (**faithfulness**) и ничего важного не потерял (**coverage**).
2. **«Сориентироваться в теме»** — ответ на исследовательский вопрос, опирающийся на реально найденные статьи, со списком источников. Успех = ответ по существу вопроса (**answer relevancy**) и обоснован найденным (**faithfulness**).

**Не-цели.** Не заменяет чтение статьи для глубокой работы; не строит полноценный векторный индекс arXiv (см. ниже, почему); не делает систематический обзор/мета-анализ; не оценивает научную новизну.

Обе работы и обе пары метрик измеряются не на инференсе, а отдельным офлайн-харнессом — см. [«Эксперименты и оценка качества»](#эксперименты-и-оценка-качества) ниже.


## Архитектура системы

Поток управления реализован как граф состояний с помощью **LangGraph**:

1.  **Intent Classifier** (structured output) — определяет цель пользователя:
    *   `summarize`: глубокая суммаризация одной статьи (Map-Reduce);
    *   `research`: вопрос, требующий поиска фактов/статей — обрабатывается агентом с function calling;
    *   `other`: обработка нерелевантных запросов (приветствия, оффтоп).
2.  **Summarize-ветка**: `resolve_target_article` — явный arXiv ID в запросе определяется регуляркой и обрабатывается сразу; если ID нет, узел **не пытается угадать статью сам**, а возвращает top-N кандидатов из поиска — выбор делает пользователь (в Streamlit) или клиент gRPC, вызывая отдельный RPC `SummarizeArticle`. Дальше — `fetch_fulltext` (ar5iv/PDF, с SQLite-кэшем) → `process_and_chunk` (Merge & Split) → `map_reduce_summarize` → `ragas_eval`.
3.  **Research-ветка (function calling вместо RAG)**: цикл `research_step`, ограниченный `max_research_iterations`. На каждом шаге модель через structured output решает: дать финальный ответ прямо сейчас, либо вызвать инструмент — `search_arxiv` (поиск статей по теме) или `get_fulltext` (прочитать конкретную статью). Результат инструмента добавляется в накопленный контекст (`evidence`), и цикл повторяется, пока модель не сочтёт данных достаточно или не будет достигнут лимит итераций. **Grounding-гарантия**: минимум `min_research_iterations` (по умолчанию 1) вызовов инструмента обязательны, прежде чем разрешён финальный ответ — без этого маленькая модель иногда отвечала, ни разу не заглянув в arXiv (вплоть до того, что повторяла сам вопрос как «ответ»). Использованные при этом статьи попадают в `sources` итогового ответа.
4.  **RAGAS-оценка качества** (`ragas_eval`) — вместо прежнего узла критика: измеряет **faithfulness** и **answer relevancy** по методике [RAGAS](https://docs.ragas.io/), не переписывая ответ.

---

## Ключевые технические решения

### 1. Полнотекстовый слой поверх живого arXiv
Раньше текст статьи брался из Postgres; теперь `modules.arxiv_source` + `modules.article_store` тянут его напрямую с arXiv:
*   **Разбор идентификаторов** (`modules.arxiv_source.identifiers`) — единая точка распознавания arXiv ID/ссылок: новый формат (`2301.12345`), старый формат до апреля 2007 (`math/0702019`, `hep-th/9901001`, whitelist известных архивов — не любое `слово/1234567`), URL (`arxiv.org/abs|pdf/...`, `ar5iv.../html/...`), DOI (`10.48550/arXiv...`). Раньше старый формат не только не распознавался, но и **портился** при разборе Atom-фида (`raw_id.rsplit("/", 1)[-1]` от `.../abs/math/0702019v1` давал `"0702019v1"` — префикс архива терялся); такая статья есть в собственном тестовом корпусе (`evaluation/suites/summarization.yaml`, кейс `sum-math-0702019`).
*   **Explicit ID/ссылка — без поиска.** `ArxivToolkit.find_candidates()` — единая точка поиска, которой пользуются и summarize-ветка, и research-инструмент `search_arxiv`: если сам запрос оказывается arXiv ID или ссылкой на статью, идёт прямой `get_by_id`, поиск (и тем более LLM-rewriter) не вызывается вовсе.
*   **Query Rewriter** (`modules.query_rewriter`) — если запрос НЕ является явным ID, LLM извлекает структурированный `SearchPlan` (термины на английском, устойчивые фразы, авторы, категории, годы — arXiv лексический и практически англоязычный, поэтому перевод обязателен) и строится лестница из 4-5 field-scoped булевых запросов (`ti:`/`abs:`/`all:`/`au:`/`cat:`/`submittedDate:`) убывающей специфичности; первый уровень с непустым результатом и используется. Планы кэшируются в SQLite (`data/query_plan_cache.sqlite`) — LLM-вызов не повторяется на тот же запрос. Живая проверка на «что такое механизм внимания в трансформерах?»: без rewriter'а — 5 из 5 результатов не по теме (лексический поиск по кириллице выдаёт шум); с rewriter'ом — все 5 по теме.
*   **Лимиты API** — клиент соблюдает требование arXiv «не чаще одного запроса в 3 секунды» (`_throttle`) и повторяет запрос при 429/таймауте, но не повторяет собственные 4xx-ошибки (например пустой `id_list`).
*   **Полный текст, порядок попыток**: сначала [ar5iv](https://ar5iv.labs.arxiv.org) (LaTeXML-рендер — секции размечены `<h2>/<h3>`, почти один в один воспроизводит прежнюю структуру `{ЗАГОЛОВОК: текст}`); при неудаче — PDF (PyMuPDF) + regex-эвристика заголовков.
*   **SQLite-кэш** (`data/arxiv_cache.sqlite`) — повторные запросы к одной статье не бьют по сети снова.

### 2. Двухстадийный Article Processor (Merge & Split)
Для решения проблемы лимита контекста и сохранения структуры статьи реализован продвинутый алгоритм подготовки данных:
*   **Stage 1: Token-Aware Merging.** Мелкие подразделы статьи объединяются с соседями, пока не достигнут порога `min_tokens`. Это минимизирует количество вызовов LLM.
*   **Stage 2: Recursive Splitting.** Если после слияния или изначально секция превышает `max_tokens`, она рекурсивно разбивается на части, с сохранением нумерации в заголовках (например, *"Methodology (Part 1)"*).
*   **Stage 3: Custom Overlaps.** К каждому финальному чанку добавляются "теневые" контексты (`past_overlap` и `future_overlap`) из соседних фрагментов, обеспечивая плавность переходов в суммаризации.

### 3. Research-агент с обязательным grounding
Нативные tool-calling API у разных провайдеров несовместимы (у OpenRouter есть OpenAI-совместимый `tools=`, у локального MLX единого стандарта нет) — поэтому function calling реализован провайдер-агностично: `modules.structured_output.generate_structured()` накладывает JSON-инструкцию поверх любого промпта, валидирует ответ Pydantic-схемой (`modules.schemas.ResearchDecision`), при невалидном JSON делает один repair-запрос, а при повторном провале — безопасный дефолт с логированием. Модель решает `action: final_answer | call_tool`, а Python-код лишь диспетчеризирует вызов через `modules.tools.ArxivToolkit`. Если модель пытается ответить раньше, чем сделала `min_research_iterations` вызовов инструмента, решение принудительно подменяется на `call_tool` — это гарантирует, что ответ хоть в какой-то мере опирается на реально найденные данные, а не на "рефлекс" маленькой модели.

Два симметричных случая на границах цикла обрабатываются явно, потому что маленькие модели регулярно в них попадают:
*   `call_tool` **без** имени инструмента или с пустыми аргументами нормализуется в `search_arxiv(query=запрос пользователя)` — иначе итерация сжигалась впустую, а в `evidence` попадал мусор вроде «неизвестный инструмент».
*   Если лимит итераций исчерпан, а модель всё ещё просит инструмент, агент делает один дополнительный вызов, явно сняв инструменты со стола, и просит сформулировать ответ по уже собранным данным. Раньше в этой ветке пользователю отдавалась заглушка «не удалось найти достаточно информации» — даже когда полные тексты статей были успешно скачаны, весь `evidence` просто выбрасывался.

### 4. RAGAS вместо критика
Прежний критик правил отчёт по найденным замечаниям; вместо этого теперь **измеряются**, без переписывания, две метрики по методике RAGAS (`modules/ragas_eval.py`):
*   **Faithfulness** — LLM разбивает ответ на атомарные утверждения, затем для каждого отдельным вызовом проверяет, подтверждается ли оно контекстом (`supported: bool`); итог — доля подтверждённых утверждений.
*   **Answer Relevancy** — LLM генерирует несколько гипотетических вопросов, на которые ответ был бы хорошим ответом (с флагом `noncommittal` для уклончивых ответов); итог — средний косинус между эмбеддингом исходного вопроса и эмбеддингами сгенерированных. Это единственное место в проекте, где снова понадобились эмбеддинги (`sentence-transformers`, маленькая модель) — не для retrieval, а только для этой метрики.

Контекст для faithfulness — исходные чанки статьи (summarize) или содержимое `evidence` (research). Он почти всегда длиннее бюджета одного вызова, поэтому под **каждое** проверяемое утверждение подбираются лексически ближайшие к нему фрагменты контекста, а не просто его начало: иначе для длинной статьи любое утверждение из её второй половины автоматически считалось бы неподтверждённым и faithfulness падал бы тем сильнее, чем длиннее статья. Отбор живёт строго внутри метрики и на ответ пользователю не влияет (в оригинальном RAGAS на этом месте стоят `retrieved_contexts`, которых без ретривера нет).

Вопрос для answer relevancy — запрос пользователя; в summarize-ветке вместо голого ID (`"Summarize arXiv:1706.03762"`) подставляется формулировка по заголовку статьи, так как у ID нет осмысленного эмбеддинга и метрика получалась заниженной независимо от качества обзора.

**Опциональность и LLM-as-a-judge.** Метрики стоят дополнительных LLM-вызовов (разбор ответа на утверждения + проверка каждого), поэтому считаются по запросу: поле `skip_metrics` в `Ask`/`SummarizeArticle`, чекбокс в Streamlit, аргумент `compute_metrics` в `agent.invoke()`/`agent.summarize_article()`; глобально всё это отключается `APP_USE_RAGAS=false`. Судьёй по умолчанию выступает та же модель, что отвечает пользователю, но её можно развести с оцениваемой моделью: `APP_RAGAS_JUDGE_BACKEND=openrouter` + `APP_RAGAS_JUDGE_MODEL=...` поднимет отдельного судью по API.

---

## Отладка и мониторинг (Observability)

### 1. Интроспекция AgentState
Состояние агента (`AgentState`) доступно на каждом шаге:
*   `target_article_id` / `article_chunks` / `debug_data` / `candidates` — для summarize-ветки;
*   `evidence` (результаты вызовов инструментов) / `iterations` / `sources` — для research-ветки;
*   `faithfulness` / `answer_relevancy` — RAGAS-метрики, общие для обеих веток.

### 2. Интеграция с LangSmith
Проект интегрирован с **LangSmith**: визуализация графа, трассировка выполнения, версионирование промптов в Hub с fallback-механизмом на локальный `modules/prompts_local.yaml` (в отличие от прежнего `prompts.yaml`, этот файл коммитится вместе с репозиторием — агент рабочий "из коробки" даже без доступа к Hub).
*   [Промпты](https://smith.langchain.com/hub/fluloeo?organizationId=527befdf-145d-42e4-b03b-bdad31b098c3)

Общие для всех узлов имена (`modules.node_names.NodeName`) и класс `AgentTraceExporter` (`modules/eval.py`) позволяют выгружать трейсы в pandas для офлайн-анализа.

---

## Эксперименты и оценка качества

Метрики качества (faithfulness/coverage/answer relevancy) считаются не на инференсе, а отдельным офлайн-харнессом (`evaluation/`) — он не влияет на то, что видит пользователь, зато даёт узловую (per-node) валидацию всего графа, а не только оценку финального ответа.

**Как наблюдаются узлы графа без LangSmith.** `evaluation.tracing.GraphRecorder` подписывается на `app.stream(state, stream_mode="debug")` LangGraph — это единственный режим стрима, дающий и пред-состояние узла, и его дельту, и task id, и тайминги, притом устойчиво к зацикленным узлам (`research_step` посещается по несколько раз за прогон, каждый визит различим). Тайминг дополнительно снимается обёрткой вокруг самих функций-узлов (`app.nodes[name].bound.func`) — таймстемпы `debug`-режима искажаются задержкой потребителя потока, обёртка от этого не зависит. Третий, самый дешёвый источник сигнала — перехват уже существующих логов `modules.*` (`research_step: normalized incomplete tool call`, `Structured output repair failed`, `Could not fetch full text` и т.д.), атрибутированных текущим узлом: агент и так логирует всё нужное, просто раньше это не попадало в состояние графа.

**Что проверяется на каждом узле.** Два типа проверок: детерминированные **checks** (бесплатные, pass/fail — например `process_and_chunk`: точное совпадение `past_overlap`/`future_overlap` с хвостом/головой соседнего чанка, перцентили токенов; `map_reduce_summarize`: ни один чанк не потерян, нет пустых map-выжимок) и LLM-judge **метрики** (`faithfulness`, **`coverage`** — новая, зеркальная faithfulness: не что добавлено лишнего, а что упущено, `answer_relevancy`), которые считаются только там, где имеют смысл (`evaluation/metrics/applicability.py` — декларативный гейт: суммаризация → faithfulness+coverage, ответ на вопрос → answer_relevancy+faithfulness; запрос неприменимой метрики стоит ноль LLM-вызовов и даёт `status="na"`, а не тихий 0.0).

**Судья — не «та же модель».** В отличие от инференс-пути (где раньше был вариант `same`, судья = отвечающая модель), харнесс требует явный `--judge-model`: сравнение 4B vs 30B, где 4B к тому же судит саму себя, не значит ничего.

**Датасет.** `article_sample.json` (50 статей) + `dataset_gemini_final.json` (эталонные обзоры gemini-2.5-pro) — оба уже в репозитории (гитигнорены как `*.json`, получить отдельно). `InMemoryArticleStore`/`FrozenSearchClient` (`evaluation/dataset/offline_store.py`) дают полностью офлайн-прогон суммаризации: ноль сетевых запросов, ноль троттлинга.

```bash
python scripts/run_eval.py list-suites
python scripts/run_eval.py run --suite summarization --offline --limit 5
python scripts/run_eval.py run --suite search_recall --no-rewriter   # baseline для сравнения с rewriter'ом
```

Каждый прогон пишется в `evaluation/runs/<UTC>__<suite>__<label>__<git-sha>/` (манифест с git-состоянием и редактированными секретами, `nodes.jsonl`/`checks.jsonl`/`metrics.jsonl`/`events.jsonl`/`cases.jsonl`, потоковая запись с `flush()` — прогон, убитый на середине, остаётся анализируемым).

---

## Стек технологий
*   **Ядро:** LangGraph, LangChain.
*   **LLM:** [MLX](https://github.com/ml-explore/mlx) (локальный инференс на Apple Silicon, бэкенд по умолчанию — `Qwen3-4B-Instruct-2507` 4-бит), OpenRouter (облачный, опционально), vLLM (GPU, опционально).
*   **Данные:** живой arXiv API (Atom + ar5iv + PDF), SQLite (локальный кэш).
*   **Оценка качества:** RAGAS-метрики (faithfulness, answer relevancy), `sentence-transformers` только для эмбеддингов answer relevancy.
*   **Backend:** gRPC.
*   **UI:** Streamlit.
*   **Structured output / валидация:** Pydantic.
*   **Мониторинг:** LangSmith.

---

## Запуск проекта

### Настройка секретов и конфигурации
Создайте файл `.env`:
```text
# LangSmith (опционально, для tracing/hub)
LANGSMITH_API_KEY=your_key

# OpenRouter (только если APP_LLM_BACKEND=openrouter)
OPENROUTER_API_KEY=your_key

# Конфигурация приложения (у всех есть разумные дефолты, см. modules/config.py)
APP_LLM_BACKEND=mlx
APP_MLX_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit
APP_USE_HUB=false
APP_USE_RAGAS=true
APP_MIN_RESEARCH_ITERATIONS=1
APP_MAX_RESEARCH_ITERATIONS=3
APP_GRPC_HOST=localhost
APP_GRPC_PORT=50051

# LLM-as-a-judge для RAGAS: same (та же модель, дефолт) | openrouter | mlx
APP_RAGAS_JUDGE_BACKEND=same
# APP_RAGAS_JUDGE_MODEL=qwen/qwen3-30b-a3b-instruct-2507

# Файловый лог каждой суммаризации (id, чанки, map-выжимки, отчёт, тайминги);
# пустая строка отключает запись
APP_SUMMARIZATION_LOG_DIR=logs/summarizations
```

**Про выбор модели.** Дефолт — `Qwen3-4B-Instruct-2507-4bit` (~2.5 ГБ). Более крупная `Qwen3-30B-A3B-Instruct-2507-4bit` (~15-17 ГБ) заметно умнее, но на 24 ГБ unified memory под обычной десктопной нагрузкой уводит систему в своп: суммаризация одной статьи растягивалась с секунд до 5-7+ минут. Если памяти больше — `APP_MLX_MODEL` переопределяет дефолт.

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск gRPC backend
```bash
python -m grpc_service.server
```
Поднимает `ArxivAgent` один раз (через `modules.bootstrap.build_agent`) и обслуживает запросы на `localhost:50051`. Протобаф-стабы уже сгенерированы в `grpc_service/generated/`; при изменении `.proto` перегенерировать: `bash grpc_service/gen_proto.sh`.

### Запуск Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```
Чистый gRPC-клиент — не тянет LangGraph/LLM напрямую, обращается к уже запущенному backend'у. Если `summarize`-запрос не содержит явного arXiv ID, UI покажет top-5 найденных статей на выбор и вызовет `SummarizeArticle` для выбранной; для `research`-ответов показывает блок «Источники» и RAGAS-метрики. Чекбокс «Считать RAGAS-метрики» выключает их для более быстрого ответа. Свёрнутый по умолчанию блок «Промежуточные выжимки по разделам (Map-стадия)» показывает мелким серым текстом (`st.caption`), что модель извлекла из каждого раздела статьи до финальной сборки обзора — полезно для отладки, когда итоговый обзор выглядит подозрительно (например, пропустил важную секцию).

> **О времени ответа.** Суммаризация статьи целиком — это Map-Reduce по нескольким десяткам чанков, и вызовы к MLX идут последовательно, без батчинга: на локальной модели это минуты, независимо от того, считаются метрики или нет. Клиентские таймауты в UI выставлены соответственно (900 с). Research-ветка заметно быстрее — там 1-3 вызова инструмента и столько же генераций.

> **Стабы protobuf кэшируются процессом.** После правки `.proto` и `bash grpc_service/gen_proto.sh` нужно перезапустить и gRPC-сервер, и Streamlit: Streamlit перезагружает сам `streamlit_app.py`, но уже импортированный `arxiv_agent_pb2` остаётся в памяти старым, и обращение к новому полю падает с `ValueError: Protocol message AskRequest has no "..." field`.

### Программное использование (например, в ноутбуке)
```python
from modules.bootstrap import build_agent
from modules.config import AppConfig

agent = build_agent(AppConfig.from_env())

# Явный ID — суммаризация сразу
result = agent.invoke("Сделай детальный обзор статьи 1706.03762")
print(result["final_answer"], result["faithfulness"], result["answer_relevancy"])

# Без явного ID — вернутся кандидаты, суммаризация отдельным вызовом
result = agent.invoke("Сделай обзор статьи про diffusion models")
chosen_id = result["candidates"][0]["arxiv_id"]
result = agent.summarize_article(chosen_id)

# Research — с гарантированным минимум одним вызовом инструмента
result = agent.invoke("Что известно про dropout как метод регуляризации?")
print(result["final_answer"], result["sources"])

# RAGAS-метрики опциональны: без них ответ заметно быстрее
result = agent.summarize_article("1706.03762", compute_metrics=False)
```

## Граф агента

```mermaid
graph TD
    Start((START)) --> Classifier{Classifier}

    Classifier -->|Intent: OTHER| OtherHandler[Other Handler]
    Classifier -->|Intent: SUMMARIZE| ResolveArticle[Resolve Target Article]
    Classifier -->|Intent: RESEARCH| ResearchStep[Research Step]

    OtherHandler --> End((END))

    ResolveArticle -->|explicit ID| FetchFulltext[Fetch Fulltext]
    ResolveArticle -->|no ID: candidates для пользователя| End
    FetchFulltext -->|error / debug| End
    FetchFulltext --> ProcessChunk[Process & Chunk]
    ProcessChunk --> MapReduce[Map-Reduce Summarize]
    MapReduce --> Ragas[RAGAS Eval]
    Ragas --> End

    ResearchStep -->|call_tool<br/>или iterations < min| ResearchStep
    ResearchStep -->|final_answer, grounded| Ragas2[RAGAS Eval]
    Ragas2 --> End

    SummarizeRPC(("SummarizeArticle(article_id)<br/>отдельный подграф")) -.-> FetchFulltext

    style Start fill:#f9f,stroke:#333
    style End fill:#f9f,stroke:#333
    style Classifier fill:#fff4dd,stroke:#d4a017
    style Ragas fill:#e1f5fe,stroke:#01579b
    style Ragas2 fill:#e1f5fe,stroke:#01579b
    style OtherHandler fill:#ffebee,stroke:#c62828
    style ResearchStep fill:#e8f5e9,stroke:#2e7d32
    style SummarizeRPC fill:#fff9c4,stroke:#f57f17
```

## Визуализация логики обработки (Детальный граф)

```mermaid
graph TD
    style Start fill:#212121,stroke:#fff,stroke-width:2px,color:#fff
    style End fill:#212121,stroke:#fff,stroke-width:2px,color:#fff
    style Classifier fill:#ffcc80,stroke:#e65100,stroke-width:2px
    style Other fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
    style ArxivAPI fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    style Sqlite fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    style RagasNode fill:#f8bbd0,stroke:#880e4f,stroke-width:2px
    style MapReduce fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px
    style ResearchLoop fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style UserPick fill:#fff9c4,stroke:#f57f17,stroke-width:2px

    Start((User Query)) --> Classifier{"Classifier (structured output)"}
    Classifier -->|"OTHER"| Other["Other Node: Default Message"]
    Other --> End((END))

    subgraph Summarize_Pipeline ["Summarize Pipeline"]
        direction TB
        Resolve["Resolve Target Article:\nregex arXiv ID?"]
        UserPick["Нет ID -> вернуть top-5 кандидатов,\nвыбор делает пользователь в UI\n(SummarizeArticle RPC)"]
        ArxivAPI[("arXiv API: search + ar5iv/PDF")]
        Sqlite[("SQLite: кэш полного текста")]

        subgraph Article_Processor ["Article Processor"]
            direction TB
            Merge["Merge: Combine < min_tokens"]
            Split["Split: Recursive Split > max_tokens"]
            Overlaps["Overlaps: Past/Future Context"]
            Merge --> Split --> Overlaps
        end

        subgraph Map_Reduce_Phase ["Map-Reduce"]
            direction LR
            C1["Chunk 1 + Overlaps"] --> M1("Map Summarizer")
            C2["Chunk 2 + Overlaps"] --> M2("Map Summarizer")
            CN["Chunk N + Overlaps"] --> MN("Map Summarizer")
            M1 & M2 & MN --> Join["Concat Summaries"] --> Reduce("Reduce: Final Synthesis")
        end

        Resolve -->|"нет ID"| UserPick
        Resolve -->|"есть ID"| ArxivAPI --> Sqlite --> Merge
        Overlaps --> Map_Reduce_Phase
    end
    Classifier -->|"SUMMARIZE"| Resolve
    UserPick --> End

    subgraph Ragas_Eval ["RAGAS Eval (structured output, без переписывания)"]
        direction TB
        Draft["Draft Answer / Report"]
        Claims["Claim Extraction: атомарные утверждения"]
        Verdicts["Per-Claim Verdict: подтверждается контекстом?"]
        Faithfulness["Faithfulness = supported / total"]
        Questions["Generated Questions от ответа"]
        Embed[("Embeddings: sentence-transformers")]
        Relevancy["Answer Relevancy = mean cosine(query, questions)"]

        Draft --> Claims --> Verdicts --> Faithfulness
        Draft --> Questions --> Embed --> Relevancy
    end
    Reduce --> Draft
    Faithfulness --> End
    Relevancy --> End

    subgraph ResearchLoop ["Research: Function Calling Loop (заменяет RAG)"]
        direction TB
        Decide{"ResearchDecision:\nfinal_answer или call_tool?"}
        Grounded{"iterations >= min_research_iterations?"}
        ToolSearch["Tool: search_arxiv(query)"]
        ToolFulltext["Tool: get_fulltext(article_id)"]
        Evidence[("evidence + sources: накопленные\nрезультаты инструментов")]
        Cap{"iterations >= max?"}

        Decide -->|"final_answer, но ещё не grounded"| ForceTool["Форсируем call_tool: search_arxiv(query)"]
        ForceTool --> ToolSearch
        Decide -->|"call_tool: search_arxiv"| ToolSearch --> Evidence
        Decide -->|"call_tool: get_fulltext"| ToolFulltext --> Evidence
        Evidence --> Cap
        Cap -->|"нет"| Decide
        Cap -->|"да, форсируем"| FinalAnswer
        Decide -->|"final_answer, grounded"| FinalAnswer["Final Answer + sources"]
    end
    Classifier -->|"RESEARCH"| Decide
    FinalAnswer --> Draft
```
