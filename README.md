# AI-агент для поиска и суммаризации научных статей (arXiv.org)

Данный репозиторий содержит исходный код продвинутой системы для анализа научной литературы на базе LLM. Проект автоматизирует цикл работы с ArXiv: от интеллектуального поиска до формирования верифицированных обзоров длинных текстов.

## Архитектура системы
Поток управления реализован как граф состояний с помощью **LangGraph**. Это позволяет агенту гибко переключаться между стратегиями обработки в зависимости от контекста:

1.  **Intent Classifier** — Определяет цель пользователя:
    *   `summarize`: глубокая суммаризация одной статьи (Map-Reduce).
    *   `question`: поиск ответа по нескольким источникам (RAG).
    *   `other`: обработка нерелевантных запросов (приветствия, оффтоп).
2.  **Multi-Query Rewriter** — Генерирует список из 3–5 семантических вариаций запроса на английском языке, расширяя область поиска.
3.  **Advanced Retriever** — Выполняет параллельный поиск в **LanceDB** по всем вариациям запроса с последующей дедупликацией чанков.
4.  **Self-Correction (Critic Node)** — Опциональный модуль аудита. Сверяет финальный отчет с исходным текстом статьи, выявляет галлюцинации и при необходимости инициирует исправление отчета.

---

## Ключевые технические решения

### 1. Система самокоррекции (Critic Loop)
Для исключения фактических ошибок в научных отчетах внедрен механизм "критика":
*   **Verification:** Пакетная проверка каждого смыслового чанка статьи на соответствие итоговому отчету.
*   **Correction:** Если найдены несоответствия (неверные метрики, искаженные выводы), агент формирует список правок и пересобирает финальный текст.

### 2. Map-Reduce с контекстным перекрытием (Overlaps)
Для работы с длинными статьями используется кастомный алгоритм обработки:
*   **Context Overlaps:** К каждому чанку на этапе Map добавляется "память" (past/future overlap) — фрагменты соседних секций. Это сохраняет логическую связность на границах разделов.
*   **Section Merging:** Короткие подразделы объединяются до порога `min_tokens`, минимизируя количество вызовов API без потери контекста.

---

## Отладка и мониторинг (Observability)

Система спроектирована как "прозрачный ящик". Весь процесс выполнения можно детально отследить двумя способами:

### 1. Интроспекция AgentState
Состояние агента (`AgentState`) доступно на каждом шаге и содержит полную историю работы:
*   `search_queries`: список всех переформулированных запросов.
*   `relevant_docs`: DataFrame со всеми найденными чанками и их метаданными.
*   `article_chunks`: полная структура статьи с добавленными контекстными перекрытиями.
*   `critic_notes`: подробный лог замечаний аудитора (если были найдены ошибки).
*   `debug_data`: промежуточные выжимки с этапа Map перед их финальным объединением.

### 2. Интеграция с LangSmith
Проект полностью интегрирован с платформой **LangSmith**, что дает следующие возможности:
*   **Визуализация графа:** Просмотр пути запроса через узлы и условные ребра.
*   **Трассировка (Tracing):** Пошаговый анализ времени выполнения и потребления токенов для каждой ноды.
*   **Управление промптами:** Все системные инструкции версионируются в Hub. Реализован **Fallback-механизм**: при отсутствии связи с облаком система автоматически переключается на локальный `prompts.yaml`.

---

## Стек технологий
*   **Ядро:** LangGraph, LangChain.
*   **LLM:** vLLM (инференс на GPU), OpenRouter (внешние модели).
*   **Хранение:** LanceDB (векторный поиск), PostgreSQL (полные тексты).
*   **NLP:** Sentence-Transformers, HuggingFace Transformers.
*   **Мониторинг:** LangSmith.

---

## Запуск проекта

### Настройка секретов
Создайте файл `.env`:
```text
OPENROUTER_API_KEY=your_key
LANGSMITH_API_KEY=your_key
HF_TOKEN=your_token
```

### Инициализация агента
```python
agent = ArxivAgent(
    llm_provider=provider,
    retriever=retriever,
    sum_pipeline=sum_pipe,
    processor=processor,
    embed_model=retrieval_model,
    db_params=db_params,
    tokenizer=tokenizer,
    prompts=hub_prompts_config, 
    use_critic=True, # Включить верификацию критиком
    use_hub=True     # Использовать LangSmith Hub
)

# Вызов агента
result = agent.invoke("Сделай детальный обзор статьи про обучение с подкреплением")
display(Markdown(result['final_answer']))
processor.visualize(result['debug_data'])
```

```mermaid
graph TD
    Start((START)) --> Classifier{Classifier}
    
    Classifier -->|Intent: OTHER| OtherHandler[Other Handler]
    Classifier -->|Intent: YES / NO| Rewriter[Rewriter]
    
    OtherHandler --> End((END))
    
    Rewriter --> Retriever[Retriever]
    
    Retriever -->|Intent: NO| QA[QA Node]
    Retriever -->|Intent: YES| Summarizer[Summarizer]
    
    QA --> End
    
    Summarizer --> IsCritic{Use Critic?}
    IsCritic -->|Yes| Critic[Critic Node]
    IsCritic -->|No| End
    
    Critic --> End

    style Start fill:#f9f,stroke:#333
    style End fill:#f9f,stroke:#333
    style Classifier fill:#fff4dd,stroke:#d4a017
    style Critic fill:#e1f5fe,stroke:#01579b
    style OtherHandler fill:#ffebee,stroke:#c62828
```

```mermaid
graph TD
    %% Точка входа
    Start((START)) --> Classifier{Intent?}

    %% ВЕТКА QA
    subgraph QA_Path [Strategy: RAG Response]
        direction TB
        Rewriter[Query Rewriter<br/><i>Gen 5 Queries</i>]
        Search[Multi-Search<br/><i>LanceDB</i>]
        
        subgraph QA_Fusion [Context Consolidation]
            C1[Chunk A]
            C2[Chunk B]
            C3[Chunk C]
            Deduplicate[<b>Deduplication</b><br/>by ID/Text]
            Merge[<b>Final Context</b><br/>Merged Chunks]
        end

        Rewriter --> Search
        Search --> C1 & C2 & C3
        C1 & C2 & C3 --> Deduplicate --> Merge
        Merge --> QA_Node[QA Answer Node]
    end

    %% ВЕТКА SUMMARIZATION
    subgraph Sum_Path [Strategy: Map-Reduce + Audit]
        direction TB
        DB[(PostgreSQL)] --> Processor[Article Processor]
        
        subgraph Chunks_With_Overlaps [Step 1: Data Preparation]
            direction LR
            P1[Past] -.-> M1[<b>Chunk 1</b>]
            M1 -.-> F1[Future]
            P2[Past] -.-> M2[<b>Chunk 2</b>]
            M2 -.-> F2[Future]
            
            note[<i>Overlaps preserve<br/>context between sections</i>]
        end

        subgraph Map_Reduce_Phase [Step 2: Summarization]
            direction TB
            M1 --> Map1[Map]
            M2 --> Map2[Map]
            Map1 & Map2 --> Reduce[<b>Reduce</b><br/>Synthesis of Draft]
        end

        subgraph Critic_Audit [Step 3: Verification Loop]
            direction TB
            Reduce --> Draft[Draft Report]
            
            %% Линии аудита (контроль)
            Draft --> Auditor{<b>Critic Node</b><br/>Fact Checker}
            Auditor -.->|Cross-Check| M1
            Auditor -.->|Cross-Check| M2
            
            Auditor -->|Found Errors| Correction[<b>Final Correction</b><br/>Fix Hallucinations]
            Auditor -->|OK| Verified[Verified Report]
        end
    end

    %% Финальные связи
    Classifier -->|NO| Rewriter
    Classifier -->|YES| DB

    QA_Node --> End((END))
    Verified --> End
    Correction --> End

    %% Стилизация
    style Start fill:#f9f,stroke:#333
    style End fill:#f9f,stroke:#333
    style Classifier fill:#fff4dd,stroke:#d4a017
    style Critic_Audit fill:#f0f4ff,stroke:#01579b,stroke-dasharray: 5 5
    style Chunks_With_Overlaps fill:#f1f8e9,stroke:#2e7d32
    style Auditor fill:#ffeb3b,stroke:#fbc02d
    style Deduplicate fill:#e1f5fe,stroke:#01579b
```
