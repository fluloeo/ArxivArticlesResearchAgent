# AI-агент для поиска и суммаризации научных статей (arXiv.org)

Данный репозиторий содержит исходный код продвинутой системы для анализа научной литературы на базе LLM. Проект автоматизирует цикл работы с ArXiv: от интеллектуального поиска до формирования верифицированных обзоров длинных текстов.
Репозиторий - часть [командного проекта](https://github.com/ai-science-articles-aggregator)

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


### 1. Двухстадийный Article Processor (Merge & Split)
Для решения проблемы лимита контекста и сохранения структуры статьи реализован продвинутый алгоритм подготовки данных:
*   **Stage 1: Token-Aware Merging.** Мелкие подразделы статьи объединяются с соседями, пока не достигнут порога `min_tokens`. Это минимизирует количество вызовов LLM.
*   **Stage 2: Recursive Splitting.** Если после слияния или изначально секция превышает `max_tokens`, она рекурсивно разбивается на части. При этом сохраняется строгая нумерация в заголовках (например, *"Methodology (Part 1)"*), что помогает модели на этапе Map сохранять контекст.
*   **Stage 3: Custom Overlaps.** К каждому финальному чанку добавляются "теневые" контексты (`past_overlap` и `future_overlap`) из соседних фрагментов, обеспечивая плавность переходов в суммаризации.

### 2. Система верификации (Critic Loop)
В отличие от простых суммаризаторов, агент проводит финальный аудит:
*   **Parallel Audit:** Каждый оригинальный фрагмент текста сравнивается с итоговым отчетом.
*   **Error Classification:** Критик помечает ошибки типов `[NUM]` (цифры), `[TERM]` (термины) или `[LOGIC]`.
*   **Correction:** На основе собранных заметок модель делает финальный "чистовик".

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
* [Промпты](https://smith.langchain.com/hub/fluloeo?organizationId=527befdf-145d-42e4-b03b-bdad31b098c3)

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
## Граф агента 

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

## Визуализация логики обработки (Детальный граф)

```mermaid
graph TD
    %% ================= СТИЛИ =================
    style Start fill:#212121,stroke:#fff,stroke-width:2px,color:#fff
    style End fill:#212121,stroke:#fff,stroke-width:2px,color:#fff
    style Classifier fill:#ffcc80,stroke:#e65100,stroke-width:2px
    style Other fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px
    style LanceDB fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    style PostgreSQL fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    style CriticNode fill:#f8bbd0,stroke:#880e4f,stroke-width:2px
    style MapReduce fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px

    %% ================= ВХОД И РОУТИНГ =================
    Start((User Query)) --> Classifier{"Classifier Node (Intent?)"}
    Classifier -->|"OTHER"| Other["Other Node: Default Message"]
    Other --> End((END))

    %% ================= ПОИСКОВЫЙ БЛОК =================
    subgraph Information_Retrieval ["Information Retrieval"]
        direction TB
        Rewriter["Rewriter Node: Multi-Query Generation"]
        
        subgraph Vector_Search ["Vector Search (LanceDB)"]
            Q1["Query 1 -> 5 results"]
            Q2["Query 2 -> 5 results"]
            QN["Query N -> 5 results"]
        end

        Dedup{"Deduplication Logic"}
        
        Classifier -->|"YES / NO"| Rewriter
        Rewriter --> Q1 & Q2 & QN
        Q1 & Q2 & QN --> Dedup
        
        Dedup -->|"Intent: NO (QA)"| DocsQA["Unique Chunks List"]
        Dedup -->|"Intent: YES (Sum)"| TopDoc["Single Top Article ID"]
    end

    %% ================= ВЕТКА QA =================
    subgraph QA_Pipeline ["QA Pipeline"]
        direction TB
        QAContext["Concat Chunks to Single Context"]
        QAGen["QA Node: Answer Generation"]
        
        DocsQA --> QAContext --> QAGen
    end
    QAGen --> End

    %% ================= ВЕТКА СУММАРИЗАЦИИ =================
    subgraph Summarization_Pipeline ["Summarization Pipeline"]
        direction TB
        PostgreSQL[("PostgreSQL: Fetch Parsed Sections")]
        
        subgraph Article_Processor ["Article Processor Pipeline"]
            direction TB
            Merge["Merge Stage: Combine < min_tokens"]
            Split["Split Stage: Recursive Split > max_tokens"]
            Overlaps["Overlap Stage: Add Past/Future Context"]
            
            Merge --> Split --> Overlaps
        end

        TopDoc --> PostgreSQL
        PostgreSQL -->|"Parsed Section Dict"| Merge
        
        subgraph Map_Reduce_Phase ["Map-Reduce Execution"]
            direction LR
            C1["Chunk 1 (Part 1) + Overlaps"] --> M1("Map Summarizer")
            C2["Chunk 1 (Part 2) + Overlaps"] --> M2("Map Summarizer")
            CN["Chunk N + Overlaps"] --> MN("Map Summarizer")
            
            M1 & M2 & MN --> Join["Concat Summaries"] --> Reduce("Reduce: Final Synthesis")
        end
        Overlaps --> Map_Reduce_Phase
    end

    %% ================= ВЕТКА КРИТИКА =================
    subgraph Critic_Audit_Loop ["Critic Audit Loop"]
        direction TB
        Draft["Draft Summary Report"]
        
        subgraph Per_Chunk_Verification ["Per-Chunk Verification"]
            Verify["Critic Verify: Parallel Audit"]
        end

        CheckErrors{"Notes empty?"}
        CriticCorrect["Critic Correction: Fix Report using Notes"]
        
        Reduce --> Draft
        Draft --> Verify
        
        %% Показываем, что оригинальные чанки подаются в критика поштучно
        C1 & C2 & CN -.->|"Iterate Original Text"| Verify 
        
        Verify --> CheckErrors
        CheckErrors -->|"Yes: OK"| FinalOk["Keep Original Report"]
        CheckErrors -->|"No: Errors Found"| CriticCorrect
    end

    FinalOk --> End
    CriticCorrect --> End
```