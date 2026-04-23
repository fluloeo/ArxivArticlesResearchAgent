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

```Mermaid
graph TD
    %% Определение узлов
    Start((START))
    Classifier[<b>Classifier</b><br/>Определение намерения]
    OtherHandler[<b>Other Handler</b><br/>Заглушка для оффтопа]
    Rewriter[<b>Rewriter</b><br/>Генерация Multi-Query]
    Retriever[<b>Retriever</b><br/>Поиск в LanceDB]
    QA[<b>QA Node</b><br/>Точечный ответ RAG]
    Summarizer[<b>Summarizer</b><br/>Map-Reduce отчет]
    Critic[<b>Critic Node</b><br/>Верификация и коррекция]
    End((END))

    %% Логика графа
    Start --> Classifier
    
    Classifier -->|Intent: OTHER| OtherHandler
    Classifier -->|Intent: YES / NO| Rewriter
    
    OtherHandler --> End
    
    Rewriter --> Retriever
    
    Retriever -->|Intent: NO| QA
    Retriever -->|Intent: YES| Summarizer
    
    QA --> End
    
    Summarizer -->|use_critic = True| Critic
    Summarizer -->|use_critic = False| End
    
    Critic --> End

    %% Стилизация
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Classifier fill:#fff4dd,stroke:#d4a017,stroke-width:2px
    style Critic fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style OtherHandler fill:#ffebee,stroke:#c62828,stroke-width:2px
    style Summarizer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style QA fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```
