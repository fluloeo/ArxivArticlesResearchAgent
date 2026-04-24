import json
import psycopg2
import pandas as pd
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from abc import ABC, abstractmethod
from langsmith import Client as LangSmithClient
import ast
ls_client = LangSmithClient()
class AgentState(TypedDict):
    query: str
    search_queries: List[str]
    intent: str
    relevant_docs: pd.DataFrame
    article_chunks: Dict[str, Any] # Те самые чанки с перекрытиями
    final_answer: str
    debug_data: dict
    critic_notes: List[str]
    

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: str, vector: List[float], limit: int = 5) -> pd.DataFrame:
        pass

class LanceDBRetriever(BaseRetriever):
    def __init__(self, table):
        self.table = table

    def search(self, query: str, vector: List[float], limit: int = 5) -> pd.DataFrame:
        # Пока обычный векторный поиск, легко поменять на .text(query) для гибрида
        return self.table.search(vector).limit(limit).to_pandas()

# --- КЛАСС АГЕНТА ---
class ArxivAgent:
    def __init__(
        self, 
        llm_provider, 
        retriever, 
        sum_pipeline, 
        processor, 
        embed_model, 
        db_params, 
        tokenizer, 
        prompts, 
        use_critic: bool = False, 
        local_prompts=None, 
        use_hub: bool = True,
        debug_mode: bool = False
    ):
        self.llm = llm_provider
        self.retriever = retriever
        self.sum_pipeline = sum_pipeline
        self.processor = processor
        self.embed_model = embed_model
        self.db_params = db_params
        self.tokenizer = tokenizer
        self.use_hub = use_hub
        self.use_critic = use_critic 
        self.debug_mode = debug_mode
        self.local_prompts = local_prompts or {}
        
        self.resolved_prompts = {}
        for node, val in prompts.items():
            # Summarizer сам обрабатывает свои промпты, пропускаем
            if node == "summarization": 
                continue 
            self.resolved_prompts[node] = self._resolve(node, val)

        self.app = self._build_graph()
        
    def _resolve(self, node: str, val: Any):
        if self.use_hub and isinstance(val, str):
            try:
                print(f"Pulling prompt from LangSmith: {val}")
                return ls_client.pull_prompt(val)
            except Exception as e:
                print(f"Failed to pull {val}: {e}. Falling back to local.")
                
        if node in self.local_prompts:
            return self.local_prompts[node]
            
        return val

    def _format_node_chat(self, node_key: str, variables: Dict[str, Any]) -> str:
            prompt_data = self.resolved_prompts[node_key]
            
            # 1. Если это объект из Хаба (ChatPromptTemplate)
            if hasattr(prompt_data, "format_messages"):
                messages = prompt_data.format_messages(**variables)
                formatted = [{"role": "system" if m.type=="system" else "user", "content": m.content} for m in messages]
            
            # 2. Если это словарь {"system": ..., "user": ...}
            elif isinstance(prompt_data, dict):
                formatted = [
                    {"role": "system", "content": str(prompt_data.get('system', ''))},
                    {"role": "user", "content": str(prompt_data.get('user', '')).format(**variables)}
                ]
            
            # 3. Если это просто строка (Fallback)
            else:
                formatted = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(prompt_data).format(**variables)}
                ]
                
            return self.tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)

    def classifier_node(self, state: AgentState):
        p = self._format_node_chat('classifier', {"query": state['query']})
        res = self.llm.generate([p], {"max_tokens": 10, "temperature": 0})[0].strip().upper()
        if not res: 
            return {"intent": "other"}
        first_word = res.split()[0].rstrip('.,!?:')
        
        if first_word == "YES":
            intent = "summarize"
        elif first_word == "NO":
            intent = "question"
        else:
            intent = "other"
            
        print(f"[DEBUG] Classifier raw: '{res}' | Result: {intent}")
        return {"intent": intent}

    def rewrite_query_node(self, state: AgentState):
        p = self._format_node_chat('rewriter', {"query": state['query']})
        # Генерируем строку (например: "Deep learning MRI, Neural networks brain, ...")
        res = self.llm.generate([p], {"max_tokens": 100, "temperature": 0})[0].strip()
        
        queries = [q.strip().strip('"').strip("'") for q in res.split(',')]
        queries = [q for q in queries if len(q) > 2]
        
        return {"search_queries": queries}

    def retriever_node(self, state: AgentState):
        queries = state['search_queries']
        all_results = []
        
        embeddings = self.embed_model.encode(queries).tolist()

        for i, query in enumerate(queries):
            res_df = self.retriever.search(query, embeddings[i], limit=5)
            all_results.append(res_df)

        combined_df = pd.concat(all_results)
        
        # ИЗМЕНЕНИЕ ЗДЕСЬ:
        if state['intent'] == 'summarize':
            # Для суммаризации нам нужна только одна, самая релевантная статья
            final_df = combined_df.drop_duplicates(subset=['article_id'])
        else:
            # Для QA нам нужны ВСЕ уникальные чанки (даже из одной статьи)
            final_df = combined_df.drop_duplicates(subset=['id']) # или subset=['text']
            
        print(f"[DEBUG] Found {len(final_df)} unique units for intent: {state['intent']}")
        return {"relevant_docs": final_df.head(10)}

    def qa_node(self, state: AgentState):
        if self.debug_mode:
            chunk_count = len(state['relevant_docs'])
            return {"final_answer": f"DEBUG MODE: Поиск завершен. Найдено чанков: {chunk_count}. Генерация ответа пропущена."}
        ctx = "\n\n".join(state['relevant_docs']['text'].tolist())
        p = self._format_node_chat('qa', {"context": ctx, "query": state['query']})
        ans = self.llm.generate([p], {"max_tokens": 1024, "temperature": 0})[0]
        return {"final_answer": ans}

    def summarization_node(self, state: AgentState):
        if state['relevant_docs'].empty: return {"final_answer": "Статьи не найдены."}
        top_article_id = str(state['relevant_docs'].iloc[0]['article_id'])
        if self.debug_mode:
            return {"final_answer": f"DEBUG MODE: Выбрана статья ID {top_article_id} для суммаризации. Генерация отчета пропущена."}
        db_data = self._fetch(top_article_id)
        
        if not db_data: 
            return {"final_answer": "Ошибка: текст статьи не найден в базе."}
        title, raw_sections, pdf_url = db_data 

        sections = raw_sections 
        if isinstance(raw_sections, str):
            try:
                sections = json.loads(raw_sections)
            except json.JSONDecodeError:
                try:
                    sections = ast.literal_eval(raw_sections)
                except Exception as e:
                    print(f"[ERROR] Failed to parse sections: {e}")
                    sections = {"Main": raw_sections}
        print(f"[DEBUG] Parsed sections: {list(sections.keys()) if isinstance(sections, dict) else 'Not a dict'}")
        if not sections:
            return {"final_answer": f"Для статьи '{title}' нет доступных разделов."}
        clean_sections = self.processor.process(sections, show_report=False)
        overlap_data = self.processor.create_overlap_dict(clean_sections)
        
        report, chunk_summaries_dict = self.sum_pipeline.run(
            overlap_data, 
            map_params={"temperature": 0, "max_tokens": 512},
            reduce_params={"temperature": 0, "max_tokens": 1500}
        )
        
        header = f"# {title}\n🔗 [PDF]({pdf_url})\n\n"
        

        return {
            "final_answer": header + report,
            "debug_data": chunk_summaries_dict,    # Сохраняем промежуточные саммари для отладки
            "article_chunks": overlap_data        # Сохраняем чанки для критика
        }

    def critic_node(self, state: AgentState):
        """Нода-критик: сначала проверяет все чанки (verify), затем исправляет (correction)."""
        if self.debug_mode:
            return {"critic_notes": ["DEBUG MODE: Критик пропущен."]}
        summary = state['final_answer']
        chunks = state['article_chunks']
        notes = []

        print(f"[CRITIC] Начало аудита {len(chunks)} фрагментов...")

        # 1. ШАГ: Верификация (Используем ключ 'critic_verify')
        verify_prompts = []
        titles = list(chunks.keys())
        
        for title, content in chunks.items():
            p = self._format_node_chat('critic_verify', {
                "title": title,
                "chunk_text": content['main_text'],
                "summary": summary
            })
            verify_prompts.append(p)
        
        check_results = self.llm.generate(verify_prompts, {"max_tokens": 300, "temperature": 0})
        
        for title, res in zip(titles, check_results):
            if "OK" not in res.strip().upper():
                notes.append(f"Ошибка в секции [{title}]: {res}")

        if not notes:
            print("[CRITIC] Ошибок не найдено.")
            return {"final_answer": summary, "critic_notes": []}
        
        # 2. ШАГ: Коррекция (Используем ключ 'critic_correction')
        print(f"[CRITIC] Найдено несоответствий: {len(notes)}. Исправление отчета...")
        correction_p = self._format_node_chat('critic_correction', {
            "summary": summary,
            "notes": "\n".join(notes)
        })
        
        corrected_summary = self.llm.generate([correction_p], {"max_tokens": 2500, "temperature": 0})[0]
        
        return {
            "final_answer": corrected_summary,
            "critic_notes": notes 
        }

    def _fetch(self, aid):
        with psycopg2.connect(**self.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT title, section_text_new, pdf_url FROM arxivdb.public.articles WHERE id = %s", (aid,))
                return cur.fetchone()
    def other_node(self, state: AgentState):
        """Обработка запросов вне научной тематики или попыток взлома."""
        # Можно взять текст из промпта или прописать здесь
        msg = "Я — специализированный научный ассистент по базе ArXiv. Пожалуйста, задайте вопрос, касающийся научных исследований, или укажите ID статьи для суммаризации."
        return {"final_answer": msg}
    def _build_graph(self):
        wf = StateGraph(AgentState)
        
        wf.add_node("classifier", self.classifier_node)
        wf.add_node("other_handler", self.other_node) # НОВАЯ НОДА
        wf.add_node("rewriter", self.rewrite_query_node)
        wf.add_node("retriever", self.retriever_node)
        wf.add_node("qa", self.qa_node)
        wf.add_node("summarizer", self.summarization_node)
        
        if self.use_critic:
            wf.add_node("critic", self.critic_node)

        wf.set_entry_point("classifier")

        # --- ЛОГИКА ПОСЛЕ КЛАССИФИКАТОРА ---
        def route_after_classifier(state):
            if state["intent"] == "other":
                return "other"
            return "proceed"

        wf.add_conditional_edges(
            "classifier",
            route_after_classifier,
            {
                "other": "other_handler",
                "proceed": "rewriter"
            }
        )

        wf.add_edge("rewriter", "retriever")
    
        def route_after_retrieval(state):
            return "summarizer" if state["intent"] == "summarize" else "qa"
            
        wf.add_conditional_edges(
            "retriever", 
            route_after_retrieval, 
            {
                "summarizer": "summarizer", 
                "qa": "qa"
            }
        )
        
        if self.use_critic:
            wf.add_edge("summarizer", "critic")
            wf.add_edge("critic", END)
        else:
            wf.add_edge("summarizer", END)
            
        wf.add_edge("qa", END)
        wf.add_edge("other_handler", END) 
        
        return wf.compile()

    def invoke(self, query: str): return self.app.invoke({"query": query})