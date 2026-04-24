import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
from langsmith import Client as LangSmithClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentTraceExporter:
    """
    Профессиональный экспортер трейсов LangGraph. 
    Обеспечивает сбор AgentState и сопоставление с IO моделей.
    """

    def __init__(
        self, 
        project_name: str, 
        include_llm_io: bool = True, 
        include_prompts: bool = True
    ):
        self.client = LangSmithClient()
        self.project_name = project_name
        self.include_llm_io = include_llm_io
        self.include_prompts = include_prompts
        
        # Названия нод для идентификации в графе
        self._TARGET_NODES = ['classifier', 'rewriter', 'qa', 'summarizer', 'summarization', 'critic']

    def fetch_dataset(self, limit: int = 20) -> pd.DataFrame:
        logger.info(f"Начинаю экспорт проекта '{self.project_name}' (limit={limit})")
        
        root_runs = self.client.list_runs(
            project_name=self.project_name,
            run_type="chain",
            is_root=True,
            limit=limit
        )

        all_records = []
        for root in root_runs:
            try:
                record = self._process_single_trace(root)
                all_records.append(record)
            except Exception as e:
                logger.error(f"Ошибка при обработке трейса {root.id}: {e}")

        df = pd.DataFrame(all_records)
        logger.info(f"Экспорт завершен. Собрано записей: {len(df)}")
        return df

    def _process_single_trace(self, root_run: Any) -> Dict[str, Any]:
        state = root_run.outputs if root_run.outputs else {}
        
        record = {
            "trace_id": str(root_run.id),
            "timestamp": root_run.start_time,
            "query": root_run.inputs.get("query"),
            "intent": state.get("intent"),
            "search_queries": state.get("search_queries"),
            "relevant_docs": state.get("relevant_docs"),
            "article_chunks": state.get("article_chunks"),
            "final_answer": state.get("final_answer"),
            "debug_data": state.get("debug_data"),
            "critic_notes": state.get("critic_notes"),
            "error": root_run.error
        }

        if self.include_llm_io:
            record.update(self._get_llm_data_map(root_run.trace_id))

        return record

    def _get_llm_data_map(self, trace_id: str) -> Dict[str, Any]:
        child_runs = list(self.client.list_runs(trace_id=trace_id))
        run_dict = {run.id: run for run in child_runs}
        
        llm_map = {
            "llm_classifier": None, "llm_rewriter": None, "llm_qa": None,
            "llm_map_summaries": [], "llm_reduce": None, "llm_critic": None
        }

        for run in child_runs:
            if run.run_type != "llm" and "vLLM" not in run.name:
                continue

            node_owner = self._find_node_owner(run, run_dict)
            if not node_owner:
                continue

            raw_prompt, raw_response = self._extract_raw_io(run)

            if node_owner in ['summarizer', 'summarization']:
                # Логика для Map (много промптов)
                if isinstance(raw_prompt, list) and len(raw_prompt) > 1:
                    if isinstance(raw_response, list):
                        if self.include_prompts:
                            llm_map["llm_map_summaries"] = [
                                {"p": p, "r": r} for p, r in zip(raw_prompt, raw_response)
                            ]
                        else:
                            # Сохраняем только чистые ответы списком
                            llm_map["llm_map_summaries"] = raw_response
                else:
                    # Логика для Reduce
                    llm_map["llm_reduce"] = self._finalize_io(raw_prompt, raw_response)
            
            else:
                target_key = f"llm_{node_owner}"
                if target_key in llm_map:
                    llm_map[target_key] = self._finalize_io(raw_prompt, raw_response)

        return llm_map

    def _finalize_io(self, prompt: Any, response: Any) -> Any:
        """Форматирует выход в зависимости от настроек прозрачности."""
        if self.include_prompts:
            return {"prompt": prompt, "response": response}
        return response

    def _find_node_owner(self, run: Any, run_dict: Dict[str, Any]) -> Optional[str]:
        curr = run
        while curr and curr.parent_run_id:
            parent = run_dict.get(curr.parent_run_id)
            if not parent: break
            name = parent.name.lower()
            for target in self._TARGET_NODES:
                if target in name: return target
            curr = parent
        return None

    def _extract_raw_io(self, run: Any) -> Tuple[Any, Any]:
        """Извлекает сырые данные промпта и ответа."""
        prompt = run.inputs.get("prompts")
        out = run.outputs or {}
        response = out.get("outputs") or out.get("output")
        
        if not response and "generations" in out:
            try:
                response = [g[0].get("text") for g in out["generations"]]
            except:
                response = out.get("generations")
        return prompt, response

    def save_to_jsonl(self, df: pd.DataFrame, filename: str):
        df.to_json(filename, orient="records", lines=True, force_ascii=False)
        logger.info(f"💾 Экспорт завершен: {filename}")
