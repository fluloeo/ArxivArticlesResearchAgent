import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from langsmith import Client as LangSmithClient

from .node_names import NodeName

logger = logging.getLogger(__name__)

_MAP_REDUCE_NODES = {NodeName.MAP_REDUCE_SUMMARIZE.value}


class AgentTraceExporter:
    """Экспортер трейсов LangGraph из LangSmith в pandas.DataFrame для последующего анализа/eval.

    Имена узлов графа берутся из modules.node_names.NodeName — того же источника, которым
    пользуется agent.py при построении графа, так что переименование узла требует правки
    только в одном месте.
    """

    def __init__(
        self,
        project_name: str,
        include_llm_io: bool = True,
        include_prompts: bool = True,
        client: Optional[LangSmithClient] = None,
    ):
        self.client = client or LangSmithClient()
        self.project_name = project_name
        self.include_llm_io = include_llm_io
        self.include_prompts = include_prompts
        self._target_nodes = [n.value for n in NodeName]

    def fetch_dataset(self, limit: int = 20) -> pd.DataFrame:
        logger.info("Начинаю экспорт проекта '%s' (limit=%d)", self.project_name, limit)

        root_runs = self.client.list_runs(project_name=self.project_name, run_type="chain", is_root=True, limit=limit)

        all_records = []
        for root in root_runs:
            try:
                all_records.append(self._process_single_trace(root))
            except Exception:
                logger.exception("Ошибка при обработке трейса %s", root.id)

        df = pd.DataFrame(all_records)
        logger.info("Экспорт завершён. Собрано записей: %d", len(df))
        return df

    def _process_single_trace(self, root_run: Any) -> Dict[str, Any]:
        state = root_run.outputs if root_run.outputs else {}

        record = {
            "trace_id": str(root_run.id),
            "timestamp": root_run.start_time,
            "query": root_run.inputs.get("query"),
            "intent": state.get("intent"),
            "target_article_id": state.get("target_article_id"),
            "candidates": state.get("candidates"),
            "article_chunks": state.get("article_chunks"),
            "evidence": state.get("evidence"),
            "sources": state.get("sources"),
            "final_answer": state.get("final_answer"),
            "debug_data": state.get("debug_data"),
            "faithfulness": state.get("faithfulness"),
            "answer_relevancy": state.get("answer_relevancy"),
            "error": root_run.error,
        }

        if self.include_llm_io:
            record.update(self._get_llm_data_map(root_run.trace_id))

        return record

    def _get_llm_data_map(self, trace_id: str) -> Dict[str, Any]:
        child_runs = list(self.client.list_runs(trace_id=trace_id))
        run_dict = {run.id: run for run in child_runs}

        llm_map: Dict[str, Any] = {f"llm_{node}": None for node in self._target_nodes}
        llm_map["llm_map_summaries"] = []
        llm_map["llm_reduce"] = None

        for run in child_runs:
            if run.run_type != "llm" and "MLX" not in run.name and "OpenRouter" not in run.name and "vLLM" not in run.name:
                continue

            node_owner = self._find_node_owner(run, run_dict)
            if not node_owner:
                continue

            raw_prompt, raw_response = self._extract_raw_io(run)

            if node_owner in _MAP_REDUCE_NODES:
                if isinstance(raw_prompt, list) and len(raw_prompt) > 1 and isinstance(raw_response, list):
                    llm_map["llm_map_summaries"] = (
                        [{"p": p, "r": r} for p, r in zip(raw_prompt, raw_response)]
                        if self.include_prompts
                        else raw_response
                    )
                else:
                    llm_map["llm_reduce"] = self._finalize_io(raw_prompt, raw_response)
            else:
                target_key = f"llm_{node_owner}"
                if target_key in llm_map:
                    llm_map[target_key] = self._finalize_io(raw_prompt, raw_response)

        return llm_map

    def _finalize_io(self, prompt: Any, response: Any) -> Any:
        if self.include_prompts:
            return {"prompt": prompt, "response": response}
        return response

    def _find_node_owner(self, run: Any, run_dict: Dict[str, Any]) -> Optional[str]:
        curr = run
        while curr and curr.parent_run_id:
            parent = run_dict.get(curr.parent_run_id)
            if not parent:
                break
            name = parent.name.lower()
            for target in self._target_nodes:
                if target in name:
                    return target
            curr = parent
        return None

    def _extract_raw_io(self, run: Any) -> Tuple[Any, Any]:
        prompt = run.inputs.get("prompts") or run.inputs.get("conversations")
        out = run.outputs or {}
        response = out.get("outputs") or out.get("output")

        if not response and "generations" in out:
            try:
                response = [g[0].get("text") for g in out["generations"]]
            except (KeyError, IndexError, TypeError):
                logger.warning("Unexpected 'generations' shape for run %s", run.id)
                response = out.get("generations")
        return prompt, response

    def save_to_jsonl(self, df: pd.DataFrame, filename: str) -> None:
        df.to_json(filename, orient="records", lines=True, force_ascii=False)
        logger.info("💾 Экспорт завершён: %s", filename)
