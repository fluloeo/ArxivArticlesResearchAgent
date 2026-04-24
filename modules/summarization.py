import time
import requests
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Union
from langsmith import Client as LangSmithClient
from openai import OpenAI
ls_client = LangSmithClient()
from langsmith import traceable
try:
    from langchainhub.client import Client
    pull = Client().pull
except:
    pull = None 

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        pass

class VLLMProvider(LLMProvider):
    def __init__(self, llm_engine, sampling_params_class, model_name: str):
        self.llm = llm_engine
        self.params_factory = sampling_params_class
        self.model_name = model_name
        self.generations_log = []

    @traceable(run_type="llm", name="vLLM_Generate")
    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        if not prompts: return[]
        vllm_params = self.params_factory(**sampling_params)
        outputs = self.llm.generate(prompts, vllm_params)
        texts = [output.outputs[0].text for output in outputs]
        for p, t in zip(prompts, texts):
            self.generations_log.append({
                "timestamp": time.time(),
                "model": self.model_name,
                "prompt": p,
                "response": t,
                "params": sampling_params
            })
        return texts
    def save_log_to_json(self, filename="debug_generations.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.generations_log, f, ensure_ascii=False, indent=2)

# class MistralProvider(LLMProvider):
#     def __init__(self, api_key: str, model_name: str = "mistral-small-latest"):
#         self.api_key = api_key
#         self.model_name = model_name
#         self.url = "https://api.mistral.ai/v1/chat/completions"

#     def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
#         if not prompts:
#             return []

#         results = []
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#             "Accept": "application/json"
#         }

#         for prompt in prompts:
#             payload = {
#                 "model": self.model_name,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": sampling_params.get("temperature", 0.7),
#                 "top_p": sampling_params.get("top_p", 1.0),
#                 "max_tokens": sampling_params.get("max_tokens", 1024),
#             }

#             try:
#                 response = requests.post(self.url, json=payload, headers=headers)
#                 response.raise_for_status()
#                 data = response.json()
                
#                 content = data['choices'][0]['message']['content']
#                 results.append(content)
            
#                 time.sleep(0.1) 
                
#             except Exception as e:
#                 print(f"Error calling Mistral API: {e}")
#                 results.append("")

#         return results

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-120b:free", use_reasoning: bool = True):
        """
        :param api_key: Ключ OpenRouter
        :param model_name: Название модели (например, "deepseek/deepseek-r1" или "openai/gpt-oss-120b:free")
        :param use_reasoning: Включить ли расширенные рассуждения (reasoning)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name
        self.use_reasoning = use_reasoning

    def generate(self, prompts: List[str], sampling_params: Dict[str, Any]) -> List[str]:
        if not prompts:
            return []

        results = []
        
        # Подготовка параметров для OpenRouter (extra_body)
        extra_body = {}
        if self.use_reasoning:
            extra_body["reasoning"] = {"enabled": True}

        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=sampling_params.get("temperature", 0.7),
                    max_tokens=sampling_params.get("max_tokens", 1024),
                    top_p=sampling_params.get("top_p", 1.0),
                    extra_body=extra_body
                )
                content = response.choices[0].message.content
                
                # По желанию: можно логгировать рассуждения, если они есть
                # reasoning = getattr(response.choices[0].message, 'reasoning_details', None)
                # if reasoning:
                #     print(f"[DEBUG Reasoning]: {reasoning[:100]}...")

                results.append(content)

                # Небольшая пауза для бесплатных лимитов
                if "free" in self.model_name:
                    time.sleep(0.1) 

            except Exception as e:
                print(f"Error calling OpenRouter ({self.model_name}): {e}")
                results.append("")

        return results

class SummarizationPipeline:
    def __init__(self, provider: LLMProvider, tokenizer, prompts: Dict[str, Any], local_prompts: Dict[str, Any] = None, use_hub: bool = True):
        self.provider = provider
        self.tokenizer = tokenizer
        self.use_hub = use_hub
        self.local_prompts = local_prompts or {}
        
        # Резолвим промпты передавая сразу ключ и значение
        self.resolved_prompts = {k: self._resolve(k, v) for k, v in prompts.items()}

    def _resolve(self, key: str, val: Any):
        # 1. Пробуем скачать из хаба, если включено
        if self.use_hub and isinstance(val, str):
            try:
                print(f"Pulling prompt from LangSmith: {val}")
                return ls_client.pull_prompt(val)
            except Exception as e:
                print(f"Failed to pull {val}: {e}. Falling back to local.")
        
        if key in self.local_prompts:
            return self.local_prompts[key]
            
        return val

    def _format_chat(self, template: Any, variables: Dict[str, Any], system_fallback: str = None) -> str:
        if hasattr(template, "format_messages"):
            messages = template.format_messages(**variables)
            formatted =[{"role": "system" if m.type=="system" else "user", "content": m.content} for m in messages]
        elif isinstance(template, dict):
            formatted =[
                {"role": "system", "content": template.get('system', '')},
                {"role": "user", "content": template.get('user', '').format(**variables)}
            ]
        else:
            sys_content = system_fallback if system_fallback else ""
            formatted =[{"role": "system", "content": str(sys_content)}, 
                            {"role": "user", "content": str(template).format(**variables)}]

        return self.tokenizer.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)

    def run(self, overlap_dict: Dict[str, Dict[str, str]], map_params: Dict[str, Any], reduce_params: Dict[str, Any]):
        titles = list(overlap_dict.keys())
        map_prompts = [self._format_chat(self.resolved_prompts['map'], {"title": t, **p}, self.resolved_prompts.get('system_map')) 
                       for t, p in overlap_dict.items()]
        
        chunk_summaries = self.provider.generate(map_prompts, map_params)
        combined = "\n\n".join([f"### {t}\n{s}" for t, s in zip(titles, chunk_summaries)])
        
        reduce_p = self._format_chat(self.resolved_prompts['reduce'], {"summaries": combined}, self.resolved_prompts.get('system_reduce'))
        final_report = self.provider.generate([reduce_p], reduce_params)[0]
        return final_report, dict(zip(titles, chunk_summaries))