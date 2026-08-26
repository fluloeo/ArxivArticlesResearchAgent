import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from .base import Conversation, LLMProvider

logger = logging.getLogger(__name__)


class MLXProvider(LLMProvider):
    """Локальный инференс на Apple Silicon через mlx-lm. Провайдер по умолчанию.

    `mlx_lm.generate` регистрирует свой Metal generation stream ОДИН РАЗ как модульный
    глобал, на том потоке, где `mlx_lm` был впервые импортирован — вызов generate() из
    любого другого потока падает с `RuntimeError: There is no Stream(gpu, 0) in current
    thread` (мы наступили на это: gRPC ThreadPoolExecutor дёргал generate() из воркер-потока,
    отличного от того, что импортировал mlx_lm при старте сервера, каждый вызов тихо
    возвращал "", и все structured-output вызовы — classifier, research_step и т.д. —
    откатывались на safe default).

    Поэтому `import mlx_lm` намеренно НЕ делается на уровне модуля — он происходит внутри
    выделенного единственного воркер-потока (см. _load), который потом обслуживает
    ВСЕ вызовы generate() до конца жизни процесса.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-inference")
        self.model, self.tokenizer = self._executor.submit(self._load).result()

    def _load(self):
        from mlx_lm import load as mlx_load

        logger.info("Loading MLX model %s", self.model_name)
        return mlx_load(self.model_name)

    def _format_prompt(self, conversation: Conversation) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                list(conversation), add_generation_prompt=True, tokenize=False
            )
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in conversation) + "\n\nassistant:"

    def generate(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        if not conversations:
            return []
        return self._executor.submit(self._generate_on_mlx_thread, conversations, sampling_params).result()

    def _generate_on_mlx_thread(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(temp=sampling_params.get("temperature", 0.0))
        max_tokens = sampling_params.get("max_tokens", 1024)

        # frequency_penalty — параметр vLLM/OpenAI, у mlx-lm его нет; ближайший аналог в его
        # API — repetition_penalty. Раньше значение просто терялось: NodeGenerationConfig
        # объявлял summarization_map с frequency_penalty=1.2, а map-фаза по факту работала
        # вообще без штрафа за повторы (конфиг «врал» про реальные параметры генерации).
        penalty = sampling_params.get("frequency_penalty") or None
        logits_processors = make_logits_processors(repetition_penalty=penalty) if penalty else None

        results = []
        for conversation in conversations:
            prompt = self._format_prompt(conversation)
            try:
                text = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                )
            except Exception:
                logger.exception("MLX generation failed for prompt of length %d", len(prompt))
                text = ""
            results.append(text)
        return results
