from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


Conversation = List[ChatMessage]


class LLMProvider(ABC):
    """Единый контракт: список диалогов -> список текстовых ответов.

    Каждая реализация сама отвечает за перевод List[ChatMessage] в свой
    нативный формат (chat template токенизатора, messages= для OpenAI-совместимого
    API и т.д.) — вызывающий код никогда не имеет дела с сырыми строками промпта.
    """

    @abstractmethod
    def generate(self, conversations: List[Conversation], sampling_params: Dict[str, Any]) -> List[str]:
        ...

    def generate_stream(self, conversation: Conversation, sampling_params: Dict[str, Any]) -> Iterator[str]:
        """Токенный стриминг ОДНОГО диалога — только для узлов со связной прозой на выходе
        (сейчас — reduce-стадия суммаризации), не для structured output (там нужен целый
        валидный JSON, стримить куски бессмысленно). Дефолт — не настоящий стриминг, а один
        yield с полным текстом: безопасный фолбэк для провайдеров без своей реализации
        (VLLMProvider) и для RecordingProvider (харнесс не стримит вовсе, но не должен падать,
        если что-то мимоходом позовёт generate_stream)."""
        yield self.generate([conversation], sampling_params)[0]
