from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypedDict


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
