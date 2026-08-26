import json
import logging
import typing
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from .llm.base import Conversation, LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _placeholder_for_type(annotation: Any) -> Any:
    origin = get_origin(annotation)

    if origin is Literal:
        return get_args(annotation)[0]
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _placeholder_for_type(non_none[0]) if non_none else None
    if origin in (list, typing.List):
        args = get_args(annotation)
        return [_placeholder_for_type(args[0])] if args else ["..."]
    if origin in (dict, typing.Dict) or annotation is dict:
        return {}
    if annotation is list:
        return ["..."]
    if annotation is bool:
        return True
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is str:
        return "..."
    return "..."


def _example_instance(schema: Type[BaseModel]) -> Dict[str, Any]:
    example = {}
    for name, field_info in schema.model_fields.items():
        if field_info.default not in (PydanticUndefined, None) and not field_info.is_required():
            example[name] = field_info.default
        else:
            example[name] = _placeholder_for_type(field_info.annotation)
    return example


def _json_instruction(schema: Type[BaseModel]) -> str:
    # Раньше сюда вставлялся полный schema.model_json_schema() — но он сам по себе валидный
    # JSON, начинающийся с {"description": ...}, и модели (в т.ч. 30B) иногда путались и
    # начинали ЭХОМ повторять схему вместо того, чтобы заполнить её конкретными значениями.
    # Пример-заготовка того же вида, что ожидается на выходе, оказался надёжнее прямого дампа схемы.
    example = json.dumps(_example_instance(schema), ensure_ascii=False)
    return (
        "\n\nОтветь СТРОГО одним JSON-объектом, без пояснений и без markdown-обёртки ```, "
        f"в точности такой структуры (это лишь пример формата, не готовый ответ):\n{example}"
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def generate_structured(
    provider: LLMProvider,
    conversations: List[Conversation],
    schema: Type[T],
    sampling_params: Dict[str, Any],
    defaults: List[T],
) -> List[T]:
    """Structured output поверх ЛЮБОГО LLMProvider (провайдер-агностичный "function calling"):
    JSON-инструкция поверх переданного промпта -> Pydantic-валидация -> один repair-запрос
    при невалидном JSON -> безопасный дефолт с логированием, если и repair не помог.

    Это заменяет прежний парсинг сырого текста ответа ("YES"/"NO", подстрока "OK"),
    который был хрупкой точкой ветвления графа.

    `defaults[i]` используется, если модель так и не вернула валидный JSON для `conversations[i]`.
    """
    if len(defaults) != len(conversations):
        raise ValueError("defaults must have the same length as conversations")
    if not conversations:
        return []

    instruction = _json_instruction(schema)
    augmented: List[Conversation] = [
        [*conv[:-1], {"role": conv[-1]["role"], "content": conv[-1]["content"] + instruction}] for conv in conversations
    ]

    raw_responses = provider.generate(augmented, sampling_params)

    results: List[Optional[T]] = [None] * len(conversations)
    repair_indices: List[int] = []
    repair_conversations: List[Conversation] = []

    for i, raw in enumerate(raw_responses):
        try:
            results[i] = schema.model_validate_json(_extract_json(raw))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            repair_indices.append(i)
            repair_conversations.append(
                [
                    *augmented[i],
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"Твой предыдущий ответ был невалиден ({e}). Верни ТОЛЬКО валидный JSON по схеме, без пояснений.",
                    },
                ]
            )

    if repair_conversations:
        repair_responses = provider.generate(repair_conversations, sampling_params)
        for idx, raw in zip(repair_indices, repair_responses):
            try:
                results[idx] = schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                logger.error(
                    "Structured output repair failed for schema=%s, falling back to default: %s",
                    schema.__name__,
                    e,
                )

    return [r if r is not None else defaults[i] for i, r in enumerate(results)]
