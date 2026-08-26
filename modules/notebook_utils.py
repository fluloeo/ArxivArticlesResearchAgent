"""Презентационные хелперы для Jupyter/IPython.

Вынесены из ArticleProcessor: `modules.processing` теперь не тянет IPython как
обязательную зависимость (её импорт здесь лениво выполняется внутри функций),
поэтому основной пакет `modules` можно использовать в gRPC-сервере/CLI, где
IPython вообще не установлен.
"""

from typing import Any, Callable, Dict, Optional


def print_report(before: int, after: int) -> None:
    from IPython.display import Markdown, display

    display(Markdown(f"📝 **Preprocessing**: Секций было: `{before}`, стало: `{after}`"))


def visualize(data_dict: Dict[str, Any], token_counter_func: Optional[Callable[[str], int]] = None) -> None:
    """Визуализация чанков ArticleProcessor (с перекрытиями или без)."""
    from IPython.display import HTML, Markdown, display

    if not data_dict:
        print("No data to visualize.")
        return

    titles = list(data_dict.keys())
    values = list(data_dict.values())
    is_complex = isinstance(values[0], dict)

    total_len = sum(len(v["main_text"] if is_complex else v) for v in values)
    display(Markdown(f"**Всего фрагментов:** `{len(data_dict)}` | **Длина:** `{total_len}` симв.\n\n---"))

    for i, title in enumerate(titles):
        val = data_dict[title]
        past, main, future = ("", val, "") if not is_complex else (val["past_overlap"], val["main_text"], val["future_overlap"])

        tokens_info = f"`Токенов: {token_counter_func(main)}` | " if token_counter_func else ""

        display(Markdown(f"### *Chunk {i+1}*: {title}\n>{tokens_info}`Символов: {len(main)}`"))

        past_h = f"<span style='background-color: #f0f0f0; color: #888;'>{past}</span>" if past else ""
        future_h = f"<span style='background-color: #f0f0f0; color: #888;'>{future}</span>" if future else ""

        html = f"""<div style="font-size: 11px; line-height: 1.2; border: 1px solid #ddd; padding: 8px; background-color: #fff;">
                   {past_h}<span>{main}</span>{future_h}</div>"""
        display(HTML(html))
        display(Markdown("\n---\n"))
