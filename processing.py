from typing import List, Dict, Any, Optional
from IPython.display import display, Markdown, HTML

class ArticleProcessor:
    def __init__(self, tokenizer, min_tokens: int = 700, overlap_len: int = 250):
        """
        Класс для подготовки текста статьи к подаче в LLM.
        
        Args:
            tokenizer: Токенизатор (например, из Transformers или vLLM).
            min_tokens: Порог объединения мелких секций.
            overlap_len: Длина контекстного перекрытия.
        """
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.overlap_len = overlap_len

    def get_token_length(self, text: str) -> int:
        """Возвращает длину текста в токенах."""
        return len(self.tokenizer.encode(text))

    def _merge_small_chunks(self, titles: List[str], chunks: List[str]) -> Dict[str, str]:
        """Внутренний метод для слияния мелких секций."""
        processed_titles = titles[:]
        processed_chunks = chunks[:]
        separator = " "
        i = 0
        
        while i < len(processed_chunks):
            current_len = self.get_token_length(processed_chunks[i])
            if current_len >= self.min_tokens or len(processed_chunks) == 1:
                i += 1
                continue
            
            left_len = self.get_token_length(processed_chunks[i-1]) if i > 0 else float('inf')
            right_len = self.get_token_length(processed_chunks[i+1]) if i < len(processed_chunks) - 1 else float('inf')
            
            if left_len < right_len:
                processed_titles[i-1] = f"{processed_titles[i-1]} + {processed_titles[i]}"
                processed_chunks[i-1] = f"{processed_chunks[i-1]}{separator}{processed_chunks[i]}"
                processed_titles.pop(i)
                processed_chunks.pop(i)
                i -= 1 
            else:
                processed_titles[i] = f"{processed_titles[i]} + {processed_titles[i+1]}"
                processed_chunks[i] = f"{processed_chunks[i]}{separator}{processed_chunks[i+1]}"
                processed_titles.pop(i+1)
                processed_chunks.pop(i+1)
                
        return dict(zip(processed_titles, processed_chunks))

    def process(self, data_dict: Dict[str, str], show_report: bool = True) -> Dict[str, str]:
        """
        Основной метод подготовки: слияние мелких секций.
        """
        titles = list(data_dict.keys())
        chunks = list(data_dict.values())
        
        token_lengths = [self.get_token_length(x) for x in chunks]
        
        # Условие: если есть слишком мелкие чанки (меньше порога)
        if any(length < self.min_tokens for length in token_lengths):
            final_dict = self._merge_small_chunks(titles, chunks)
            if show_report:
                self._print_report(len(chunks), len(final_dict))
            return final_dict
        
        return data_dict

    def create_overlap_dict(self, data_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Создает структуру с контекстными перекрытиями (past/future)."""
        titles = list(data_dict.keys())
        chunks = list(data_dict.values())
        result = {}

        for i in range(len(chunks)):
            result[titles[i]] = {
                "past_overlap": chunks[i-1][-self.overlap_len:] if i > 0 else "",
                "main_text": chunks[i],
                "future_overlap": chunks[i+1][:self.overlap_len] if i < len(chunks) - 1 else ""
            }
        return result

    def _print_report(self, before: int, after: int):
        """Выводит Markdown-отчет о слиянии."""
        display(Markdown(f"📝 **Preprocessing**: Секций было: `{before}`, стало: `{after}`"))

    @staticmethod
    def visualize(data_dict: Dict[str, Any], token_counter_func=None) -> None:
        """
        Статический метод для визуализации чанков. 
        Передаем функцию подсчета токенов извне, чтобы не зависеть от self.
        """
        titles = list(data_dict.keys())
        values = list(data_dict.values())
        is_complex = isinstance(values[0], dict)
        
        total_len = sum(len(v['main_text'] if is_complex else v) for v in values)
        display(Markdown(f"**Всего фрагментов:** `{len(data_dict)}` | **Длина:** `{total_len}` симв.\n\n---"))
        
        for i, title in enumerate(titles):
            val = data_dict[title]
            past, main, future = ("", val, "") if not is_complex else (val['past_overlap'], val['main_text'], val['future_overlap'])
            
            # Если передана функция для токенов - считаем
            tokens_info = f"`Токенов: {token_counter_func(main)}` | " if token_counter_func else ""
            
            display(Markdown(f"### *Chunk {i+1}*: {title}\n>{tokens_info}`Символов: {len(main)}`"))
            
            past_h = f"<span style='background-color: #f0f0f0; color: #888;'>{past}</span>" if past else ""
            future_h = f"<span style='background-color: #f0f0f0; color: #888;'>{future}</span>" if future else ""
            
            html = f"""<div style="font-size: 11px; line-height: 1.2; border: 1px solid #ddd; padding: 8px; background-color: #fff;">
                       {past_h}<span>{main}</span>{future_h}</div>"""
            display(HTML(html))
            display(Markdown("\n---\n"))