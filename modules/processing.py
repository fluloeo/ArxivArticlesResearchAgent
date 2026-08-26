from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ArticleProcessor:
    def __init__(self, tokenizer, min_tokens: int = 700, max_tokens: int = 2000, overlap_len: int = 250):
        """
        Класс для подготовки текста статьи к подаче в LLM.
        
        Args:
            tokenizer: Токенизатор.
            min_tokens: Минимальный порог (для слияния).
            max_tokens: Максимальный порог (для разбиения).
            overlap_len: Длина перекрытия (в символах).
        """
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_len = overlap_len

    def get_token_length(self, text: str) -> int:
        """Возвращает длину текста в токенах."""
        return len(self.tokenizer.encode(text))

    def _merge_small_chunks(self, titles: List[str], chunks: List[str]) -> List[Tuple[str, str]]:
        """Внутренний метод для слияния мелких секций."""
        if not chunks:
            return []
        processed_titles = titles[:]
        processed_chunks = chunks[:]
        separator = "\n\n"
        i = 0
        
        while i < len(processed_chunks):
            current_len = self.get_token_length(processed_chunks[i])
            if current_len >= self.min_tokens or len(processed_chunks) == 1:
                i += 1
                continue
            
            left_len = self.get_token_length(processed_chunks[i-1]) if i > 0 else float('inf')
            right_len = self.get_token_length(processed_chunks[i+1]) if i < len(processed_chunks) - 1 else float('inf')

            if left_len == float('inf') and right_len == float('inf'):
                i += 1
                continue

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
                
        return list(zip(processed_titles, processed_chunks))

    def _split_large_chunks(self, merged_list: List[tuple]) -> Dict[str, str]:
        """Разбиение слишком больших секций с контролем названий."""
        final_dict = {}
        
        # Настраиваем сплиттер: считаем по токенам, оверлап 0 (будем делать свой позже)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=0,
            length_function=self.get_token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        for title, text in merged_list:
            tokens_count = self.get_token_length(text)
            if tokens_count > self.max_tokens:
                sub_chunks = splitter.split_text(text)
                for idx, sub_text in enumerate(sub_chunks):
                    new_title = f"{title} (Part {idx + 1})"
                    final_dict[new_title] = sub_text
            else:
                final_dict[title] = text
                
        return final_dict

    def process(self, data_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Полный цикл: Слияние мелких -> Разбиение крупных.
        """
        initial_titles = list(data_dict.keys())
        initial_chunks = list(data_dict.values())
        merged_list = self._merge_small_chunks(initial_titles, initial_chunks)
        return self._split_large_chunks(merged_list)

    def create_overlap_dict(self, data_dict: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Создает структуру с контекстными перекрытиями (past/future)."""
        titles = list(data_dict.keys())
        chunks = list(data_dict.values())
        result = {}

        for i in range(len(chunks)):
            result[titles[i]] = {
                "past_overlap": chunks[i-1][-self.overlap_len:] if (i > 0 and len(chunks[i-1]) > 0) else "",
                "main_text": chunks[i],
                "future_overlap": chunks[i+1][:self.overlap_len] if (i < len(chunks) - 1 and len(chunks[i+1]) > 0) else ""
            }
        return result