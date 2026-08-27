import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import grpc
import streamlit as st

# `streamlit run ui/streamlit_app.py` кладёт в sys.path директорию скрипта (ui/), а не
# корень репозитория — добавляем его явно, чтобы импорт пакета grpc_service резолвился
# независимо от текущей рабочей директории.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grpc_service.generated import arxiv_agent_pb2, arxiv_agent_pb2_grpc  # noqa: E402

GRPC_HOST = os.environ.get("APP_GRPC_HOST", "localhost")
GRPC_PORT = os.environ.get("APP_GRPC_PORT", "50051")

st.set_page_config(page_title="ArXiv Research Agent", page_icon="📚", layout="centered")

# Точечный CSS поверх дефолтной темы Streamlit — своих виджетов не заменяет (текстовые
# поля/кнопки/статус остаются нативными компонентами Streamlit ради стриминга и состояния),
# только визуальная полировка: типографика, hero-баннер, скругления, акцентный цвет.
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

.hero {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 1.1rem 1.4rem; margin-bottom: 1.1rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 55%, #0ea5e9 100%);
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
}
.hero-icon { font-size: 2.4rem; line-height: 1; }
.hero h1 { color: #fff; font-size: 1.5rem; margin: 0 0 0.15rem 0; font-weight: 700; }
.hero p { color: rgba(255,255,255,0.9); margin: 0; font-size: 0.92rem; }

.stButton > button {
    border-radius: 10px; font-weight: 600; border: none;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #4338ca, #6366f1); }

div[data-testid="stTextInput"] input {
    border-radius: 10px;
}

div[data-testid="stExpander"], div[data-baseweb="accordion"] {
    border-radius: 12px !important; overflow: hidden;
}

.streaming-cursor { opacity: 0.55; }

footer, #MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_stub():
    channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
    return arxiv_agent_pb2_grpc.ArxivAgentServiceStub(channel)


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "…"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}с"
    m, s = divmod(seconds, 60)
    return f"{m}м {s:02d}с"


def _progress_line(progress) -> str:
    """Строка в духе tqdm: при известном total — счётчик, процент и ETA по уже
    накопленному темпу; иначе — просто прошедшее время (для стадий без внутреннего цикла,
    например fetch_fulltext)."""
    if progress.total > 0:
        pct = 100 * progress.current / progress.total
        eta = _fmt_duration(progress.eta_s) if progress.eta_s >= 0 else "оцениваю…"
        return f"{progress.message} — {progress.current}/{progress.total} ({pct:.0f}%) · осталось ~{eta}"
    return f"{progress.message} · {_fmt_duration(progress.elapsed_s)} прошло"


def stream_response(rpc_iter: Iterator, status_label: str):
    """Итерирует server-streaming RPC (AskEvent), обновляя UI по ходу: живой статус-лог с
    прогрессом/ETA (аналог tqdm), печатающийся по мере готовности финальный текст, список
    map-выжимок по готовности. Возвращает финальный AskResponse либо None при ошибке
    (уже показанной пользователю)."""
    status = st.status(status_label, expanded=True)
    answer_placeholder = st.empty()
    map_expander = st.expander("🧩 Промежуточные выжимки по разделам (Map-стадия)", expanded=False)

    answer_text = ""
    final_response = None
    try:
        for event in rpc_iter:
            kind = event.WhichOneof("payload")
            if kind == "progress":
                line = _progress_line(event.progress)
                status.update(label=line)
                status.write(line)
            elif kind == "delta":
                answer_text += event.delta.text
                answer_placeholder.markdown(answer_text + ' <span class="streaming-cursor">▌</span>', unsafe_allow_html=True)
            elif kind == "map_summary":
                chunk = event.map_summary
                with map_expander:
                    st.markdown(f"**{chunk.title}**")
                    st.caption(chunk.summary)
            elif kind == "final":
                final_response = event.final
    except grpc.RpcError as e:
        status.update(label="Ошибка соединения", state="error")
        detail = e.details() if hasattr(e, "details") else str(e)
        st.error(f"gRPC-бэкенд недоступен или вернул ошибку: {detail}\n\nЗапущен ли `python -m grpc_service.server`?")
        return None

    if final_response is None:
        status.update(label="Поток завершился без ответа", state="error")
        st.error("Сервер закрыл соединение, не прислав финальный ответ.")
        return None

    if final_response.error:
        status.update(label="Ошибка агента", state="error")
        answer_placeholder.empty()
        st.error(final_response.error)
        return final_response

    status.update(label="Готово", state="complete", expanded=False)
    if final_response.final_answer:
        answer_placeholder.markdown(final_response.final_answer)
    else:
        answer_placeholder.empty()
    return final_response


def render_extras(response) -> None:
    if response.sources:
        with st.expander("📎 Источники", expanded=True):
            for source in response.sources:
                st.write("-", source)
    if response.tool_calls:
        with st.expander("🛠️ Вызовы инструментов (function calling)"):
            for call in response.tool_calls:
                st.code(call)


st.markdown(
    '<div class="hero"><span class="hero-icon">📚</span>'
    "<div><h1>ArXiv Research Agent</h1>"
    "<p>Суммаризация статей (Map-Reduce) и research-агент с function calling поверх arXiv API.</p></div></div>",
    unsafe_allow_html=True,
)

if "pending_candidates" not in st.session_state:
    st.session_state.pending_candidates = None

with st.container(border=True):
    query = st.text_input(
        "Ваш запрос",
        placeholder="Например: «сделай обзор статьи 1706.03762» или «что такое dropout?»",
        label_visibility="collapsed",
    )
    ask_clicked = st.button("Спросить →", type="primary", use_container_width=True)

if ask_clicked and not query.strip():
    st.warning("Введите запрос.")
elif ask_clicked:
    stub = get_stub()
    st.session_state.pending_candidates = None
    response = stream_response(
        stub.Ask(arxiv_agent_pb2.AskRequest(query=query), timeout=900), "Агент запускается…"
    )
    if response and not response.error:
        if response.candidates:
            st.session_state.pending_candidates = list(response.candidates)
        else:
            render_extras(response)

if st.session_state.pending_candidates:
    st.subheader("Нашёл несколько статей — выберите, какую суммаризировать:")
    candidates = st.session_state.pending_candidates
    labels = [f"{c.arxiv_id}: {c.title}" for c in candidates]
    choice_idx = st.radio(
        "Статьи", options=range(len(candidates)), format_func=lambda i: labels[i], label_visibility="collapsed"
    )
    with st.container(border=True):
        st.markdown("**Abstract**")
        st.write(candidates[choice_idx].abstract)

    if st.button("Суммаризировать выбранную статью", type="primary", use_container_width=True):
        stub = get_stub()
        chosen_id = candidates[choice_idx].arxiv_id
        st.session_state.pending_candidates = None
        response = stream_response(
            stub.SummarizeArticle(arxiv_agent_pb2.SummarizeArticleRequest(article_id=chosen_id), timeout=900),
            f"Суммаризирую {chosen_id}…",
        )
        if response and not response.error:
            render_extras(response)

st.markdown(
    '<p style="text-align:center; opacity:0.5; font-size:0.8rem; margin-top:2rem;">'
    "Ответ формируется потоково — узел за узлом, токен за токеном на стадии синтеза отчёта."
    "</p>",
    unsafe_allow_html=True,
)
