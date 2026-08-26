import os
import sys
from pathlib import Path

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


@st.cache_resource
def get_stub():
    channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
    return arxiv_agent_pb2_grpc.ArxivAgentServiceStub(channel)


def call_rpc(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except grpc.RpcError as e:
        detail = e.details() if hasattr(e, "details") else str(e)
        return None, f"gRPC-бэкенд недоступен или вернул ошибку: {detail}\n\nЗапущен ли `python -m grpc_service.server`?"


def render_response(response) -> None:
    if response.error:
        st.error(response.error)
        return

    st.markdown(response.final_answer)

    if response.sources:
        with st.expander("📎 Источники", expanded=True):
            for source in response.sources:
                st.write("-", source)

    if response.HasField("faithfulness") or response.HasField("answer_relevancy"):
        with st.expander("📊 RAGAS-метрики"):
            if response.HasField("faithfulness"):
                st.metric("Faithfulness", f"{response.faithfulness:.2f}")
            if response.HasField("answer_relevancy"):
                st.metric("Answer Relevancy", f"{response.answer_relevancy:.2f}")

    if response.tool_calls:
        with st.expander("🛠️ Вызовы инструментов (function calling)"):
            for call in response.tool_calls:
                st.code(call)


st.title("📚 ArXiv Research Agent")
st.caption("Суммаризация статей (Map-Reduce + RAGAS) и research-агент с function calling поверх arXiv API.")

if "pending_candidates" not in st.session_state:
    st.session_state.pending_candidates = None

query = st.text_input(
    "Ваш запрос", placeholder="Например: «сделай обзор статьи 1706.03762» или «что такое dropout?»"
)
compute_metrics = st.checkbox(
    "Считать RAGAS-метрики (faithfulness / answer relevancy)",
    value=True,
    help="Требует дополнительных LLM-вызовов (разбор ответа на утверждения + проверка каждого "
    "по контексту) — заметно увеличивает время ответа. Выключите для более быстрого ответа.",
)
ask_clicked = st.button("Спросить", type="primary")

if ask_clicked and not query.strip():
    st.warning("Введите запрос.")

elif ask_clicked:
    stub = get_stub()
    with st.spinner("Агент работает — это может занять несколько минут (поиск/загрузка статей, генерация, RAGAS)..."):
        response, error = call_rpc(
            stub.Ask, arxiv_agent_pb2.AskRequest(query=query, skip_metrics=not compute_metrics), timeout=900
        )

    if error:
        st.error(error)
        st.session_state.pending_candidates = None
    elif response.candidates:
        st.session_state.pending_candidates = list(response.candidates)
    else:
        st.session_state.pending_candidates = None
        render_response(response)

if st.session_state.pending_candidates:
    st.subheader("Нашёл несколько статей — выберите, какую суммаризировать:")
    candidates = st.session_state.pending_candidates
    labels = [f"{c.arxiv_id}: {c.title}" for c in candidates]
    choice_idx = st.radio(
        "Статьи",
        options=range(len(candidates)),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
    )
    abstract = candidates[choice_idx].abstract
    st.caption(abstract[:400] + "…" if len(abstract) > 400 else abstract)

    if st.button("Суммаризировать выбранную статью", type="primary"):
        stub = get_stub()
        chosen_id = candidates[choice_idx].arxiv_id
        with st.spinner(f"Суммаризирую {chosen_id} — это может занять несколько минут..."):
            response, error = call_rpc(
                stub.SummarizeArticle,
                arxiv_agent_pb2.SummarizeArticleRequest(article_id=chosen_id, skip_metrics=not compute_metrics),
                timeout=900,
            )
        st.session_state.pending_candidates = None
        if error:
            st.error(error)
        else:
            render_response(response)
