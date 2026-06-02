"""Streamlit chat UI for mi-copilot.

Launch via the CLI:
    mi-copilot ui

Or directly:
    streamlit run mi_copilot/llm_assistant/app.py
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from mi_copilot.llm_assistant.chat import answer_stream
from mi_copilot.llm_assistant.retrieve import retrieve

load_dotenv()

st.set_page_config(
    page_title="mi-copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Sidebar ============
with st.sidebar:
    st.title("🤖 mi-copilot")
    st.caption("RAG-based LLM assistant for Mechanistic Interpretability")
    st.divider()

    st.subheader("⚙️ Retrieval settings")
    top_k = st.slider("Top-K chunks", 3, 12, 6,
                      help="More chunks = more context, but more cost & potential noise.")
    source_options = {
        "All sources (default)": None,
        "TransformerLens docs": "transformerlens",
        "MI papers (Nanda, Charton, ...)": "papers",
        "mi-copilot's own docs": "mi_copilot_repo",
    }
    source_label = st.selectbox("Source filter", list(source_options.keys()))
    source_filter = source_options[source_label]

    st.divider()
    st.subheader("📊 Knowledge base")
    st.markdown(
        "- **859 chunks** in ChromaDB\n"
        "- 4 MI papers (Nanda, Charton, Friedman, Weiss)\n"
        "- 21 TransformerLens docs\n"
        "- mi-copilot's own README + outlines\n"
    )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("📦 [GitHub](https://github.com/j43hyun9/mi-copilot)")
    st.caption("LLM: OpenAI GPT-4.1 · Embed: multilingual-e5-small")


# ============ Main ============
st.title("MI 분야 질의 어시스턴트")
st.markdown(
    "Mechanistic Interpretability 분야의 논문 · 도구 · 재현법을 자연어로 묻고, "
    "출처가 인용된 답변을 받으세요. 한국어/영어 모두 지원합니다."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example queries hint when empty
if not st.session_state.messages:
    with st.expander("💡 예시 질문", expanded=False):
        st.markdown(
            "- Nanda 2023 의 modular addition 회로 분석을 어떻게 재현하나요?\n"
            "- TransformerLens 의 HookedTransformer 는 어떻게 사용하나요?\n"
            "- Charton 의 GCD 모델은 어떤 알고리즘을 사용하나요?\n"
            "- What is the difference between activation patching and ablation?\n"
            "- Grokking 현상에서 progress measure 가 왜 필요한가요?\n"
        )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 검색된 출처 — {len(msg['sources'])} chunks"):
                for h in msg["sources"]:
                    st.markdown(f"**`{h.source}`** · `{h.file}` · chunk #{h.chunk_idx}  "
                                f"(distance: {h.distance:.3f})")
                    st.markdown(f"> {h.text[:400]}{'...' if len(h.text) > 400 else ''}")
                    st.divider()


# ============ Input ============
if prompt := st.chat_input("질문을 입력하세요... (한국어/영어 모두 가능)"):
    # User turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant turn — stream
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 검색 중..."):
                hits, token_gen = answer_stream(
                    prompt, k=top_k, source_filter=source_filter
                )
            full_response = st.write_stream(token_gen)

            with st.expander(f"📚 검색된 출처 — {len(hits)} chunks", expanded=False):
                for h in hits:
                    st.markdown(f"**`{h.source}`** · `{h.file}` · chunk #{h.chunk_idx}  "
                                f"(distance: {h.distance:.3f})")
                    st.markdown(f"> {h.text[:400]}{'...' if len(h.text) > 400 else ''}")
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": hits,
            })
        except Exception as e:
            err = f"❌ Error: {e}"
            st.error(err)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err,
            })
