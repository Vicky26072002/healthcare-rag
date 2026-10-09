"""Streamlit chat UI for MedRAG.  Run: streamlit run app/streamlit_app.py"""

import streamlit as st

from medrag.config import Settings
from medrag.pipeline import RAGPipeline

st.set_page_config(page_title="MedRAG", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading index...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline.load(Settings.from_env())


st.title("🩺 MedRAG - Healthcare Knowledge Assistant")
st.caption(
    "Hybrid retrieval (dense embeddings + BM25 + cross-encoder re-ranking) over "
    "MedQuAD - 47K question-answer pairs from 12 NIH websites."
)

with st.sidebar:
    st.header("Retrieval settings")
    mode = st.radio("Mode", ["hybrid", "dense", "sparse"], index=0)
    top_k = st.slider("Top-k chunks", 1, 10, 5)
    rerank = st.toggle("Cross-encoder re-ranking", value=True)
    st.divider()
    st.caption("Built by VIGNESHKUMAR S")

try:
    pipeline = load_pipeline()
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nRun `python -m medrag ingest` first.")
    st.stop()

question = st.chat_input("Ask a health question, e.g. 'What are the warning signs of dengue?'")
if question:
    with st.chat_message("user"):
        st.write(question)
    has_reranker = pipeline.retriever.reranker is not None
    answer = pipeline.ask(question, top_k=top_k, mode=mode, rerank=rerank and has_reranker)
    with st.chat_message("assistant"):
        if answer.emergency_flags:
            st.error("Possible emergency symptoms detected. Call 112 / 108 or go to the nearest hospital.")
        st.write(answer.answer)
        with st.expander(f"Sources ({len(answer.sources)})"):
            for s in answer.sources:
                section = f" - {s.section}" if s.section else ""
                st.markdown(f"**[{s.index}] {s.source}{section}** (score {s.score})")
                if s.url:
                    st.caption(s.url)
                st.write(s.text)
        st.info(answer.disclaimer)
