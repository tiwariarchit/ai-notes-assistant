from io import BytesIO
from typing import List

import pypdf
import streamlit as st

from app.llm import answer_with_context
from app.rag_pipeline import RagPipeline


st.set_page_config(page_title="AI Notes Search Engine", layout="wide")

st.title("🧠 AI Notes Search Engine")
st.write(
    "Upload your PDF notes, run semantic search over them, and get RAG-based answers powered by an LLM."
)


def extract_text_from_uploaded_pdfs(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    texts: List[str] = []
    for f in files:
        reader = pypdf.PdfReader(BytesIO(f.getvalue()))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        texts.append("\n".join(pages_text))
    return texts


with st.sidebar:
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop one or more PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=10, value=5, step=1)


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if uploaded_files:
    raw_texts = extract_text_from_uploaded_pdfs(uploaded_files)
    pipeline = RagPipeline()
    pipeline.build_from_texts(raw_texts)
    st.session_state.pipeline = pipeline
    st.success(f"Indexed {len(uploaded_files)} PDF file(s). You can ask questions below.")
else:
    st.info("Upload at least one PDF to get started.")


st.header("2. Ask a question")
question = st.text_input("Ask anything about your uploaded notes")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Ask AI", disabled=not bool(question.strip())):
        if st.session_state.pipeline is None:
            st.warning("Please upload PDFs first so I can index your notes.")
        else:
            with st.spinner("Thinking with your notes..."):
                contexts = st.session_state.pipeline.retrieve(question, k=top_k)
                answer = answer_with_context(question, contexts)

            st.subheader("Answer")
            st.write(answer)

with col2:
    st.subheader("Retrieved context")
    if st.session_state.pipeline is not None and question.strip():
        contexts = st.session_state.pipeline.retrieve(question, k=top_k)
        for i, chunk in enumerate(contexts, start=1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk.strip())

