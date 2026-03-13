import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="AI Notes Assistant")

st.title("AI Notes Assistant")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF file")
question = st.text_input("Ask a question")

if uploaded_file:

    st.success("PDF uploaded successfully!")

    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Split by paragraphs instead of fixed size
    chunks = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    if question:

        question_embedding = model.encode([question])

        similarities = cosine_similarity(question_embedding, embeddings)

        top_indices = similarities[0].argsort()[-5:][::-1]

        st.write("### AI Answer")

        results = []

        for i in top_indices:
            chunk = chunks[i]

            # keyword filtering
            if any(word in chunk.lower() for word in question.lower().split()):
                results.append(chunk)

        if results:
            for r in results:
                st.write("• " + r)
        else:
            for i in top_indices:
                st.write("• " + chunks[i])