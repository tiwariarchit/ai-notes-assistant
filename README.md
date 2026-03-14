# AI Notes Assistant

AI Notes Assistant is an AI-powered document search system that allows users to upload PDF files and ask questions about their content.

The system uses semantic search and vector embeddings to retrieve relevant information from documents. It demonstrates the concept of Retrieval-Augmented Generation (RAG) by combining document retrieval with AI-powered responses.

This project was built as part of an AI/ML internship evaluation to demonstrate vector search, semantic retrieval, and document-based question answering.


## Problem Statement

Large documents such as notes, research papers, and resumes contain valuable information but searching through them manually is inefficient. Traditional keyword search often fails when the query wording differs from the document text.

This project solves this problem using semantic search, allowing users to ask natural language questions and retrieve relevant information from documents.

Project layout
--------------

- `app/pdf_loader.py` – load PDF files from the `data` directory (CLI use).
- `app/text_splitter.py` – split raw text into overlapping character chunks.
- `app/vector_store.py` – build an in-memory vector store using `sentence-transformers`.
- `app/rag_pipeline.py` – simple end-to-end RAG pipeline; supports directory or in-memory texts.
- `app/llm.py` – small helper around OpenAI chat models for RAG answers.
- `ui.py` – Typer-based CLI entry point.
- `streamlit_app.py` – web UI for uploading PDFs and asking questions.
- `data/` – put your `.pdf` notes here if you use the CLI.

## System Architecture

PDF Upload  
↓  
Text Extraction (PyPDF)  
↓  
Text Chunking  
↓  
Vector Embeddings (Sentence Transformers)  
↓  
Similarity Search (Cosine Similarity)  
↓  
Relevant Context Retrieval  
↓  
Answer Display
Setup
-----

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key (for example on macOS / Linux):

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

4. (Optional) For CLI usage, add some PDF files into the `data/` directory.

Usage – Web app
---------------

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

In the browser:

- Upload one or more PDF notes.
- Ask any question about them.
- The app does semantic search over your notes and returns a RAG-based answer.

Usage – CLI (optional)
----------------------

Build and query in one go (in-memory only):

```bash
python ui.py ask "What are the main ideas in these notes?"
```

You can also just build (mainly for checking everything works):

```bash
python ui.py build
```


