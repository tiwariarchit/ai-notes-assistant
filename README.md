AI Notes Search Engine
======================

A minimal prototype for a local Retrieval-Augmented Generation (RAG) style notes assistant over your PDFs.

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


