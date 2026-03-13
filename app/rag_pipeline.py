from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .pdf_loader import load_pdfs
from .text_splitter import simple_chunk
from .vector_store import VectorStore


@dataclass
class RagPipeline:
    """
    Simple end-to-end RAG pipeline:
    - Load PDFs (from disk or uploaded file bytes)
    - Split into chunks
    - Build vector store
    - Retrieve relevant chunks
    """

    data_dir: Path | None = None
    store: VectorStore | None = None

    def build_from_directory(self) -> None:
        if self.data_dir is None:
            raise ValueError("data_dir must be set to build from directory.")

        raw_docs: List[str] = load_pdfs(self.data_dir)
        chunks = simple_chunk(raw_docs)
        self.store = VectorStore.from_texts(chunks)

    def build_from_texts(self, texts: Sequence[str]) -> None:
        chunks = simple_chunk(texts)
        self.store = VectorStore.from_texts(chunks)

    def retrieve(self, question: str, k: int = 5) -> List[str]:
        if self.store is None:
            raise RuntimeError("RAG pipeline not built yet. Call build_*() first.")

        results = self.store.similarity_search(question, k=k)
        return [doc for doc, _score in results]


__all__ = ["RagPipeline"]

