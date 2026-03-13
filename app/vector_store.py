from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class VectorStore:
    """
    Minimal in-memory vector store for prototyping.

    - Uses a SentenceTransformer encoder.
    - Stores embeddings + original chunks.
    """

    model: SentenceTransformer
    embeddings: np.ndarray
    documents: List[str]

    @classmethod
    def from_texts(cls, texts: Sequence[str], model_name: str = "all-MiniLM-L6-v2") -> "VectorStore":
        model = SentenceTransformer(model_name)
        embeddings = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return cls(model=model, embeddings=embeddings, documents=list(texts))

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        # avoid division by zero
        sims = (self.embeddings @ query_vec) / np.where(norms == 0, 1e-10, norms)
        idx = np.argsort(-sims)[:k]
        return [(self.documents[i], float(sims[i])) for i in idx]


__all__ = ["VectorStore"]

