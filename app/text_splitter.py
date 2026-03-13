from typing import Iterable, List


def simple_chunk(
    texts: Iterable[str],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    Naive text splitter for prototyping.

    Splits each input text into overlapping character chunks.
    """
    chunks: List[str] = []
    for text in texts:
        start = 0
        length = len(text)
        while start < length:
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= length:
                break
            start = end - chunk_overlap
    return chunks


__all__ = ["simple_chunk"]

