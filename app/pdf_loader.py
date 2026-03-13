from pathlib import Path
from typing import Iterable, List


def load_pdfs(directory: Path) -> List[str]:
    """
    Load all PDF files from a directory and return their text contents.

    This is a thin abstraction so it can be swapped for more
    sophisticated loaders later.
    """
    import pypdf  # imported lazily to avoid hard dependency at import-time

    texts: List[str] = []
    directory = Path(directory)

    for pdf_path in directory.glob("*.pdf"):
        reader = pypdf.PdfReader(str(pdf_path))
        pages: Iterable[str] = (page.extract_text() or "" for page in reader.pages)
        texts.append("\n".join(pages))

    return texts


__all__ = ["load_pdfs"]

