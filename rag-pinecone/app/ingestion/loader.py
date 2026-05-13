"""
Document loader supporting local files and web URLs.

Supports:
- PDF files via PyMuPDF
- Plain text (.txt) and Markdown (.md) files
- Web URLs via httpx + BeautifulSoup
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A loaded document with its text content and metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


def load_from_file(file_path: str) -> list[Document]:
    """Load documents from a local file path.

    Args:
        file_path: Absolute or relative path to a file.

    Returns:
        List of Document objects with source metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".txt", ".md"):
        return _load_text(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Supported: .pdf, .txt, .md"
        )


def load_from_url(url: str, timeout: float = 30.0) -> list[Document]:
    """Load documents from a web URL.

    Args:
        url: The URL to fetch and parse.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of Document objects with source metadata.
    """
    logger.info("Fetching URL: %s", url)
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text.strip():
        logger.warning("No text extracted from URL: %s", url)
        return []

    title = soup.title.string if soup.title else url

    return [
        Document(
            text=text,
            metadata={
                "source": url,
                "title": title,
                "page_number": 1,
            },
        )
    ]


def load(source: str) -> list[Document]:
    """Auto-detect source type and load documents.

    Args:
        source: A file path or URL string.

    Returns:
        List of Document objects.
    """
    if source.startswith(("http://", "https://")):
        return load_from_url(source)
    return load_from_file(source)


# ── Private helpers ──────────────────────────────────────────


def _load_pdf(path: Path) -> list[Document]:
    """Extract text from a PDF file using PyMuPDF."""
    import fitz  # PyMuPDF

    docs: list[Document] = []
    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if text.strip():
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": str(path),
                            "page_number": page_num,
                            "total_pages": len(pdf),
                        },
                    )
                )
    logger.info("Loaded %d pages from PDF: %s", len(docs), path)
    return docs


def _load_text(path: Path) -> list[Document]:
    """Load a plain text or markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning("Empty file: %s", path)
        return []

    return [
        Document(
            text=text,
            metadata={
                "source": str(path),
                "page_number": 1,
            },
        )
    ]
