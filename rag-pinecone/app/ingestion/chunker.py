"""
Recursive character text splitter with tiktoken-based token counting.

Splits documents into chunks of ≤512 tokens with ~50 token overlap,
preserving all source metadata on every chunk.
"""

from __future__ import annotations

import logging
from typing import Optional

import tiktoken

from app.ingestion.loader import Document

logger = logging.getLogger(__name__)

# Tokeniser used for counting — matches the encoding used by modern models.
ENCODING_NAME = "cl100k_base"
DEFAULT_CHUNK_SIZE = 512     # tokens
DEFAULT_CHUNK_OVERLAP = 50   # tokens

# Separators tried in order — recursive splitting strategy.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_length(text: str, encoding: tiktoken.Encoding) -> int:
    """Return the number of tokens in *text*."""
    return len(encoding.encode(text))


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding: tiktoken.Encoding,
    separators: list[str],
) -> list[str]:
    """Recursively split *text* into chunks of ≤ *chunk_size* tokens.

    Tries each separator in order, splitting on the first one that
    produces pieces small enough. If no separator works, falls back
    to character-level splitting.
    """
    # Base case — text already fits in one chunk
    if _token_length(text, encoding) <= chunk_size:
        return [text] if text.strip() else []

    # Pick the best separator
    separator = separators[-1]  # fallback
    for sep in separators:
        if sep in text:
            separator = sep
            break

    parts = text.split(separator) if separator else list(text)
    remaining_separators = (
        separators[separators.index(separator) + 1:]
        if separator in separators
        else separators[-1:]
    )

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for part in parts:
        part_length = _token_length(part, encoding)

        # If a single part exceeds chunk_size, split it recursively
        if part_length > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunk_text = separator.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

            # Recurse with finer separators
            sub_chunks = _split_text(
                part, chunk_size, chunk_overlap, encoding, remaining_separators
            )
            chunks.extend(sub_chunks)
            continue

        # Would adding this part overflow the chunk?
        separator_tokens = _token_length(separator, encoding) if current_chunk else 0
        if current_length + separator_tokens + part_length > chunk_size:
            # Flush current chunk
            if current_chunk:
                chunk_text = separator.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Keep overlap from the tail of the current chunk
                overlap_parts: list[str] = []
                overlap_len = 0
                for prev_part in reversed(current_chunk):
                    pl = _token_length(prev_part, encoding)
                    if overlap_len + pl > chunk_overlap:
                        break
                    overlap_parts.insert(0, prev_part)
                    overlap_len += pl

                current_chunk = overlap_parts
                current_length = overlap_len

        current_chunk.append(part)
        current_length += _token_length(part, encoding) + (
            _token_length(separator, encoding) if len(current_chunk) > 1 else 0
        )

    # Flush remaining
    if current_chunk:
        chunk_text = separator.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split a list of documents into smaller chunks.

    Args:
        documents: Source documents to split.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        A flat list of Document chunks with preserved + augmented metadata.
    """
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    chunks: list[Document] = []

    for doc in documents:
        text_chunks = _split_text(
            doc.text, chunk_size, chunk_overlap, encoding, SEPARATORS
        )
        for idx, chunk_text in enumerate(text_chunks):
            chunk_meta = {
                **doc.metadata,
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
            }
            chunks.append(Document(text=chunk_text, metadata=chunk_meta))

    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
